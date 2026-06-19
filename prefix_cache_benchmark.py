from __future__ import annotations

import argparse
import asyncio
import json
import os
import random

import pandas as pd
from openai import AsyncOpenAI
from common.prefix_cache_common import (
    RequestResult,
    RequestSpec,
    generate_request_schedule,
    get_base_url,
    parse_int_range,
    parse_pct_range,
    run_benchmark,
    send_request,
)


def print_summary(
    results: list[RequestResult], csv_output: str | None, json_output: bool
):
    df = pd.DataFrame([r.__dict__ for r in results])
    successful = df[df["successful"]]
    reuse = successful[successful["is_prefix_reuse"]]
    fresh = successful[~successful["is_prefix_reuse"]]

    CSI = "\x1b["
    RESET = CSI + "0m"

    print(f"\n{CSI}36;1m=== PREFIX CACHE BENCHMARK RESULTS ==={RESET}")
    print(f"  Total requests:        {len(df)}")
    print(f"  Successful:            {len(successful)}")
    print(f"  Fresh (no reuse):      {len(fresh)}")
    print(f"  Prefix-reuse:          {len(reuse)}")

    if not successful.empty:
        print(f"\n{CSI}36mAll requests:{RESET}")
        print(f"  Mean TTFT:   {successful['ttft'].mean():.3f}s")
        print(f"  Median TTFT: {successful['ttft'].median():.3f}s")
        print(f"  P99 TTFT:    {successful['ttft'].quantile(0.99):.3f}s")

    if not fresh.empty:
        print(f"\n{CSI}33mFresh requests:{RESET}")
        print(f"  Mean TTFT:   {fresh['ttft'].mean():.3f}s")
        print(f"  Median TTFT: {fresh['ttft'].median():.3f}s")
        print(f"  P99 TTFT:    {fresh['ttft'].quantile(0.99):.3f}s")

    if not reuse.empty:
        print(f"\n{CSI}32mPrefix-reuse requests:{RESET}")
        print(f"  Mean TTFT:   {reuse['ttft'].mean():.3f}s")
        print(f"  Median TTFT: {reuse['ttft'].median():.3f}s")
        print(f"  P99 TTFT:    {reuse['ttft'].quantile(0.99):.3f}s")
        print(
            f"  Mean reused prefix len: {reuse['reuse_prefix_len'].mean():.0f} tokens"
        )

    if not fresh.empty and not reuse.empty:
        speedup = fresh["ttft"].mean() / reuse["ttft"].mean()
        print(f"\n{CSI}35mTTFT speedup (fresh / reuse): {speedup:.2f}x{RESET}")

    total_time = df["request_end"].max() - df["request_start"].min()
    print(f"\n  Wall-clock time: {total_time:.3f}s")

    df["latency"] = df["request_end"] - df["request_start"]

    if csv_output:
        df.to_csv(csv_output, index=False)
        print(f"\n  Per-request data written to {csv_output}")

    if json_output:
        summary = {
            "total_requests": len(df),
            "successful": int(len(successful)),
            "all_mean_ttft": float(successful["ttft"].mean())
            if not successful.empty
            else None,
            "all_median_ttft": float(successful["ttft"].median())
            if not successful.empty
            else None,
            "all_p99_ttft": float(successful["ttft"].quantile(0.99))
            if not successful.empty
            else None,
            "fresh_mean_ttft": float(fresh["ttft"].mean()) if not fresh.empty else None,
            "reuse_mean_ttft": float(reuse["ttft"].mean()) if not reuse.empty else None,
            "ttft_speedup": float(fresh["ttft"].mean() / reuse["ttft"].mean())
            if not fresh.empty and not reuse.empty
            else None,
            "wall_clock_s": float(total_time),
        }
        print(json.dumps(summary))


def print_schedule(specs) -> None:
    CSI = "\x1b["
    RESET = CSI + "0m"
    YELLOW = CSI + "33m"
    GREEN = CSI + "32m"

    children: dict[int, list[int]] = {}
    for s in specs:
        if s.reuse_source_id is not None:
            children.setdefault(s.reuse_source_id, []).append(s.request_id)

    print(f"\n{CSI}36;1m=== REQUEST SCHEDULE ==={RESET}")
    print(
        "  trace req_id contains the 'vllm_id' token below "
        "(full id is chatcmpl-<vllm_id> or cmpl-<vllm_id>)."
    )
    print(
        f"  {'req_id':>7}  {'vllm_id':<12}  {'kind':<5}  "
        f"{'parent':>7}  {'prefix_len':>10}  reused_by"
    )
    for s in specs:
        if s.reuse_source_id is not None:
            color, label = GREEN, "REUSE"
            parent = str(s.reuse_source_id)
            prefix = str(s.reuse_prefix_len or 0)
        else:
            color, label = YELLOW, "FRESH"
            parent = "-"
            prefix = "-"
        reused_by = ",".join(str(c) for c in children.get(s.request_id, [])) or "-"
        kind = f"{color}{label:<5}{RESET}"
        vllm_id = f"bench-{s.request_id}"
        print(
            f"  {s.request_id:>7}  {vllm_id:<12}  {kind}  "
            f"{parent:>7}  {prefix:>10}  {reused_by}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prefix-caching benchmark for LLM serving engines.",
    )

    p.add_argument("--host", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument(
        "--model",
        type=str,
        default="auto",
        help="Model name, or 'auto' to query /v1/models.",
    )
    p.add_argument(
        "--num-requests",
        type=int,
        required=True,
        help="Total number of requests to send.",
    )
    p.add_argument(
        "--doc-size",
        type=str,
        required=True,
        help="Document size in tokens: single int or lo-hi range "
        "(e.g. 10240 or 10240-40960).",
    )
    p.add_argument(
        "--prefix-reuse-pct",
        type=float,
        default=0.0,
        help="Fraction of requests (0.0-1.0) that reuse a prefix from a prior request.",
    )
    p.add_argument(
        "--prefix-size",
        type=str,
        default="0.5",
        help="Reused prefix as a fraction of the source doc size "
        "(0.0-1.0): single value or lo-hi range "
        "(e.g. 0.5 or 0.25-0.75).",
    )
    p.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Poisson arrival rate in requests/second.  If omitted, "
        "all requests are sent as fast as concurrency allows.",
    )
    p.add_argument(
        "--output-len", type=int, default=1, help="Max tokens to generate per request."
    )
    p.add_argument(
        "--pre-warmup-requests",
        type=int,
        default=5,
        help="Number of untimed requests to send before the measured benchmark. "
        "Use 0 to disable.",
    )
    p.add_argument(
        "--pre-warmup-doc-size",
        type=int,
        default=256,
        help="Document size in tokens for each untimed pre-warmup request.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum number of in-flight requests.",
    )
    p.add_argument(
        "--completions",
        action="store_true",
        help="Use completions API instead of chat completions.",
    )
    p.add_argument(
        "--eos-token-id", type=int, default=None, help="EOS token id to bias against."
    )
    p.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Write per-request CSV to this path.",
    )
    p.add_argument(
        "--json-output", action="store_true", help="Print JSON summary line to stdout."
    )
    p.add_argument(
        "--print-schedule",
        action="store_true",
        help="Print each request id, whether it is fresh or a prefix-reuse, "
        "its parent (reuse source) id, and the reused prefix length.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    return p


async def run_pre_warmup_requests(
    client: AsyncOpenAI,
    model: str,
    count: int,
    doc_tokens: int,
    output_len: int,
    completions_mode: bool,
    eos_token_id: int | None,
) -> None:
    for request_id in range(count):
        spec = RequestSpec(
            request_id=-(request_id + 1),
            doc_tokens=doc_tokens,
            scheduled_time=0.0,
        )
        prompt = f"pre-warmup-{request_id} " + " ".join(["hi"] * doc_tokens)
        await send_request(
            client=client,
            model=model,
            prompt=prompt,
            spec=spec,
            output_len=output_len,
            completions_mode=completions_mode,
            eos_token_id=eos_token_id,
            benchmark_start=0.0,
        )


async def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.host is not None and args.port is not None and args.base_url is not None:
        parser.error("Cannot use --host/--port and --base-url together.")

    rng = random.Random(args.seed)
    doc_lo, doc_hi = parse_int_range(args.doc_size)
    pfx_lo, pfx_hi = parse_pct_range(args.prefix_size)

    base_url = get_base_url(args)
    print(f"Base URL: {base_url}")

    api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=None)

    model = args.model
    if model == "auto":
        models = await client.models.list()
        model = models.data[0].id
        print(f"Auto-selected model: {model}")

    if args.pre_warmup_requests < 0:
        parser.error("--pre-warmup-requests must be >= 0")
    if args.pre_warmup_doc_size < 1:
        parser.error("--pre-warmup-doc-size must be >= 1")

    if args.pre_warmup_requests:
        print(
            f"Sending {args.pre_warmup_requests} untimed pre-warmup request(s) "
            f"({args.pre_warmup_doc_size} tokens each)"
        )
        await run_pre_warmup_requests(
            client=client,
            model=model,
            count=args.pre_warmup_requests,
            doc_tokens=args.pre_warmup_doc_size,
            output_len=args.output_len,
            completions_mode=args.completions,
            eos_token_id=args.eos_token_id,
        )

    specs = generate_request_schedule(
        num_requests=args.num_requests,
        doc_size_lo=doc_lo,
        doc_size_hi=doc_hi,
        prefix_reuse_pct=args.prefix_reuse_pct,
        prefix_pct_lo=pfx_lo,
        prefix_pct_hi=pfx_hi,
        arrival_rate=args.arrival_rate,
        rng=rng,
    )

    n_reuse = sum(1 for s in specs if s.reuse_source_id is not None)
    print(
        f"Scheduled {len(specs)} requests "
        f"({n_reuse} prefix-reuse, {len(specs) - n_reuse} fresh)"
    )
    if args.arrival_rate:
        total_span = specs[-1].scheduled_time if specs else 0
        print(
            f"Poisson arrival rate: {args.arrival_rate} req/s  "
            f"(schedule spans {total_span:.1f}s)"
        )

    if args.print_schedule:
        print_schedule(specs)

    results = await run_benchmark(
        client=client,
        model=model,
        specs=specs,
        output_len=args.output_len,
        max_concurrency=args.max_concurrency,
        completions_mode=args.completions,
        eos_token_id=args.eos_token_id,
    )

    print_summary(results, args.csv_output, args.json_output)


if __name__ == "__main__":
    asyncio.run(main())
