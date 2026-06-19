"""Replay SCBench sessions against a vLLM OpenAI endpoint as a multi-turn cache test.

SCBench (microsoft/SCBench) is shared-context: each session is one long context
plus a list of follow-up turns that reuse it. This replays a simplified multi-turn
mode: turn 0 sends ``context + Q0``; each later turn appends the prior turn's
golden answer and the next question to the whole previous prompt, so the previous
prompt is a byte-identical, cache-hittable prefix. A turn is classified "reuse"
when it trails an earlier turn of the same session; the summary splits TTFT into
fresh vs reuse like bailian_replay/longbench_v2_replay, so bench.py parses all
three with one code path.

This measures performance, not answer accuracy: output-len is small and the
generated answer is not scored. Golden (dataset) answers build the history, so the
prompt chain is deterministic and the run is offline (no answer feedback loop).

Arrival models (pick one):
  --closed-loop     default; no pacing, --max-concurrency is the only limit.
  --arrival-rate R  Poisson arrivals at R req/s.

Reconstruction is offline and GPU-free, so this script imports no vLLM/torch
CUDA code.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import sys
from pathlib import Path

# Make `common` importable when run as a standalone script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import SCBENCH_CONFIGS, parse_scbench_records  # noqa: E402
from common.prefix_cache_common import RequestSpec, get_base_url, run_benchmark  # noqa: E402


def load_parquet(parquet_dir, config):
    """Read a config's parquet (plus any .partNN shards) from a local dir."""
    import pandas as pd

    main = os.path.join(parquet_dir, f"{config}.parquet")
    shards = [main] + sorted(glob.glob(os.path.join(parquet_dir, f"{config}.part*.parquet")))
    shards = [s for s in shards if os.path.exists(s)]
    if not shards:
        raise SystemExit(
            f"No parquet for config {config!r} in {parquet_dir} "
            f"(run scripts/scbench_download.py first)."
        )
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def build_specs(records, arrival_model, arrival_rate, rng):
    """Attach a scheduled_time to each turn record per the chosen arrival model.

    Records arrive in emission order (turn k after turn k-1 of the same session),
    which run_benchmark's sequential dispatch preserves, so a reuse turn reaches the
    scheduler right behind its source.
    """
    specs = []
    prompts = []
    elapsed = 0.0
    for rec in records:
        if arrival_model == "poisson":
            elapsed += rng.expovariate(arrival_rate)
            scheduled = elapsed
        else:  # closed-loop
            scheduled = 0.0
        specs.append(
            RequestSpec(
                request_id=rec.request_id,
                doc_tokens=rec.prompt_token_len,
                reuse_source_id=rec.reuse_source_id,
                reuse_prefix_len=rec.reuse_prefix_len,
                scheduled_time=scheduled,
            )
        )
        prompts.append(rec.prompt)
    return specs, prompts


def print_spec_debug(specs, prompts, records):
    print("\n=== SPEC DEBUG (dispatch order) ===")
    header = (f"{'idx':>4} {'req_id':>7} {'sess':>5} {'turn':>5} {'source':>7} "
              f"{'prefix_len':>10} {'doc_tokens':>10} {'chars':>9}  kind")
    print(header)
    print("-" * len(header))
    for idx, (spec, prompt, rec) in enumerate(zip(specs, prompts, records)):
        source = spec.reuse_source_id if spec.reuse_source_id is not None else "-"
        prefix = spec.reuse_prefix_len if spec.reuse_prefix_len else "-"
        kind = "reuse" if spec.reuse_source_id is not None else "base"
        if kind == "base":
            print("-" * len(header))
        print(
            f"{idx:>4} {spec.request_id:>7} {rec.session_id:>5} {rec.turn:>5} "
            f"{str(source):>7} {str(prefix):>10} {spec.doc_tokens:>10} "
            f"{len(prompt):>9}  {kind}"
        )
    print("-" * len(header))
    base_tokens = sum(s.doc_tokens for s in specs if s.reuse_source_id is None)
    reuse_tokens = sum(s.doc_tokens for s in specs if s.reuse_source_id is not None)
    reused = sum(s.reuse_prefix_len or 0 for s in specs if s.reuse_source_id is not None)
    total = base_tokens + reuse_tokens
    print(f"Total requests: {len(specs)}")
    print(f"Base-turn tokens: {base_tokens}  Reuse-turn tokens: {reuse_tokens}")
    if total:
        print(f"Cache-reusable prefix tokens: {reused} ({reused / total:.1%} of {total} total)")


def summarize(results, records, csv_output, json_output):
    import pandas as pd

    df = pd.DataFrame([r.__dict__ for r in results])
    rec_df = pd.DataFrame([r.__dict__ for r in records])
    if not rec_df.empty:
        df = df.merge(
            rec_df[["request_id", "session_id", "turn", "config", "context_token_len"]],
            on="request_id",
            how="left",
        )

    successful = df[df["successful"]]
    reuse = successful[successful["is_prefix_reuse"]]
    fresh = successful[~successful["is_prefix_reuse"]]

    def mean(col, frame):
        return float(frame[col].mean()) if not frame.empty else None

    def q99(col, frame):
        return float(frame[col].quantile(0.99)) if not frame.empty else None

    total_time = (
        float(df["request_end"].max() - df["request_start"].min())
        if not df.empty
        else 0.0
    )

    print("\n=== SCBENCH REPLAY ===")
    print(f"  Total requests: {len(df)}  Successful: {len(successful)}")
    print(f"  Fresh: {len(fresh)}  Reuse: {len(reuse)}")
    if not successful.empty:
        print(f"  All   mean TTFT: {mean('ttft', successful):.3f}s")
    if not fresh.empty:
        print(f"  Fresh mean TTFT: {mean('ttft', fresh):.3f}s")
    if not reuse.empty:
        print(f"  Reuse mean TTFT: {mean('ttft', reuse):.3f}s")
    if not fresh.empty and not reuse.empty and mean("ttft", reuse):
        print(f"  TTFT speedup (fresh/reuse): {mean('ttft', fresh) / mean('ttft', reuse):.2f}x")
    print(f"  Wall-clock: {total_time:.3f}s")

    if csv_output:
        df.to_csv(csv_output, index=False)
        print(f"  Per-request data -> {csv_output}")

    if json_output:
        fresh_mean = mean("ttft", fresh)
        reuse_mean = mean("ttft", reuse)
        summary = {
            "total_requests": len(df),
            "successful": int(len(successful)),
            "fresh_count": int(len(fresh)),
            "reuse_count": int(len(reuse)),
            "all_mean_ttft": mean("ttft", successful),
            "all_median_ttft": (
                float(successful["ttft"].median()) if not successful.empty else None
            ),
            "all_p99_ttft": q99("ttft", successful),
            "fresh_mean_ttft": fresh_mean,
            "reuse_mean_ttft": reuse_mean,
            "reuse_p99_ttft": q99("ttft", reuse),
            "ttft_speedup": (
                fresh_mean / reuse_mean if fresh_mean and reuse_mean else None
            ),
            "wall_clock_s": total_time,
        }
        print(json.dumps(summary))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default="auto", help="Model name, or 'auto' to query /v1/models.")
    p.add_argument("--config", required=True, choices=SCBENCH_CONFIGS, help="SCBench subtask to replay.")
    p.add_argument("--parquet-dir", default="dataset/scbench", help="Dir holding <config>.parquet (default: dataset/scbench).")
    p.add_argument("--tokenizer", default=None, help="Tokenizer name (defaults to --model once resolved).")
    p.add_argument("--max-sessions", type=int, default=0, help="Cap sessions replayed (0 = all). Total requests is sessions x turns.")
    p.add_argument("--max-turns", type=int, default=0, help="Cap turns per session (0 = all).")
    p.add_argument("--max-context-tokens", type=int, default=0, help="Middle-truncate each context to this many tokens (0 = no truncation). Set below the model window.")
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--output-len", type=int, default=1)
    arrival = p.add_mutually_exclusive_group()
    arrival.add_argument("--closed-loop", action="store_true", default=True, help="No pacing; concurrency is the only limit (default).")
    arrival.add_argument("--arrival-rate", type=float, default=None, help="Override: Poisson arrivals at this req/s.")
    p.add_argument("--completions", action="store_true")
    p.add_argument("--eos-token-id", type=int, default=None)
    p.add_argument("--csv-output", default=None)
    p.add_argument("--json-output", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug-specs", action="store_true", help="Print the dispatch-order schedule and exit without sending requests.")
    return p


async def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    from openai import AsyncOpenAI

    base_url = get_base_url(args)
    client = AsyncOpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"), timeout=None)

    model = args.model
    if model == "auto":
        models = await client.models.list()
        model = models.data[0].id
        print(f"Auto-selected model: {model}")

    # Local import: transformers tokenizer is heavy and unneeded for --help.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model)

    df = load_parquet(args.parquet_dir, args.config)
    records = parse_scbench_records(
        df,
        args.config,
        tokenizer,
        max_sessions=args.max_sessions,
        max_turns=args.max_turns,
        max_context_tokens=args.max_context_tokens,
    )

    arrival_model = "poisson" if args.arrival_rate is not None else "closed"
    print(f"Arrival model: {arrival_model}")

    specs, prompts = build_specs(records, arrival_model, args.arrival_rate, rng)
    print(f"Sessions parsed -> {len(specs)} total requests")

    if args.debug_specs:
        print_spec_debug(specs, prompts, records)
        return

    results = await run_benchmark(
        client=client,
        model=model,
        specs=specs,
        prompts=prompts,
        output_len=args.output_len,
        max_concurrency=args.max_concurrency,
        completions_mode=args.completions,
        eos_token_id=args.eos_token_id,
        sequential_dispatch=True,
    )

    summarize(results, records, args.csv_output, args.json_output)


if __name__ == "__main__":
    asyncio.run(main())
