"""Replay LongBench v2 against a vLLM OpenAI endpoint as a long-document cache test.

Each of the 503 multiple-choice questions becomes one request whose prompt puts
the long context first, so the document is a reusable KV-cache prefix. A question
is classified "reuse" when an earlier question carried a byte-identical context;
the summary splits TTFT into fresh vs reuse like bailian_replay, so bench.py
parses both with one code path.

This measures performance, not answer accuracy: output-len is small and the
generated answer is not scored. Reuse is the dataset's own context sharing
(documents that back more than one question); the printed reuse fraction is data
dependent.

Every base document becomes a short multi-turn run: after the document, a random
--chain-followups run of follow-ups each append a random --chain-append-tokens
suffix of filler, so follow-up k sends the whole prompt of k-1 plus that suffix.
The prior prompt is then a full cache-hittable prefix, which is the reuse pattern
LongBench's own data barely produces. Follow-ups always run; the flags only set
their count and size. --max-base-docs caps the base documents, and the request
total is base docs plus their follow-ups.

Arrival models (pick one):
  --closed-loop     default; no pacing, --max-concurrency is the only limit
                    (LongBench has no timestamps, mirroring the official runner).
  --arrival-rate R  Poisson arrivals at R req/s.

Reconstruction is offline and GPU-free, so this script imports no vLLM/torch
CUDA code.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

# Make `common` importable when run as a standalone script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import parse_longbench_v2_records  # noqa: E402
from common.prefix_cache_common import (  # noqa: E402
    RequestSpec,
    get_base_url,
    parse_int_range,
    run_benchmark,
)

# Default chain shape when the --chain-* flags are omitted.
CHAIN_FOLLOWUPS_RANGE = (2, 5)
CHAIN_APPEND_TOKENS_RANGE = (500, 10000)


def _chain_filler(num_tokens):
    """Filler text appended to grow a chain prompt by ~num_tokens tokens.

    Content is irrelevant to the cache: only prefix identity and length drive
    block reuse, so one filler word ~= one token (the assumption _build_document
    in prefix_cache_common already makes). The leading newline keeps the suffix's
    first token from merging into the prior prompt's last token, so the shared
    prefix stays byte-identical down the chain.
    """
    return "\n" + " ".join(["hi"] * num_tokens)


def build_specs(records, arrival_model, arrival_rate, rng,
                followups_range=CHAIN_FOLLOWUPS_RANGE,
                append_range=CHAIN_APPEND_TOKENS_RANGE,
                max_context_tokens=0, two_phase=False):
    """Expand records into (specs, prompts), growing follow-ups off each base doc.

    Default mode chains: follow-up k sends the whole text of k-1 plus a random
    filler suffix, so the entire prior prompt is a cache-hittable prefix. The
    follow-up count and each suffix length are drawn from rng (seeded in main), so
    the schedule repeats across runs. A zero-width followups_range degenerates to
    one request per record (used by tests).

    `two_phase` flattens to single-hop reuse: every follow-up extends or trims its
    base doc directly (not the previous follow-up), and all base docs are emitted
    before any follow-up. Sending every base first lets their KV evict from the
    GPU before the reuse phase, so reuse hits the external offload tier instead of
    the GPU prefix cache (the path a back-to-back chain keeps GPU-resident).

    `max_context_tokens` (>0) caps the prompt. The base context is already
    middle-truncated to it at load time. A follow-up that would exceed the window
    trims that many tokens off the end of its source instead, so it stays a
    byte-identical prefix of the source (whole shorter prompt is a cache hit) and
    doc_tokens shrinks. This keeps every base doc producing follow-ups even when it
    already sits at the budget, where there is no headroom to grow.

    scheduled_time is assigned in emission order; run_benchmark's sequential
    dispatch sends specs in that order and sleeps until each scheduled_time.
    """
    next_followup_id = len(records)
    groups = []  # (base_pair, [followup_pair, ...]); pair = (spec, prompt)

    for rec in records:
        base_pair = (
            RequestSpec(
                request_id=rec.request_id,
                doc_tokens=rec.context_token_len,
                reuse_source_id=rec.reuse_source_id,
                reuse_prefix_len=rec.reuse_prefix_len,
            ),
            rec.prompt,
        )
        followups = []
        prev_id = rec.request_id
        prev_prompt = rec.prompt
        prev_tokens = rec.context_token_len
        n_followups = rng.randint(*followups_range) if followups_range[1] > 0 else 0
        for _ in range(n_followups):
            delta = rng.randint(*append_range)
            # two_phase reuses the base; chained mode reuses the previous follow-up.
            src_id = rec.request_id if two_phase else prev_id
            src_prompt = rec.prompt if two_phase else prev_prompt
            src_tokens = rec.context_token_len if two_phase else prev_tokens

            at_budget = max_context_tokens and src_tokens >= max_context_tokens
            if at_budget:
                if delta >= src_tokens:
                    break
                new_tokens = src_tokens - delta
                # Cut at a char boundary by the chars/token ratio so the kept text
                # stays a byte-identical prefix and its block hashes still match.
                keep_chars = int(len(src_prompt) * new_tokens / src_tokens)
                new_prompt = src_prompt[:keep_chars]
            else:
                if max_context_tokens:
                    delta = min(delta, max_context_tokens - src_tokens)
                if delta <= 0:
                    break
                new_tokens = src_tokens + delta
                new_prompt = src_prompt + _chain_filler(delta)
            followups.append((
                RequestSpec(
                    request_id=next_followup_id,
                    doc_tokens=new_tokens,
                    reuse_source_id=src_id,
                    # One prompt is always a prefix of the other, so the shared
                    # length is the shorter of the two.
                    reuse_prefix_len=min(new_tokens, src_tokens),
                ),
                new_prompt,
            ))
            prev_id = next_followup_id
            prev_prompt = new_prompt
            prev_tokens = new_tokens
            next_followup_id += 1
        groups.append((base_pair, followups))

    if two_phase:
        ordered = [g[0] for g in groups] + [f for g in groups for f in g[1]]
    else:
        ordered = [pair for g in groups for pair in (g[0], *g[1])]

    elapsed = 0.0
    specs = []
    prompts = []
    for spec, prompt in ordered:
        if arrival_model == "poisson":
            elapsed += rng.expovariate(arrival_rate)
            spec.scheduled_time = elapsed
        specs.append(spec)
        prompts.append(prompt)
    return specs, prompts


def print_spec_debug(specs, prompts):
    """Print the dispatch-order schedule, one line per request, so reuse chains
    can be eyeballed. `words` is a whitespace split, which tracks tokens only for
    the filler. Base docs and their follow-ups are grouped by blank separators.
    """
    print("\n=== SPEC DEBUG (dispatch order) ===")
    header = f"{'idx':>4} {'req_id':>7} {'source':>7} {'prefix_len':>10} {'doc_tokens':>10} {'words':>8} {'chars':>9}  kind"
    print(header)
    print("-" * len(header))
    for idx, (spec, prompt) in enumerate(zip(specs, prompts)):
        words = len(prompt.split())
        source = spec.reuse_source_id if spec.reuse_source_id is not None else "-"
        prefix = spec.reuse_prefix_len if spec.reuse_prefix_len is not None else "-"
        kind = "reuse" if spec.reuse_source_id is not None else "base"
        if kind == "base":
            print("-" * len(header))
        print(
            f"{idx:>4} {spec.request_id:>7} {str(source):>7} {str(prefix):>10} "
            f"{spec.doc_tokens:>10} {words:>8} {len(prompt):>9}  {kind}"
        )
    print("-" * len(header))
    base_tokens = sum(s.doc_tokens for s in specs if s.reuse_source_id is None)
    reuse_tokens = sum(s.doc_tokens for s in specs if s.reuse_source_id is not None)
    reused = sum(s.reuse_prefix_len or 0 for s in specs if s.reuse_source_id is not None)
    total = base_tokens + reuse_tokens
    print(f"Total requests: {len(specs)}")
    print(f"Base-doc tokens: {base_tokens}  Reuse-request tokens: {reuse_tokens}")
    print(f"Cache-reusable prefix tokens: {reused} ({reused / total:.1%} of {total} total)")
    print("Base-doc tokens must exceed GPU KV capacity to evict before reuse "
          "(~144 KB/token for Qwen3-4B fp16).")


def summarize(results, records, csv_output, json_output):
    import pandas as pd

    df = pd.DataFrame([r.__dict__ for r in results])
    rec_df = pd.DataFrame([r.__dict__ for r in records])
    if not rec_df.empty:
        df = df.merge(
            rec_df[["request_id", "domain", "difficulty", "context_token_len"]],
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

    print("\n=== LONGBENCH V2 REPLAY ===")
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default="auto", help="Model name, or 'auto' to query /v1/models.")
    p.add_argument("--tokenizer", default=None, help="Tokenizer name (defaults to --model once resolved).")
    p.add_argument("--max-base-docs", type=int, default=0, help="Cap base documents pulled from the dataset (0 = all). The request total is base docs plus their follow-ups.")
    p.add_argument("--chain-followups", default=f"{CHAIN_FOLLOWUPS_RANGE[0]}-{CHAIN_FOLLOWUPS_RANGE[1]}", help="Follow-ups per base doc as MIN-MAX (e.g. 2-5), drawn per doc. Each follow-up reuses the whole prior prompt as a cache prefix.")
    p.add_argument("--chain-append-tokens", default=f"{CHAIN_APPEND_TOKENS_RANGE[0]}-{CHAIN_APPEND_TOKENS_RANGE[1]}", help="Tokens appended at each follow-up as MIN-MAX, drawn per step.")
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--output-len", type=int, default=1)
    p.add_argument("--domain", default=None, help="Filter to one LongBench v2 domain.")
    p.add_argument("--difficulty", default=None, choices=["easy", "hard"], help="Filter to one difficulty.")
    p.add_argument("--max-context-tokens", type=int, default=0, help="Middle-truncate each context to this many tokens (0 = no truncation). Set below the model window.")
    arrival = p.add_mutually_exclusive_group()
    arrival.add_argument("--closed-loop", action="store_true", default=True, help="No pacing; concurrency is the only limit (default).")
    arrival.add_argument("--arrival-rate", type=float, default=None, help="Override: Poisson arrivals at this req/s.")
    p.add_argument("--completions", action="store_true")
    p.add_argument("--eos-token-id", type=int, default=None)
    p.add_argument("--csv-output", default=None)
    p.add_argument("--json-output", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--two-phase", action="store_true", help="Single-hop reuse: every follow-up reuses its base doc directly, and all base docs are sent before any reuse. Lets base KV evict from the GPU so reuse hits the external offload tier instead of the GPU prefix cache.")
    p.add_argument("--debug-specs", action="store_true", help="Print the dispatch-order schedule (request order, reuse source, prefix len, token/char length) and exit without sending requests.")
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

    records = parse_longbench_v2_records(
        tokenizer,
        max_entries=args.max_base_docs,
        domain=args.domain,
        difficulty=args.difficulty,
        max_context_tokens=args.max_context_tokens,
    )

    arrival_model = "poisson" if args.arrival_rate is not None else "closed"
    print(f"Arrival model: {arrival_model}")

    specs, prompts = build_specs(
        records,
        arrival_model,
        args.arrival_rate,
        rng,
        followups_range=parse_int_range(args.chain_followups),
        append_range=parse_int_range(args.chain_append_tokens),
        max_context_tokens=args.max_context_tokens,
        two_phase=args.two_phase,
    )
    print(f"Base docs: {len(records)}  Total requests: {len(specs)}"
          f"{'  (two-phase)' if args.two_phase else ''}")

    if args.debug_specs:
        print_spec_debug(specs, prompts)
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
