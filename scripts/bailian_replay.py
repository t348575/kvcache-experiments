"""Replay a Bailian/Qwen production usage trace against a vLLM OpenAI endpoint.

Each trace line is reconstructed into a prompt whose block structure preserves the
trace's prefix reuse (identical block hashes -> identical token blocks), so reuse
in the trace drives real KV-cache reuse in the server. A request is classified as
"reuse" when its leading blocks were already emitted by an earlier request; the
summary splits TTFT into fresh vs reuse exactly like prefix_cache_benchmark.py, so
bench.py parses both with one code path.

Arrival models (pick one):
  --time-scale S    honor the trace's own per-request timestamps, multiplied by S
                    (default; S<1 compresses time, S>1 stretches it).
  --arrival-rate R  ignore trace timestamps; Poisson arrivals at R req/s.
  --closed-loop     ignore timestamps; fire all requests bounded only by
                    --max-concurrency (closed-loop, max in-flight = concurrency).

Tokenization needs the served model's tokenizer; reconstruction is offline and
GPU-free, so this script imports no vLLM/torch CUDA code.
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

from common.dataset import QWEN_TRACE_FILES, parse_qwen_trace_records  # noqa: E402
from common.prefix_cache_common import RequestSpec, get_base_url, run_benchmark  # noqa: E402


def resolve_eos_token_ids(tokenizer, model_name):
    """Collect every token id that would stop generation, so biasing against all of
    them forces a request to run to its full max_tokens.

    vLLM stops on any id in the model's generation_config.eos_token_id (chat models
    like Qwen3 carry several, e.g. <|im_end|> and <|endoftext|>), so the tokenizer's
    single eos_token_id is not enough. Pull both and dedupe; generation config is
    best-effort (skipped if unavailable).
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    try:
        from transformers import GenerationConfig

        gen_cfg = GenerationConfig.from_pretrained(model_name)
        eos = gen_cfg.eos_token_id
        if isinstance(eos, (list, tuple)):
            ids.update(int(i) for i in eos)
        elif eos is not None:
            ids.add(int(eos))
    except Exception as e:
        print(f"  (generation config EOS lookup skipped: {e})")
    return sorted(ids)


def build_specs(records, arrival_model, time_scale, arrival_rate, rng,
                use_dataset_output_len=False, max_model_len=0):
    """Attach a scheduled_time to each record per the chosen arrival model.

    scheduled_time is relative to benchmark start; run_benchmark sleeps until it.
    Reuse classification is carried via reuse_source_id/reuse_prefix_len so the
    resulting RequestResult.is_prefix_reuse is set without a second pass.

    When ``use_dataset_output_len`` is set, each request's max output tokens comes
    from the trace's own output_length (records with a missing/zero output_length
    fall back to the benchmark-wide --output-len). When ``max_model_len`` > 0 that
    per-request value is capped to ``max_model_len - prompt_tokens`` (min 1) so a
    long-output record cannot ask for more tokens than the context window allows.
    """
    specs = []
    base_ts = records[0].timestamp if records else 0.0
    elapsed = 0.0
    for rec in records:
        if arrival_model == "timestamp":
            scheduled = (rec.timestamp - base_ts) * time_scale
        elif arrival_model == "poisson":
            elapsed += rng.expovariate(arrival_rate)
            scheduled = elapsed
        else:  # closed-loop: no pacing, semaphore is the only limit
            scheduled = 0.0
        if use_dataset_output_len and rec.output_length > 0:
            output_len = rec.output_length
            if max_model_len > 0:
                output_len = min(output_len, max(1, max_model_len - len(rec.tokens)))
        else:
            output_len = None
        specs.append(
            RequestSpec(
                request_id=rec.request_id,
                doc_tokens=len(rec.tokens),
                reuse_source_id=0 if rec.is_reuse else None,
                reuse_prefix_len=rec.reuse_prefix_blocks * 16,
                scheduled_time=scheduled,
                output_len=output_len,
            )
        )
    return specs


def summarize(results, records, csv_output, json_output):
    import pandas as pd

    df = pd.DataFrame([r.__dict__ for r in results])
    rec_df = pd.DataFrame([r.__dict__ for r in records])
    if not rec_df.empty:
        df = df.merge(
            rec_df[["request_id", "turn", "req_type", "reuse_prefix_blocks", "chat_id"]],
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

    print("\n=== BAILIAN TRACE REPLAY ===")
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
    p.add_argument("--trace-path", required=True, help="Directory holding the qwen_*_blksz_16.jsonl files.")
    p.add_argument("--task", required=True, choices=sorted(QWEN_TRACE_FILES))
    p.add_argument("--tokenizer", default=None, help="Tokenizer name (defaults to --model once resolved).")
    p.add_argument("--max-requests", type=int, default=0, help="Cap requests replayed (0 = all).")
    p.add_argument("--offset", type=int, default=0, help="Skip the first N trace lines; replay the window starting here.")
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--output-len", type=int, default=1,
                   help="Max output tokens per request (fallback when --use-dataset-output-len is set but a record has no output_length).")
    p.add_argument("--max-model-len", type=int, default=0,
                   help="Context window; caps per-request dataset output_length to max-model-len minus prompt tokens (0 = no cap).")
    p.add_argument("--use-dataset-output-len", action="store_true",
                   help="Use each trace record's own output_length as its max output tokens (falls back to --output-len when 0/missing).")
    p.add_argument("--suppress-eos", action="store_true",
                   help="Bias against EOS so every request generates its full max_tokens (auto-derives EOS ids from the tokenizer + generation config; needed to match the dataset's output_length exactly).")
    arrival = p.add_mutually_exclusive_group()
    arrival.add_argument("--time-scale", type=float, default=1.0, help="Replay trace timestamps x this factor (default model).")
    arrival.add_argument("--arrival-rate", type=float, default=None, help="Override: Poisson arrivals at this req/s.")
    arrival.add_argument("--closed-loop", action="store_true", help="Override: no pacing; concurrency is the only limit.")
    p.add_argument("--completions", action="store_true")
    p.add_argument("--eos-token-id", type=int, default=None)
    p.add_argument("--csv-output", default=None)
    p.add_argument("--json-output", action="store_true")
    p.add_argument("--seed", type=int, default=42)
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

    trace_file = os.path.join(args.trace_path, f"{QWEN_TRACE_FILES[args.task]}.jsonl")
    print(f"Parsing trace: {trace_file}")
    records = parse_qwen_trace_records(
        trace_file, tokenizer, max_records=args.max_requests, offset=args.offset
    )
    n_reuse = sum(1 for r in records if r.is_reuse)
    print(
        f"Loaded {len(records)} requests ({n_reuse} reuse, {len(records) - n_reuse} fresh)"
        f" [offset={args.offset}, max_requests={args.max_requests}]"
    )

    if args.arrival_rate is not None:
        arrival_model = "poisson"
    elif args.closed_loop:
        arrival_model = "closed"
    else:
        arrival_model = "timestamp"
    print(f"Arrival model: {arrival_model}")

    specs = build_specs(
        records, arrival_model, args.time_scale, args.arrival_rate, rng,
        use_dataset_output_len=args.use_dataset_output_len,
        max_model_len=args.max_model_len,
    )
    if args.use_dataset_output_len:
        n_dataset = sum(1 for s in specs if s.output_len is not None)
        n_capped = sum(
            1 for s, r in zip(specs, records)
            if s.output_len is not None and s.output_len < r.output_length
        )
        msg = (f"Output tokens: dataset output_length for {n_dataset}/{len(specs)} reqs"
               f" (rest fall back to --output-len={args.output_len})")
        if args.max_model_len > 0:
            msg += f"; {n_capped} capped to max-model-len={args.max_model_len}"
        print(msg)
    prompts = [r.prompt for r in records]

    eos_token_id = args.eos_token_id
    if args.suppress_eos and eos_token_id is None:
        eos_token_id = resolve_eos_token_ids(tokenizer, args.tokenizer or model)
        print(f"Suppressing EOS ids {eos_token_id} so every request hits its full max_tokens")

    results = await run_benchmark(
        client=client,
        model=model,
        specs=specs,
        prompts=prompts,
        output_len=args.output_len,
        max_concurrency=args.max_concurrency,
        completions_mode=args.completions,
        eos_token_id=eos_token_id,
    )

    summarize(results, records, args.csv_output, args.json_output)


if __name__ == "__main__":
    asyncio.run(main())
