#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kv_connector_harness import (
    RuntimeConnectorConfig,
    RuntimeKVConnectorHarness,
    build_request_specs,
    parse_float_range,
    parse_int_range,
    parse_json_dict,
    results_to_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scenario workloads against the same vLLM runtime KV connectors.",
    )
    parser.add_argument(
        "--connector", required=True, help="vLLM KV connector class name."
    )
    parser.add_argument(
        "--connector-module-path",
        default=None,
        help="Optional external module path for the connector class.",
    )
    parser.add_argument(
        "--connector-extra-config",
        default="{}",
        help="JSON object passed as kv_connector_extra_config.",
    )
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument(
        "--doc-size",
        type=str,
        required=True,
        help="Document size in tokens: single value or range like 4096-32768.",
    )
    parser.add_argument(
        "--store-pct",
        type=float,
        default=0.3,
        help="Fraction of requests that are fresh documents expected to be stored.",
    )
    parser.add_argument(
        "--prefix-reuse",
        type=str,
        default="1.0",
        help="Fraction of a prior document to reuse for load-style requests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="facebook/opt-125m")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--cache-dtype", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument(
        "--register-cache-mode",
        choices=["auto", "cross_layer", "layer_dict"],
        default="auto",
        help="How to initialize worker-side KV caches before running the connector.",
    )
    parser.add_argument(
        "--cross-layer-backend-path",
        default="vllm.v1.attention.backends.flash_attn:FlashAttentionBackend",
        help="Attention backend used when register-cache-mode is cross_layer.",
    )
    parser.add_argument("--csv-output", type=str, default=None)
    parser.add_argument("--json-output", action="store_true")
    return parser


def summarize(rows: list[dict]) -> dict:
    successful = [row for row in rows if row["successful"]]
    store_reqs = [row for row in rows if row["reuse_source_id"] is None]
    load_reqs = [row for row in rows if row["reuse_source_id"] is not None]
    return {
        "total_requests": len(rows),
        "successful_requests": len(successful),
        "store_requests": len(store_reqs),
        "load_requests": len(load_reqs),
        "finished_sending": sum(1 for row in rows if row["finished_sending"]),
        "finished_recving": sum(1 for row in rows if row["finished_recving"]),
        "total_duration_s": sum(float(row["duration_s"]) for row in rows),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()

    doc_size_range = parse_int_range(args.doc_size)
    prefix_reuse_range = parse_float_range(args.prefix_reuse)
    connector_extra_config = parse_json_dict(args.connector_extra_config)

    specs = build_request_specs(
        num_requests=args.num_requests,
        doc_size_range=doc_size_range,
        store_pct=args.store_pct,
        prefix_reuse_range=prefix_reuse_range,
        seed=args.seed,
    )
    config = RuntimeConnectorConfig(
        connector_name=args.connector,
        connector_module_path=args.connector_module_path,
        connector_extra_config=connector_extra_config,
        model_name=args.model_name,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        dtype=args.dtype,
        cache_dtype=args.cache_dtype,
        device=args.device,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        register_cache_mode=args.register_cache_mode,
        cross_layer_backend_path=args.cross_layer_backend_path,
    )

    harness = RuntimeKVConnectorHarness(config)
    try:
        results = harness.run_requests(specs)
    finally:
        harness.shutdown()

    rows = results_to_rows(results)
    summary = summarize(rows)

    print(f"Connector: {args.connector}")
    print(f"Requests: {summary['total_requests']}")
    print(f"Successful: {summary['successful_requests']}")
    print(f"Store requests: {summary['store_requests']}")
    print(f"Load requests: {summary['load_requests']}")
    print(f"Finished sending: {summary['finished_sending']}")
    print(f"Finished recving: {summary['finished_recving']}")
    print(f"Total duration: {summary['total_duration_s']:.3f}s")

    if args.csv_output:
        csv_path = str(Path(args.csv_output))
        write_csv(csv_path, rows)
        print(f"Wrote per-request CSV to {csv_path}")

    if args.json_output:
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
