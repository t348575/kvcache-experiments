#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kv_connector_harness import (
    ScenarioRequestSpec,
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
        "--request-specs-json",
        type=str,
        default=None,
        help="Optional JSON file containing an explicit list of request specs.",
    )
    parser.add_argument(
        "--doc-size",
        type=str,
        default=None,
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
    parser.add_argument(
        "--model-name", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--csv-output", type=str, default=None)
    parser.add_argument(
        "--per-instance-workers",
        type=int,
        default=1,
        help="Number of worker threads used to execute this instance's request list.",
    )
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
        "scheduler_steps": sum(int(row["scheduler_steps"]) for row in rows),
        "total_scheduled_tokens": sum(
            int(row["total_scheduled_tokens"]) for row in rows
        ),
        "max_scheduled_tokens_per_step": max(
            (int(row["max_scheduled_tokens_per_step"]) for row in rows),
            default=0,
        ),
        "finished_sending": sum(1 for row in rows if row["finished_sending"]),
        "finished_recving": sum(1 for row in rows if row["finished_recving"]),
        "connector_load_ops": sum(int(row["connector_load_ops"]) for row in rows),
        "connector_load_bytes": sum(int(row["connector_load_bytes"]) for row in rows),
        "connector_load_time_s": sum(
            float(row["connector_load_time_s"]) for row in rows
        ),
        "connector_store_ops": sum(int(row["connector_store_ops"]) for row in rows),
        "connector_store_bytes": sum(int(row["connector_store_bytes"]) for row in rows),
        "connector_store_time_s": sum(
            float(row["connector_store_time_s"]) for row in rows
        ),
        "total_duration_s": sum(float(row["duration_s"]) for row in rows),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_request_specs(path: str) -> list[ScenarioRequestSpec]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("request specs JSON must contain a list")

    specs: list[ScenarioRequestSpec] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each request spec must be a JSON object")
        specs.append(
            ScenarioRequestSpec(
                request_id=int(item["request_id"]),
                doc_tokens=int(item["doc_tokens"]),
                reuse_source_id=(
                    None
                    if item.get("reuse_source_id") is None
                    else int(item["reuse_source_id"])
                ),
                reuse_prefix_len=int(item["reuse_prefix_len"]),
                prompt_token_ids=tuple(int(token) for token in item["prompt_token_ids"]),
                logical_request_id=(
                    None
                    if item.get("logical_request_id") is None
                    else int(item["logical_request_id"])
                ),
                prompt_key=(
                    None
                    if item.get("prompt_key") is None
                    else str(item["prompt_key"])
                ),
                traffic_scope=str(item.get("traffic_scope", "default")),
                target_instance=(
                    None
                    if item.get("target_instance") is None
                    else int(item["target_instance"])
                ),
            )
        )
    return specs


def split_specs_for_workers(
    specs: list[ScenarioRequestSpec], worker_count: int
) -> list[list[ScenarioRequestSpec]]:
    worker_specs: list[list[ScenarioRequestSpec]] = [[] for _ in range(worker_count)]
    for index, spec in enumerate(specs):
        worker_specs[index % worker_count].append(spec)
    return [chunk for chunk in worker_specs if chunk]


def run_worker_requests(
    config: RuntimeConnectorConfig,
    worker_index: int,
    specs: list[ScenarioRequestSpec],
) -> tuple[int, list[dict], str, str]:
    harness = RuntimeKVConnectorHarness(config)
    try:
        results = harness.run_requests(specs)
        placement = harness.placement
    finally:
        harness.shutdown()
    return (
        worker_index,
        results_to_rows(results),
        placement.runtime_kv_device,
        placement.runtime_kv_medium,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.per_instance_workers < 1:
        raise ValueError("--per-instance-workers must be >= 1")

    connector_extra_config = parse_json_dict(args.connector_extra_config)

    if args.request_specs_json:
        specs = load_request_specs(args.request_specs_json)
    else:
        if args.doc_size is None:
            raise ValueError("--doc-size is required unless --request-specs-json is provided")
        doc_size_range = parse_int_range(args.doc_size)
        prefix_reuse_range = parse_float_range(args.prefix_reuse)
        specs = build_request_specs(
            num_requests=args.num_requests,
            doc_size_range=doc_size_range,
            store_pct=args.store_pct,
            prefix_reuse_range=prefix_reuse_range,
            seed=args.seed,
        )

    if len(specs) != args.num_requests:
        raise ValueError(
            f"expected {args.num_requests} request specs, got {len(specs)}"
        )
    config = RuntimeConnectorConfig(
        connector_name=args.connector,
        connector_module_path=args.connector_module_path,
        connector_extra_config=connector_extra_config,
        model_name=args.model_name,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )

    worker_chunks = split_specs_for_workers(specs, args.per_instance_workers)
    worker_rows: list[tuple[int, list[dict], str, str]] = []
    with ThreadPoolExecutor(max_workers=len(worker_chunks)) as executor:
        futures = [
            executor.submit(run_worker_requests, config, worker_index, chunk)
            for worker_index, chunk in enumerate(worker_chunks)
        ]
        for future in futures:
            worker_rows.append(future.result())

    worker_rows.sort(key=lambda item: item[0])
    rows = [row for _, worker_result_rows, _, _ in worker_rows for row in worker_result_rows]
    rows.sort(key=lambda row: int(row["request_id"]))
    placement_device = worker_rows[0][2]
    placement_medium = worker_rows[0][3]
    summary = summarize(rows)
    summary["runtime_kv_device"] = placement_device
    summary["runtime_kv_medium"] = placement_medium
    summary["shared_requests"] = sum(1 for row in rows if row["traffic_scope"] == "shared")
    summary["unique_requests"] = sum(1 for row in rows if row["traffic_scope"] == "unique")
    summary["per_instance_workers"] = len(worker_chunks)

    print(f"Connector: {args.connector}")
    print(f"Requests: {summary['total_requests']}")
    print(f"Successful: {summary['successful_requests']}")
    print(f"Store requests: {summary['store_requests']}")
    print(f"Load requests: {summary['load_requests']}")
    print(f"Scheduler steps: {summary['scheduler_steps']}")
    print(f"Total scheduled tokens: {summary['total_scheduled_tokens']}")
    print(f"Max scheduled tokens/step: {summary['max_scheduled_tokens_per_step']}")
    print(f"Finished sending: {summary['finished_sending']}")
    print(f"Finished recving: {summary['finished_recving']}")
    print(f"Connector load ops: {summary['connector_load_ops']}")
    print(f"Connector load bytes: {summary['connector_load_bytes']}")
    print(f"Connector load time: {summary['connector_load_time_s']:.6f}s")
    print(f"Connector store ops: {summary['connector_store_ops']}")
    print(f"Connector store bytes: {summary['connector_store_bytes']}")
    print(f"Connector store time: {summary['connector_store_time_s']:.6f}s")
    print(f"Runtime KV device: {summary['runtime_kv_device']}")
    print(f"Runtime KV medium: {summary['runtime_kv_medium']}")
    print(f"Shared requests: {summary['shared_requests']}")
    print(f"Unique requests: {summary['unique_requests']}")
    print(f"Per-instance workers: {summary['per_instance_workers']}")
    print(f"Total duration: {summary['total_duration_s']:.3f}s")

    if args.csv_output:
        csv_path = str(Path(args.csv_output))
        write_csv(csv_path, rows)
        print(f"Wrote per-request CSV to {csv_path}")

    if args.json_output:
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
