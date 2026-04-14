#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from prefix_cache_common import (
    RequestSpec,
    generate_request_schedule,
    parse_int_range,
    parse_pct_range,
    run_benchmark,
)
from vllm_cluster_common import (
    INSTANCE_GPUS,
    INSTANCE_PORTS,
    VLLM_SERVER_CONFIG,
    build_servers,
    collect_per_instance_metrics,
    fetch_model_name,
    launch_cluster,
    parse_csv_floats,
    scrape_metrics,
    stop_cluster,
    summarise_results,
    write_request_csv,
)


def build_instance_specs(
    instance_id: int,
    requests_per_instance: int,
    doc_lo: int,
    doc_hi: int,
    prefix_reuse_pct: float,
    prefix_pct_lo: float,
    prefix_pct_hi: float,
    arrival_rate: float | None,
    scheduled_offset: float,
    request_id_offset: int,
) -> list[RequestSpec]:
    specs = generate_request_schedule(
        num_requests=requests_per_instance,
        doc_size_lo=doc_lo,
        doc_size_hi=doc_hi,
        prefix_reuse_pct=prefix_reuse_pct,
        prefix_pct_lo=prefix_pct_lo,
        prefix_pct_hi=prefix_pct_hi,
        arrival_rate=arrival_rate,
        rng=random.Random(1000 + instance_id),
    )
    for index, spec in enumerate(specs):
        spec.request_id = request_id_offset + index
        spec.scheduled_time += scheduled_offset
    return specs


async def run_instance_workload(
    base_url: str,
    model: str,
    specs: list[RequestSpec],
    output_len: int,
    max_concurrency: int,
    completions_mode: bool,
    eos_token_id: int | None,
) -> list[dict[str, Any]]:
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"),
        timeout=None,
    )
    results = await run_benchmark(
        client=client,
        model=model,
        specs=specs,
        output_len=output_len,
        max_concurrency=max_concurrency,
        completions_mode=completions_mode,
        eos_token_id=eos_token_id,
    )
    return [asdict(result) for result in results]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scenario A benchmark: isolated storage baseline per vLLM instance."
    )
    parser.add_argument("--storage-root", type=str, required=True)
    parser.add_argument("--instances", type=int, default=4, choices=[1, 4])
    parser.add_argument("--requests-per-instance", type=int, default=32)
    parser.add_argument("--doc-size", type=str, default="4096")
    parser.add_argument("--prefix-reuse-pct", type=float, default=0.0)
    parser.add_argument("--prefix-size", type=str, default="0.5")
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--start-skews", type=str, default="0,0,0,0")
    parser.add_argument("--completions", action="store_true")
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wipe-storage", action="store_true")
    parser.add_argument(
        "--no-launch", action="store_true", help="Use already-running servers."
    )
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    all_start_skews = parse_csv_floats(args.start_skews)
    start_skews = all_start_skews[: args.instances]
    servers = build_servers(start_skews, num_instances=args.instances)

    output_dir = Path(args.output_dir or f"scenario_a_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_root = Path(args.storage_root)
    if args.wipe_storage and storage_root.exists():
        shutil.rmtree(storage_root)
    storage_root.mkdir(parents=True, exist_ok=True)

    storage_paths = {
        server.instance_id: str(storage_root / f"instance_{server.instance_id}")
        for server in servers
    }
    for path in storage_paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    doc_lo, doc_hi = parse_int_range(args.doc_size)
    prefix_pct_lo, prefix_pct_hi = parse_pct_range(args.prefix_size)

    procs = []
    try:
        if not args.no_launch:
            procs = launch_cluster(servers, output_dir, storage_paths)

        model_name = await fetch_model_name(servers[0].base_url)
        print(f"Model: {model_name}")

        before_metrics = {
            server.instance_id: scrape_metrics(server.base_url) for server in servers
        }

        specs_by_instance: dict[int, list[RequestSpec]] = {}
        next_request_id = 0
        for server in servers:
            specs = build_instance_specs(
                instance_id=server.instance_id,
                requests_per_instance=args.requests_per_instance,
                doc_lo=doc_lo,
                doc_hi=doc_hi,
                prefix_reuse_pct=args.prefix_reuse_pct,
                prefix_pct_lo=prefix_pct_lo,
                prefix_pct_hi=prefix_pct_hi,
                arrival_rate=args.arrival_rate,
                scheduled_offset=server.start_skew_s,
                request_id_offset=next_request_id,
            )
            next_request_id += len(specs)
            specs_by_instance[server.instance_id] = specs

        grouped_results = await asyncio.gather(
            *[
                run_instance_workload(
                    base_url=server.base_url,
                    model=model_name,
                    specs=specs_by_instance[server.instance_id],
                    output_len=args.output_len,
                    max_concurrency=args.max_concurrency,
                    completions_mode=args.completions,
                    eos_token_id=args.eos_token_id,
                )
                for server in servers
            ]
        )

        after_metrics = {
            server.instance_id: scrape_metrics(server.base_url) for server in servers
        }

        rows_by_instance: dict[int, list[dict[str, Any]]] = {}
        request_rows: list[dict[str, Any]] = []
        for server, instance_rows in zip(servers, grouped_results):
            rows_by_instance[server.instance_id] = []
            for row in instance_rows:
                row["instance_id"] = server.instance_id
                row["port"] = server.port
                row["storage_path"] = storage_paths[server.instance_id]
                row["latency"] = row["request_end"] - row["request_start"]
                rows_by_instance[server.instance_id].append(row)
                request_rows.append(row)

        request_csv = output_dir / "requests.csv"
        if request_rows:
            write_request_csv(request_csv, request_rows)

        summary = {
            "scenario": "A_isolated_storage",
            "config": {
                "storage_root": str(storage_root),
                "instances": args.instances,
                "requests_per_instance": args.requests_per_instance,
                "doc_size": args.doc_size,
                "prefix_reuse_pct": args.prefix_reuse_pct,
                "prefix_size": args.prefix_size,
                "arrival_rate": args.arrival_rate,
                "max_concurrency": args.max_concurrency,
                "output_len": args.output_len,
                "start_skews": start_skews,
                "instance_ports": INSTANCE_PORTS[: args.instances],
                "instance_gpus": INSTANCE_GPUS[: args.instances],
                "storage_paths": storage_paths,
                "vllm_server_config": VLLM_SERVER_CONFIG,
            },
            "aggregate_results": summarise_results(request_rows),
            "per_instance": collect_per_instance_metrics(
                servers,
                before_metrics,
                after_metrics,
                rows_by_instance,
            ),
            "request_csv": str(request_csv),
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)

        aggregate = summary["aggregate_results"]
        print("\n=== Scenario A Summary ===")
        print(f"Output directory: {output_dir}")
        print(f"Total requests:   {aggregate['total_requests']}")
        print(f"Successful:       {aggregate['successful_requests']}")
        print(f"Throughput:       {aggregate['request_throughput_rps']}")
        print(
            f"TTFT p50/p95/p99: {aggregate['ttft_p50_s']} / {aggregate['ttft_p95_s']} / {aggregate['ttft_p99_s']}"
        )
        print(
            f"Latency p50/p95/p99: {aggregate['latency_p50_s']} / {aggregate['latency_p95_s']} / {aggregate['latency_p99_s']}"
        )
        print(f"Summary JSON:     {summary_path}")
    finally:
        stop_cluster(procs)


if __name__ == "__main__":
    asyncio.run(main())
