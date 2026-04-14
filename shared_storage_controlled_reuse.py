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
    build_prompts,
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


def generate_global_hotset_schedule(
    total_requests: int,
    doc_size_lo: int,
    doc_size_hi: int,
    prefix_reuse_pct: float,
    prefix_pct_lo: float,
    prefix_pct_hi: float,
    arrival_rate: float | None,
    hotset_size: int,
    rng: random.Random,
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    elapsed = 0.0
    hot_sources: list[RequestSpec] = []

    for request_id in range(total_requests):
        doc_tokens = rng.randint(doc_size_lo, doc_size_hi)
        spec = RequestSpec(request_id=request_id, doc_tokens=doc_tokens)

        if hot_sources and rng.random() < prefix_reuse_pct:
            source = rng.choice(hot_sources)
            frac = rng.uniform(prefix_pct_lo, prefix_pct_hi)
            spec.reuse_source_id = source.request_id
            spec.reuse_prefix_len = max(1, int(frac * source.doc_tokens))
        else:
            hot_sources.append(spec)
            if len(hot_sources) > hotset_size:
                hot_sources.pop(0)

        if arrival_rate is not None and arrival_rate > 0:
            elapsed += rng.expovariate(arrival_rate)
        spec.scheduled_time = elapsed
        specs.append(spec)

    return specs


def assign_requests_to_instances(
    servers,
    global_specs: list[RequestSpec],
    global_prompts: list[str],
    assignment_mode: str,
    warmup_to_instance0: int,
) -> tuple[dict[int, list[RequestSpec]], dict[int, list[str]]]:
    specs_by_instance: dict[int, list[RequestSpec]] = {
        s.instance_id: [] for s in servers
    }
    prompts_by_instance: dict[int, list[str]] = {s.instance_id: [] for s in servers}

    follower_ids = [s.instance_id for s in servers[1:]]
    follower_count = max(1, len(follower_ids))

    for index, (spec, prompt) in enumerate(zip(global_specs, global_prompts)):
        if assignment_mode == "round_robin":
            target = servers[index % len(servers)].instance_id
        elif assignment_mode == "warmup_handoff":
            if index < warmup_to_instance0 or not follower_ids:
                target = servers[0].instance_id
            else:
                target = follower_ids[(index - warmup_to_instance0) % follower_count]
        else:
            raise ValueError(f"Unsupported assignment_mode: {assignment_mode}")

        local_spec = RequestSpec(
            request_id=spec.request_id,
            doc_tokens=spec.doc_tokens,
            reuse_source_id=spec.reuse_source_id,
            reuse_prefix_len=spec.reuse_prefix_len,
            scheduled_time=spec.scheduled_time,
        )
        specs_by_instance[target].append(local_spec)
        prompts_by_instance[target].append(prompt)

    for server in servers:
        for spec in specs_by_instance[server.instance_id]:
            spec.scheduled_time += server.start_skew_s

    return specs_by_instance, prompts_by_instance


async def run_instance_workload(
    base_url: str,
    model: str,
    specs: list[RequestSpec],
    prompts: list[str],
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
        prompts=prompts,
        output_len=output_len,
        max_concurrency=max_concurrency,
        completions_mode=completions_mode,
        eos_token_id=eos_token_id,
    )
    return [asdict(result) for result in results]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scenario C benchmark: shared storage with controlled prefix reuse."
    )
    parser.add_argument("--shared-storage-path", type=str, required=True)
    parser.add_argument("--requests-per-instance", type=int, default=32)
    parser.add_argument("--doc-size", type=str, default="4096")
    parser.add_argument("--prefix-reuse-pct", type=float, default=0.5)
    parser.add_argument("--prefix-size", type=str, default="0.5")
    parser.add_argument("--hotset-size", type=int, default=10)
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--start-skews", type=str, default="0,0,0,0")
    parser.add_argument(
        "--assignment-mode",
        type=str,
        default="round_robin",
        choices=["round_robin", "warmup_handoff"],
    )
    parser.add_argument("--warmup-to-instance0", type=int, default=10)
    parser.add_argument("--completions", action="store_true")
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wipe-shared-storage", action="store_true")
    parser.add_argument(
        "--no-launch", action="store_true", help="Use already-running servers."
    )
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start_skews = parse_csv_floats(args.start_skews)
    servers = build_servers(start_skews)

    output_dir = Path(args.output_dir or f"scenario_c_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wipe_shared_storage and os.path.exists(args.shared_storage_path):
        shutil.rmtree(args.shared_storage_path)
    Path(args.shared_storage_path).mkdir(parents=True, exist_ok=True)

    doc_lo, doc_hi = parse_int_range(args.doc_size)
    prefix_pct_lo, prefix_pct_hi = parse_pct_range(args.prefix_size)
    total_requests = args.requests_per_instance * len(servers)
    rng = random.Random(args.seed)

    global_specs = generate_global_hotset_schedule(
        total_requests=total_requests,
        doc_size_lo=doc_lo,
        doc_size_hi=doc_hi,
        prefix_reuse_pct=args.prefix_reuse_pct,
        prefix_pct_lo=prefix_pct_lo,
        prefix_pct_hi=prefix_pct_hi,
        arrival_rate=args.arrival_rate,
        hotset_size=args.hotset_size,
        rng=rng,
    )
    global_prompts = build_prompts(global_specs)
    specs_by_instance, prompts_by_instance = assign_requests_to_instances(
        servers=servers,
        global_specs=global_specs,
        global_prompts=global_prompts,
        assignment_mode=args.assignment_mode,
        warmup_to_instance0=args.warmup_to_instance0,
    )

    storage_paths = {server.instance_id: args.shared_storage_path for server in servers}

    procs = []
    try:
        if not args.no_launch:
            procs = launch_cluster(servers, output_dir, storage_paths)

        model_name = await fetch_model_name(servers[0].base_url)
        print(f"Model: {model_name}")

        before_metrics = {
            server.instance_id: scrape_metrics(server.base_url) for server in servers
        }

        grouped_results = await asyncio.gather(
            *[
                run_instance_workload(
                    base_url=server.base_url,
                    model=model_name,
                    specs=specs_by_instance[server.instance_id],
                    prompts=prompts_by_instance[server.instance_id],
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
                row["latency"] = row["request_end"] - row["request_start"]
                rows_by_instance[server.instance_id].append(row)
                request_rows.append(row)

        request_csv = output_dir / "requests.csv"
        if request_rows:
            write_request_csv(request_csv, request_rows)

        summary = {
            "scenario": "C_controlled_reuse_shared_storage",
            "config": {
                "shared_storage_path": args.shared_storage_path,
                "requests_per_instance": args.requests_per_instance,
                "doc_size": args.doc_size,
                "prefix_reuse_pct": args.prefix_reuse_pct,
                "prefix_size": args.prefix_size,
                "hotset_size": args.hotset_size,
                "arrival_rate": args.arrival_rate,
                "max_concurrency": args.max_concurrency,
                "output_len": args.output_len,
                "start_skews": start_skews,
                "assignment_mode": args.assignment_mode,
                "warmup_to_instance0": args.warmup_to_instance0,
                "instance_ports": INSTANCE_PORTS,
                "instance_gpus": INSTANCE_GPUS,
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
        print("\n=== Scenario C Summary ===")
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
