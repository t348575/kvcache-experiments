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

from prefix_cache_common import RequestSpec, run_benchmark
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


def prompt_from_sharegpt_item(item: dict[str, Any]) -> str:
    conversations = item.get("conversations", [])
    turns: list[str] = []
    for conv in conversations:
        role = str(conv.get("from", "")).lower()
        value = str(conv.get("value", "")).strip()
        if role in {"human", "user"} and value:
            turns.append(value)
    return "\n\n".join(turns).strip()


def prompt_from_json_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    for key in ("prompt", "text", "input", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if "conversations" in item:
        return prompt_from_sharegpt_item(item)
    return ""


def load_trace_prompts(dataset_path: str, max_prompts: int | None) -> list[str]:
    path = Path(dataset_path)
    prompts: list[str] = []

    if path.suffix == ".jsonl":
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                prompt = prompt_from_json_item(json.loads(line))
                if prompt:
                    prompts.append(prompt)
                if max_prompts is not None and len(prompts) >= max_prompts:
                    break
        return prompts

    if path.suffix == ".json":
        with open(path) as handle:
            data = json.load(handle)
        if isinstance(data, list):
            items = data
        else:
            items = data.get("data", [])
        for item in items:
            prompt = prompt_from_json_item(item)
            if prompt:
                prompts.append(prompt)
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
        return prompts

    with open(path) as handle:
        for line in handle:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    return prompts


def maybe_inject_repeated_prefix(
    prompts: list[str],
    repeated_prefix: str,
    inject_fraction: float,
    rng: random.Random,
) -> list[str]:
    if not repeated_prefix or inject_fraction <= 0:
        return prompts
    updated: list[str] = []
    prefix = repeated_prefix.strip()
    for prompt in prompts:
        if rng.random() < inject_fraction:
            updated.append(f"{prefix}\n\n{prompt}")
        else:
            updated.append(prompt)
    return updated


def build_global_specs(
    total_requests: int,
    arrival_rate: float | None,
    rng: random.Random,
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    elapsed = 0.0
    for request_id in range(total_requests):
        if arrival_rate is not None and arrival_rate > 0:
            elapsed += rng.expovariate(arrival_rate)
        specs.append(
            RequestSpec(
                request_id=request_id,
                doc_tokens=0,
                scheduled_time=elapsed,
            )
        )
    return specs


def distribute_prompts(
    servers,
    prompts: list[str],
    global_specs: list[RequestSpec],
) -> tuple[dict[int, list[RequestSpec]], dict[int, list[str]]]:
    specs_by_instance: dict[int, list[RequestSpec]] = {
        s.instance_id: [] for s in servers
    }
    prompts_by_instance: dict[int, list[str]] = {s.instance_id: [] for s in servers}

    for index, (prompt, spec) in enumerate(zip(prompts, global_specs)):
        server = servers[index % len(servers)]
        local_spec = RequestSpec(
            request_id=spec.request_id,
            doc_tokens=len(prompt.split()),
            reuse_source_id=None,
            reuse_prefix_len=0,
            scheduled_time=spec.scheduled_time + server.start_skew_s,
        )
        specs_by_instance[server.instance_id].append(local_spec)
        prompts_by_instance[server.instance_id].append(prompt)
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
        description="Scenario D benchmark: realistic trace replay over shared storage."
    )
    parser.add_argument("--shared-storage-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--start-skews", type=str, default="0,0,0,0")
    parser.add_argument("--repeated-prefix-file", type=str, default=None)
    parser.add_argument("--repeated-prefix-text", type=str, default="")
    parser.add_argument("--inject-prefix-fraction", type=float, default=0.0)
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

    output_dir = Path(args.output_dir or f"scenario_d_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wipe_shared_storage and os.path.exists(args.shared_storage_path):
        shutil.rmtree(args.shared_storage_path)
    Path(args.shared_storage_path).mkdir(parents=True, exist_ok=True)

    prompts = load_trace_prompts(args.dataset_path, args.max_prompts)
    rng = random.Random(args.seed)
    repeated_prefix = args.repeated_prefix_text
    if args.repeated_prefix_file:
        with open(args.repeated_prefix_file) as handle:
            repeated_prefix = handle.read()
    prompts = maybe_inject_repeated_prefix(
        prompts,
        repeated_prefix=repeated_prefix,
        inject_fraction=args.inject_prefix_fraction,
        rng=rng,
    )
    if not prompts:
        raise RuntimeError("No prompts could be loaded from the dataset path")

    global_specs = build_global_specs(len(prompts), args.arrival_rate, rng)
    specs_by_instance, prompts_by_instance = distribute_prompts(
        servers, prompts, global_specs
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
            "scenario": "D_trace_replay_shared_storage",
            "config": {
                "shared_storage_path": args.shared_storage_path,
                "dataset_path": args.dataset_path,
                "max_prompts": args.max_prompts,
                "arrival_rate": args.arrival_rate,
                "max_concurrency": args.max_concurrency,
                "output_len": args.output_len,
                "start_skews": start_skews,
                "inject_prefix_fraction": args.inject_prefix_fraction,
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
        print("\n=== Scenario D Summary ===")
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
