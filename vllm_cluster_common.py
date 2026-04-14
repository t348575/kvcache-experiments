#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from openai import AsyncOpenAI
from prometheus_client.parser import text_string_to_metric_families


SERVER_STARTUP_TIMEOUT = 240
INSTANCE_PORTS = [8000, 8001, 8002, 8003]
INSTANCE_GPUS = ["0", "1", "2", "3"]

# Edit shared vLLM settings here for all scenario scripts.
VLLM_SERVER_CONFIG: dict[str, Any] = {
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "max_model_len": 92000,
    "gpu_memory_utilization": 0.9,
    "kv_transfer_config": {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "SharedStorageOffloadingSpec",
            "spec_module_path": "llmd_fs_backend.spec",
            "use_odirect": True
        },
    },
    "extra_vllm_args": [
        "--no-enable-prefix-caching",
        "--distributed_executor_backend",
        "mp"
    ],
    "extra_env": {},
}


@dataclass
class ServerSpec:
    instance_id: int
    gpu_id: str
    port: int
    base_url: str
    start_skew_s: float


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_vllm_arg(raw: str) -> tuple[str, str | None]:
    if not raw.startswith("--"):
        raise ValueError(f"Extra vLLM args must start with '--': {raw}")
    if "=" in raw:
        flag, value = raw.split("=", 1)
        return flag, value
    return raw, None


def wait_for_server(port: int, timeout: int) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = pct / 100.0 * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def flatten_metric_sample(metric_name: str, sample: Any) -> tuple[str, float]:
    if sample.labels:
        labels = ",".join(f"{k}={sample.labels[k]}" for k in sorted(sample.labels))
        key = f"{metric_name}{{{labels}}}"
    else:
        key = metric_name
    return key, float(sample.value)


def scrape_metrics(base_url: str) -> dict[str, float]:
    metrics_url = base_url.removesuffix("/v1") + "/metrics"
    response = requests.get(metrics_url, timeout=10)
    response.raise_for_status()

    snapshot: dict[str, float] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            key, value = flatten_metric_sample(family.name, sample)
            snapshot[key] = value
    return snapshot


def diff_metrics(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    diff: dict[str, float] = {}
    for key, end_value in after.items():
        start_value = before.get(key, 0.0)
        delta = end_value - start_value
        if abs(delta) > 1e-12:
            diff[key] = delta
    return diff


def extract_high_signal_metrics(delta: dict[str, float]) -> dict[str, float]:
    selected: dict[str, float] = {}
    interesting = (
        "prefix_cache",
        "external",
        "request_prompt_tokens",
        "request_generation_tokens",
        "request_success",
        "request_failure",
        "queue",
        "kv_cache",
        "num_requests",
    )
    for key, value in sorted(delta.items()):
        if any(token in key for token in interesting):
            selected[key] = value
    return selected


def make_kv_transfer_config(shared_storage_path: str | None) -> dict[str, Any] | None:
    kv_cfg = VLLM_SERVER_CONFIG.get("kv_transfer_config")
    if kv_cfg is None:
        return None
    cfg = json.loads(json.dumps(kv_cfg))
    extra_cfg = cfg.setdefault("kv_connector_extra_config", {})
    if shared_storage_path is None:
        extra_cfg.pop("shared_storage_path", None)
    else:
        extra_cfg["shared_storage_path"] = shared_storage_path
    return cfg


def build_vllm_command(port: int, shared_storage_path: str | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(VLLM_SERVER_CONFIG["model"]),
        "--port",
        str(port),
        "--max-model-len",
        str(VLLM_SERVER_CONFIG["max_model_len"]),
        "--gpu-memory-utilization",
        str(VLLM_SERVER_CONFIG["gpu_memory_utilization"]),
    ]

    kv_cfg = make_kv_transfer_config(shared_storage_path)
    if kv_cfg is not None:
        cmd.extend(["--kv-transfer-config", json.dumps(kv_cfg, separators=(",", ":"))])

    for raw_arg in VLLM_SERVER_CONFIG["extra_vllm_args"]:
        flag, value = parse_vllm_arg(raw_arg)
        cmd.append(flag)
        if value is not None:
            cmd.append(value)
    return cmd


def build_servers(
    start_skews: list[float], num_instances: int | None = None
) -> list[ServerSpec]:
    count = num_instances if num_instances is not None else len(INSTANCE_PORTS)
    if count > len(INSTANCE_PORTS):
        raise ValueError(f"Only {len(INSTANCE_PORTS)} instance slots are configured")
    if len(start_skews) != count:
        raise ValueError(f"Expected {count} start skews, got {len(start_skews)}")
    return [
        ServerSpec(
            instance_id=index,
            gpu_id=INSTANCE_GPUS[index],
            port=INSTANCE_PORTS[index],
            base_url=f"http://localhost:{INSTANCE_PORTS[index]}/v1",
            start_skew_s=start_skews[index],
        )
        for index in range(count)
    ]


def start_server(
    spec: ServerSpec,
    shared_storage_path: str | None,
    output_dir: Path,
) -> tuple[subprocess.Popen[Any], Any]:
    cmd = build_vllm_command(spec.port, shared_storage_path)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
    env.setdefault("OPENAI_API_KEY", "sk-dummy")
    for key, value in VLLM_SERVER_CONFIG["extra_env"].items():
        env[key] = str(value)

    log_path = output_dir / f"server_{spec.instance_id}.log"
    log_handle = open(log_path, "a", buffering=1)
    print(
        f"[server {spec.instance_id}] gpu={spec.gpu_id} port={spec.port} start: {' '.join(cmd)}",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_handle


def stop_server(proc: subprocess.Popen[Any], log_handle: Any) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
    finally:
        log_handle.close()


def launch_cluster(
    servers: list[ServerSpec],
    output_dir: Path,
    storage_paths: dict[int, str | None],
) -> list[tuple[subprocess.Popen[Any], Any]]:
    procs: list[tuple[subprocess.Popen[Any], Any]] = []
    for server in servers:
        procs.append(
            start_server(server, storage_paths.get(server.instance_id), output_dir)
        )
    for server in servers:
        print(
            f"[server {server.instance_id}] waiting for /health on :{server.port}",
            flush=True,
        )
        if not wait_for_server(server.port, SERVER_STARTUP_TIMEOUT):
            raise RuntimeError(
                f"vLLM instance {server.instance_id} on port {server.port} did not become healthy"
            )
    return procs


def stop_cluster(procs: list[tuple[subprocess.Popen[Any], Any]]) -> None:
    for proc, log_handle in reversed(procs):
        stop_server(proc, log_handle)


async def fetch_model_name(base_url: str) -> str:
    requested_model = str(VLLM_SERVER_CONFIG["model"])
    if requested_model != "auto":
        return requested_model
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"),
        timeout=None,
    )
    models = await client.models.list()
    return models.data[0].id


def summarise_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [r for r in rows if r["successful"]]
    ttfts = [float(r["ttft"]) for r in successful]
    latencies = [float(r["request_end"] - r["request_start"]) for r in successful]

    wall_clock_s = 0.0
    if rows:
        wall_clock_s = max(r["request_end"] for r in rows) - min(
            r["request_start"] for r in rows
        )

    return {
        "total_requests": len(rows),
        "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "wall_clock_s": wall_clock_s,
        "request_throughput_rps": (len(successful) / wall_clock_s)
        if wall_clock_s > 0
        else None,
        "ttft_mean_s": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "ttft_p99_s": percentile(ttfts, 99),
        "latency_mean_s": (sum(latencies) / len(latencies)) if latencies else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
    }


def write_request_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_per_instance_metrics(
    servers: list[ServerSpec],
    before_metrics: dict[int, dict[str, float]],
    after_metrics: dict[int, dict[str, float]],
    rows_by_instance: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for server in servers:
        metric_delta = diff_metrics(
            before_metrics[server.instance_id],
            after_metrics[server.instance_id],
        )
        summary[str(server.instance_id)] = {
            "server": asdict(server),
            "results": summarise_results(rows_by_instance.get(server.instance_id, [])),
            "interesting_metric_deltas": extract_high_signal_metrics(metric_delta),
            "all_metric_deltas": metric_delta,
        }
    return summary
