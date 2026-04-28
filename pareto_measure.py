#!/usr/bin/env python3
"""
pareto_measure.py
─────────────────
Orchestrates the minimal set of measurements needed to build a KV-cache
Pareto frontier (when does prefix caching pay off?).

Two curves are measured:

  A  cold_prefill  f(N)  — TTFT to cold-compute N tokens (no caching)
                           Server: prefix caching disabled.
  B  cache_hit     g(P)  — TTFT when P tokens are served from the native
                           KV offload cache + ~1-token fresh suffix.
                           Server: native KV offloading enabled.

The Pareto frontier in (doc_size × prefix_fraction) space is the locus where:
    f(D) = g(frac * D) + f((1 - frac) * D)

An optional concurrency sweep (Curves A & B at several concurrency levels)
shows how the frontier shifts under production-like load.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from benchmark_common import pct, resolve_scratch_path_for_config, save_profile_artifacts, write_dataclass_csv

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PRESETS
# ─────────────────────────────────────────────────────────────────────────────
# Each preset contains all model-specific parameters needed to run the server
# and size the CPU KV offload cache.
#
# KV sizing fields:
#   kv_num_layers   — num_hidden_layers
#   kv_num_kv_heads — num_key_value_heads  (GQA head count)
#   kv_head_dim     — hidden_size / num_attention_heads
#   kv_dtype_bytes  — 2 for fp16/bf16, 1 for fp8

MODEL_PRESETS: dict[str, dict] = {
    "llama-3.2-3b": {
        "model":            "meta-llama/Llama-3.2-3B-Instruct",
        "kv_num_layers":    28,
        "kv_num_kv_heads":  8,
        "kv_head_dim":      128,
        "kv_dtype_bytes":   2,
        "kv_cache_dtype":   "auto",
    },
    "llama-3.1-8b": {
        "model":            "meta-llama/Llama-3.1-8B-Instruct",
        "kv_num_layers":    32,
        "kv_num_kv_heads":  8,
        "kv_head_dim":      128,
        "kv_dtype_bytes":   2,
        "kv_cache_dtype":   "auto",
    },
    "Mistral-Small-3.1-24B": {
        "model":            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "kv_num_layers":    40,
        "kv_num_kv_heads":  8,
        "kv_head_dim":      128,
        "kv_dtype_bytes":   2,
        "kv_cache_dtype":   "auto",
    },
    "qwen2.5-32b-fp16": {
        "model":            "Qwen/Qwen2.5-32B-Instruct",
        "kv_num_layers":    64,
        "kv_num_kv_heads":  8,
        "kv_head_dim":      128,
        "kv_dtype_bytes":   2,
        "kv_cache_dtype":   "auto",
    },
    "llama-3.1-70b-fp8": {
        "model":            "nvidia/Llama-3.1-70B-Instruct-FP8",
        "kv_num_layers":    80,
        "kv_num_kv_heads":  8,
        "kv_head_dim":      128,
        "kv_dtype_bytes":   1,
        "kv_cache_dtype":   "fp8",
    },
    "gemma-3-27b-fp8": {
        "model":            "google/gemma-3-27b-it",
        "kv_num_layers":    62,
        "kv_num_kv_heads":  16,
        "kv_head_dim":      128,
        "kv_dtype_bytes":   1,
        "kv_cache_dtype":   "fp8",
    },
    "qwen3-32b-fp8": {
        "model":            "Qwen/Qwen3-32B",
        "kv_num_layers":    64,
        "kv_num_kv_heads":  8,
        "kv_head_dim":      128,
        "kv_dtype_bytes":   1,
        "kv_cache_dtype":   "fp8",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Model / KV params are set at startup via select_model_preset().
MODEL           = None
KV_NUM_LAYERS   = None
KV_NUM_KV_HEADS = None
KV_HEAD_DIM     = None
KV_DTYPE_BYTES  = None
KV_CACHE_DTYPE  = None

MAX_MODEL_LEN  = 92000
GPU_MEM_UTIL   = 0.9
VLLM_PORT      = 8000
OUTPUT_TOKENS  = 1      # keep short — we care about TTFT, not decode throughput
SERVER_STARTUP_TIMEOUT = 200  # seconds
KV_BUFFER       = 0.25  # allocate 25% more CPU memory than strictly needed

BENCHMARK_SCRIPT = "prefix_cache_benchmark.py"

# simple-profiler writes this file in the working directory.
# Both baseline and native_offload use upstream vLLM (engine-core process).
PROFILE_JSON = "merged.json"

# Set True to only run cold-prefill jobs (for pareto_estimate target models)
COLD_ONLY = False

# Measurement points for the serial curves (A & B)
DOC_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 81920]

# How many query requests per (doc_size, concurrency) point
N_REPEATS_SERIAL      = 3   # concurrency = 1
N_REPEATS_CONCURRENCY = 3   # concurrency > 1

# Concurrency sweep — set CONCURRENCY_LEVELS = [] to skip entirely
CONCURRENCY_DOC_SIZES = []
CONCURRENCY_LEVELS    = []

# Native KV offload cache size (GB)
NATIVE_OFFLOAD_SIZE = 128

_TS       = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"pareto_measure_{_TS}"
OUTPUT_CSV = f"pareto_measure_{_TS}.csv"

# ─────────────────────────────────────────────────────────────────────────────
# SERVER CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

SERVER_CONFIGS: dict[str, dict] = {
    "baseline": {
        "--no-enable-prefix-caching": None,
    },
    "native_offload": {
        "--no-enable-prefix-caching":          None,
        "--kv-offloading-backend":             "native",
        "--kv-offloading-size":                str(NATIVE_OFFLOAD_SIZE),
        "--disable-hybrid-kv-cache-manager":   None,
    },
    "storage_offload": {
        "--no-enable-prefix-caching":          None,
        "--kv-transfer-config": """{
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
            "spec_name": "SharedStorageOffloadingSpec",
            "spec_module_path": "llmd_fs_backend.spec",
            "shared_storage_path": "/scratch-node/jkanichai.21435621/llm-d-fs",
            "use_odirect": true
            }
        }""",
        "--distributed_executor_backend": "mp"
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    curve:            str    # "cold_prefill" | "cache_hit"
    server_config:    str    # "baseline" | "native_offload"
    doc_size:         int    # total tokens per request document
    concurrency:      int    # max inflight requests
    n_requests:       int    # total sent (includes 1 warmup for cache_hit)
    prefix_reuse_pct: float  # 0.0 (cold) | 1.0 (cache hit)
    prefix_size:      str    # "0" (cold) | "1.0" (cache hit)


@dataclass
class Result:
    curve:         str
    server_config: str
    doc_size:      int
    prefix_len:    int           # 0 for cold; ≈ doc_size for cache hit
    prefix_frac:   float         # 0.0 for cold; 1.0 for cache hit
    concurrency:   int
    n_queries:     int
    n_successful:  int
    ttft_mean_s:   Optional[float] = None
    ttft_median_s: Optional[float] = None
    ttft_p05_s:    Optional[float] = None
    ttft_p95_s:    Optional[float] = None
    ttft_p99_s:       Optional[float] = None
    gpu_transfer_csv: Optional[str]   = None   # path to per-kernel transfer CSV
    error:            Optional[str]   = None


# ─────────────────────────────────────────────────────────────────────────────
# JOB PLAN
# ─────────────────────────────────────────────────────────────────────────────

def build_job_plan(server_configs: list[str]) -> list[Job]:
    jobs: list[Job] = []

    # Each job runs against a freshly started vLLM instance (clean cache).

    # ── Curve A: cold prefill, serial ────────────────────────────────────────
    if "baseline" in server_configs:
        for d in DOC_SIZES:
            jobs.append(Job(
                curve="cold_prefill", server_config="baseline",
                doc_size=d, concurrency=1,
                n_requests=N_REPEATS_SERIAL,
                prefix_reuse_pct=0.0, prefix_size="0",
            ))

    if COLD_ONLY:
        return jobs

    # ── Curve B: cache hit, serial ────────────────────────────────────────────
    # Request 0 is a fresh "warmup" that populates the KV cache.
    # Requests 1..N_REPEATS_SERIAL all reuse that prefix (is_prefix_reuse=True).
    for d in DOC_SIZES:
        for cfg in ["native_offload", "storage_offload"]:
            if cfg in server_configs:
                jobs.append(Job(
                    curve="cache_hit", server_config=cfg,
                    doc_size=d, concurrency=1,
                    n_requests=1 + N_REPEATS_SERIAL,
                    prefix_reuse_pct=1.0, prefix_size="1.0",
                ))

    # ── Curve C: concurrency sweep ─────────────────────────────────────────────
    for d in CONCURRENCY_DOC_SIZES:
        for c in CONCURRENCY_LEVELS:
            n = c * N_REPEATS_CONCURRENCY
            if "baseline" in server_configs:
                jobs.append(Job(
                    curve="cold_prefill", server_config="baseline",
                    doc_size=d, concurrency=c,
                    n_requests=n,
                    prefix_reuse_pct=0.0, prefix_size="0",
                ))
            for cfg in ["native_offload", "storage_offload"]:
                if cfg in server_configs:
                    jobs.append(Job(
                        curve="cache_hit", server_config=cfg,
                        doc_size=d, concurrency=c,
                        n_requests=1 + n,
                        prefix_reuse_pct=1.0, prefix_size="1.0",
                    ))

    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# CPU KV OFFLOAD SIZING
# ─────────────────────────────────────────────────────────────────────────────

def compute_offload_size_gb(job: Job) -> float:
    """
    Compute the CPU memory (GB) needed to hold the KV cache for all tokens
    that could be in the offload store at once for this job, plus KV_BUFFER.

    For cache_hit jobs the warmup request's full KV must be retained in the
    offload store while the query requests are served, so we size for
    (1 + concurrency) * doc_size tokens.  For cold_prefill jobs the offload
    store is unused, but we still return a minimum non-zero value so the
    server config remains valid.
    """
    if job.curve == "cache_hit":
        # The native offload evicts ALL completed request KV blocks to CPU —
        # not just the warmup.  Size for the warmup + all concurrent queries.
        max_tokens = (1 + job.concurrency) * job.doc_size
    else:
        max_tokens = 0  # cold_prefill doesn't use the CPU offload at all

    kv_bytes = max_tokens * 2 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM * KV_DTYPE_BYTES
    size_gb  = kv_bytes * (1 + KV_BUFFER) / 1024 ** 3

    # Round up to nearest 0.5 GB, minimum 1 GB
    size_gb = max(1.0, math.ceil(size_gb * 2) / 2)
    print(f"  CPU KV offload: {max_tokens:,} tokens × "
          f"{2 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM * KV_DTYPE_BYTES / 1024:.1f} KB/tok "
          f"+ {KV_BUFFER:.0%} buffer → {size_gb:.1f} GB")
    return size_gb


# ─────────────────────────────────────────────────────────────────────────────
# SERVER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_vllm_command(vllm_args: dict) -> list[str]:
    cmd = [
        "vllm", "serve",
        MODEL,
        "--port",                   str(VLLM_PORT),
        "--max-model-len",          str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--kv-cache-dtype",         KV_CACHE_DTYPE,
    ]
    for flag, value in vllm_args.items():
        cmd.append(flag)
        if value is not None:
            if isinstance(value, (dict, list)):
                cmd.append(json.dumps(value, separators=(",", ":")))
            else:
                cmd.append(str(value))
    print(f"running {cmd}")
    return cmd


def wait_for_server(port: int, timeout: int) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def start_server(config_name: str, log_path: str, job: "Job") -> subprocess.Popen:
    vllm_args = dict(SERVER_CONFIGS[config_name])
    if config_name == "native_offload":
        offload_gb = compute_offload_size_gb(job)
        vllm_args["--kv-offloading-size"] = str(offload_gb)
    cmd = build_vllm_command(vllm_args)
    print(f"\n  ▶  Starting vLLM [{config_name}]: {' '.join(cmd)}")
    print(f"     Log: {log_path}")
    tee_out = subprocess.Popen(["tee", "-a", log_path], stdin=subprocess.PIPE)
    tee_err = subprocess.Popen(["tee", "-a", log_path], stdin=subprocess.PIPE)
    return subprocess.Popen(cmd, stdout=tee_out.stdin, stderr=tee_err.stdin)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print("  ■  vLLM server stopped.")


def wipe_shared_storage(config_name: str) -> None:
    cfg = SERVER_CONFIGS[config_name]
    kv_transfer_cfg = cfg.get("--kv-transfer-config", {})
    try:
        if isinstance(kv_transfer_cfg, str):
            cfg_dict = json.loads(kv_transfer_cfg)
        else:
            cfg_dict = kv_transfer_cfg
        shared_path = cfg_dict.get("kv_connector_extra_config", {}).get("shared_storage_path")
    except (json.JSONDecodeError, TypeError, KeyError):
        shared_path = None
    if shared_path and os.path.exists(shared_path):
        shutil.rmtree(shared_path)
        print(f"  🗑  Wiped shared_storage_path: {shared_path}")
    elif shared_path:
        print(f"  🗑  shared_storage_path does not exist: {shared_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER & RESULT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(job: Job, csv_path: str) -> Optional[list[dict]]:
    cmd = [
        sys.executable, BENCHMARK_SCRIPT,
        "--port",             str(VLLM_PORT),
        "--model",            "auto",
        "--num-requests",     str(job.n_requests),
        "--doc-size",         str(job.doc_size),
        "--prefix-reuse-pct", str(job.prefix_reuse_pct),
        "--prefix-size",      job.prefix_size,
        "--output-len",       str(OUTPUT_TOKENS),
        "--max-concurrency",  str(job.concurrency),
        "--csv-output",       csv_path,
    ]
    print(f"\n  ▶  {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr)
    if proc.returncode != 0:
        print(f"  ✗  Benchmark exited with code {proc.returncode}")
        return None
    try:
        with open(csv_path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"  ✗  Could not read '{csv_path}': {e}")
        return None


def parse_rows(rows: list[dict], job: Job) -> Result:
    # For cache_hit: query rows are those with is_prefix_reuse=True.
    # For cold_prefill: all rows are query rows (no warmup).
    if job.curve == "cache_hit":
        query_rows = [r for r in rows
                      if r["is_prefix_reuse"] == "True" and r["successful"] == "True"]
    else:
        query_rows = [r for r in rows if r["successful"] == "True"]

    ttfts = [float(r["ttft"]) for r in query_rows]

    prefix_len = 0
    if query_rows and job.curve == "cache_hit":
        prefix_len = int(float(query_rows[0]["reuse_prefix_len"]))

    result = Result(
        curve=job.curve,
        server_config=job.server_config,
        doc_size=job.doc_size,
        prefix_len=prefix_len,
        prefix_frac=job.prefix_reuse_pct,
        concurrency=job.concurrency,
        n_queries=len(query_rows),
        n_successful=len(query_rows),
    )

    if ttfts:
        result.ttft_mean_s   = sum(ttfts) / len(ttfts)
        result.ttft_median_s = pct(ttfts, 50)
        result.ttft_p05_s    = pct(ttfts, 5)
        result.ttft_p95_s    = pct(ttfts, 95)
        result.ttft_p99_s    = pct(ttfts, 99)
    else:
        result.error = "no successful query rows"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE TRACE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def save_profile(job_idx: int) -> Optional[str]:
    """
    Copy and parse PROFILE_JSON (written by simple-profiler) for the current job.
    Returns the path to the saved per-kernel transfer CSV, or None if not found.
    """
    try:
        transfer_csv = save_profile_artifacts(PROFILE_JSON, OUTPUT_DIR, f"job_{job_idx:04d}")
        if transfer_csv:
            with open(transfer_csv, newline="") as f:
                n_events = sum(1 for _ in csv.DictReader(f))
            print(f"  📊  GPU transfer trace → {transfer_csv}  ({n_events} events)")
        return transfer_csv
    except Exception as e:
        print(f"  ⚠   Could not parse profile JSON: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(results: list[Result], path: str) -> None:
    write_dataclass_csv(results, path)
    print(f"  💾  Results → {path}  ({len(results)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def select_model_preset() -> None:
    """Interactively prompt the user to choose a model preset."""
    global MODEL, KV_NUM_LAYERS, KV_NUM_KV_HEADS, KV_HEAD_DIM, KV_DTYPE_BYTES, KV_CACHE_DTYPE
    names = list(MODEL_PRESETS.keys())
    print("\nAvailable model presets:")
    for i, name in enumerate(names, 1):
        p = MODEL_PRESETS[name]
        dtype_label = p.get("kv_cache_dtype", "auto")
        print(f"  {i}) {name}  ({p['model']})  [kv_cache_dtype={dtype_label}]")
    while True:
        raw = input(f"Select preset [1-{len(names)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            preset = MODEL_PRESETS[names[int(raw) - 1]]
            break
        print(f"  Enter a number between 1 and {len(names)}.")
    MODEL           = preset["model"]
    KV_NUM_LAYERS   = preset["kv_num_layers"]
    KV_NUM_KV_HEADS = preset["kv_num_kv_heads"]
    KV_HEAD_DIM     = preset["kv_head_dim"]
    KV_DTYPE_BYTES  = preset["kv_dtype_bytes"]
    KV_CACHE_DTYPE  = preset.get("kv_cache_dtype", "auto")
    print(f"  → {MODEL}  (kv_cache_dtype={KV_CACHE_DTYPE})")


def parse_args():
    parser = argparse.ArgumentParser(description="Pareto measurement for KV-cache prefix caching")
    parser.add_argument(
        "--wipe-shared-storage",
        action="store_true",
        help="After each storage_offload job, rm -rf the shared_storage_path",
    )
    parser.add_argument(
        "--server-configs",
        nargs="+",
        default=["baseline", "native_offload", "storage_offload"],
        choices=["baseline", "native_offload", "storage_offload"],
        help="Which server configs to run (default: baseline, native_offload, and storage_offload)",
    )
    parser.add_argument(
        "--shared-storage-path",
        type=str,
        default=None,
        help="Override the shared_storage_path for storage_offload (overrides auto-resolution)",
    )
    return parser.parse_args()


def resolve_scratch_path(override_path: Optional[str]) -> None:
    """
    Resolve the shared_storage_path for storage_offload.

    If --shared-storage-path is provided, use that directly.
    Otherwise, if the current path starts with /scratch-node,
    replace the hardcoded user directory with the first /scratch-node/jkanichai*
    entry, then append llm-d-fs.
    """
    cfg = SERVER_CONFIGS.get("storage_offload")
    if cfg is None:
        return
    wrapper = {"vllm_args": cfg}
    resolved = resolve_scratch_path_for_config(wrapper, override_path)
    if resolved:
        print(f"  shared_storage_path → {resolved}")


def main():
    args = parse_args()
    select_model_preset()
    resolve_scratch_path(args.shared_storage_path)

    if not Path(BENCHMARK_SCRIPT).exists():
        print(f"ERROR: '{BENCHMARK_SCRIPT}' not found.")
        sys.exit(1)

    jobs = build_job_plan(args.server_configs)

    print(f"\n  Server configs : {', '.join(args.server_configs)}")
    print(f"  Wipe storage   : {'yes' if args.wipe_shared_storage else 'no'}")

    cold_s  = [j for j in jobs if j.curve == "cold_prefill" and j.concurrency == 1]
    hit_s   = [j for j in jobs if j.curve == "cache_hit"    and j.concurrency == 1]
    cold_c  = [j for j in jobs if j.curve == "cold_prefill" and j.concurrency > 1]
    hit_c   = [j for j in jobs if j.curve == "cache_hit"    and j.concurrency > 1]

    print(f"\n{'═'*70}")
    print(f"  Pareto measurement plan — {len(jobs)} jobs total")
    print(f"{'─'*70}")
    print(f"  Curve A  cold_prefill (serial)      : {len(cold_s):>3} jobs  "
          f"({len(DOC_SIZES)} sizes × {N_REPEATS_SERIAL} reps)")
    print(f"  Curve B  cache_hit    (serial)      : {len(hit_s):>3} jobs  "
          f"({len(DOC_SIZES)} sizes × {N_REPEATS_SERIAL} reps)")
    if CONCURRENCY_LEVELS:
        print(f"  Curve C  cold_prefill (concurrency) : {len(cold_c):>3} jobs  "
              f"({len(CONCURRENCY_DOC_SIZES)} sizes × {len(CONCURRENCY_LEVELS)} conc levels)")
        print(f"  Curve C  cache_hit    (concurrency) : {len(hit_c):>3} jobs  "
              f"({len(CONCURRENCY_DOC_SIZES)} sizes × {len(CONCURRENCY_LEVELS)} conc levels)")
    print(f"{'─'*70}")
    print(f"  Doc sizes (tokens) : {DOC_SIZES}")
    print(f"  Concurrency sweep  : {CONCURRENCY_LEVELS}  over doc sizes {CONCURRENCY_DOC_SIZES}")
    print(f"  Output             : {OUTPUT_CSV}")
    print(f"{'═'*70}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results: list[Result] = []

    # Each job gets its own fresh vLLM instance to guarantee a clean cache state.
    for job_idx, job in enumerate(jobs, 1):
        print(f"\n{'─'*70}")
        print(f"  Job {job_idx}/{len(jobs)} "
              f"[{job.server_config}]  curve={job.curve}  "
              f"doc_size={job.doc_size}  concurrency={job.concurrency}")

        csv_path = os.path.join(
            OUTPUT_DIR,
            f"job_{job_idx:04d}_{job.curve}_{job.doc_size}_c{job.concurrency}.csv",
        )
        server_log = os.path.join(
            OUTPUT_DIR,
            f"job_{job_idx:04d}_server.log",
        )

        server_proc = None
        try:
            server_proc = start_server(job.server_config, server_log, job)
            print(f"  ⏳ Waiting up to {SERVER_STARTUP_TIMEOUT}s for vLLM…")
            if not wait_for_server(VLLM_PORT, SERVER_STARTUP_TIMEOUT):
                raise RuntimeError(
                    f"vLLM did not become healthy within {SERVER_STARTUP_TIMEOUT}s"
                )
            print("  ✓  Server is ready.")

            rows = run_benchmark(job, csv_path)
            if rows is None:
                result = Result(
                    curve=job.curve, server_config=job.server_config,
                    doc_size=job.doc_size, prefix_len=0,
                    prefix_frac=job.prefix_reuse_pct,
                    concurrency=job.concurrency,
                    n_queries=0, n_successful=0,
                    error="benchmark subprocess failed",
                )
            else:
                result = parse_rows(rows, job)

        except Exception as e:
            print(f"  ✗  {e}")
            result = Result(
                curve=job.curve, server_config=job.server_config,
                doc_size=job.doc_size, prefix_len=0,
                prefix_frac=job.prefix_reuse_pct,
                concurrency=job.concurrency,
                n_queries=0, n_successful=0,
                error=str(e),
            )
        finally:
            if server_proc:
                stop_server(server_proc)
            if args.wipe_shared_storage and job.server_config == "storage_offload":
                wipe_shared_storage(job.server_config)

        if job.server_config == "storage_offload":
            print(f"  ⏳ Sleeping 60s before next storage_offload job…")
            time.sleep(60)

        # simple-profiler writes PROFILE_JSON only after vLLM exits, so copy it now.
        if result and not result.error:
            result.gpu_transfer_csv = save_profile(job_idx)

        all_results.append(result)

        if result.error:
            print(f"  ✗  {result.error}")
        else:
            print(
                f"  ✓  mean={result.ttft_mean_s:.3f}s  "
                f"median={result.ttft_median_s:.3f}s  "
                f"p99={result.ttft_p99_s:.3f}s  "
                f"n={result.n_successful}"
            )

        # Write after every job so partial results survive crashes.
        write_csv(all_results, OUTPUT_CSV)

    # Final write + summary table
    write_csv(all_results, OUTPUT_CSV)

    print(f"\n{'═'*70}")
    print(f"  {'CURVE':<16} {'CONFIG':<16} {'DOC_SIZE':>9} {'CONC':>5} "
          f"{'MEAN_TTFT':>10} {'MEDIAN':>9} {'P99':>9}")
    print(f"  {'-'*16} {'-'*16} {'-'*9} {'-'*5} {'-'*10} {'-'*9} {'-'*9}")
    for r in all_results:
        if r.error:
            print(f"  {r.curve:<16} {r.server_config:<16} {r.doc_size:>9} "
                  f"{r.concurrency:>5}  ERROR: {r.error}")
        else:
            print(
                f"  {r.curve:<16} {r.server_config:<16} {r.doc_size:>9} "
                f"{r.concurrency:>5} "
                f"{r.ttft_mean_s:>9.3f}s "
                f"{r.ttft_median_s:>8.3f}s "
                f"{r.ttft_p99_s:>8.3f}s"
            )
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
