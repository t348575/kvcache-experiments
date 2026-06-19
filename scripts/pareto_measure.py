"""Measure the two TTFT curves needed for a KV-cache Pareto frontier.

  A  cold_prefill  f(N): TTFT to cold-compute N tokens, prefix caching off.
  B  cache_hit     g(P): TTFT when P tokens load from the native KV offload
                         cache plus a ~1-token fresh suffix, offloading on.

The frontier in (doc_size x prefix_fraction) space is where
    f(D) = g(frac * D) + f((1 - frac) * D)

An optional concurrency sweep runs both curves at several concurrency levels.
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

from common.benchmark_common import pct, resolve_storage_paths, save_profile_artifacts, write_dataclass_csv
from common import vllm_server
from common.model_info import fetch_model_geometry

# Model / KV params are resolved at startup by select_model_preset():
#   - model id + alias come from the config "models" map
#   - kv_cache_dtype is prompted interactively (auto / fp8)
#   - kv geometry (layers, kv-heads, head-dim) is fetched from the model's
#     Hugging Face config.json (see common/model_info.fetch_model_geometry)
MODEL           = None
KV_NUM_LAYERS   = None
KV_NUM_KV_HEADS = None
KV_HEAD_DIM     = None
KV_DTYPE_BYTES  = None
KV_CACHE_DTYPE  = None

# simple-profiler writes this file in the working directory.
# Both baseline and native_offload use upstream vLLM (engine-core process).
PROFILE_JSON = "merge.json"

DEFAULT_CONFIG_PATH = "pareto_config.json"

# Run parameters + server configs are loaded from the JSON config by
# apply_loaded_config(). Declared here so module-level references resolve.
MAX_MODEL_LEN          = None
GPU_MEM_UTIL           = None
VLLM_PORT              = None
OUTPUT_TOKENS          = None
SERVER_STARTUP_TIMEOUT = None
KV_BUFFER              = None
COLD_ONLY              = None
BENCHMARK_SCRIPT       = None
DOC_SIZES              = None
N_REPEATS_SERIAL       = None
N_REPEATS_CONCURRENCY  = None
CONCURRENCY_DOC_SIZES  = None
CONCURRENCY_LEVELS     = None
MODELS                 = None
SERVER_CONFIGS         = None
OUTPUT_DIR             = None
OUTPUT_CSV             = None


def load_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("Top-level pareto config must be a JSON object")
    if not isinstance(config.get("configs"), dict) or not config["configs"]:
        raise ValueError("Config file must contain a non-empty 'configs' object")
    if not isinstance(config.get("models"), dict) or not config["models"]:
        raise ValueError("Config file must contain a non-empty 'models' object")
    return config


def apply_loaded_config(config: dict) -> None:
    global MAX_MODEL_LEN, GPU_MEM_UTIL, VLLM_PORT, OUTPUT_TOKENS
    global SERVER_STARTUP_TIMEOUT, KV_BUFFER, COLD_ONLY, BENCHMARK_SCRIPT
    global DOC_SIZES, N_REPEATS_SERIAL, N_REPEATS_CONCURRENCY
    global CONCURRENCY_DOC_SIZES, CONCURRENCY_LEVELS, MODELS, SERVER_CONFIGS
    global OUTPUT_DIR, OUTPUT_CSV

    MAX_MODEL_LEN          = int(config["max_model_len"])
    GPU_MEM_UTIL           = float(config["gpu_mem_util"])
    VLLM_PORT              = int(config["vllm_port"])
    OUTPUT_TOKENS          = int(config.get("output_tokens", 1))
    SERVER_STARTUP_TIMEOUT = int(config.get("server_startup_timeout", 200))
    KV_BUFFER              = float(config.get("kv_buffer", 0.25))
    COLD_ONLY              = bool(config.get("cold_only", False))
    BENCHMARK_SCRIPT       = config.get("benchmark_script", "prefix_cache_benchmark.py")
    DOC_SIZES              = list(config["doc_sizes"])
    N_REPEATS_SERIAL       = int(config.get("n_repeats_serial", 3))
    N_REPEATS_CONCURRENCY  = int(config.get("n_repeats_concurrency", 3))
    CONCURRENCY_DOC_SIZES  = list(config.get("concurrency_doc_sizes", []))
    CONCURRENCY_LEVELS     = list(config.get("concurrency_levels", []))
    MODELS                 = dict(config["models"])
    SERVER_CONFIGS         = dict(config["configs"])

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = str(config.get("output_prefix", "pareto_measure"))
    OUTPUT_DIR = f"{output_prefix}_{run_timestamp}"
    OUTPUT_CSV = f"{output_prefix}_{run_timestamp}.csv"


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


def build_job_plan(server_configs: list[str]) -> list[Job]:
    jobs: list[Job] = []
    selected_configs = [cfg for cfg in SERVER_CONFIGS if cfg in server_configs]
    cache_configs = [cfg for cfg in selected_configs if cfg != "baseline"]

    # Each job runs against a freshly started vLLM instance (clean cache).

    # ── Curve A: cold prefill, serial ────────────────────────────────────────
    if "baseline" in selected_configs:
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
        for cfg in cache_configs:
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
            if "baseline" in selected_configs:
                jobs.append(Job(
                    curve="cold_prefill", server_config="baseline",
                    doc_size=d, concurrency=c,
                    n_requests=n,
                    prefix_reuse_pct=0.0, prefix_size="0",
                ))
            for cfg in cache_configs:
                jobs.append(Job(
                    curve="cache_hit", server_config=cfg,
                    doc_size=d, concurrency=c,
                    n_requests=1 + n,
                    prefix_reuse_pct=1.0, prefix_size="1.0",
                ))

    return jobs


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
        # The native offload evicts ALL completed request KV blocks to CPU, not
        # just the warmup. Size for the warmup + all concurrent queries.
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


def build_vllm_command(vllm_args: dict) -> list[str]:
    cmd = vllm_server.build_vllm_command(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        gpu_mem_util=GPU_MEM_UTIL,
        vllm_args=vllm_args,
        port=VLLM_PORT,
        launcher=["vllm", "serve"],
        model_as_flag=False,
        extra_base_args=["--kv-cache-dtype", KV_CACHE_DTYPE],
        skip_keys=("is_storage",),
    )
    print(f"running {cmd}")
    return cmd


def wait_for_server(port: int, timeout: int) -> bool:
    return vllm_server.wait_for_server(port, timeout)


def start_server(config_name: str, log_path: str, job: "Job") -> subprocess.Popen:
    vllm_args = dict(SERVER_CONFIGS[config_name])
    if config_name == "native_offload":
        offload_gb = compute_offload_size_gb(job)
        vllm_args["--kv-offloading-size"] = str(offload_gb)
    cmd = build_vllm_command(vllm_args)
    print(f"\n  ▶  Starting vLLM [{config_name}]: {' '.join(cmd)}")
    print(f"     Log: {log_path}")
    return vllm_server.start_server(cmd, log_path)


def is_storage_config(config_name: str) -> bool:
    return bool(SERVER_CONFIGS[config_name].get("is_storage"))


def get_shared_storage_path(config_name: str) -> Optional[str]:
    cfg = SERVER_CONFIGS[config_name]
    kv_transfer_cfg = cfg.get("--kv-transfer-config")
    if kv_transfer_cfg is None:
        return None
    try:
        if isinstance(kv_transfer_cfg, str):
            kv_transfer_cfg = json.loads(kv_transfer_cfg)
    except json.JSONDecodeError:
        return None
    return (
        kv_transfer_cfg.get("kv_connector_extra_config", {}).get("shared_storage_path")
        or kv_transfer_cfg.get("shared_storage_path")
    )


def stop_server(proc: subprocess.Popen) -> None:
    vllm_server.stop_server(proc)


def wipe_shared_storage(config_name: str) -> None:
    shared_path = get_shared_storage_path(config_name)
    if shared_path and os.path.exists(shared_path):
        shutil.rmtree(shared_path)
        print(f"  🗑  Wiped shared_storage_path: {shared_path}")
    elif shared_path:
        print(f"  🗑  shared_storage_path does not exist: {shared_path}")


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
    proc = vllm_server.run_benchmark_subprocess(cmd)
    if proc.returncode != 0:
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


def write_csv(results: list[Result], path: str) -> None:
    write_dataclass_csv(results, path)
    print(f"  💾  Results → {path}  ({len(results)} rows)")


def select_model_preset() -> None:
    """Interactively choose a model, pick its KV cache dtype, fetch its geometry."""
    global MODEL, KV_NUM_LAYERS, KV_NUM_KV_HEADS, KV_HEAD_DIM, KV_DTYPE_BYTES, KV_CACHE_DTYPE
    names = list(MODELS.keys())
    print("\nAvailable models:")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {name}  ({MODELS[name]})")
    while True:
        raw = input(f"Select model [1-{len(names)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            MODEL = MODELS[names[int(raw) - 1]]
            break
        print(f"  Enter a number between 1 and {len(names)}.")

    while True:
        raw = input("KV cache dtype [auto/fp8] (default auto): ").strip().lower()
        raw = raw or "auto"
        if raw in ("auto", "fp8"):
            KV_CACHE_DTYPE = raw
            break
        print("  Enter 'auto' or 'fp8'.")
    KV_DTYPE_BYTES = 1 if KV_CACHE_DTYPE == "fp8" else 2

    KV_NUM_LAYERS, KV_NUM_KV_HEADS, KV_HEAD_DIM = fetch_model_geometry(MODEL)
    print(f"  → {MODEL}  (kv_cache_dtype={KV_CACHE_DTYPE})")
    print(f"     geometry: layers={KV_NUM_LAYERS}  kv_heads={KV_NUM_KV_HEADS}  "
          f"head_dim={KV_HEAD_DIM}  dtype_bytes={KV_DTYPE_BYTES}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pareto measurement for KV-cache prefix caching")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to pareto JSON config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--wipe-shared-storage",
        action="store_true",
        help="After each storage job, rm -rf the shared_storage_path",
    )
    parser.add_argument(
        "--server-configs",
        nargs="+",
        default=None,
        help="Which server configs to run (default: all configs in the config file)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Base directory substituted for the {data_dir} token in storage paths",
    )
    return parser.parse_args()


def resolve_storage_paths_in_configs(data_dir: str) -> None:
    """Substitute the {data_dir} token in the shared_storage_path of storage configs."""
    for name, cfg in SERVER_CONFIGS.items():
        if not cfg.get("is_storage"):
            continue
        wrapper = {"vllm_args": cfg}
        for resolved in resolve_storage_paths(wrapper, data_dir):
            print(f"  {name} storage path → {resolved}")


def main():
    args = parse_args()
    apply_loaded_config(load_config(args.config))

    if args.server_configs is None:
        server_configs = list(SERVER_CONFIGS)
    else:
        invalid = [c for c in args.server_configs if c not in SERVER_CONFIGS]
        if invalid:
            print(f"ERROR: unknown server configs {invalid}; "
                  f"available: {list(SERVER_CONFIGS)}")
            sys.exit(1)
        server_configs = args.server_configs

    select_model_preset()
    resolve_storage_paths_in_configs(args.data_dir)

    if not Path(BENCHMARK_SCRIPT).exists():
        print(f"ERROR: '{BENCHMARK_SCRIPT}' not found.")
        sys.exit(1)

    jobs = build_job_plan(server_configs)

    print(f"\n  Server configs : {', '.join(server_configs)}")
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
            if args.wipe_shared_storage and is_storage_config(job.server_config):
                wipe_shared_storage(job.server_config)

        if is_storage_config(job.server_config):
            print(f"  ⏳ Sleeping 5s before next storage job…")
            time.sleep(5)

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
