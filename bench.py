"""Run vLLM KV-cache offloading benchmarks defined in a JSON config.

Each config is expanded over any list-valued vllm_args/env/benchmark args
(cartesian product). For every expansion: start a vLLM server, wait for health,
run the benchmark driver, record TTFT/throughput to a timestamped CSV. Each
config runs N_REPETITIONS times, restarting vLLM each run.

Usage: python bench.py --config bench_config.json
"""

import argparse
import csv
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from common.benchmark_common import (
    resolve_storage_paths,
    save_profile_artifacts,
    write_dataclass_csv,
)
from common import vllm_server

# All values below are populated from the JSON config by apply_loaded_config().
DEFAULT_CONFIG_PATH = "bench_config.json"

MODEL: Optional[str] = None
MAX_MODEL_LEN: Optional[int] = None
GPU_MEM_UTIL: Optional[float] = None
DATA_DIR: Optional[str] = None
VLLM_PORT: Optional[int] = None
N_REPETITIONS: Optional[int] = None
SERVER_STARTUP_TIMEOUT: Optional[int] = None

# Flush mode is not supported with prefix_cache_benchmark.py; main() errors if set.
FLUSH_MODE = False

OUTPUT_DIR: Optional[str] = None
OUTPUT_CSV: Optional[str] = None

PREFIX_CACHE_SCRIPT: Optional[str] = None
PREFIX_CACHE_DEFAULTS: dict[str, Any] = {}

SHAREGPT_DATASET_PATH: Optional[str] = None
SHAREGPT_DEFAULTS: dict[str, Any] = {}

BAILIAN_TRACE_PATH: Optional[str] = None
BAILIAN_SCRIPT: Optional[str] = None
BAILIAN_DEFAULTS: dict[str, Any] = {}

LONGBENCH_SCRIPT: Optional[str] = None
LONGBENCH_DEFAULTS: dict[str, Any] = {}

CONFIGS: list[dict[str, Any]] = []


def _expand_dict(d: dict) -> list[tuple[dict, list[tuple[str, Any]]]]:
    """One (resolved_dict, varied_pairs) per cartesian combination of list values.

    varied_pairs lists only keys that had multiple choices, for sub-config naming.
    """
    keys = list(d.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in d.values()]
    combos = []
    for combo in itertools.product(*value_lists):
        resolved = dict(zip(keys, combo))
        varied = [(k, v) for k, v in zip(keys, combo) if isinstance(d[k], list)]
        combos.append((resolved, varied))
    return combos


def _value_for_name(value: Any) -> str:
    if isinstance(value, dict):
        if "shared_storage_path" in value.get("kv_connector_extra_config", {}):
            return value["kv_connector_extra_config"]["shared_storage_path"]
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark scenarios from a JSON config file")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to benchmark JSON config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory substituted for the {data_dir} token in storage paths; "
             "overrides the config's top-level 'data_dir'",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a prior results CSV; skip (config_name, repetition) pairs "
             "that completed without error and only run the missing ones",
    )
    return parser.parse_args()


def load_completed_keys(path: str) -> set[tuple[str, int]]:
    """Return (config_name, repetition) pairs from a prior results CSV that
    completed without error. Used by --resume to skip already-run experiments;
    rows carrying an error value are treated as missing so they get re-run."""
    completed: set[tuple[str, int]] = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("error") or "").strip():
                continue
            try:
                completed.add((row["config_name"], int(row["repetition"])))
            except (KeyError, ValueError):
                continue
    return completed


def _require_config_key(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
    return config[key]


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Top-level benchmark config must be a JSON object")
    if not isinstance(config.get("configs"), list) or not config["configs"]:
        raise ValueError("Config file must contain a non-empty 'configs' array")
    return config


def apply_loaded_config(config: dict[str, Any]) -> None:
    global MODEL, MAX_MODEL_LEN, GPU_MEM_UTIL, DATA_DIR, VLLM_PORT, N_REPETITIONS
    global SERVER_STARTUP_TIMEOUT, FLUSH_MODE, PREFIX_CACHE_SCRIPT, PREFIX_CACHE_DEFAULTS
    global SHAREGPT_DATASET_PATH, SHAREGPT_DEFAULTS, CONFIGS, OUTPUT_DIR, OUTPUT_CSV
    global BAILIAN_TRACE_PATH, BAILIAN_SCRIPT, BAILIAN_DEFAULTS
    global LONGBENCH_SCRIPT, LONGBENCH_DEFAULTS
    global SCBENCH_SCRIPT, SCBENCH_DEFAULTS

    MODEL = _require_config_key(config, "model")
    MAX_MODEL_LEN = int(_require_config_key(config, "max_model_len"))
    GPU_MEM_UTIL = float(_require_config_key(config, "gpu_mem_util"))
    DATA_DIR = config.get("data_dir")
    VLLM_PORT = int(_require_config_key(config, "vllm_port"))
    N_REPETITIONS = int(config.get("n_repetitions", 1))
    SERVER_STARTUP_TIMEOUT = int(config.get("server_startup_timeout", 200))

    FLUSH_MODE = bool(config.get("flush_mode", False))

    PREFIX_CACHE_SCRIPT = _require_config_key(config, "prefix_cache_script")
    PREFIX_CACHE_DEFAULTS = dict(config.get("prefix_cache_defaults", {}))
    SHAREGPT_DATASET_PATH = _require_config_key(config, "sharegpt_dataset_path")
    SHAREGPT_DEFAULTS = dict(config.get("sharegpt_defaults", {}))
    BAILIAN_TRACE_PATH = config.get("bailian_trace_path")
    BAILIAN_SCRIPT = config.get("bailian_script", "scripts/bailian_replay.py")
    BAILIAN_DEFAULTS = dict(config.get("bailian_defaults", {}))
    LONGBENCH_SCRIPT = config.get("longbench_script", "scripts/longbench_v2_replay.py")
    LONGBENCH_DEFAULTS = dict(config.get("longbench_defaults", {}))
    SCBENCH_SCRIPT = config.get("scbench_script", "scripts/scbench_replay.py")
    SCBENCH_DEFAULTS = dict(config.get("scbench_defaults", {}))
    CONFIGS = list(config["configs"])

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = str(config.get("output_prefix", "kv_benchmark"))
    OUTPUT_DIR = f"{output_prefix}_{run_timestamp}"
    OUTPUT_CSV = f"{output_prefix}_results_{run_timestamp}.csv"


def resolve_storage_paths_in_configs(data_dir: Optional[str]) -> None:
    needs_data_dir = any("{data_dir}" in p for c in CONFIGS for p in get_storage_paths(c))
    if needs_data_dir and not data_dir:
        print("ERROR: a storage config uses the {data_dir} token but no data_dir was provided. "
              "Set top-level 'data_dir' in the config or pass --data-dir.")
        sys.exit(1)
    if not data_dir:
        return
    for config in CONFIGS:
        for resolved in resolve_storage_paths(config, data_dir):
            print(f"  {config.get('name', 'config')} storage path -> {resolved}")


def get_storage_paths(config: dict[str, Any]) -> list[str]:
    """All on-disk storage roots a config writes to (shared_storage_path,
    tiering secondary_tiers[].root_dir, LMCache LMCACHE_LOCAL_DISK)."""
    paths: list[str] = []
    kv_transfer_cfg = config.get("vllm_args", {}).get("--kv-transfer-config")
    if isinstance(kv_transfer_cfg, str):
        try:
            kv_transfer_cfg = json.loads(kv_transfer_cfg)
        except json.JSONDecodeError:
            kv_transfer_cfg = None
    if isinstance(kv_transfer_cfg, dict):
        extra = kv_transfer_cfg.get("kv_connector_extra_config", {})
        shared = extra.get("shared_storage_path") or kv_transfer_cfg.get("shared_storage_path")
        if shared:
            paths.append(shared)
        for tier in extra.get("secondary_tiers", []):
            if isinstance(tier, dict) and tier.get("root_dir"):
                paths.append(tier["root_dir"])
    disk = config.get("env", {}).get("LMCACHE_LOCAL_DISK")
    if disk:
        paths.append(disk[len("file://"):] if disk.startswith("file://") else disk)
    return paths


def wipe_shared_storage(config: dict[str, Any]) -> None:
    paths = get_storage_paths(config)
    if not paths:
        return
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"  Wiped shared storage: {path}")
        else:
            print(f"  Shared storage path not present: {path}")
    print("  Sleeping 5s after shared storage wipe...")
    time.sleep(5)


def _merge_prefix_cache_args(config: dict) -> dict:
    merged = dict(PREFIX_CACHE_DEFAULTS)
    merged.update(config.get("prefix_cache_args", {}))
    return merged


def _merge_sharegpt_args(config: dict) -> dict:
    merged = dict(SHAREGPT_DEFAULTS)
    merged.update(config.get("sharegpt_args", {}))
    return merged


def _merge_bailian_args(config: dict) -> dict:
    merged = dict(BAILIAN_DEFAULTS)
    merged.update(config.get("bailian_args", {}))
    return merged


def _merge_longbench_args(config: dict) -> dict:
    merged = dict(LONGBENCH_DEFAULTS)
    merged.update(config.get("longbench_args", {}))
    return merged


def _merge_scbench_args(config: dict) -> dict:
    merged = dict(SCBENCH_DEFAULTS)
    merged.update(config.get("scbench_args", {}))
    return merged


def expand_config(cfg: dict) -> list[dict]:
    """
    Expand a single config into one concrete config per combination of
    list-valued vllm_args, env, and benchmark-specific args entries.

    Benchmark type is inferred from key presence:
      - "sharegpt_args" present  → sharegpt
      - "prefix_cache_args" present → prefix_cache
      - neither                  → prefix_cache with defaults
    """
    if "sharegpt_args" in cfg:
        benchmark    = "sharegpt"
        bench_combos = _expand_dict(_merge_sharegpt_args(cfg))
        bench_key    = "sharegpt_args"
    elif "bailian_args" in cfg:
        benchmark    = "bailian"
        bench_combos = _expand_dict(_merge_bailian_args(cfg))
        bench_key    = "bailian_args"
    elif "longbench_args" in cfg:
        benchmark    = "longbench"
        bench_combos = _expand_dict(_merge_longbench_args(cfg))
        bench_key    = "longbench_args"
    elif "scbench_args" in cfg:
        benchmark    = "scbench"
        bench_combos = _expand_dict(_merge_scbench_args(cfg))
        bench_key    = "scbench_args"
    elif "prefix_cache_args" in cfg:
        benchmark    = "prefix_cache"
        bench_combos = _expand_dict(_merge_prefix_cache_args(cfg))
        bench_key    = "prefix_cache_args"
    else:
        raise ValueError(
            f"Config '{cfg['name']}' has none of 'sharegpt_args', 'bailian_args', "
            f"'longbench_args', 'scbench_args', or 'prefix_cache_args'. Add one to "
            f"specify which benchmark to run."
        )

    vllm_combos  = _expand_dict(cfg.get("vllm_args", {}))
    env_combos   = _expand_dict(cfg.get("env", {}))

    expanded = []
    for (vllm_args, vllm_varied), (env, env_varied), (bench_args, bench_varied) in (
        itertools.product(vllm_combos, env_combos, bench_combos)
    ):
        varied_parts = []
        for k, v in vllm_varied:
            varied_parts.append(f"{k.lstrip('-').replace('-','_')}={_value_for_name(v)}")
        for k, v in env_varied:
            varied_parts.append(f"{k}={_value_for_name(v)}")
        for k, v in bench_varied:
            varied_parts.append(f"{k.lstrip('-').replace('-','_')}={_value_for_name(v)}")

        sub_name = f"{cfg['name']}[{','.join(varied_parts)}]" if varied_parts else cfg["name"]

        expanded.append({
            "name":              sub_name,
            "base_config_name":  cfg["name"],
            "description":       cfg.get("description", ""),
            "benchmark":         benchmark,
            "vllm_args":         vllm_args,
            "env":               env,
            bench_key:            bench_args,
            "profile_json":      cfg.get("profile_json", None),
        })
    return expanded


def all_expanded_configs(configs: list[dict]) -> list[dict]:
    result = []
    for cfg in configs:
        result.extend(expand_config(cfg))
    return result


@dataclass
class BenchmarkResult:
    config_name:              str
    base_config_name:         str
    config_description:       str
    benchmark:                str
    repetition:               int
    doc_len:                  Optional[int]   = None
    request_rate:             Optional[float] = None
    batch_size:               Optional[int]   = None
    chunk_size:               Optional[int]   = None
    bailian_task:             Optional[str]   = None
    longbench_domain:         Optional[str]   = None
    scbench_config:           Optional[str]   = None
    warmup_mean_ttft_s:       Optional[float] = None
    warmup_total_time_s:      Optional[float] = None
    warmup_prompt_count:      Optional[int]   = None
    warmup_successful_count:  Optional[int]   = None
    query_mean_ttft_s:        Optional[float] = None
    query_total_time_s:       Optional[float] = None
    query_prompt_count:       Optional[int]   = None
    query_successful_count:   Optional[int]   = None
    # prefix_cache_benchmark.py TTFT breakdown (warmup=fresh/cold, query=reuse/hit)
    overall_mean_ttft_s:      Optional[float] = None
    reuse_mean_ttft_s:        Optional[float] = None
    ttft_speedup_x:           Optional[float] = None
    time_reduction_pct:       Optional[float] = None
    per_request_csv:          Optional[str]   = None
    # one row per profiler kernel call
    gpu_transfer_csv:         Optional[str]   = None
    sg_completed:               Optional[int]   = None
    sg_duration_s:              Optional[float] = None
    sg_request_rate:            Optional[float] = None
    sg_request_throughput:      Optional[float] = None
    sg_output_throughput:       Optional[float] = None
    sg_total_token_throughput:  Optional[float] = None
    sg_mean_ttft_ms:            Optional[float] = None
    sg_median_ttft_ms:          Optional[float] = None
    sg_p95_ttft_ms:             Optional[float] = None
    sg_p99_ttft_ms:             Optional[float] = None
    sg_mean_tpot_ms:            Optional[float] = None
    sg_p99_tpot_ms:             Optional[float] = None
    sg_mean_itl_ms:             Optional[float] = None
    sg_p99_itl_ms:              Optional[float] = None
    sg_result_json:             Optional[str]   = None
    error:                    Optional[str]   = None


def build_vllm_command(vllm_args: dict, port: Optional[int] = None) -> list[str]:
    return vllm_server.build_vllm_command(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        gpu_mem_util=GPU_MEM_UTIL,
        vllm_args=vllm_args,
        port=port if port is not None else VLLM_PORT,
    )


def build_prefix_cache_command(pc_args: dict, port: Optional[int] = None) -> list[str]:
    p = port if port is not None else VLLM_PORT
    cmd = [
        sys.executable, PREFIX_CACHE_SCRIPT,
        "--model", MODEL,
        "--port",  str(p),
    ]
    for flag, value in pc_args.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(vllm_server.stringify_for_cli(value))
    return cmd


def build_sharegpt_command(sharegpt_args: dict, result_json_path: str, port: Optional[int] = None) -> list[str]:
    p = port if port is not None else VLLM_PORT
    cmd = [
        "vllm", "bench", "serve",
        "--backend",      "vllm",
        "--model",         MODEL,
        "--port",          str(p),
        "--dataset-name",  "sharegpt",
        "--dataset-path",  SHAREGPT_DATASET_PATH,
        "--endpoint", "/v1/completions",
        "--save-result",
        "--result-filename", result_json_path,
    ]
    for flag, value in sharegpt_args.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(vllm_server.stringify_for_cli(value))
    return cmd


def run_sharegpt_benchmark(sharegpt_args: dict, result_json_path: str, port: Optional[int] = None) -> Optional[dict]:
    cmd = build_sharegpt_command(sharegpt_args, result_json_path, port=port)
    print(f"\n  ▶  Running sharegpt benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        print(f"  ✗  ShareGPT benchmark exited with code {result.returncode}")
        return None
    return parse_sharegpt_result(result_json_path)


def parse_sharegpt_result(json_path: str) -> Optional[dict]:
    try:
        with open(json_path) as f:
            data = json.load(f)
        # vllm bench serve may wrap results in a list
        if isinstance(data, list):
            data = data[0]
        return {
            "sg_completed":               data.get("completed"),
            "sg_duration_s":              data.get("duration"),
            "sg_request_throughput":      data.get("request_throughput"),
            "sg_output_throughput":       data.get("output_throughput"),
            "sg_total_token_throughput":  data.get("total_token_throughput"),
            "sg_mean_ttft_ms":            data.get("mean_ttft_ms"),
            "sg_median_ttft_ms":          data.get("median_ttft_ms"),
            "sg_p95_ttft_ms":             data.get("p95_ttft_ms"),
            "sg_p99_ttft_ms":             data.get("p99_ttft_ms"),
            "sg_mean_tpot_ms":            data.get("mean_tpot_ms"),
            "sg_p99_tpot_ms":             data.get("p99_tpot_ms"),
            "sg_mean_itl_ms":             data.get("mean_itl_ms"),
            "sg_p99_itl_ms":              data.get("p99_itl_ms"),
            "sg_request_rate":            data.get("request_rate"),
        }
    except Exception as e:
        print(f"  ⚠  Could not parse sharegpt result '{json_path}': {e}")
        return None


def build_bailian_command(bailian_args: dict, csv_path: str, port: Optional[int] = None) -> list[str]:
    p = port if port is not None else VLLM_PORT
    cmd = [
        sys.executable, BAILIAN_SCRIPT,
        "--model", MODEL,
        "--port",  str(p),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trace-path", BAILIAN_TRACE_PATH,
        "--csv-output", csv_path,
        "--json-output",
    ]
    for flag, value in bailian_args.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(vllm_server.stringify_for_cli(value))
    return cmd


def run_bailian_benchmark(bailian_args: dict, csv_path: str, port: Optional[int] = None) -> Optional[dict]:
    cmd = build_bailian_command(bailian_args, csv_path, port=port)
    print(f"\n  ▶  Running bailian replay: {' '.join(cmd)}")
    result = vllm_server.run_benchmark_subprocess(cmd)
    if result.returncode != 0:
        print(f"  ✗  Bailian replay exited with code {result.returncode}")
        return None
    parsed = parse_prefix_cache_summary(result.stdout)
    if parsed is None:
        print("  ✗  Failed to parse bailian JSON summary from stdout")
    return parsed


def build_longbench_command(longbench_args: dict, csv_path: str, port: Optional[int] = None) -> list[str]:
    p = port if port is not None else VLLM_PORT
    cmd = [
        sys.executable, LONGBENCH_SCRIPT,
        "--model", MODEL,
        "--port",  str(p),
        "--csv-output", csv_path,
        "--json-output",
    ]
    for flag, value in longbench_args.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(vllm_server.stringify_for_cli(value))
    return cmd


def run_longbench_benchmark(longbench_args: dict, csv_path: str, port: Optional[int] = None) -> Optional[dict]:
    cmd = build_longbench_command(longbench_args, csv_path, port=port)
    print(f"\n  ▶  Running LongBench v2 replay: {' '.join(cmd)}")
    result = vllm_server.run_benchmark_subprocess(cmd)
    if result.returncode != 0:
        print(f"  ✗  LongBench v2 replay exited with code {result.returncode}")
        return None
    parsed = parse_prefix_cache_summary(result.stdout)
    if parsed is None:
        print("  ✗  Failed to parse LongBench v2 JSON summary from stdout")
    return parsed


def build_scbench_command(scbench_args: dict, csv_path: str, port: Optional[int] = None) -> list[str]:
    p = port if port is not None else VLLM_PORT
    cmd = [
        sys.executable, SCBENCH_SCRIPT,
        "--model", MODEL,
        "--port",  str(p),
        "--csv-output", csv_path,
        "--json-output",
    ]
    for flag, value in scbench_args.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(vllm_server.stringify_for_cli(value))
    return cmd


def run_scbench_benchmark(scbench_args: dict, csv_path: str, port: Optional[int] = None) -> Optional[dict]:
    cmd = build_scbench_command(scbench_args, csv_path, port=port)
    print(f"\n  ▶  Running SCBench replay: {' '.join(cmd)}")
    result = vllm_server.run_benchmark_subprocess(cmd)
    if result.returncode != 0:
        print(f"  ✗  SCBench replay exited with code {result.returncode}")
        return None
    parsed = parse_prefix_cache_summary(result.stdout)
    if parsed is None:
        print("  ✗  Failed to parse SCBench JSON summary from stdout")
    return parsed


def wait_for_server(port: int, timeout: int) -> bool:
    return vllm_server.wait_for_server(port, timeout)


def start_server(config: dict, log_path: str, port: Optional[int] = None) -> subprocess.Popen:
    cmd = build_vllm_command(config["vllm_args"], port=port)
    env = {**os.environ, **{k: str(v) for k, v in config.get("env", {}).items()}}
    print(f"\n  ▶  Starting vLLM: {' '.join(cmd)}")
    if config.get("env"):
        print(f"     Env overrides: { {k: v for k, v in config['env'].items()} }")
    print(f"     Server log: {log_path}")
    return vllm_server.start_server(cmd, log_path, env=env)


def stop_server(proc: subprocess.Popen) -> None:
    vllm_server.stop_server(proc)


def run_prefix_cache(pc_args: dict, csv_path: str, port: Optional[int] = None) -> Optional[dict]:
    cmd = build_prefix_cache_command(pc_args, port=port) + ["--csv-output", csv_path, "--json-output"]
    result = vllm_server.run_benchmark_subprocess(cmd)
    if result.returncode != 0:
        return None
    parsed = parse_prefix_cache_summary(result.stdout)
    if parsed is None:
        print("  ✗  Failed to parse prefix_cache_benchmark JSON summary from stdout")
    return parsed


def parse_prefix_cache_summary(stdout: str) -> Optional[dict]:
    """Parse the JSON summary line emitted by prefix_cache_benchmark.py --json-output.

    The script prints one json.dumps(summary) line; map its TTFT fields onto
    BenchmarkResult attributes (warmup=fresh/cold, query=reuse/cache-hit).
    """
    summary = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "all_mean_ttft" in obj:
            summary = obj

    if summary is None:
        return None

    return {
        "warmup_mean_ttft_s":     summary.get("fresh_mean_ttft"),
        "query_mean_ttft_s":      summary.get("reuse_mean_ttft"),
        "reuse_mean_ttft_s":      summary.get("reuse_mean_ttft"),
        "overall_mean_ttft_s":    summary.get("all_mean_ttft"),
        "ttft_speedup_x":         summary.get("ttft_speedup"),
        "query_total_time_s":     summary.get("wall_clock_s"),
        "query_prompt_count":     summary.get("total_requests"),
        "query_successful_count": summary.get("successful"),
    }


def save_profile(profile_json: Optional[str], run_num: int) -> Optional[str]:
    try:
        return save_profile_artifacts(profile_json, OUTPUT_DIR, f"run_{run_num:03d}")
    except Exception as e:
        print(f"  ⚠  Could not parse profile JSON '{profile_json}': {e}")
        return None


def cleanup_profile_registry(profile_json: Optional[str]) -> None:
    """Delete the profiler's <profile_json>.registry sidecar between runs so the
    next run does not reuse stale registry state."""
    if not profile_json:
        return
    registry = f"{profile_json}.registry"
    if os.path.exists(registry):
        os.remove(registry)
        print(f"  Deleted profile registry: {registry}")


def write_csv(results: list[BenchmarkResult], path: str) -> None:
    write_dataclass_csv(results, path)
    print(f"\n✅  Results written to: {path}")

def extract_result_metadata(config: dict) -> dict[str, Any]:
    benchmark = config.get("benchmark", "prefix_cache")
    prefix_cache_args = config.get("prefix_cache_args", {})
    sharegpt_args = config.get("sharegpt_args", {})

    doc_size = prefix_cache_args.get("--doc-size")
    doc_len = None
    if doc_size is not None:
        try:
            doc_len = int(str(doc_size).split("-", 1)[0])
        except ValueError:
            doc_len = None
    bailian_args = config.get("bailian_args", {})
    longbench_args = config.get("longbench_args", {})
    scbench_args = config.get("scbench_args", {})
    request_rate = (
        sharegpt_args.get("--request-rate")
        or bailian_args.get("--arrival-rate")
        or longbench_args.get("--arrival-rate")
        or scbench_args.get("--arrival-rate")
    )
    batch_size = config.get("vllm_args", {}).get("--max-num-batched-tokens")
    chunk_size = config.get("env", {}).get("LMCACHE_CHUNK_SIZE")

    return {
        "doc_len": doc_len,
        "request_rate": float(request_rate) if request_rate is not None else None,
        "batch_size": int(batch_size) if batch_size is not None else None,
        "chunk_size": int(chunk_size) if chunk_size is not None else None,
        "bailian_task": bailian_args.get("--task"),
        "longbench_domain": longbench_args.get("--domain"),
        "scbench_config": scbench_args.get("--config"),
    }


def _run_one_experiment(config: dict, rep: int, run_num: int, port: int) -> "BenchmarkResult":
    result = BenchmarkResult(
        config_name=config["name"],
        base_config_name=config["base_config_name"],
        config_description=config["description"],
        benchmark=config.get("benchmark", "prefix_cache"),
        repetition=rep,
        **extract_result_metadata(config),
    )
    try:
        benchmark = config.get("benchmark", "prefix_cache")
        if benchmark == "sharegpt":
            result_json = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_sharegpt.json")
            parsed = run_sharegpt_benchmark(config["sharegpt_args"], result_json, port=port)
            if parsed:
                for key, val in parsed.items():
                    if hasattr(result, key):
                        setattr(result, key, val)
                result.sg_result_json = result_json
        elif benchmark == "bailian":
            csv_path = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}.csv")
            result.per_request_csv = csv_path
            parsed = run_bailian_benchmark(config["bailian_args"], csv_path, port=port)
            if parsed:
                for key, val in parsed.items():
                    if hasattr(result, key):
                        setattr(result, key, val)
        elif benchmark == "longbench":
            csv_path = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}.csv")
            result.per_request_csv = csv_path
            parsed = run_longbench_benchmark(config["longbench_args"], csv_path, port=port)
            if parsed:
                for key, val in parsed.items():
                    if hasattr(result, key):
                        setattr(result, key, val)
        elif benchmark == "scbench":
            csv_path = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}.csv")
            result.per_request_csv = csv_path
            parsed = run_scbench_benchmark(config["scbench_args"], csv_path, port=port)
            if parsed:
                for key, val in parsed.items():
                    if hasattr(result, key):
                        setattr(result, key, val)
        else:
            csv_path = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}.csv")
            result.per_request_csv = csv_path
            parsed = run_prefix_cache(config["prefix_cache_args"], csv_path, port=port)
            if parsed:
                for key, val in parsed.items():
                    if hasattr(result, key):
                        setattr(result, key, val)

        if benchmark == "sharegpt":
            if result.sg_completed is None:
                result.error = "Could not parse sharegpt result JSON"
        else:
            if result.overall_mean_ttft_s is None:
                label = {
                    "bailian": "bailian",
                    "longbench": "longbench",
                    "scbench": "scbench",
                }.get(benchmark, "prefix_cache_benchmark")
                result.error = f"Could not parse TTFT from {label} summary"
    except Exception as e:
        result.error = str(e)
        print(f"\n  ✗  Error: {e}")
    return result


def _print_result_summary(result: BenchmarkResult, config: dict) -> None:
    if result.error:
        print(f"\n  Result: ERROR — {result.error}")
    elif config.get("benchmark") == "sharegpt":
        print(f"\n  Result summary (sharegpt):")
        print(f"    Completed          : {result.sg_completed}")
        print(f"    Duration           : {result.sg_duration_s}s")
        print(f"    Request throughput : {result.sg_request_throughput} req/s")
        print(f"    Output throughput  : {result.sg_output_throughput} tok/s")
        print(f"    Mean TTFT          : {result.sg_mean_ttft_ms}ms")
        print(f"    P99  TTFT          : {result.sg_p99_ttft_ms}ms")
        print(f"    Mean TPOT          : {result.sg_mean_tpot_ms}ms")
    else:
        print(f"\n  Result summary:")
        print(f"    Warmup TTFT : {result.warmup_mean_ttft_s}s")
        print(f"    Query TTFT  : {result.query_mean_ttft_s}s")
        print(f"    Speedup     : {result.ttft_speedup_x}x")
        print(f"    Time saved  : {result.time_reduction_pct}%")


def _print_final_summary(all_results: list[BenchmarkResult]) -> None:
    print(f"\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Config':<45} {'Rep':>4}  {'W-TTFT':>8}  {'Q-TTFT':>8}  {'Speedup':>8}  {'TimeSaved':>10}")
    print(f"  {'-'*45} {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in all_results:
        name = r.config_name[:45]
        if r.error:
            print(f"  {name:<45} {r.repetition:>4}  ERROR: {r.error}")
        else:
            print(
                f"  {name:<45} {r.repetition:>4}"
                f"  {str(r.warmup_mean_ttft_s)+'s':>8}"
                f"  {str(r.query_mean_ttft_s)+'s':>8}"
                f"  {str(r.ttft_speedup_x)+'x':>8}"
                f"  {str(r.time_reduction_pct)+'%':>10}"
            )


def main():
    args = parse_args()
    loaded_config = load_config(args.config)
    apply_loaded_config(loaded_config)
    resolve_storage_paths_in_configs(args.data_dir or DATA_DIR)

    if not Path(PREFIX_CACHE_SCRIPT).exists():
        print(f"ERROR: workload script not found at '{PREFIX_CACHE_SCRIPT}'.")
        print("Set 'prefix_cache_script' to prefix_cache_benchmark.py.")
        sys.exit(1)

    if FLUSH_MODE:
        print("ERROR: flush_mode is not supported with prefix_cache_benchmark.py "
              "(no --warmup-only/--query-only rounds). Set flush_mode=false.")
        sys.exit(1)

    configs = all_expanded_configs(CONFIGS)
    completed = load_completed_keys(args.resume) if args.resume else set()
    experiments = [
        (c, r)
        for c in configs
        for r in range(1, N_REPETITIONS + 1)
        if (c["name"], r) not in completed
    ]
    total_runs = len(experiments)

    print(f"\n{'═'*70}")
    print(f"  Benchmark plan: {len(configs)} config(s) × {N_REPETITIONS} rep(s) = {total_runs} run(s)")
    if args.resume:
        skipped = len(configs) * N_REPETITIONS - total_runs
        print(f"  Resume: '{args.resume}' → skipping {skipped} completed run(s)")
    for i, c in enumerate(configs, 1):
        print(f"  [{i:>2}] {c['name']}")
    print(f"{'═'*70}")

    if total_runs == 0:
        print("  Nothing to run — all configs already completed in the resume CSV.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Snapshot the config used for this run so results stay reproducible.
    config_copy = os.path.join(OUTPUT_DIR, os.path.basename(args.config))
    shutil.copy2(args.config, config_copy)
    print(f"  Saved config copy: {config_copy}")

    all_results: list[BenchmarkResult] = []

    for run_num, (config, rep) in enumerate(experiments, 1):
        print(f"\n{'═'*70}")
        print(f"  Config : {config['name']}")
        print(f"  Desc   : {config['description']}")
        print(f"  Run    : {rep}/{N_REPETITIONS}  (overall {run_num}/{total_runs})")
        print(f"{'═'*70}")

        server_proc = None
        profile_json_path = config.get("profile_json")
        result = BenchmarkResult(
            config_name=config["name"],
            base_config_name=config["base_config_name"],
            config_description=config["description"],
            benchmark=config.get("benchmark", "prefix_cache"),
            repetition=rep,
            **extract_result_metadata(config),
        )
        try:
            server_log = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_server.log")
            server_proc = start_server(config, server_log)
            print(f"\n  ⏳ Waiting up to {SERVER_STARTUP_TIMEOUT}s for vLLM to be ready...")
            if not wait_for_server(VLLM_PORT, SERVER_STARTUP_TIMEOUT):
                raise RuntimeError(f"vLLM did not become healthy within {SERVER_STARTUP_TIMEOUT}s")
            print("  ✓  Server is ready.")
            result = _run_one_experiment(config, rep, run_num, VLLM_PORT)
        except Exception as e:
            result.error = str(e)
            print(f"\n  ✗  Error: {e}")
        finally:
            if server_proc:
                stop_server(server_proc)

        result.gpu_transfer_csv = save_profile(profile_json_path, run_num)
        cleanup_profile_registry(profile_json_path)
        wipe_shared_storage(config)
        all_results.append(result)
        write_csv(all_results, OUTPUT_CSV)
        _print_result_summary(result, config)

    _print_final_summary(all_results)


if __name__ == "__main__":
    main()
