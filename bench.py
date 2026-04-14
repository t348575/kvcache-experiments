"""
kv_benchmark.py
───────────────
Benchmarks vLLM KV-cache CPU offloading strategies using LMCache's
Long Doc QA workload generator.

For each named configuration the script will:
  1. Expand any list-valued arguments into all combinations (cartesian product)
     across vllm_args, env, AND long_doc_qa_args
  2. For each combination, spawn a vLLM server with those arguments
  3. Wait for the server to become healthy
  4. Run long_doc_qa.py and capture TTFT / throughput metrics
  5. Repeat steps 2-4 for N_REPETITIONS, restarting vLLM each time
  6. Write all results to a timestamped CSV file

Defining multi-value tests
──────────────────────────
Any argument in `vllm_args`, `env`, or `long_doc_qa_args` can be a LIST.
The script will test every combination (cartesian product) automatically.

long_doc_qa_args
────────────────
Each config inherits from LONG_DOC_QA_DEFAULTS, then applies its own
`long_doc_qa_args` on top (merged, not replaced). This means you only
need to specify the keys you want to change or sweep per config.

  "--flag": None            →  appended as a bare flag
  "--arg":  "value"         →  appended as --arg value
  "--arg":  ["a", "b"]     →  two sub-configs, one with --arg a, one with --arg b

Example — sweep repeat-count across all configs:

    LONG_DOC_QA_DEFAULTS = {
        "--num-documents":  "20",
        "--repeat-count":   ["1", "5", "10"],   # ← 3 sub-configs per config
        ...
    }

Example — override just one arg for a specific config:

    {
        "name": "lmcache_cpu",
        "long_doc_qa_args": {
            "--document-length": ["5000", "10000"],   # sweep doc sizes for this config only
        },
        ...
    }

Usage
─────
  python kv_benchmark.py

Requirements
────────────
  pip install vllm lmcache requests
  git clone https://github.com/LMCache/LMCache.git   # for long_doc_qa.py
"""

import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit this section to add / remove test scenarios
# ─────────────────────────────────────────────────────────────────────────────

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_MODEL_LEN = 92000
GPU_MEM_UTIL = 0.9
VLLM_PORT = 8000
N_REPETITIONS = 1            # how many times each expanded config is tested
SERVER_STARTUP_TIMEOUT = 200  # seconds to wait for vLLM to become healthy

# ── Flush mode ────────────────────────────────────────────────────────────────
# When FLUSH_MODE is True, each run follows this sequence (instead of the
# normal single long_doc_qa invocation):
#   1. Start server
#   2. Warmup-only pass  (using the config's own long_doc_qa_args)
#   3. Flush pass        (static: 2 docs × 90 k tokens, warmup-only, 1 inflight)
#   4. Query-only pass   (using the config's own long_doc_qa_args)
#   5. Stop server
# The goal is to evict any GPU KV-cache populated during the warmup before
# the timed query round begins.
FLUSH_MODE = False

FLUSH_NUM_DOCS    = 2
FLUSH_DOC_LENGTH  = 90000
FLUSH_MAX_INFLIGHT = 1
_RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f"kv_benchmark_{_RUN_TIMESTAMP}"
OUTPUT_CSV = f"kv_benchmark_results_{_RUN_TIMESTAMP}.csv"

LONG_DOC_QA_SCRIPT = "lmcache-longdoc.py"

# Global defaults for long_doc_qa — inherited by every config.
# Any key can be a list to sweep across all configs.
# Per-config `long_doc_qa_args` overrides individual keys.
LONG_DOC_QA_DEFAULTS: dict[str, Any] = {
    "--num-documents":         "5",
    "--document-length":       "10000",
    "--output-len":            "1",
    "--repeat-count":          "5",
    "--repeat-mode":           "random",
    "--max-inflight-requests": "4",
}

# ── ShareGPT / vllm bench serve ───────────────────────────────────────────────
# Path to the ShareGPT JSON dataset file.
SHAREGPT_DATASET_PATH = "dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

# Global defaults for vllm bench serve — inherited by every sharegpt config.
# Per-config `sharegpt_args` overrides individual keys.
SHAREGPT_DEFAULTS: dict[str, Any] = {
    "--num-prompts":    "100",
    "--request-rate":   "32",
    "--percentile-metrics": "ttft,tpot,itl",
    "--metric-percentiles": "95,99",
    "--output-len": 1
}

# ── Define your test scenarios here ──────────────────────────────────────────
#
# benchmark       : "long_doc_qa" (default) | "sharegpt"
# vllm_args       : dict  { "--flag": None | "value" | ["v1", "v2", ...] }
# env             : dict  { "VAR":   "value"         | ["v1", "v2", ...] }
# long_doc_qa_args: dict  { "--arg": "value"         | ["v1", "v2", ...] }
#                   (merged on top of LONG_DOC_QA_DEFAULTS; only used when benchmark="long_doc_qa")
# sharegpt_args   : dict  { "--arg": "value"         | ["v1", "v2", ...] }
#                   (merged on top of SHAREGPT_DEFAULTS; only used when benchmark="sharegpt")
# profile_json    : str | None — simple-profiler output file for this config, or None
#                   "results_worker0.json"       — LMCache-patched vLLM (worker process)
#                   "results_engine_core_0.json" — upstream vLLM (engine-core process)
#
# Any list value in any of the three dicts triggers cartesian-product expansion.
# Sub-config names are auto-generated from whichever parameters were swept.
#
# ShareGPT example:
#   {
#       "name": "offloading_sharegpt",
#       "benchmark": "sharegpt",
#       "vllm_args": { "--kv-offloading-backend": "native", ... },
#       "sharegpt_args": { "--num-prompts": ["100", "500"] },
#   }
#
CONFIGS = [
    {
        "name": "baseline",
        "description": "No CPU offloading, no GPU prefix caching",
        "vllm_args": {
            "--no-enable-prefix-caching": None
        },
        "profile_json": "results_engine_core_0.json",
        # "sharegpt_args": {
        #     "--request-rate": ["8", "16", "32"]
        # }
        "long_doc_qa_args": {
            "--document-length": ["1024", "10240", "40960", "81920"]
        }
    },
    {
        "name": "offloading",
        "description": "vLLM built-in OffloadingConnector",
        "profile_json": "results_engine_core_0.json",
        "vllm_args": {
            "--no-enable-prefix-caching": None,
            "--kv-offloading-backend":    "native",
            "--kv-offloading-size":       "128",
            "--disable-hybrid-kv-cache-manager": None,
        },
        "profile_json": "results_engine_core_0.json",
        # "sharegpt_args": {
        #     "--request-rate": ["8", "16", "32"]
        # }
        "long_doc_qa_args": {
            "--document-length": ["1024", "10240", "40960", "81920"],
        },
    },
    {
        "name": "lmcache_cpu",
        "description": "LMCache CPU offloading — sweep chunk sizes and doc lengths",
        "profile_json": "results_worker0.json",
        "vllm_args": {
            "--no-enable-prefix-caching": None,
            "--kv-transfer-config":
                '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
        },
        "env": {
            "LMCACHE_LOCAL_CPU":          "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "128",
            "LMCACHE_CHUNK_SIZE":         ["64"]
        },
        "profile_json": "results_worker0.json",
        # "sharegpt_args": {
        #     "--request-rate": ["8", "16", "32"]
        # }
        "long_doc_qa_args": {
            "--document-length": ["1024", "10240", "40960", "81920"],
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG EXPANSION  (cartesian product over all list-valued parameters)
# ─────────────────────────────────────────────────────────────────────────────

def _expand_dict(d: dict) -> list[tuple[dict, list[tuple[str, Any]]]]:
    """
    Given a dict that may contain list values, return a list of
    (resolved_dict, [(key, chosen_value), ...]) tuples — one per combination.
    The second element only includes keys that had multiple choices (for naming).
    """
    keys = list(d.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in d.values()]
    combos = []
    for combo in itertools.product(*value_lists):
        resolved = dict(zip(keys, combo))
        varied = [(k, v) for k, v in zip(keys, combo) if isinstance(d[k], list)]
        combos.append((resolved, varied))
    return combos


def _merge_long_doc_qa_args(config: dict) -> dict:
    merged = dict(LONG_DOC_QA_DEFAULTS)
    merged.update(config.get("long_doc_qa_args", {}))
    return merged


def _merge_sharegpt_args(config: dict) -> dict:
    merged = dict(SHAREGPT_DEFAULTS)
    merged.update(config.get("sharegpt_args", {}))
    return merged


def expand_config(cfg: dict) -> list[dict]:
    """
    Expand a single config into one concrete config per combination of
    list-valued vllm_args, env, and benchmark-specific args entries.

    Benchmark type is inferred from key presence:
      - "sharegpt_args" present  → sharegpt
      - "long_doc_qa_args" present → long_doc_qa
      - neither                  → long_doc_qa with defaults
    """
    if "sharegpt_args" in cfg:
        benchmark    = "sharegpt"
        bench_combos = _expand_dict(_merge_sharegpt_args(cfg))
        bench_key    = "sharegpt_args"
    elif "long_doc_qa_args" in cfg:
        benchmark    = "long_doc_qa"
        bench_combos = _expand_dict(_merge_long_doc_qa_args(cfg))
        bench_key    = "long_doc_qa_args"
    else:
        raise ValueError(
            f"Config '{cfg['name']}' has neither 'sharegpt_args' nor 'long_doc_qa_args'. "
            f"Add one to specify which benchmark to run."
        )

    vllm_combos  = _expand_dict(cfg.get("vllm_args", {}))
    env_combos   = _expand_dict(cfg.get("env", {}))

    expanded = []
    for (vllm_args, vllm_varied), (env, env_varied), (bench_args, bench_varied) in (
        itertools.product(vllm_combos, env_combos, bench_combos)
    ):
        varied_parts = []
        for k, v in vllm_varied:
            varied_parts.append(f"{k.lstrip('-').replace('-','_')}={v}")
        for k, v in env_varied:
            varied_parts.append(f"{k}={v}")
        for k, v in bench_varied:
            varied_parts.append(f"{k.lstrip('-').replace('-','_')}={v}")

        sub_name = f"{cfg['name']}[{','.join(varied_parts)}]" if varied_parts else cfg["name"]

        expanded.append({
            "name":         sub_name,
            "description":  cfg.get("description", ""),
            "benchmark":    benchmark,
            "vllm_args":    vllm_args,
            "env":          env,
            bench_key:      bench_args,
            "profile_json": cfg.get("profile_json", None),
        })
    return expanded


def all_expanded_configs(configs: list[dict]) -> list[dict]:
    result = []
    for cfg in configs:
        result.extend(expand_config(cfg))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    config_name:              str
    config_description:       str
    repetition:               int
    # warmup round
    warmup_mean_ttft_s:       Optional[float] = None
    warmup_total_time_s:      Optional[float] = None
    warmup_prompt_count:      Optional[int]   = None
    warmup_successful_count:  Optional[int]   = None
    # query round (cache hits)
    query_mean_ttft_s:        Optional[float] = None
    query_total_time_s:       Optional[float] = None
    query_prompt_count:       Optional[int]   = None
    query_successful_count:   Optional[int]   = None
    # derived (long_doc_qa)
    ttft_speedup_x:           Optional[float] = None
    time_reduction_pct:       Optional[float] = None
    per_request_csv:          Optional[str]   = None
    # Path to per-transfer CSV written from simple-profiler JSON (one row per kernel call)
    gpu_transfer_csv:         Optional[str]   = None
    # sharegpt / vllm bench serve metrics
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

# ─────────────────────────────────────────────────────────────────────────────
# SERVER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_vllm_command(vllm_args: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port",  str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL)
    ]
    for flag, value in vllm_args.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(str(value))
    return cmd


def build_long_doc_qa_command(lmdqa_args: dict) -> list[str]:
    cmd = [
        sys.executable, LONG_DOC_QA_SCRIPT,
        "--model", MODEL,
        "--port",  str(VLLM_PORT),
    ]
    for flag, value in lmdqa_args.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(str(value))
    return cmd


def build_sharegpt_command(sharegpt_args: dict, result_json_path: str) -> list[str]:
    cmd = [
        "vllm", "bench", "serve",
        "--backend",      "vllm",
        "--model",         MODEL,
        "--port",          str(VLLM_PORT),
        "--dataset-name",  "sharegpt",
        "--dataset-path",  SHAREGPT_DATASET_PATH,
        "--endpoint", "/v1/completions",
        "--save-result",
        "--result-filename", result_json_path,
    ]
    for flag, value in sharegpt_args.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(str(value))
    return cmd


def run_sharegpt_benchmark(sharegpt_args: dict, result_json_path: str) -> Optional[dict]:
    cmd = build_sharegpt_command(sharegpt_args, result_json_path)
    print(f"\n  ▶  Running sharegpt benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout + "\n" + result.stderr)
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


def wait_for_server(port: int, timeout: int) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def start_server(config: dict, log_path: str) -> subprocess.Popen:
    cmd = build_vllm_command(config["vllm_args"])
    env = {**os.environ, **{k: str(v) for k, v in config.get("env", {}).items()}}
    print(f"\n  ▶  Starting vLLM: {' '.join(cmd)}")
    if config.get("env"):
        print(f"     Env overrides: { {k: v for k, v in config['env'].items()} }")
    print(f"     Server log: {log_path}")
    # Both stdout and stderr are piped through separate `tee` processes so they
    # appear on the terminal and are saved to the log file (append mode so both
    # streams end up in the same file).
    tee_out = subprocess.Popen(["tee", "-a", log_path], stdin=subprocess.PIPE)
    tee_err = subprocess.Popen(["tee", "-a", log_path], stdin=subprocess.PIPE)
    return subprocess.Popen(cmd, env=env, stdout=tee_out.stdin, stderr=tee_err.stdin)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    print("  ■  vLLM server stopped.")

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_long_doc_qa(lmdqa_args: dict, csv_path: str) -> Optional[dict]:
    cmd = build_long_doc_qa_command(lmdqa_args) + ["--csv-output", csv_path]
    print(f"\n  ▶  Running benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout + "\n" + result.stderr)
    try:
        return parse_csv(csv_path)
    except Exception as e:
        print(f"  ✗  Failed to read CSV output: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(csv_path: str) -> dict:
    warmup_rows, query_rows = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            (warmup_rows if row["round"] == "warmup" else query_rows).append(row)

    def mean_ttft(rows):
        vals = [float(r["ttft"]) for r in rows if r["successful"] == "True"]
        return sum(vals) / len(vals) if vals else None

    def total_time(rows):
        ends = [float(r["request_end"]) for r in rows]
        return max(ends) if ends else None

    def success_count(rows):
        return sum(1 for r in rows if r["successful"] == "True")

    parsed = {
        "warmup_mean_ttft_s":      mean_ttft(warmup_rows),
        "warmup_total_time_s":     total_time(warmup_rows),
        "warmup_prompt_count":     len(warmup_rows),
        "warmup_successful_count": success_count(warmup_rows),
        "query_mean_ttft_s":       mean_ttft(query_rows),
        "query_total_time_s":      total_time(query_rows),
        "query_prompt_count":      len(query_rows),
        "query_successful_count":  success_count(query_rows),
    }

    w, q = parsed["warmup_mean_ttft_s"], parsed["query_mean_ttft_s"]
    if w and q and q > 0:
        parsed["ttft_speedup_x"] = round(w / q, 3)

    wt, qt = parsed["warmup_total_time_s"], parsed["query_total_time_s"]
    if wt and qt and wt > 0:
        parsed["time_reduction_pct"] = round((1 - qt / wt) * 100, 2)

    return parsed


def parse_profile_json(json_path: str) -> list[dict]:
    """
    Parse a simple-profiler trace JSON and return one record per GPU transfer
    kernel call.  Each record has:
        direction   : "to_gpu" or "from_gpu"
        ts_us       : event start timestamp (µs)
        dur_us      : kernel duration (µs)
        num_bytes   : bytes transferred (if available, else empty)

    Recognised event names:
      - VLLMPagedMemGPUConnectorV2.to_gpu.kernel / .from_gpu.kernel
      - cuda_transfer(cpu_to_gpu)                / cuda_transfer(gpu_to_cpu)
    """
    with open(json_path) as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data

    TO_GPU_NAMES   = {"VLLMPagedMemGPUConnectorV2.to_gpu.kernel", "cuda_transfer(cpu_to_gpu)"}
    FROM_GPU_NAMES = {"VLLMPagedMemGPUConnectorV2.from_gpu.kernel", "cuda_transfer(gpu_to_cpu)"}

    transfers = []
    for e in events:
        if not isinstance(e, dict):
            continue
        name = e.get("name", "")
        if name in TO_GPU_NAMES:
            direction = "to_gpu"
        elif name in FROM_GPU_NAMES:
            direction = "from_gpu"
        else:
            continue
        transfers.append({
            "direction": direction,
            "ts_us":     e.get("ts", ""),
            "dur_us":    e.get("dur", ""),
            "num_bytes": (e.get("args") or {}).get("num_bytes", ""),
        })
    return transfers


def write_gpu_transfer_csv(transfers: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["direction", "ts_us", "dur_us", "num_bytes"])
        writer.writeheader()
        writer.writerows(transfers)


def find_and_parse_profile(profile_json: Optional[str]) -> Optional[list[dict]]:
    """Parse the config's profile_json file if set, or return None."""
    if not profile_json:
        return None
    try:
        return parse_profile_json(profile_json)
    except Exception as e:
        print(f"  ⚠  Could not parse profile JSON '{profile_json}': {e}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = list(BenchmarkResult.__dataclass_fields__.keys())

def write_csv(results: list[BenchmarkResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\n✅  Results written to: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FLUSH-MODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_long_doc_qa_warmup_only(lmdqa_args: dict, csv_path: str) -> None:
    """Run long_doc_qa in warmup-only mode (no query round, no CSV result needed)."""
    args = dict(lmdqa_args)
    cmd = build_long_doc_qa_command(args) + ["--warmup-only", "--csv-output", csv_path]
    print(f"\n  ▶  [flush] Warmup-only pass: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout + "\n" + result.stderr)


def run_flush_pass(csv_path: str) -> None:
    """Run a static flush pass (2 docs × 90k tokens, warmup-only, 1 inflight)."""
    flush_args = dict(LONG_DOC_QA_DEFAULTS)
    flush_args["--num-documents"]         = str(FLUSH_NUM_DOCS)
    flush_args["--document-length"]       = str(FLUSH_DOC_LENGTH)
    flush_args["--max-inflight-requests"] = str(FLUSH_MAX_INFLIGHT)
    cmd = build_long_doc_qa_command(flush_args) + ["--warmup-only", "--csv-output", csv_path]
    print(f"\n  ▶  [flush] Flush pass (evict GPU cache): {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout + "\n" + result.stderr)


def run_long_doc_qa_query_only(lmdqa_args: dict, csv_path: str) -> Optional[dict]:
    """Run long_doc_qa in query-only mode and return parsed metrics."""
    args = dict(lmdqa_args)
    cmd = build_long_doc_qa_command(args) + ["--query-only", "--csv-output", csv_path]
    print(f"\n  ▶  [flush] Query-only pass: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout + "\n" + result.stderr)
    try:
        return parse_csv(csv_path)
    except Exception as e:
        print(f"  ✗  Failed to read CSV output: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not Path(LONG_DOC_QA_SCRIPT).exists():
        print(f"ERROR: long_doc_qa.py not found at '{LONG_DOC_QA_SCRIPT}'.")
        print("Clone LMCache first:  git clone https://github.com/LMCache/LMCache.git")
        sys.exit(1)

    configs = all_expanded_configs(CONFIGS)
    total_runs = len(configs) * N_REPETITIONS

    print(f"\n{'═'*70}")
    print(f"  Benchmark plan: {len(configs)} config(s) × {N_REPETITIONS} rep(s) = {total_runs} total runs")
    for i, c in enumerate(configs, 1):
        print(f"  [{i:>2}] {c['name']}")
    print(f"{'═'*70}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results: list[BenchmarkResult] = []
    run_num = 0

    for config in configs:
        for rep in range(1, N_REPETITIONS + 1):
            run_num += 1
            print(f"\n{'═'*70}")
            print(f"  Config : {config['name']}")
            print(f"  Desc   : {config['description']}")
            print(f"  Run    : {rep}/{N_REPETITIONS}  (overall {run_num}/{total_runs})")
            print(f"{'═'*70}")

            result = BenchmarkResult(
                config_name=config["name"],
                config_description=config["description"],
                repetition=rep,
            )

            server_proc = None
            try:
                server_log = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_server.log")
                server_proc = start_server(config, server_log)
                print(f"\n  ⏳ Waiting up to {SERVER_STARTUP_TIMEOUT}s for vLLM to be ready...")
                if not wait_for_server(VLLM_PORT, SERVER_STARTUP_TIMEOUT):
                    raise RuntimeError(f"vLLM did not become healthy within {SERVER_STARTUP_TIMEOUT}s")
                print("  ✓  Server is ready.")

                benchmark = config.get("benchmark", "long_doc_qa")

                if benchmark == "sharegpt":
                    result_json = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_sharegpt.json")
                    parsed = run_sharegpt_benchmark(config["sharegpt_args"], result_json)
                    if parsed:
                        for key, val in parsed.items():
                            if hasattr(result, key):
                                setattr(result, key, val)
                        result.sg_result_json = result_json
                else:
                    csv_path = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}.csv")
                    result.per_request_csv = csv_path

                    if FLUSH_MODE:
                        # Step 2: warmup-only (populate offload cache)
                        warmup_csv = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_warmup.csv")
                        run_long_doc_qa_warmup_only(config["long_doc_qa_args"], warmup_csv)
                        # Step 3: flush pass (evict GPU KV-cache with large unrelated docs)
                        flush_csv = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_flush.csv")
                        run_flush_pass(flush_csv)
                        # Step 4: query-only (measure cache-hit TTFT)
                        parsed = run_long_doc_qa_query_only(config["long_doc_qa_args"], csv_path)
                    else:
                        parsed = run_long_doc_qa(config["long_doc_qa_args"], csv_path)

                    if parsed:
                        for key, val in parsed.items():
                            if hasattr(result, key):
                                setattr(result, key, val)

                profile_json_path = config.get("profile_json")
                if profile_json_path and os.path.exists(profile_json_path):
                    saved_json = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_profile.json")
                    shutil.copy2(profile_json_path, saved_json)
                transfers = find_and_parse_profile(profile_json_path)
                if transfers is not None:
                    transfer_csv = os.path.join(OUTPUT_DIR, f"run_{run_num:03d}_gpu_transfers.csv")
                    write_gpu_transfer_csv(transfers, transfer_csv)
                    result.gpu_transfer_csv = transfer_csv

                if benchmark == "sharegpt":
                    if result.sg_completed is None:
                        result.error = "Could not parse sharegpt result JSON"
                else:
                    if result.query_mean_ttft_s is None:
                        result.error = "Could not parse query TTFT from CSV"

            except Exception as e:
                result.error = str(e)
                print(f"\n  ✗  Error: {e}")
            finally:
                if server_proc:
                    stop_server(server_proc)

            all_results.append(result)

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

    write_csv(all_results, OUTPUT_CSV)

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


if __name__ == "__main__":
    main()