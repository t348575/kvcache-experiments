#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import csv
import json
import io
import signal
import contextlib
import shlex
import shutil
import subprocess
import sys
import time
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kv_connector_harness import (
    RuntimeConnectorConfig,
    RuntimeKVConnectorHarness,
    ScenarioRequestSpec,
    estimate_kv_cache_bytes,
    get_free_cuda_bytes,
    parse_float_range,
    parse_int_range,
    resolve_local_hf_config_path,
    results_to_rows,
)


DEFAULT_STORE_PCTS = [0.3, 0.5, 0.7]
DEFAULT_PREFIX_REUSES = [0.5, 1.0]
VALID_FILE_IO_MODES = {"full", "page_io", "metadata_only"}


@dataclass(frozen=True)
class PressureJob:
    doc_size: str
    store_pct: float
    prefix_reuse: str
    num_requests: int
    file_io_mode: str
    skip_gpu_copy: bool


TRACE_COUNTERS = {
    "issued_ops": "block_rq_issue_ops",
    "issued_bytes": "block_rq_issue_bytes",
    "read_ops": "block_read_ops",
    "read_bytes": "block_read_bytes",
    "write_ops": "block_write_ops",
    "write_bytes": "block_write_bytes",
}


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated string list"
        )
    return items


def parse_csv_file_io_modes(value: str) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated file I/O mode list"
        )
    invalid = [item for item in items if item not in VALID_FILE_IO_MODES]
    if invalid:
        raise argparse.ArgumentTypeError(
            "invalid file I/O mode(s): "
            + ", ".join(invalid)
            + "; expected one of: "
            + ", ".join(sorted(VALID_FILE_IO_MODES))
        )
    return items


def parse_csv_floats(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated float list"
        )
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_csv_bools(value: str) -> list[bool]:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated boolean list"
        )
    try:
        return [mapping[item] for item in items]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid boolean value: {exc.args[0]}"
        ) from exc


def parse_percentage(value: float, *, label: str) -> float:
    if value < 0.0 or value > 100.0:
        raise ValueError(f"{label} must be in [0, 100]")
    return value / 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch concurrent kv_connector_scenarios.py processes against the "
            "same shared filesystem path to pressure filesystem metadata."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch kv_connector_scenarios.py",
    )
    parser.add_argument(
        "--scenario-script",
        default="./kv_connector_scenarios.py",
        help="Path to kv_connector_scenarios.py",
    )
    parser.add_argument(
        "--connector",
        required=True,
        help="vLLM KV connector class name passed through to the scenario script",
    )
    parser.add_argument(
        "--connector-module-path",
        help="Optional external module path for the connector class",
    )
    parser.add_argument(
        "--connector-extra-config",
        default="{}",
        help="Base JSON object for kv_connector_extra_config; shared_storage_path is injected",
    )
    parser.add_argument(
        "--shared-storage-path",
        required=True,
        help="Shared filesystem path used by all concurrent scenario processes",
    )
    parser.add_argument(
        "--instances",
        type=int,
        required=True,
        help="Number of kv_connector_scenarios.py instances to launch concurrently per job",
    )
    parser.add_argument(
        "--doc-sizes",
        type=parse_csv_strings,
        default=["4096-16384"],
        help="Comma-separated document sizes or ranges passed to --doc-size",
    )
    parser.add_argument(
        "--store-pcts",
        type=parse_csv_floats,
        default=DEFAULT_STORE_PCTS,
        help="Comma-separated store ratios; each job is a load/store mixture",
    )
    parser.add_argument(
        "--prefix-reuses",
        type=parse_csv_strings,
        default=[str(value) for value in DEFAULT_PREFIX_REUSES],
        help="Comma-separated prefix reuse ratios/ranges passed to --prefix-reuse",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=400,
        help="Requests per concurrent scenario process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base seed; each instance gets a deterministic offset",
    )
    parser.add_argument(
        "--stagger-ms",
        type=int,
        default=0,
        help="Optional delay between process launches within a concurrent job",
    )
    parser.add_argument(
        "--unique-request-pct",
        type=float,
        default=0.0,
        help=(
            "Percentage of requests that should come from an instance-private workload "
            "instead of the shared cross-instance workload."
        ),
    )
    parser.add_argument(
        "--per-instance-workers",
        type=int,
        default=1,
        help="Number of worker threads each scenario process should use.",
    )
    parser.add_argument(
        "--wipe-shared-storage-before-job",
        action="store_true",
        help="Delete and recreate the shared storage path before each job",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue the sweep even if one job fails",
    )
    parser.add_argument(
        "--shared-runtime",
        action="store_true",
        help="Run logical instances inside one process and share one GPU KV arena",
    )
    parser.add_argument(
        "--trace-metadata",
        action="store_true",
        help="Record one job-wide bpftrace stream of block read/write ops and bytes",
    )
    parser.add_argument(
        "--bpftrace",
        default="bpftrace",
        help="Path to the bpftrace binary",
    )
    parser.add_argument(
        "--sudo-prefix",
        default="sudo",
        help="Command prefix used to launch bpftrace, or empty string for none",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for logs, per-instance CSVs, and job summaries",
    )
    parser.add_argument(
        "--file-io-modes",
        type=parse_csv_file_io_modes,
        default=["full"],
        help=(
            "Comma-separated llm-d file I/O experiment modes: full, page_io, "
            "metadata_only"
        ),
    )
    parser.add_argument(
        "--skip-gpu-copy-options",
        type=parse_csv_bools,
        default=[False],
        help=(
            "Comma-separated booleans controlling llm-d skip_gpu_copy, for "
            "example false,true"
        ),
    )

    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--hf-config-path",
        default=None,
        help=(
            "Local directory or config.json used for model metadata resolution. "
            "If omitted, remote model IDs are resolved once outside vLLM."
        ),
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument(
        "--cpu-emulated-gpu-copy-bandwidth-gbps",
        type=float,
        default=0.0,
        help=(
            "Synthetic GPU<->CPU copy bandwidth used when GPU copies are enabled "
            "but an instance falls back to CPU placement"
        ),
    )
    return parser.parse_args()


def parse_extra_config(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("connector extra config must be a JSON object")
    return parsed


def build_jobs(args: argparse.Namespace) -> list[PressureJob]:
    jobs: list[PressureJob] = []
    for doc_size in args.doc_sizes:
        for store_pct in args.store_pcts:
            for prefix_reuse in args.prefix_reuses:
                for file_io_mode in args.file_io_modes:
                    for skip_gpu_copy in args.skip_gpu_copy_options:
                        jobs.append(
                            PressureJob(
                                doc_size=doc_size,
                                store_pct=store_pct,
                                prefix_reuse=prefix_reuse,
                                num_requests=args.num_requests,
                                file_io_mode=file_io_mode,
                                skip_gpu_copy=skip_gpu_copy,
                            )
                        )
    return jobs


def job_name(job_index: int, job: PressureJob) -> str:
    doc_label = job.doc_size.replace("-", "to")
    store_label = str(job.store_pct).replace(".", "p")
    reuse_label = job.prefix_reuse.replace("-", "to").replace(".", "p")
    gpu_label = "nogpu" if job.skip_gpu_copy else "gpu"
    mode_label = job.file_io_mode.replace("_", "-")
    return (
        f"{job_index:03d}_doc{doc_label}_store{store_label}_reuse{reuse_label}"
        f"_{mode_label}_{gpu_label}"
    )


def build_job_extra_config(
    base_extra_config: dict[str, Any],
    shared_storage_path: Path,
    job: PressureJob,
) -> dict[str, Any]:
    extra_config = dict(base_extra_config)
    extra_config["shared_storage_path"] = str(shared_storage_path)
    extra_config["file_io_mode"] = job.file_io_mode
    extra_config["skip_gpu_copy"] = job.skip_gpu_copy
    return extra_config


def _dispatcher_unique_token(request_id: int, token_idx: int) -> int:
    return request_id * 1_000_000 + token_idx + 1


def _dispatcher_corpus_token(doc_id: int, token_idx: int) -> int:
    return doc_id * 1_000_000 + token_idx + 1


@dataclass(frozen=True)
class _CorpusDoc:
    doc_id: int
    doc_tokens: int
    prompt_token_ids: tuple[int, ...]
    prompt_key: str


def build_dispatched_request_specs(
    *,
    instances: int,
    num_requests_per_instance: int,
    doc_size_range: tuple[int, int],
    store_pct: float,
    prefix_reuse_range: tuple[float, float],
    unique_request_ratio: float,
    seed: int,
) -> list[list[ScenarioRequestSpec]]:
    if not 0.0 <= store_pct <= 1.0:
        raise ValueError("store_pct must be in [0.0, 1.0]")
    if not 0.0 <= unique_request_ratio <= 1.0:
        raise ValueError("unique_request_ratio must be in [0.0, 1.0]")

    rng = random.Random(seed)
    specs_by_instance: list[list[ScenarioRequestSpec]] = [[] for _ in range(instances)]
    shared_docs: list[_CorpusDoc] = []
    unique_docs_by_instance: dict[int, list[_CorpusDoc]] = {
        instance_idx: [] for instance_idx in range(instances)
    }
    next_doc_id = 0
    total_requests = instances * num_requests_per_instance

    for logical_request_id in range(total_requests):
        target_instance = logical_request_id % instances
        traffic_scope = "unique" if rng.random() < unique_request_ratio else "shared"
        scope_docs = (
            unique_docs_by_instance[target_instance]
            if traffic_scope == "unique"
            else shared_docs
        )
        doc_tokens = rng.randint(*doc_size_range)
        should_store = rng.random() < store_pct or not scope_docs

        if should_store:
            doc_id = next_doc_id
            next_doc_id += 1
            prompt_key = f"{traffic_scope}-doc-{doc_id}"
            prompt_token_ids = tuple(
                _dispatcher_corpus_token(doc_id, token_idx)
                for token_idx in range(doc_tokens)
            )
            corpus_doc = _CorpusDoc(
                doc_id=doc_id,
                doc_tokens=doc_tokens,
                prompt_token_ids=prompt_token_ids,
                prompt_key=prompt_key,
            )
            scope_docs.append(corpus_doc)
            spec = ScenarioRequestSpec(
                request_id=logical_request_id,
                doc_tokens=doc_tokens,
                reuse_source_id=None,
                reuse_prefix_len=0,
                prompt_token_ids=prompt_token_ids,
                logical_request_id=logical_request_id,
                prompt_key=prompt_key,
                traffic_scope=traffic_scope,
                target_instance=target_instance,
            )
        else:
            source = rng.choice(scope_docs)
            reuse_frac = rng.uniform(*prefix_reuse_range)
            reuse_prefix_len = min(
                doc_tokens,
                source.doc_tokens,
                max(1, int(source.doc_tokens * reuse_frac)),
            )
            suffix_len = max(0, doc_tokens - reuse_prefix_len)
            prompt_token_ids = source.prompt_token_ids[:reuse_prefix_len] + tuple(
                _dispatcher_unique_token(logical_request_id, token_idx + reuse_prefix_len)
                for token_idx in range(suffix_len)
            )
            spec = ScenarioRequestSpec(
                request_id=logical_request_id,
                doc_tokens=doc_tokens,
                reuse_source_id=source.doc_id,
                reuse_prefix_len=reuse_prefix_len,
                prompt_token_ids=prompt_token_ids,
                logical_request_id=logical_request_id,
                prompt_key=(
                    source.prompt_key
                    if suffix_len == 0 and reuse_prefix_len == source.doc_tokens
                    else f"{source.prompt_key}:reuse:{reuse_prefix_len}:req-{logical_request_id}"
                ),
                traffic_scope=traffic_scope,
                target_instance=target_instance,
            )

        specs_by_instance[target_instance].append(spec)

    return specs_by_instance


def write_request_specs_json(path: Path, specs: list[ScenarioRequestSpec]) -> None:
    payload = []
    for spec in specs:
        payload.append(
            {
                "request_id": spec.request_id,
                "doc_tokens": spec.doc_tokens,
                "reuse_source_id": spec.reuse_source_id,
                "reuse_prefix_len": spec.reuse_prefix_len,
                "prompt_token_ids": list(spec.prompt_token_ids),
                "logical_request_id": spec.logical_request_id,
                "prompt_key": spec.prompt_key,
                "traffic_scope": spec.traffic_scope,
                "target_instance": spec.target_instance,
            }
        )
    path.write_text(json.dumps(payload))


def make_runtime_config(args: argparse.Namespace, extra_config: dict[str, Any]) -> RuntimeConnectorConfig:
    return RuntimeConnectorConfig(
        connector_name=args.connector,
        connector_module_path=args.connector_module_path,
        connector_extra_config=extra_config,
        model_name=args.model_name,
        hf_config_path=args.hf_config_path,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )


def select_runtime_kv_device(config: RuntimeConnectorConfig, skip_gpu_copy: bool) -> str:
    if skip_gpu_copy:
        return "cpu"
    required_bytes = estimate_kv_cache_bytes(config)
    for device_idx, free_bytes in enumerate(get_free_cuda_bytes()):
        if free_bytes >= required_bytes:
            return f"cuda:{device_idx}"
    return "cpu"


def apply_runtime_placement(
    extra_config: dict[str, Any],
    runtime_kv_device: str,
    cpu_emulated_gpu_copy_bandwidth_gbps: float,
) -> dict[str, Any]:
    placed_config = dict(extra_config)
    placed_config["runtime_kv_device"] = runtime_kv_device
    placed_config["runtime_kv_medium"] = (
        "cpu" if runtime_kv_device == "cpu" else "gpu"
    )
    placed_config["cpu_emulated_gpu_copy_bandwidth_gbps"] = (
        cpu_emulated_gpu_copy_bandwidth_gbps
        if runtime_kv_device == "cpu" and not bool(extra_config.get("skip_gpu_copy", False))
        else 0.0
    )
    return placed_config


def make_scenario_command(
    *,
    args: argparse.Namespace,
    job: PressureJob,
    instance_idx: int,
    csv_output: Path,
    request_specs_json: Path,
    extra_config: dict[str, Any],
) -> list[str]:
    cmd = [
        args.python,
        args.scenario_script,
        "--connector",
        args.connector,
        "--connector-extra-config",
        json.dumps(extra_config),
        "--num-requests",
        str(job.num_requests),
        "--request-specs-json",
        str(request_specs_json),
        "--store-pct",
        str(job.store_pct),
        "--prefix-reuse",
        job.prefix_reuse,
        "--seed",
        str(args.seed + instance_idx),
        "--model-name",
        args.model_name,
        "--block-size",
        str(args.block_size),
        "--num-blocks",
        str(args.num_blocks),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--per-instance-workers",
        str(args.per_instance_workers),
        "--csv-output",
        str(csv_output),
        "--json-output",
    ]
    if args.hf_config_path:
        cmd.extend(["--hf-config-path", args.hf_config_path])
    if args.connector_module_path:
        cmd.extend(["--connector-module-path", args.connector_module_path])
    return cmd


def summarize_request_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["successful"]]
    return {
        "total_requests": len(rows),
        "successful_requests": len(successful),
        "shared_requests": sum(1 for row in rows if row.get("traffic_scope") == "shared"),
        "unique_requests": sum(1 for row in rows if row.get("traffic_scope") == "unique"),
        "finished_sending": sum(1 for row in rows if row["finished_sending"]),
        "finished_recving": sum(1 for row in rows if row["finished_recving"]),
        "scheduler_steps": sum(int(row["scheduler_steps"]) for row in rows),
        "total_scheduled_tokens": sum(
            int(row["total_scheduled_tokens"]) for row in rows
        ),
        "max_scheduled_tokens_per_step": max(
            (int(row["max_scheduled_tokens_per_step"]) for row in rows),
            default=0,
        ),
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
    }


def run_shared_runtime_job(
    *,
    args: argparse.Namespace,
    run_name: str,
    run_dir: Path,
    job: PressureJob,
    shared_storage_path: Path,
    base_extra_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int, int, int, int]:
    doc_size_range = parse_int_range(job.doc_size)
    prefix_reuse_range = parse_float_range(job.prefix_reuse)
    specs_by_instance = build_dispatched_request_specs(
        instances=args.instances,
        num_requests_per_instance=job.num_requests,
        doc_size_range=doc_size_range,
        store_pct=job.store_pct,
        prefix_reuse_range=prefix_reuse_range,
        unique_request_ratio=parse_percentage(
            args.unique_request_pct, label="unique_request_pct"
        ),
        seed=args.seed,
    )

    extra_config = build_job_extra_config(base_extra_config, shared_storage_path, job)
    runtime_kv_device = select_runtime_kv_device(
        make_runtime_config(args, extra_config),
        job.skip_gpu_copy,
    )
    extra_config = apply_runtime_placement(
        extra_config,
        runtime_kv_device,
        args.cpu_emulated_gpu_copy_bandwidth_gbps,
    )
    config = make_runtime_config(args, extra_config)

    instance_results: list[list[dict[str, Any]]] = [[] for _ in range(args.instances)]
    instance_durations = [0.0 for _ in range(args.instances)]
    failed_instances = 0
    stderr_buffer = io.StringIO()

    with contextlib.redirect_stderr(stderr_buffer):
        harness = RuntimeKVConnectorHarness(config)
        try:
            for round_idx in range(job.num_requests):
                for instance_idx in range(args.instances):
                    spec = specs_by_instance[instance_idx][round_idx]
                    started_at = time.time()
                    result = harness.run_request(spec)
                    instance_durations[instance_idx] += time.time() - started_at
                    instance_results[instance_idx].append(results_to_rows([result])[0])
        finally:
            harness.shutdown()

    stderr_text = stderr_buffer.getvalue()
    instance_rows: list[dict[str, Any]] = []
    total_finished_sending = 0
    total_finished_recving = 0
    total_successful_requests = 0
    total_requests = 0

    for instance_idx, rows in enumerate(instance_results):
        instance_dir = run_dir / f"instance_{instance_idx:03d}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = instance_dir / "stdout.log"
        stderr_path = instance_dir / "stderr.log"
        csv_path = instance_dir / "requests.csv"

        summary = summarize_request_rows(rows)
        summary_text = json.dumps(summary) + "\n"
        stdout_path.write_text(summary_text)
        stderr_path.write_text(stderr_text)
        write_csv(rows, csv_path)

        total_finished_sending += int(summary.get("finished_sending", 0))
        total_finished_recving += int(summary.get("finished_recving", 0))
        total_successful_requests += int(summary.get("successful_requests", 0))
        total_requests += int(summary.get("total_requests", 0))

        failed = any(not row["successful"] for row in rows)
        if failed:
            failed_instances += 1

        instance_rows.append(
            {
                "run_name": run_name,
                "instance_idx": instance_idx,
                "returncode": 0 if not failed else 1,
                "status": "ok" if not failed else "failed",
                "duration_s": instance_durations[instance_idx],
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "finished_sending": summary.get("finished_sending"),
                "finished_recving": summary.get("finished_recving"),
                "successful_requests": summary.get("successful_requests"),
                "total_requests": summary.get("total_requests"),
                "shared_requests": summary.get("shared_requests"),
                "unique_requests": summary.get("unique_requests"),
                "connector_load_ops": summary.get("connector_load_ops"),
                "connector_load_bytes": summary.get("connector_load_bytes"),
                "connector_load_time_s": summary.get("connector_load_time_s"),
                "connector_store_ops": summary.get("connector_store_ops"),
                "connector_store_bytes": summary.get("connector_store_bytes"),
                "connector_store_time_s": summary.get("connector_store_time_s"),
                "stderr_tail": " | ".join(stderr_text.splitlines()[-5:]),
                "doc_size": job.doc_size,
                "store_pct": job.store_pct,
                "prefix_reuse": job.prefix_reuse,
                "num_requests": job.num_requests,
                "file_io_mode": job.file_io_mode,
                "skip_gpu_copy": job.skip_gpu_copy,
                "runtime_kv_device": runtime_kv_device,
                "runtime_kv_medium": extra_config.get("runtime_kv_medium"),
            }
        )

    return (
        instance_rows,
        failed_instances,
        total_finished_sending,
        total_finished_recving,
        total_successful_requests,
        total_requests,
    )


def run_job(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    shared_storage_path: Path,
    job_index: int,
    total_jobs: int,
    job: PressureJob,
    base_extra_config: dict[str, Any],
) -> dict[str, Any]:
    run_name = job_name(job_index, job)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.wipe_shared_storage_before_job and shared_storage_path.exists():
        shutil.rmtree(shared_storage_path)
    shared_storage_path.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, subprocess.Popen[str], Path, Path, Path, float, str]] = []
    start_time = time.time()
    trace_output_path = run_dir / "metadata_trace.log"
    trace_csv_path = run_dir / "metadata_trace.csv"
    trace_process: subprocess.Popen[str] | None = None
    doc_size_range = parse_int_range(job.doc_size)
    prefix_reuse_range = parse_float_range(job.prefix_reuse)
    unique_request_ratio = parse_percentage(
        args.unique_request_pct, label="unique_request_pct"
    )
    specs_by_instance = build_dispatched_request_specs(
        instances=args.instances,
        num_requests_per_instance=job.num_requests,
        doc_size_range=doc_size_range,
        store_pct=job.store_pct,
        prefix_reuse_range=prefix_reuse_range,
        unique_request_ratio=unique_request_ratio,
        seed=args.seed,
    )

    print(
        f"[{job_index}/{total_jobs}] {run_name}: launching {args.instances} instances "
        f"against {shared_storage_path}",
        file=sys.stderr,
        flush=True,
    )

    try:
        if args.trace_metadata:
            trace_process = start_bpftrace(
                bpftrace_path=args.bpftrace,
                sudo_prefix=args.sudo_prefix,
                output_path=trace_output_path,
            )

        if args.shared_runtime:
            (
                instance_rows,
                failed_instances,
                total_finished_sending,
                total_finished_recving,
                total_successful_requests,
                total_requests,
            ) = run_shared_runtime_job(
                args=args,
                run_name=run_name,
                run_dir=run_dir,
                job=job,
                shared_storage_path=shared_storage_path,
                base_extra_config=base_extra_config,
            )
        else:
            for instance_idx in range(args.instances):
                instance_dir = run_dir / f"instance_{instance_idx:03d}"
                instance_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = instance_dir / "stdout.log"
                stderr_path = instance_dir / "stderr.log"
                csv_path = instance_dir / "requests.csv"
                request_specs_path = instance_dir / "request_specs.json"
                write_request_specs_json(request_specs_path, specs_by_instance[instance_idx])

                extra_config = build_job_extra_config(
                    base_extra_config, shared_storage_path, job
                )
                runtime_kv_device = select_runtime_kv_device(
                    make_runtime_config(args, extra_config),
                    job.skip_gpu_copy,
                )
                extra_config = apply_runtime_placement(
                    extra_config,
                    runtime_kv_device,
                    args.cpu_emulated_gpu_copy_bandwidth_gbps,
                )

                cmd = make_scenario_command(
                    args=args,
                    job=job,
                    instance_idx=instance_idx,
                    csv_output=csv_path,
                    request_specs_json=request_specs_path,
                    extra_config=extra_config,
                )

                stdout_handle = stdout_path.open("w")
                stderr_handle = stderr_path.open("w")
                proc = subprocess.Popen(
                    cmd,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )
                stdout_handle.close()
                stderr_handle.close()
                processes.append(
                    (
                        instance_idx,
                        proc,
                        stdout_path,
                        stderr_path,
                        request_specs_path,
                        time.time(),
                        runtime_kv_device,
                    )
                )

                if args.stagger_ms > 0 and instance_idx + 1 < args.instances:
                    time.sleep(args.stagger_ms / 1000.0)

            instance_rows = []
            failed_instances = 0
            total_finished_sending = 0
            total_finished_recving = 0
            total_successful_requests = 0
            total_requests = 0

            for (
                instance_idx,
                proc,
                stdout_path,
                stderr_path,
                request_specs_path,
                launched_at,
                runtime_kv_device,
            ) in processes:
                returncode = proc.wait()
                duration_s = time.time() - launched_at
                stdout_text = stdout_path.read_text() if stdout_path.exists() else ""
                stderr_text = stderr_path.read_text() if stderr_path.exists() else ""

                summary: dict[str, Any] = {}
                for line in reversed(stdout_text.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        summary = parsed
                        break

                total_finished_sending += int(summary.get("finished_sending", 0))
                total_finished_recving += int(summary.get("finished_recving", 0))
                total_successful_requests += int(summary.get("successful_requests", 0))
                total_requests += int(summary.get("total_requests", 0))

                row = {
                    "run_name": run_name,
                    "instance_idx": instance_idx,
                    "returncode": returncode,
                    "status": "ok" if returncode == 0 else "failed",
                    "duration_s": duration_s,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "finished_sending": summary.get("finished_sending"),
                    "finished_recving": summary.get("finished_recving"),
                    "successful_requests": summary.get("successful_requests"),
                    "total_requests": summary.get("total_requests"),
                    "connector_load_ops": summary.get("connector_load_ops"),
                    "connector_load_bytes": summary.get("connector_load_bytes"),
                    "connector_load_time_s": summary.get("connector_load_time_s"),
                    "connector_store_ops": summary.get("connector_store_ops"),
                    "connector_store_bytes": summary.get("connector_store_bytes"),
                    "connector_store_time_s": summary.get("connector_store_time_s"),
                    "stderr_tail": " | ".join(stderr_text.splitlines()[-5:]),
                    "doc_size": job.doc_size,
                    "store_pct": job.store_pct,
                    "prefix_reuse": job.prefix_reuse,
                    "num_requests": job.num_requests,
                    "file_io_mode": job.file_io_mode,
                    "skip_gpu_copy": job.skip_gpu_copy,
                    "runtime_kv_device": runtime_kv_device,
                    "runtime_kv_medium": "cpu" if runtime_kv_device == "cpu" else "gpu",
                    "per_instance_workers": summary.get("per_instance_workers"),
                    "shared_requests": summary.get("shared_requests"),
                    "unique_requests": summary.get("unique_requests"),
                    "request_specs_path": str(request_specs_path),
                }
                instance_rows.append(row)
                if returncode != 0:
                    failed_instances += 1
    finally:
        if trace_process is not None:
            stop_bpftrace(trace_process)

    manifest_path = run_dir / "instances.csv"
    write_csv(instance_rows, manifest_path)

    elapsed_s = time.time() - start_time
    total_connector_load_ops = sum(
        int(row.get("connector_load_ops") or 0) for row in instance_rows
    )
    total_connector_load_bytes = sum(
        int(row.get("connector_load_bytes") or 0) for row in instance_rows
    )
    total_connector_load_time_s = sum(
        float(row.get("connector_load_time_s") or 0.0) for row in instance_rows
    )
    total_connector_store_ops = sum(
        int(row.get("connector_store_ops") or 0) for row in instance_rows
    )
    total_connector_store_bytes = sum(
        int(row.get("connector_store_bytes") or 0) for row in instance_rows
    )
    total_connector_store_time_s = sum(
        float(row.get("connector_store_time_s") or 0.0) for row in instance_rows
    )
    total_shared_requests = sum(
        int(row.get("shared_requests") or 0) for row in instance_rows
    )
    total_unique_requests = sum(
        int(row.get("unique_requests") or 0) for row in instance_rows
    )
    trace_rows = parse_bpftrace_output(trace_output_path) if args.trace_metadata else []
    if trace_rows:
        write_csv(trace_rows, trace_csv_path)
    trace_summary = summarize_trace_rows(trace_rows)
    placement_counts: dict[str, int] = {}
    for row in instance_rows:
        placement = str(row.get("runtime_kv_device", "unknown"))
        placement_counts[placement] = placement_counts.get(placement, 0) + 1
    summary = {
        "run_name": run_name,
        "status": "ok" if failed_instances == 0 else "failed",
        "instances": args.instances,
        "failed_instances": failed_instances,
        "duration_s": elapsed_s,
        "requests_total": total_requests,
        "requests_successful": total_successful_requests,
        "finished_sending": total_finished_sending,
        "finished_recving": total_finished_recving,
        "shared_requests": total_shared_requests,
        "unique_requests": total_unique_requests,
        "unique_request_pct": args.unique_request_pct,
        "dispatch_mode": "round_robin",
        "per_instance_workers": args.per_instance_workers,
        "connector_load_ops": total_connector_load_ops,
        "connector_load_bytes": total_connector_load_bytes,
        "connector_load_time_s": total_connector_load_time_s,
        "connector_store_ops": total_connector_store_ops,
        "connector_store_bytes": total_connector_store_bytes,
        "connector_store_time_s": total_connector_store_time_s,
        "doc_size": job.doc_size,
        "store_pct": job.store_pct,
        "prefix_reuse": job.prefix_reuse,
        "num_requests_per_instance": job.num_requests,
        "file_io_mode": job.file_io_mode,
        "skip_gpu_copy": job.skip_gpu_copy,
        "runtime_kv_device_counts": placement_counts,
        "shared_storage_path": str(shared_storage_path),
        "connector_extra_config": build_job_extra_config(
            base_extra_config, shared_storage_path, job
        ),
        "manifest_path": str(manifest_path),
        "metadata_trace_path": str(trace_output_path) if args.trace_metadata else None,
        "metadata_trace_csv": str(trace_csv_path) if trace_rows else None,
        **trace_summary,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def start_bpftrace(
    *, bpftrace_path: str, sudo_prefix: str, output_path: Path
) -> subprocess.Popen[str]:
    script = build_bpftrace_program()
    cmd: list[str] = []
    if sudo_prefix:
        cmd.extend(shlex.split(os.path.expandvars(sudo_prefix)))
    cmd.extend([bpftrace_path, "-e", script])
    output_handle = output_path.open("w")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        output_handle.close()
    time.sleep(1.0)
    return proc


def stop_bpftrace(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def build_bpftrace_program() -> str:
    return "\n".join(
        [
            'BEGIN { printf("TRACE_START\\n"); }',
            "tracepoint:block:block_rq_issue /args.bytes > 0/ {",
            "  @issued_ops = count();",
            "  @issued_bytes = sum(args.bytes);",
            '  if (strncmp(args.rwbs, "R", 1) == 0) {',
            "    @read_ops = count();",
            "    @read_bytes = sum(args.bytes);",
            '  } else if (strncmp(args.rwbs, "W", 1) == 0) {',
            "    @write_ops = count();",
            "    @write_bytes = sum(args.bytes);",
            "  }",
            "}",
            "interval:s:1 {",
            '  time("time: %s\\n");',
            "  print(@issued_ops);",
            "  print(@issued_bytes);",
            "  print(@read_ops);",
            "  print(@read_bytes);",
            "  print(@write_ops);",
            "  print(@write_bytes);",
            "  clear(@issued_ops);",
            "  clear(@issued_bytes);",
            "  clear(@read_ops);",
            "  clear(@read_bytes);",
            "  clear(@write_ops);",
            "  clear(@write_bytes);",
            "}",
        ]
    )


def parse_bpftrace_output(trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in trace_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("time: "):
            if current is not None:
                rows.append(current)
            current = {
                "time_ns": int(line.split(":", 1)[1].strip()),
                "issued_ops": 0,
                "issued_bytes": 0,
                "read_ops": 0,
                "read_bytes": 0,
                "write_ops": 0,
                "write_bytes": 0,
            }
            continue
        if current is None:
            continue
        if line.startswith("@issued_ops:"):
            current["issued_ops"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("@issued_bytes:"):
            current["issued_bytes"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("@read_ops:"):
            current["read_ops"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("@read_bytes:"):
            current["read_bytes"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("@write_ops:"):
            current["write_ops"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("@write_bytes:"):
            current["write_bytes"] = int(line.split(":", 1)[1].strip())
    if current is not None:
        rows.append(current)

    if not rows:
        return rows
    base = rows[0]["time_ns"]
    for row in rows:
        row["second"] = (row["time_ns"] - base) / 1e9
        row["total_rw_ops"] = row["read_ops"] + row["write_ops"]
        row["total_rw_bytes"] = row["read_bytes"] + row["write_bytes"]
    return rows


def summarize_trace_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trace_samples": 0,
            "read_ops_per_sec_mean": None,
            "read_ops_per_sec_max": None,
            "read_bytes_per_sec_mean": None,
            "read_bytes_per_sec_max": None,
            "write_ops_per_sec_mean": None,
            "write_ops_per_sec_max": None,
            "write_bytes_per_sec_mean": None,
            "write_bytes_per_sec_max": None,
            "total_rw_ops_per_sec_mean": None,
            "total_rw_ops_per_sec_max": None,
            "total_rw_bytes_per_sec_mean": None,
            "total_rw_bytes_per_sec_max": None,
        }
    return {
        "trace_samples": len(rows),
        "read_ops_per_sec_mean": sum(row["read_ops"] for row in rows) / len(rows),
        "read_ops_per_sec_max": max(row["read_ops"] for row in rows),
        "read_bytes_per_sec_mean": sum(row["read_bytes"] for row in rows) / len(rows),
        "read_bytes_per_sec_max": max(row["read_bytes"] for row in rows),
        "write_ops_per_sec_mean": sum(row["write_ops"] for row in rows) / len(rows),
        "write_ops_per_sec_max": max(row["write_ops"] for row in rows),
        "write_bytes_per_sec_mean": sum(row["write_bytes"] for row in rows) / len(rows),
        "write_bytes_per_sec_max": max(row["write_bytes"] for row in rows),
        "total_rw_ops_per_sec_mean": sum(row["total_rw_ops"] for row in rows)
        / len(rows),
        "total_rw_ops_per_sec_max": max(row["total_rw_ops"] for row in rows),
        "total_rw_bytes_per_sec_mean": sum(row["total_rw_bytes"] for row in rows)
        / len(rows),
        "total_rw_bytes_per_sec_max": max(row["total_rw_bytes"] for row in rows),
    }


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    parse_percentage(args.unique_request_pct, label="unique_request_pct")
    if args.per_instance_workers < 1:
        raise ValueError("--per-instance-workers must be >= 1")
    if args.shared_runtime and args.per_instance_workers != 1:
        raise ValueError(
            "--per-instance-workers only applies to subprocess mode; leave it at 1 with --shared-runtime"
        )
    scenario_script = Path(args.scenario_script)
    if not scenario_script.exists():
        raise FileNotFoundError(f"scenario script not found: {scenario_script}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"kv_connector_fs_pressure_{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_storage_path = Path(args.shared_storage_path)
    base_extra_config = parse_extra_config(args.connector_extra_config)
    args.hf_config_path = resolve_local_hf_config_path(
        RuntimeConnectorConfig(
            connector_name=args.connector,
            model_name=args.model_name,
            hf_config_path=args.hf_config_path,
        )
    )
    print(f"Using local HF config path: {args.hf_config_path}")
    jobs = build_jobs(args)

    summaries: list[dict[str, Any]] = []
    for job_index, job in enumerate(jobs, start=1):
        summary = run_job(
            args=args,
            output_dir=output_dir,
            shared_storage_path=shared_storage_path,
            job_index=job_index,
            total_jobs=len(jobs),
            job=job,
            base_extra_config=base_extra_config,
        )
        summaries.append(summary)
        if summary["status"] != "ok" and not args.keep_going:
            break

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json_text = json.dumps(summaries, indent=2) + "\n"
    summary_json.write_text(summary_json_text)
    write_csv(summaries, summary_csv)

    print(f"Wrote job summaries to {summary_json}")
    print(f"Wrote job CSV to {summary_csv}")
    print(summary_json_text, end="")


if __name__ == "__main__":
    main()
