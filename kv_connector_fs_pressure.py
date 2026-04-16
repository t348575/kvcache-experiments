#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_STORE_PCTS = [0.3, 0.5, 0.7]
DEFAULT_PREFIX_REUSES = [0.5, 1.0]


@dataclass(frozen=True)
class PressureJob:
    doc_size: str
    store_pct: float
    prefix_reuse: str
    num_requests: int


TRACE_COUNTERS = {
    "alloc": "ext4_mb_new_blocks",
    "bitmap": "ext4_read_block_bitmap_nowait",
    "jbd2": "jbd2_journal_commit_transaction",
}


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated string list"
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
        "--trace-metadata",
        action="store_true",
        help="Record one job-wide bpftrace stream of EXT4 metadata counters",
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

    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=16)
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
                jobs.append(
                    PressureJob(
                        doc_size=doc_size,
                        store_pct=store_pct,
                        prefix_reuse=prefix_reuse,
                        num_requests=args.num_requests,
                    )
                )
    return jobs


def job_name(job_index: int, job: PressureJob) -> str:
    doc_label = job.doc_size.replace("-", "to")
    store_label = str(job.store_pct).replace(".", "p")
    reuse_label = job.prefix_reuse.replace("-", "to").replace(".", "p")
    return f"{job_index:03d}_doc{doc_label}_store{store_label}_reuse{reuse_label}"


def make_scenario_command(
    *,
    args: argparse.Namespace,
    job: PressureJob,
    instance_idx: int,
    csv_output: Path,
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
        "--doc-size",
        job.doc_size,
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
        "--csv-output",
        str(csv_output),
        "--json-output",
    ]
    if args.connector_module_path:
        cmd.extend(["--connector-module-path", args.connector_module_path])
    return cmd


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

    processes: list[tuple[int, subprocess.Popen[str], Path, Path, float]] = []
    start_time = time.time()
    trace_output_path = run_dir / "metadata_trace.log"
    trace_csv_path = run_dir / "metadata_trace.csv"
    trace_process: subprocess.Popen[str] | None = None

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

        for instance_idx in range(args.instances):
            instance_dir = run_dir / f"instance_{instance_idx:03d}"
            instance_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = instance_dir / "stdout.log"
            stderr_path = instance_dir / "stderr.log"
            csv_path = instance_dir / "requests.csv"

            extra_config = dict(base_extra_config)
            extra_config["shared_storage_path"] = str(shared_storage_path)

            cmd = make_scenario_command(
                args=args,
                job=job,
                instance_idx=instance_idx,
                csv_output=csv_path,
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
                (instance_idx, proc, stdout_path, stderr_path, time.time())
            )

            if args.stagger_ms > 0 and instance_idx + 1 < args.instances:
                time.sleep(args.stagger_ms / 1000.0)

        instance_rows: list[dict[str, Any]] = []
        failed_instances = 0
        total_finished_sending = 0
        total_finished_recving = 0
        total_successful_requests = 0
        total_requests = 0

        for instance_idx, proc, stdout_path, stderr_path, launched_at in processes:
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
                "stderr_tail": " | ".join(stderr_text.splitlines()[-5:]),
                "doc_size": job.doc_size,
                "store_pct": job.store_pct,
                "prefix_reuse": job.prefix_reuse,
                "num_requests": job.num_requests,
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
    trace_rows = parse_bpftrace_output(trace_output_path) if args.trace_metadata else []
    if trace_rows:
        write_csv(trace_rows, trace_csv_path)
    trace_summary = summarize_trace_rows(trace_rows)
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
        "doc_size": job.doc_size,
        "store_pct": job.store_pct,
        "prefix_reuse": job.prefix_reuse,
        "num_requests_per_instance": job.num_requests,
        "shared_storage_path": str(shared_storage_path),
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
        cmd.extend(sudo_prefix.split())
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
            "kprobe:ext4_mb_new_blocks { @alloc = count(); }",
            "kprobe:ext4_read_block_bitmap_nowait { @bitmap = count(); }",
            "kprobe:jbd2_journal_commit_transaction { @jbd2 = count(); }",
            "interval:s:1 {",
            '  time("time: %s\\n");',
            "  print(@alloc);",
            "  print(@bitmap);",
            "  print(@jbd2);",
            "  clear(@alloc);",
            "  clear(@bitmap);",
            "  clear(@jbd2);",
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
                "alloc": 0,
                "bitmap": 0,
                "jbd2": 0,
            }
            continue
        if current is None:
            continue
        if line.startswith("@alloc:"):
            current["alloc"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("@bitmap:"):
            current["bitmap"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("@jbd2:"):
            current["jbd2"] = int(line.split(":", 1)[1].strip())
    if current is not None:
        rows.append(current)

    if not rows:
        return rows
    base = rows[0]["time_ns"]
    for row in rows:
        row["second"] = (row["time_ns"] - base) / 1e9
        row["total_metadata_ops"] = row["alloc"] + row["bitmap"] + row["jbd2"]
    return rows


def summarize_trace_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "metadata_samples": 0,
            "alloc_ops_per_sec_mean": None,
            "alloc_ops_per_sec_max": None,
            "bitmap_ops_per_sec_mean": None,
            "bitmap_ops_per_sec_max": None,
            "jbd2_ops_per_sec_mean": None,
            "jbd2_ops_per_sec_max": None,
            "total_metadata_ops_per_sec_mean": None,
            "total_metadata_ops_per_sec_max": None,
        }
    return {
        "metadata_samples": len(rows),
        "alloc_ops_per_sec_mean": sum(row["alloc"] for row in rows) / len(rows),
        "alloc_ops_per_sec_max": max(row["alloc"] for row in rows),
        "bitmap_ops_per_sec_mean": sum(row["bitmap"] for row in rows) / len(rows),
        "bitmap_ops_per_sec_max": max(row["bitmap"] for row in rows),
        "jbd2_ops_per_sec_mean": sum(row["jbd2"] for row in rows) / len(rows),
        "jbd2_ops_per_sec_max": max(row["jbd2"] for row in rows),
        "total_metadata_ops_per_sec_mean": sum(
            row["total_metadata_ops"] for row in rows
        )
        / len(rows),
        "total_metadata_ops_per_sec_max": max(
            row["total_metadata_ops"] for row in rows
        ),
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
    summary_json.write_text(json.dumps(summaries, indent=2) + "\n")
    write_csv(summaries, summary_csv)

    print(f"Wrote job summaries to {summary_json}")
    print(f"Wrote job CSV to {summary_csv}")


if __name__ == "__main__":
    main()
