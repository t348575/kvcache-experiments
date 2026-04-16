#!/usr/bin/env python3

"""Run sweep matrices for fs_metadata_bench and summarize results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


DEFAULT_OPS = ["lookup_txn", "publish_txn", "evict_txn"]
DEFAULT_THREADS = [1, 2, 4, 8]


@dataclass(frozen=True)
class SweepJob:
    op: str
    threads: int
    files: int
    fanout: int
    payload_bytes: int
    sync_file: bool
    sync_dir: bool
    warmup_sec: int
    duration_sec: int
    seed: int


def parse_csv_ints(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated integer list"
        )
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated string list"
        )
    return items


def parse_sync_modes(value: str) -> list[tuple[bool, bool]]:
    modes: list[tuple[bool, bool]] = []
    for item in parse_csv_strings(value):
        if item == "none":
            modes.append((False, False))
        elif item == "file":
            modes.append((True, False))
        elif item == "dir":
            modes.append((False, True))
        elif item in {"file+dir", "dir+file"}:
            modes.append((True, True))
        else:
            raise argparse.ArgumentTypeError(
                "sync modes must be drawn from none,file,dir,file+dir"
            )
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a matrix sweep for fs_metadata_bench and write JSON + CSV summaries."
    )
    parser.add_argument(
        "--binary",
        default="./fs_metadata_bench",
        help="Path to the fs_metadata_bench binary",
    )
    parser.add_argument(
        "--root-base",
        required=True,
        help="Base workspace path passed to the benchmark; each run gets a subdirectory",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for sweep outputs; defaults to fs_metadata_sweep_<timestamp>",
    )
    parser.add_argument(
        "--ops",
        type=parse_csv_strings,
        default=DEFAULT_OPS,
        help="Comma-separated operations to benchmark",
    )
    parser.add_argument(
        "--threads",
        type=parse_csv_ints,
        default=DEFAULT_THREADS,
        help="Comma-separated thread counts",
    )
    parser.add_argument(
        "--files",
        type=parse_csv_ints,
        default=[10000],
        help="Comma-separated working-set sizes",
    )
    parser.add_argument(
        "--fanouts",
        type=parse_csv_ints,
        default=[256],
        help="Comma-separated fanout values",
    )
    parser.add_argument(
        "--payload-bytes",
        type=parse_csv_ints,
        default=[4096],
        help="Comma-separated payload sizes for write paths",
    )
    parser.add_argument(
        "--sync-modes",
        type=parse_sync_modes,
        default=[(False, False)],
        help="Comma-separated durability modes: none,file,dir,file+dir",
    )
    parser.add_argument("--warmup-sec", type=int, default=2)
    parser.add_argument("--duration-sec", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Pass --cleanup to each benchmark run",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue the sweep even if a run fails",
    )
    return parser.parse_args()


def resolve_binary(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path.resolve()
    resolved = shutil.which(path_str)
    if resolved:
        return Path(resolved).resolve()
    raise FileNotFoundError(f"benchmark binary not found: {path_str}")


def make_jobs(args: argparse.Namespace) -> list[SweepJob]:
    jobs: list[SweepJob] = []
    for op, threads, files, fanout, payload_bytes, sync_mode in itertools.product(
        args.ops,
        args.threads,
        args.files,
        args.fanouts,
        args.payload_bytes,
        args.sync_modes,
    ):
        sync_file, sync_dir = sync_mode
        jobs.append(
            SweepJob(
                op=op,
                threads=threads,
                files=files,
                fanout=fanout,
                payload_bytes=payload_bytes,
                sync_file=sync_file,
                sync_dir=sync_dir,
                warmup_sec=args.warmup_sec,
                duration_sec=args.duration_sec,
                seed=args.seed,
            )
        )
    return jobs


def sync_mode_label(sync_file: bool, sync_dir: bool) -> str:
    if sync_file and sync_dir:
        return "file+dir"
    if sync_file:
        return "file"
    if sync_dir:
        return "dir"
    return "none"


def run_job(
    binary: Path,
    root_base: Path,
    output_dir: Path,
    job_index: int,
    total_jobs: int,
    job: SweepJob,
    cleanup: bool,
) -> dict:
    run_name = (
        f"{job_index:03d}_{job.op}_t{job.threads}_f{job.files}_d{job.fanout}"
        f"_p{job.payload_bytes}_{sync_mode_label(job.sync_file, job.sync_dir)}"
    )
    run_root = root_base / run_name
    json_path = output_dir / f"{run_name}.json"

    cmd = [
        str(binary),
        "--root",
        str(run_root),
        "--op",
        job.op,
        "--threads",
        str(job.threads),
        "--warmup-sec",
        str(job.warmup_sec),
        "--duration-sec",
        str(job.duration_sec),
        "--files",
        str(job.files),
        "--fanout",
        str(job.fanout),
        "--payload-bytes",
        str(job.payload_bytes),
        "--seed",
        str(job.seed),
        "--json-out",
        str(json_path),
        "--reset",
    ]
    if job.sync_file:
        cmd.append("--sync-file")
    if job.sync_dir:
        cmd.append("--sync-dir")
    if cleanup:
        cmd.append("--cleanup")

    print(
        f"[{job_index}/{total_jobs}] {run_name}: {' '.join(cmd)}",
        file=sys.stderr,
        flush=True,
    )

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    row = {
        "run_name": run_name,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "stdout": completed.stdout.strip(),
        "json_path": str(json_path),
        **asdict(job),
    }

    if completed.returncode == 0:
        with json_path.open() as handle:
            payload = json.load(handle)
        latency = payload.get("latency_us", {})
        row.update(
            {
                "setup_sec": payload.get("setup_sec"),
                "elapsed_sec": payload.get("elapsed_sec"),
                "attempts": payload.get("attempts"),
                "operations": payload.get("operations"),
                "errors": payload.get("errors"),
                "ops_per_sec": payload.get("ops_per_sec"),
                "bytes_written": payload.get("bytes_written"),
                "bytes_read": payload.get("bytes_read"),
                "latency_count": latency.get("count"),
                "latency_min_us": latency.get("min"),
                "latency_mean_us": latency.get("mean"),
                "latency_p50_us": latency.get("p50"),
                "latency_p95_us": latency.get("p95"),
                "latency_p99_us": latency.get("p99"),
                "latency_max_us": latency.get("max"),
                "error_samples": " | ".join(payload.get("error_samples", [])),
            }
        )
    return row


def write_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "run_name",
        "status",
        "returncode",
        "op",
        "threads",
        "files",
        "fanout",
        "payload_bytes",
        "sync_file",
        "sync_dir",
        "warmup_sec",
        "duration_sec",
        "seed",
        "setup_sec",
        "elapsed_sec",
        "attempts",
        "operations",
        "errors",
        "ops_per_sec",
        "bytes_written",
        "bytes_read",
        "latency_count",
        "latency_min_us",
        "latency_mean_us",
        "latency_p50_us",
        "latency_p95_us",
        "latency_p99_us",
        "latency_max_us",
        "error_samples",
        "json_path",
        "stderr",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    binary = resolve_binary(args.binary)
    root_base = Path(args.root_base).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"fs_metadata_sweep_{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(args)
    manifest = {
        "binary": str(binary),
        "root_base": str(root_base),
        "output_dir": str(output_dir.resolve()),
        "job_count": len(jobs),
        "jobs": [asdict(job) for job in jobs],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rows: list[dict] = []
    failures = 0
    for job_index, job in enumerate(jobs, start=1):
        row = run_job(
            binary, root_base, output_dir, job_index, len(jobs), job, args.cleanup
        )
        rows.append(row)
        if row["status"] != "ok":
            failures += 1
            if not args.keep_going:
                break

    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote CSV summary: {csv_path}")
    print(f"Wrote JSON summary: {json_path}")
    if failures:
        print(f"Sweep finished with {failures} failed run(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
