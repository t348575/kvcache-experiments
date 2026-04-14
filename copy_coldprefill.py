#!/usr/bin/env python3
"""
Copy cold_prefill rows (and all their associated files) from a source run
directory into a target directory, renumbering jobs to continue from the
target's existing job count.

Usage:
    python copy_coldprefill.py <source_subdir> <target_dir> [source_base_dir]

Example:
    python copy_coldprefill.py mistral ./h100-combined/
    python copy_coldprefill.py mistral ./h100-combined/ /home/joseph/kvcache-experiments/h100-run5

For each cold_prefill row in <source_base_dir>/<source_subdir>.csv that is not
already present in <target_dir>/<source_subdir>.csv (deduplicated by
(curve, server_config, doc_size, concurrency)):
  - It is assigned the next sequential job number (e.g. if the target
    already has jobs 1..12, the first copied row becomes job 13).
  - All associated files are copied into <target_dir>/<source_subdir>/
    and renamed to the new job number:
      job_{src:04d}_cold_prefill_*.csv  →  job_{new:04d}_cold_prefill_*.csv
      job_{src:04d}_server.log          →  job_{new:04d}_server.log
      job_{src:04d}_profile.json         →  job_{new:04d}_profile.json
      gpu_transfer_csv (if any)         →  gpu_{new:04d}_*.csv
"""

import csv
import os
import re
import shutil
import sys


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def last_job_idx(target_dir: str, source_subdir: str) -> int:
    """
    Find the highest job index already in <target_dir>/<source_subdir>/ by
    scanning all filenames matching job_NNNN_*.csv.
    Returns 0 if no jobs exist yet.
    """
    subdir = os.path.join(target_dir, source_subdir)
    if not os.path.exists(subdir):
        return 0
    max_idx = 0
    for fname in os.listdir(subdir):
        m = re.match(r"^job_(\d+)_", fname)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def existing_keys(path: str) -> set[tuple]:
    """Return {(curve, server_config, doc_size, concurrency)} already in the CSV."""
    if not os.path.exists(path):
        return set()
    return {
        (r["curve"], r["server_config"], r["doc_size"], r["concurrency"])
        for r in read_csv(path)
    }


def find_src_job_files(
    src_dir: str, doc_size: str, conc: str
) -> tuple[str, int] | None:
    """
    Find the source job CSV matching the given doc_size and concurrency.
    Returns (src_csv_path, src_job_idx) or None if not found.
    """
    for fname in os.listdir(src_dir):
        if not fname.endswith(".csv"):
            continue
        if f"_cold_prefill_{doc_size}_c{conc}." not in fname:
            continue
        m = re.match(r"^job_(\d+)_cold_prefill_(\d+)_c(\d+)\.csv$", fname)
        if not m:
            continue
        src_idx = int(m.group(1))
        return os.path.join(src_dir, fname), src_idx
    return None


def copy_row_files(
    src_dir: str,
    src_job_idx: int,
    new_job_idx: int,
    doc_size: str,
    conc: str,
    gpu_csv: str,
    target_subdir: str,
) -> None:
    """
    Copy and rename all files for a single job into target_subdir:
      job_{src:04d}_cold_prefill_{doc_size}_c{conc}.csv
    → job_{new:04d}_cold_prefill_{doc_size}_c{conc}.csv
    Same for server.log, profile.json, and gpu_transfer_csv.
    """
    os.makedirs(target_subdir, exist_ok=True)

    def copy_renamed(src_fname: str, dst_fname: str) -> None:
        src_path = os.path.join(src_dir, src_fname)
        dst_path = os.path.join(target_subdir, dst_fname)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

    # Job CSV
    src_csv = f"job_{src_job_idx:04d}_cold_prefill_{doc_size}_c{conc}.csv"
    dst_csv = f"job_{new_job_idx:04d}_cold_prefill_{doc_size}_c{conc}.csv"
    copy_renamed(src_csv, dst_csv)

    # Server log
    src_log = f"job_{src_job_idx:04d}_server.log"
    dst_log = f"job_{new_job_idx:04d}_server.log"
    copy_renamed(src_log, dst_log)

    # Profile JSON
    src_prof = f"job_{src_job_idx:04d}_profile.json"
    dst_prof = f"job_{new_job_idx:04d}_profile.json"
    copy_renamed(src_prof, dst_prof)

    # GPU transfer CSV
    if gpu_csv and os.path.exists(gpu_csv):
        gpu_name = os.path.basename(gpu_csv)
        # Preserve original gpu csv name structure but with new job index
        dst_gpu = f"gpu_{new_job_idx:04d}_{'_'.join(gpu_name.split('_')[1:])}"
        shutil.copy2(gpu_csv, os.path.join(target_subdir, dst_gpu))


DEFAULT_SOURCE_BASE_DIR = "/home/joefe/repos/kvcache-experiments/data/surf/h100-run4"


def copy_coldprefill(
    source_subdir: str,
    target_dir: str,
    source_base_dir: str = DEFAULT_SOURCE_BASE_DIR,
) -> None:
    base = source_base_dir
    src_csv = os.path.join(base, f"{source_subdir}.csv")
    src_files_dir = os.path.join(base, source_subdir)

    if not os.path.exists(src_csv):
        print(f"ERROR: {src_csv} does not exist")
        sys.exit(1)

    tgt_csv = os.path.join(target_dir, f"{source_subdir}.csv")
    tgt_subdir = os.path.join(target_dir, source_subdir)

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(tgt_subdir, exist_ok=True)

    all_rows = read_csv(src_csv)
    cold = [r for r in all_rows if r.get("curve") == "cold_prefill"]
    print(
        f"  {source_subdir}.csv: {len(cold)} cold_prefill rows (of {len(all_rows)} total)"
    )

    existing = existing_keys(tgt_csv)
    new_rows = []
    for r in cold:
        key = (r["curve"], r["server_config"], r["doc_size"], r["concurrency"])
        if key in existing:
            print(f"  Skipping duplicate: {key}")
            continue
        new_rows.append(r)
        existing.add(key)

    if not new_rows:
        print("  Nothing new to add.")
        return

    next_job_idx = last_job_idx(target_dir, source_subdir) + 1
    print(f"  Target last job: {next_job_idx - 1}, starting at {next_job_idx}")

    for r in new_rows:
        doc_size = r["doc_size"]
        conc = r["concurrency"]
        gpu_csv = r.get("gpu_transfer_csv", "").strip()

        result = find_src_job_files(src_files_dir, doc_size, conc)
        if result is None:
            print(f"  WARNING: no job file found for doc_size={doc_size} conc={conc}")
            continue
        _, src_job_idx = result

        copy_row_files(
            src_files_dir,
            src_job_idx,
            next_job_idx,
            doc_size,
            conc,
            gpu_csv,
            tgt_subdir,
        )
        print(
            f"  Copied job {src_job_idx} → job {next_job_idx} "
            f"(cold_prefill {doc_size} c{conc})"
        )

        next_job_idx += 1

    # Append new rows to target CSV
    fieldnames = list(new_rows[0].keys())
    if os.path.exists(tgt_csv):
        existing_rows = read_csv(tgt_csv)
        write_csv(existing_rows + new_rows, tgt_csv, fieldnames)
    else:
        write_csv(new_rows, tgt_csv, fieldnames)

    print(f"  Added {len(new_rows)} rows → {tgt_csv}")


def main():
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python copy_coldprefill.py <source_subdir> <target_dir> [source_base_dir]"
        )
        print("Example: python copy_coldprefill.py mistral ./h100-combined/")
        print(
            "Example: python copy_coldprefill.py mistral ./h100-combined/ "
            "/home/joseph/kvcache-experiments/h100-run5"
        )
        sys.exit(1)

    source_base_dir = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_SOURCE_BASE_DIR
    copy_coldprefill(sys.argv[1], sys.argv[2], source_base_dir)


if __name__ == "__main__":
    main()
