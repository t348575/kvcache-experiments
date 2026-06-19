from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from analyze_prefix_csv import summarize_csv
from analyze_profile_json import summarize_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired prefix benchmark CSV and profile JSON files in a directory."
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing <run>.csv and <run>.json files.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional combined one-row-per-run summary CSV path.",
    )
    return parser.parse_args()


def pick(rows: list[dict[str, object]], **filters: str) -> dict[str, object] | None:
    for row in rows:
        if all(str(row.get(key, "")) == value for key, value in filters.items()):
            return row
    return None


def value(row: dict[str, object] | None, key: str) -> float:
    if row is None:
        return math.nan
    raw = row.get(key, math.nan)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def build_run_summary(csv_path: Path, json_path: Path | None) -> dict[str, object]:
    csv_rows = summarize_csv(csv_path)
    successful = pick(csv_rows, group="successful")
    fresh = pick(csv_rows, group="fresh")
    reuse = pick(csv_rows, group="reuse")

    row: dict[str, object] = {
        "run": csv_path.stem,
        "csv": str(csv_path),
        "json": str(json_path) if json_path else "",
        "requests": value(successful, "requests"),
        "wall_clock_s": value(successful, "wall_clock_s"),
        "throughput_req_s": value(successful, "throughput_req_s"),
        "ttft_mean_s": value(successful, "ttft_mean"),
        "ttft_p95_s": value(successful, "ttft_p95"),
        "ttft_p99_s": value(successful, "ttft_p99"),
        "latency_mean_s": value(successful, "latency_mean"),
        "dispatch_delay_mean_s": value(successful, "dispatch_delay_s_mean"),
        "dispatch_delay_p95_s": value(successful, "dispatch_delay_s_p95"),
        "fresh_ttft_mean_s": value(fresh, "ttft_mean"),
        "reuse_ttft_mean_s": value(reuse, "ttft_mean"),
        "reuse_prefix_mean_tokens": value(reuse, "reuse_prefix_len_mean"),
    }
    row["fresh_over_reuse_ttft"] = row["fresh_ttft_mean_s"] / row["reuse_ttft_mean_s"] if row["reuse_ttft_mean_s"] and math.isfinite(row["reuse_ttft_mean_s"]) else math.nan

    if json_path is not None and json_path.exists():
        profile_rows, request_rows = summarize_events(json_path)
        profile_requests = pick(profile_rows, category="request", event="request", direction="")
        forward = pick(profile_rows, category="model", event="forward", direction="")
        load = pick(profile_rows, category="kv_offload", event="load_e2e", direction="")
        store = pick(profile_rows, category="kv_offload", event="save_e2e", direction="")
        file_read = pick(profile_rows, category="fs", event="file_read", direction="")
        file_write = pick(profile_rows, category="fs", event="file_write", direction="")
        fs_load = pick(profile_rows, category="fs_transfer", event="fs_transfer", direction="storage_to_gpu") or pick(profile_rows, category="xnvme_transfer", event="xnvme_transfer", direction="storage_to_gpu")
        fs_store = pick(profile_rows, category="fs_transfer", event="fs_transfer", direction="gpu_to_storage") or pick(profile_rows, category="xnvme_transfer", event="xnvme_transfer", direction="gpu_to_storage")

        wait_values = [float(r["wait_to_forward_ms"]) for r in request_rows if math.isfinite(float(r["wait_to_forward_ms"]))]
        profile_ttft_values = [float(r["request_to_sample_end_ms"]) for r in request_rows if math.isfinite(float(r["request_to_sample_end_ms"]))]

        row.update({
            "profile_request_mean_ms": value(profile_requests, "mean_ms"),
            "profile_request_p95_ms": value(profile_requests, "p95_ms"),
            "profile_ttft_approx_mean_ms": sum(profile_ttft_values) / len(profile_ttft_values) if profile_ttft_values else math.nan,
            "profile_wait_to_forward_mean_ms": sum(wait_values) / len(wait_values) if wait_values else math.nan,
            "forward_mean_ms": value(forward, "mean_ms"),
            "kv_load_total_ms": value(load, "total_ms"),
            "kv_load_mean_ms": value(load, "mean_ms"),
            "kv_store_total_ms": value(store, "total_ms"),
            "kv_store_mean_ms": value(store, "mean_ms"),
            "file_read_total_ms": value(file_read, "total_ms"),
            "file_read_mean_ms": value(file_read, "mean_ms"),
            "file_read_effective_gbps": value(file_read, "effective_gbps"),
            "file_write_total_ms": value(file_write, "total_ms"),
            "file_write_mean_ms": value(file_write, "mean_ms"),
            "file_write_effective_gbps": value(file_write, "effective_gbps"),
            "storage_load_mean_ms": value(fs_load, "mean_ms"),
            "storage_load_effective_gbps": value(fs_load, "effective_gbps"),
            "storage_store_mean_ms": value(fs_store, "mean_ms"),
            "storage_store_effective_gbps": value(fs_store, "effective_gbps"),
        })
        row["kv_store_over_load_total"] = row["kv_store_total_ms"] / row["kv_load_total_ms"] if row["kv_load_total_ms"] and math.isfinite(row["kv_load_total_ms"]) else math.nan

    return row


def fmt(value: object, unit: str = "") -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.3f}{unit}"
    return str(value)


def print_table(rows: list[dict[str, object]]) -> None:
    print("run, ttft_mean, p95, req_profile, wait, load, store, file_read, file_write, throughput")
    for row in rows:
        print(
            f"{row['run']}, "
            f"{fmt(row.get('ttft_mean_s'), 's')}, "
            f"{fmt(row.get('ttft_p95_s'), 's')}, "
            f"{fmt(row.get('profile_request_mean_ms'), 'ms')}, "
            f"{fmt(row.get('profile_wait_to_forward_mean_ms'), 'ms')}, "
            f"{fmt(row.get('kv_load_mean_ms'), 'ms')}, "
            f"{fmt(row.get('kv_store_mean_ms'), 'ms')}, "
            f"{fmt(row.get('file_read_mean_ms'), 'ms')}, "
            f"{fmt(row.get('file_write_mean_ms'), 'ms')}, "
            f"{fmt(row.get('throughput_req_s'), ' req/s')}"
        )


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    rows: list[dict[str, object]] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        json_path = csv_path.with_suffix(".json")
        rows.append(build_run_summary(csv_path, json_path if json_path.exists() else None))
    print_table(rows)
    if args.output_csv:
        output = Path(args.output_csv).expanduser().resolve()
        write_csv(rows, output)
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
