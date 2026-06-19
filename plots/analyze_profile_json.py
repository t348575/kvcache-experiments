from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


REQUEST_RE = re.compile(r"request\(id=([^)]*)\)")
DIRECTION_RE = re.compile(r"\(([^,)]*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Chrome trace JSON profiles from KV cache benchmark runs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON files or directories containing JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional event-family summary CSV path.",
    )
    parser.add_argument(
        "--output-request-csv",
        help="Optional per-profile-request timing CSV path.",
    )
    return parser.parse_args()


def collect_jsons(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(raw)
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.suffix.lower() != ".json":
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    return paths


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [event for event in events if isinstance(event, dict)]


def base_name(name: str) -> str:
    return name.split("(", 1)[0]


def event_direction(name: str) -> str:
    match = DIRECTION_RE.search(name)
    if not match:
        return ""
    direction = match.group(1)
    return direction if "_to_" in direction else ""


def req_id_for_event(event: dict[str, Any]) -> str:
    args = event.get("args") or {}
    if args.get("req_id"):
        return str(args["req_id"])
    match = REQUEST_RE.match(str(event.get("name", "")))
    return match.group(1) if match else ""


def pct(values: list[float], percentile: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    idx = percentile / 100.0 * (len(clean) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return clean[lo]
    return clean[lo] + (idx - lo) * (clean[hi] - clean[lo])


def stats(values: list[float]) -> dict[str, float | int]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"count": 0, "total_ms": math.nan, "mean_ms": math.nan, "median_ms": math.nan, "p95_ms": math.nan, "p99_ms": math.nan, "max_ms": math.nan}
    return {
        "count": len(clean),
        "total_ms": sum(clean),
        "mean_ms": mean(clean),
        "median_ms": median(clean),
        "p95_ms": pct(clean, 95),
        "p99_ms": pct(clean, 99),
        "max_ms": max(clean),
    }


def classify_event(event: dict[str, Any]) -> tuple[str, str, str]:
    name = str(event.get("name", ""))
    cat = str(event.get("cat", ""))
    base = base_name(name)
    direction = event_direction(name)

    if cat == "request" and base == "request":
        return "request", base, ""
    if cat == "model" and base in {"forward", "sample"}:
        return "model", base, ""
    if cat == "kv_offload" and base in {"load_e2e", "save_e2e", "transfer_e2e"}:
        return "kv_offload", base, direction if base == "transfer_e2e" else ""
    if cat in {"fs", "fs_transfer", "xnvme_transfer"}:
        return cat, base, direction if cat != "fs" else ""
    if cat == "cuda" and base in {"cuda_transfer", "cuda_staging"}:
        return "cuda", base, direction
    return cat, base, direction


def summarize_events(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events = load_events(path)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("ph") not in {None, "X"}:
            continue
        grouped[classify_event(event)].append(event)

    summary_rows: list[dict[str, object]] = []
    for (cat, name, direction), group in sorted(grouped.items()):
        durations_ms = [float(e.get("dur", math.nan)) / 1000.0 for e in group]
        bytes_values = [float((e.get("args") or {}).get("num_bytes", 0) or 0) for e in group]
        row: dict[str, object] = {
            "run": path.stem,
            "source_json": str(path),
            "category": cat,
            "event": name,
            "direction": direction,
            "bytes_total_gb": sum(bytes_values) / 1e9,
        }
        row.update(stats(durations_ms))
        if sum(bytes_values) > 0 and math.isfinite(float(row["total_ms"])) and float(row["total_ms"]) > 0:
            row["effective_gbps"] = sum(bytes_values) / (float(row["total_ms"]) / 1000.0) / 1e9
        else:
            row["effective_gbps"] = math.nan
        bw_values = [float((e.get("args") or {}).get("bw_GBps", math.nan)) for e in group]
        file_bw_values = [float((e.get("args") or {}).get("file_io_bw_GBps", math.nan)) for e in group]
        cuda_bw_values = [float((e.get("args") or {}).get("cuda_copy_bw_GBps", math.nan)) for e in group]
        row["bw_gbps_mean"] = mean([v for v in bw_values if math.isfinite(v)]) if any(math.isfinite(v) for v in bw_values) else math.nan
        row["file_io_bw_gbps_mean"] = mean([v for v in file_bw_values if math.isfinite(v)]) if any(math.isfinite(v) for v in file_bw_values) else math.nan
        row["cuda_copy_bw_gbps_mean"] = mean([v for v in cuda_bw_values if math.isfinite(v)]) if any(math.isfinite(v) for v in cuda_bw_values) else math.nan
        summary_rows.append(row)

    request_rows = summarize_requests(path, events)
    return summary_rows, request_rows


def summarize_requests(path: Path, events: list[dict[str, Any]]) -> list[dict[str, object]]:
    by_req: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "request_start_us": math.nan,
        "request_dur_ms": math.nan,
        "first_forward_start_us": math.nan,
        "first_sample_start_us": math.nan,
        "first_sample_end_us": math.nan,
        "load_ms": 0.0,
        "store_ms": 0.0,
        "file_read_ms": 0.0,
        "file_write_ms": 0.0,
    })

    for event in events:
        req_id = req_id_for_event(event)
        if not req_id:
            continue
        name = base_name(str(event.get("name", "")))
        cat = str(event.get("cat", ""))
        ts = float(event.get("ts", math.nan))
        dur_ms = float(event.get("dur", math.nan)) / 1000.0
        row = by_req[req_id]

        if cat == "request" and name == "request":
            row["request_start_us"] = min(float(row["request_start_us"]), ts) if math.isfinite(float(row["request_start_us"])) else ts
            row["request_dur_ms"] = dur_ms
        elif cat == "model" and name == "forward":
            row["first_forward_start_us"] = min(float(row["first_forward_start_us"]), ts) if math.isfinite(float(row["first_forward_start_us"])) else ts
        elif cat == "model" and name == "sample":
            row["first_sample_start_us"] = min(float(row["first_sample_start_us"]), ts) if math.isfinite(float(row["first_sample_start_us"])) else ts
            sample_end = ts + float(event.get("dur", 0.0) or 0.0)
            row["first_sample_end_us"] = min(float(row["first_sample_end_us"]), sample_end) if math.isfinite(float(row["first_sample_end_us"])) else sample_end
        elif cat == "kv_offload" and name == "load_e2e":
            row["load_ms"] = float(row["load_ms"]) + dur_ms
        elif cat == "kv_offload" and name == "save_e2e":
            row["store_ms"] = float(row["store_ms"]) + dur_ms
        elif cat == "fs" and name == "file_read":
            row["file_read_ms"] = float(row["file_read_ms"]) + dur_ms
        elif cat == "fs" and name == "file_write":
            row["file_write_ms"] = float(row["file_write_ms"]) + dur_ms

    rows: list[dict[str, object]] = []
    for req_id, row in sorted(by_req.items()):
        request_start = float(row["request_start_us"])
        forward_start = float(row["first_forward_start_us"])
        sample_start = float(row["first_sample_start_us"])
        sample_end = float(row["first_sample_end_us"])
        rows.append({
            "run": path.stem,
            "req_id": req_id,
            "request_dur_ms": row["request_dur_ms"],
            "wait_to_forward_ms": (forward_start - request_start) / 1000.0 if math.isfinite(request_start) and math.isfinite(forward_start) else math.nan,
            "request_to_sample_start_ms": (sample_start - request_start) / 1000.0 if math.isfinite(request_start) and math.isfinite(sample_start) else math.nan,
            "request_to_sample_end_ms": (sample_end - request_start) / 1000.0 if math.isfinite(request_start) and math.isfinite(sample_end) else math.nan,
            "load_ms": row["load_ms"],
            "store_ms": row["store_ms"],
            "file_read_ms": row["file_read_ms"],
            "file_write_ms": row["file_write_ms"],
        })
    return rows


def fmt(value: object, unit: str = "") -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.3f}{unit}"
    return str(value)


def row_for(rows: list[dict[str, object]], run: str, category: str, event: str, direction: str = "") -> dict[str, object] | None:
    for row in rows:
        if row["run"] == run and row["category"] == category and row["event"] == event and row["direction"] == direction:
            return row
    return None


def print_summary(summary_rows: list[dict[str, object]], request_rows: list[dict[str, object]]) -> None:
    runs = sorted({str(row["run"]) for row in summary_rows})
    for run in runs:
        print(f"\n{run}")
        reqs = [row for row in request_rows if row["run"] == run]
        if reqs:
            for key, label in [
                ("request_dur_ms", "profile request time"),
                ("wait_to_forward_ms", "wait to first forward"),
                ("request_to_sample_end_ms", "profile TTFT approximation"),
            ]:
                s = stats([float(row[key]) for row in reqs])
                print(f"  {label} mean/p95/p99: {fmt(s['mean_ms'], 'ms')} / {fmt(s['p95_ms'], 'ms')} / {fmt(s['p99_ms'], 'ms')}")

        for category, event, direction, label in [
            ("kv_offload", "load_e2e", "", "KV load e2e"),
            ("kv_offload", "save_e2e", "", "KV store e2e"),
            ("fs", "file_read", "", "file read"),
            ("fs", "file_write", "", "file write"),
            ("fs_transfer", "fs_transfer", "storage_to_gpu", "fs transfer load"),
            ("fs_transfer", "fs_transfer", "gpu_to_storage", "fs transfer store"),
            ("xnvme_transfer", "xnvme_transfer", "storage_to_gpu", "xnvme transfer load"),
            ("xnvme_transfer", "xnvme_transfer", "gpu_to_storage", "xnvme transfer store"),
        ]:
            row = row_for(summary_rows, run, category, event, direction)
            if not row or not row["count"]:
                continue
            print(f"  {label}: n={row['count']} total={fmt(row['total_ms'], 'ms')} mean={fmt(row['mean_ms'], 'ms')} p95={fmt(row['p95_ms'], 'ms')} bytes={fmt(row['bytes_total_gb'], 'GB')} eff={fmt(row['effective_gbps'], 'GB/s')}")


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
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
    summary_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    for json_path in collect_jsons(args.inputs):
        summary, requests = summarize_events(json_path)
        summary_rows.extend(summary)
        request_rows.extend(requests)
    print_summary(summary_rows, request_rows)
    if args.output_csv:
        output = Path(args.output_csv).expanduser().resolve()
        write_csv(summary_rows, output)
        print(f"\nwrote {output}")
    if args.output_request_csv:
        output = Path(args.output_request_csv).expanduser().resolve()
        write_csv(request_rows, output)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
