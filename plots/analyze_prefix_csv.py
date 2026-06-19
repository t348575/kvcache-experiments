from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


REQUEST_METRICS = (
    "ttft",
    "dispatch_delay_s",
    "doc_tokens",
    "reuse_prefix_len",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize prefix_cache_benchmark per-request CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files or directories containing CSV files.",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for machine-readable summary rows.",
    )
    return parser.parse_args()


def collect_csvs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(raw)
        candidates = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.suffix.lower() != ".csv":
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
    return paths


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


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
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "n": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "p95": pct(clean, 95),
        "p99": pct(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, object]] = []
        for raw in reader:
            row: dict[str, object] = dict(raw)
            row["request_id"] = int(float(raw["request_id"]))
            row["doc_tokens"] = to_float(raw.get("doc_tokens"))
            row["reuse_prefix_len"] = to_float(raw.get("reuse_prefix_len"))
            row["scheduled_time"] = to_float(raw.get("scheduled_time"))
            row["request_start"] = to_float(raw.get("request_start"))
            row["request_end"] = to_float(raw.get("request_end"))
            row["ttft"] = to_float(raw.get("ttft"))
            row["is_prefix_reuse"] = parse_bool(raw.get("is_prefix_reuse", ""))
            row["successful"] = parse_bool(raw.get("successful", ""))
            rows.append(row)

    if rows:
        benchmark_start = min(
            float(r["request_start"]) - float(r["scheduled_time"])
            for r in rows
            if math.isfinite(float(r["request_start"]))
        )
        for row in rows:
            expected_start = benchmark_start + float(row["scheduled_time"])
            row["dispatch_delay_s"] = max(0.0, float(row["request_start"]) - expected_start)
    return rows


def group_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    successful = [r for r in rows if r["successful"]]
    return {
        "all": rows,
        "successful": successful,
        "fresh": [r for r in successful if not r["is_prefix_reuse"]],
        "reuse": [r for r in successful if r["is_prefix_reuse"]],
    }


def summarize_csv(path: Path) -> list[dict[str, object]]:
    rows = read_rows(path)
    groups = group_rows(rows)
    output: list[dict[str, object]] = []
    for group_name, group in groups.items():
        summary: dict[str, object] = {
            "run": path.stem,
            "source_csv": str(path),
            "group": group_name,
            "requests": len(group),
            "successful": sum(1 for r in group if r["successful"]),
        }
        for metric in REQUEST_METRICS:
            metric_stats = stats([float(r.get(metric, math.nan)) for r in group])
            for key, value in metric_stats.items():
                summary[f"{metric}_{key}"] = value
        if group:
            first = min(float(r["request_start"]) for r in group)
            last = max(float(r["request_end"]) for r in group)
            summary["wall_clock_s"] = last - first
            summary["throughput_req_s"] = len(group) / (last - first) if last > first else math.nan
        else:
            summary["wall_clock_s"] = math.nan
            summary["throughput_req_s"] = math.nan
        output.append(summary)
    return output


def fmt(value: object, unit: str = "") -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.3f}{unit}"
    return str(value)


def print_summary(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row["group"] != "successful":
            continue
        print(f"\n{row['run']}")
        print(f"  successful: {row['successful']}/{row['requests']}  wall: {fmt(row['wall_clock_s'], 's')}  throughput: {fmt(row['throughput_req_s'], ' req/s')}")
        print(f"  TTFT mean/p95/p99: {fmt(row['ttft_mean'], 's')} / {fmt(row['ttft_p95'], 's')} / {fmt(row['ttft_p99'], 's')}")
        print(f"  dispatch delay mean/p95/p99: {fmt(row['dispatch_delay_s_mean'], 's')} / {fmt(row['dispatch_delay_s_p95'], 's')} / {fmt(row['dispatch_delay_s_p99'], 's')}")

        by_group = {r["group"]: r for r in rows if r["run"] == row["run"]}
        fresh = by_group.get("fresh")
        reuse = by_group.get("reuse")
        if fresh and reuse and fresh["requests"] and reuse["requests"]:
            speedup = float(fresh["ttft_mean"]) / float(reuse["ttft_mean"])
            print(f"  fresh TTFT mean: {fmt(fresh['ttft_mean'], 's')}  reuse TTFT mean: {fmt(reuse['ttft_mean'], 's')}  fresh/reuse: {fmt(speedup, 'x')}")
            print(f"  reuse prefix mean/p95: {fmt(reuse['reuse_prefix_len_mean'], ' tokens')} / {fmt(reuse['reuse_prefix_len_p95'], ' tokens')}")


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
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
    csvs = collect_csvs(args.inputs)
    rows: list[dict[str, object]] = []
    for csv_path in csvs:
        rows.extend(summarize_csv(csv_path))
    print_summary(rows)
    if args.output_csv:
        output = Path(args.output_csv).expanduser().resolve()
        write_summary_csv(rows, output)
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
