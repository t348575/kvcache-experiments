"""Merge pareto_measure.py result CSVs and their per-job artifacts.

Each pareto_measure.py run writes a summary CSV next to a directory with the
same stem. Summary row N corresponds to files named job_NNNN_* in that
directory. This script combines multiple runs, replacing failed rows with
successful reruns for the same measurement configuration.
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path


KEY_FIELDS = ("curve", "server_config", "doc_size", "prefix_frac", "concurrency")
PARETO_CSV_RE = re.compile(r"^pareto_measure_\d{8}_\d{6}\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge pareto_measure.py summary CSVs and per-job artifacts."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input pareto_measure CSVs or directories containing pareto_measure_*.csv files.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Merged summary CSV to write. Defaults to <first input dir>/pareto_measure_merged.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for merged per-job artifacts. Defaults to the output CSV stem.",
    )
    parser.add_argument(
        "--prefer",
        choices=("successful", "last"),
        default="successful",
        help="Duplicate policy. 'successful' replaces errors with successful rows; 'last' always keeps the later input.",
    )
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Do not fail if a successful row has no matching job artifact files.",
    )
    return parser.parse_args()


def collect_input_csvs(inputs: list[str]) -> list[Path]:
    csvs: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {raw}")

        if path.is_dir():
            matches = [
                match for match in sorted(path.glob("pareto_measure_*.csv"))
                if PARETO_CSV_RE.match(match.name)
            ]
        elif path.suffix.lower() == ".csv":
            matches = [path]
        else:
            raise ValueError(f"Unsupported input, expected CSV or directory: {raw}")

        for match in matches:
            resolved = match.resolve()
            if resolved not in seen:
                seen.add(resolved)
                csvs.append(resolved)

    if not csvs:
        raise ValueError("No pareto_measure CSVs found in the provided inputs")
    return csvs


def default_output_csv(csvs: list[Path]) -> Path:
    return csvs[0].parent / "pareto_measure_merged.csv"


def result_dir_for(csv_path: Path) -> Path:
    return csv_path.with_suffix("")


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    missing = [field for field in KEY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Missing key columns {missing} in row from pareto_measure CSV")
    return tuple(row.get(field, "") for field in KEY_FIELDS)


def is_successful(row: dict[str, str]) -> bool:
    return not row.get("error", "").strip()


def should_replace(existing: dict[str, str], candidate: dict[str, str], prefer: str) -> bool:
    if prefer == "last":
        return True
    if is_successful(candidate) and not is_successful(existing):
        return True
    if is_successful(candidate) == is_successful(existing):
        return True
    return False


def read_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [], []
        return list(reader.fieldnames), list(reader)


def collect_rows(
    csvs: list[Path], prefer: str
) -> tuple[list[str], list[tuple[Path, int, dict[str, str]]], int, int]:
    fieldnames: list[str] = []
    selected: dict[tuple[str, ...], tuple[Path, int, dict[str, str]]] = {}
    order: list[tuple[str, ...]] = []
    total_rows = 0
    replaced_rows = 0

    for csv_path in csvs:
        current_fieldnames, rows = read_rows(csv_path)
        for field in current_fieldnames:
            if field not in fieldnames:
                fieldnames.append(field)

        for row_idx, row in enumerate(rows, start=1):
            total_rows += 1
            key = row_key(row)
            existing = selected.get(key)
            if existing is None:
                order.append(key)
                selected[key] = (csv_path, row_idx, row)
                continue
            if should_replace(existing[2], row, prefer):
                selected[key] = (csv_path, row_idx, row)
                replaced_rows += 1

    return fieldnames, [selected[key] for key in order], total_rows, replaced_rows


def rewrite_artifact_value(value: str, src_dir: Path, dst_dir: Path, src_idx: int, dst_idx: int) -> str:
    if not value:
        return ""

    raw = Path(value)
    source = raw if raw.is_absolute() else (src_dir.parent / raw)
    if not source.exists():
        source = raw if raw.is_absolute() else (src_dir / raw.name)
    if not source.exists():
        return value

    new_name = re.sub(rf"job_{src_idx:04d}", f"job_{dst_idx:04d}", source.name, count=1)
    if new_name == source.name:
        new_name = f"job_{dst_idx:04d}_{source.name}"
    return str(dst_dir / new_name)


def copy_job_artifacts(src_csv: Path, src_idx: int, dst_dir: Path, dst_idx: int) -> int:
    src_dir = result_dir_for(src_csv)
    if not src_dir.exists():
        return 0

    copied = 0
    prefix = f"job_{src_idx:04d}_"
    for src_path in sorted(src_dir.glob(f"{prefix}*")):
        dst_name = src_path.name.replace(prefix, f"job_{dst_idx:04d}_", 1)
        shutil.copy2(src_path, dst_dir / dst_name)
        copied += 1
    return copied


def write_merged(
    fieldnames: list[str],
    selected_rows: list[tuple[Path, int, dict[str, str]]],
    output_csv: Path,
    output_dir: Path,
    allow_missing_artifacts: bool,
) -> tuple[int, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    missing_artifact_rows = 0

    for dst_idx, (src_csv, src_idx, row) in enumerate(selected_rows, start=1):
        merged = dict(row)
        copied = copy_job_artifacts(src_csv, src_idx, output_dir, dst_idx)
        if copied == 0 and is_successful(row):
            missing_artifact_rows += 1
            message = (
                f"warning: no artifacts found for successful row {src_idx} in {src_csv} "
                f"({row.get('curve')}, {row.get('server_config')}, doc_size={row.get('doc_size')})"
            )
            if allow_missing_artifacts:
                print(message, file=sys.stderr)
            else:
                raise FileNotFoundError(message)

        merged["gpu_transfer_csv"] = rewrite_artifact_value(
            merged.get("gpu_transfer_csv", ""), result_dir_for(src_csv), output_dir, src_idx, dst_idx
        )
        rows.append(merged)

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), missing_artifact_rows


def main() -> None:
    args = parse_args()
    csvs = collect_input_csvs(args.inputs)
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else default_output_csv(csvs).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else output_csv.with_suffix("")

    fieldnames, selected_rows, total_rows, replaced_rows = collect_rows(csvs, args.prefer)
    if "gpu_transfer_csv" not in fieldnames:
        fieldnames.append("gpu_transfer_csv")

    merged_count, missing_artifact_rows = write_merged(
        fieldnames,
        selected_rows,
        output_csv,
        output_dir,
        args.allow_missing_artifacts,
    )

    print(f"Read {total_rows} row(s) from {len(csvs)} CSV(s)")
    print(f"Replaced {replaced_rows} duplicate row(s) using prefer={args.prefer}")
    print(f"Wrote {merged_count} merged row(s) to {output_csv}")
    print(f"Copied per-job artifacts to {output_dir}")
    if missing_artifact_rows:
        print(f"Rows with missing artifacts: {missing_artifact_rows}")


if __name__ == "__main__":
    main()
