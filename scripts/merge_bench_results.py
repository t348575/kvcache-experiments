import argparse
import csv
import shutil
import sys
from pathlib import Path


ARTIFACT_COLUMNS = ("per_request_csv", "gpu_transfer_csv", "sg_result_json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge benchmark summary CSVs into one plot_bench-compatible dataset."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input benchmark summary CSVs and/or directories containing them.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the merged summary CSV to write.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory for copied per-run CSV/JSON artifacts. Defaults to <output-csv stem>_artifacts.",
    )
    return parser.parse_args()


def collect_input_csvs(inputs: list[str]) -> list[Path]:
    csvs: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {raw}")

        matches: list[Path]
        if path.is_dir():
            matches = sorted(path.rglob("*_results_*.csv"))
        elif path.suffix.lower() == ".csv":
            matches = [path]
        else:
            raise ValueError(f"Unsupported input (expected CSV or directory): {raw}")

        for match in matches:
            resolved = match.resolve()
            if resolved not in seen:
                seen.add(resolved)
                csvs.append(resolved)

    if not csvs:
        raise ValueError("No benchmark summary CSVs found in the provided inputs")
    return csvs


def resolve_artifact_path(csv_path: Path, raw_value: str) -> Path | None:
    if not raw_value:
        return None

    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    relative_to_csv = (csv_path.parent / raw_value).resolve()
    if relative_to_csv.exists():
        return relative_to_csv

    relative_to_cwd = Path(raw_value).resolve()
    if relative_to_cwd.exists():
        return relative_to_cwd

    return None


def copy_artifact(csv_path: Path, row_idx: int, field: str, raw_value: str, artifacts_dir: Path) -> str:
    if not raw_value:
        return ""

    source = resolve_artifact_path(csv_path, raw_value)
    if source is None:
        print(
            f"warning: could not resolve {field}={raw_value!r} from {csv_path}",
            file=sys.stderr,
        )
        return ""

    dest_name = f"{csv_path.stem}__row{row_idx:05d}__{field}{source.suffix}"
    dest_path = artifacts_dir / dest_name
    shutil.copy2(source, dest_path)
    return str(dest_path)


def merge_csvs(csv_paths: list[Path], output_csv: Path, artifacts_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue

            for field in reader.fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)

            for row_idx, row in enumerate(reader, start=1):
                merged = dict(row)
                for artifact_field in ARTIFACT_COLUMNS:
                    merged[artifact_field] = copy_artifact(
                        csv_path, row_idx, artifact_field, row.get(artifact_field, ""), artifacts_dir
                    )
                merged["source_csv"] = str(csv_path)
                rows.append(merged)

    if "source_csv" not in fieldnames:
        fieldnames.append("source_csv")

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else output_csv.with_name(f"{output_csv.stem}_artifacts")

    csv_paths = collect_input_csvs(args.inputs)
    merge_csvs(csv_paths, output_csv, artifacts_dir)

    print(f"Merged {len(csv_paths)} summary CSV(s) into {output_csv}")
    print(f"Copied referenced artifacts into {artifacts_dir}")


if __name__ == "__main__":
    main()
