#!/usr/bin/env python3

"""Plot ext4-focused filesystem activity from bpftrace stack dumps.

This script turns the per-bucket stack counts in ``trace.out`` into a CSV and
an SVG time plot. In the labels, "Direct I/O" refers to I/O that bypasses the
page cache, which is what the ``ext4_dio_*`` / ``iomap_dio_*`` stack paths
represent in this trace.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


CATEGORY_PREDICATES = {
    "dio_write_submit": lambda text: "ext4_dio_write_iter" in text
    and "iomap_dio_submit_bio" in text,
    "dio_write_flush": lambda text: "ext4_dio_write_iter" in text
    and "blk_finish_plug+49" in text
    and "iomap_dio_submit_bio" not in text,
    "dio_read_submit": lambda text: "ext4_file_read_iter" in text
    and "iomap_dio_submit_bio" in text,
    "dio_read_flush": lambda text: "ext4_file_read_iter" in text
    and "blk_finish_plug+49" in text
    and "iomap_dio_submit_bio" not in text,
    "ext4_alloc_bitmap": lambda text: "ext4_mb_new_blocks" in text
    or "ext4_read_block_bitmap_nowait" in text,
    "jbd2_commit": lambda text: "jbd2_journal_commit_transaction" in text,
    "writeback": lambda text: "wb_writeback" in text,
    "nvme_dispatch_worker": lambda text: "worker_thread" in text
    and "nvme_queue_rq+104" in text,
}

PLOT_PANELS = [
    {
        "title": "EXT4 I/O call rate",
        "series": ["dio_write_submit", "dio_read_submit"],
    },
    {
        "title": "ext4 metadata / background activity",
        "series": [
            "ext4_alloc_bitmap",
            "jbd2_commit",
            "writeback",
            "nvme_dispatch_worker",
        ],
    },
]

SERIES_LABELS = {
    "dio_write_submit": "Direct I/O write submit",
    "dio_write_flush": "Direct I/O write flush",
    "dio_read_submit": "Direct I/O read submit",
    "dio_read_flush": "Direct I/O read flush",
    "ext4_alloc_bitmap": "ext4 alloc/bitmap",
    "jbd2_commit": "jbd2 commit",
    "writeback": "writeback",
    "nvme_dispatch_worker": "nvme dispatch worker",
}

SERIES_COLORS = {
    "dio_write_submit": "#2563eb",
    "dio_write_flush": "#0f766e",
    "dio_read_submit": "#dc2626",
    "dio_read_flush": "#b45309",
    "ext4_alloc_bitmap": "#7c3aed",
    "jbd2_commit": "#ea580c",
    "writeback": "#059669",
    "nvme_dispatch_worker": "#4b5563",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse bpftrace stack dumps from trace.out and emit ext4-focused time-series CSV/SVG plots."
    )
    parser.add_argument(
        "input", nargs="?", default="trace.out", help="Path to the bpftrace dump file"
    )
    parser.add_argument("--csv", dest="csv_path", help="Output CSV path")
    parser.add_argument("--svg", dest="svg_path", help="Output SVG path")
    return parser.parse_args()


def default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_fs_events.{suffix}")


def parse_trace(trace_path: Path) -> list[dict[str, object]]:
    lines = trace_path.read_text().splitlines()
    samples: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("time: "):
            samples.append({"time": int(line.split(":", 1)[1].strip()), "blocks": []})
            index += 1
            continue
        if line.startswith("@io_graph["):
            if not samples:
                raise ValueError("encountered stack data before the first time bucket")
            block_lines = [line]
            index += 1
            while index < len(lines):
                block_line = lines[index]
                block_lines.append(block_line)
                if block_line.startswith("]: "):
                    count = int(block_line.split(":", 1)[1].strip())
                    break
                index += 1
            else:
                raise ValueError("unterminated @io_graph block")
            samples[-1]["blocks"].append(
                {"text": "\n".join(block_lines), "count": count}
            )
        index += 1
    if not samples:
        raise ValueError(f"no time buckets found in {trace_path}")
    return samples


def build_series(
    samples: list[dict[str, object]],
) -> tuple[list[float], dict[str, list[int]]]:
    base_time = int(samples[0]["time"])
    seconds = [(int(sample["time"]) - base_time) / 1e9 for sample in samples]
    series = {name: [] for name in CATEGORY_PREDICATES}

    for sample in samples:
        for name in series:
            series[name].append(0)
        for block in sample["blocks"]:
            text = block["text"]
            count = int(block["count"])
            for name, predicate in CATEGORY_PREDICATES.items():
                if predicate(text):
                    series[name][-1] += count

    return seconds, series


def write_csv(
    csv_path: Path, seconds: list[float], series: dict[str, list[int]]
) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["second", *CATEGORY_PREDICATES]
        writer.writerow(header)
        for idx, second in enumerate(seconds):
            writer.writerow(
                [f"{second:.9f}", *[series[name][idx] for name in CATEGORY_PREDICATES]]
            )


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(
    svg_path: Path, trace_path: Path, seconds: list[float], series: dict[str, list[int]]
) -> None:
    width = 1400
    left = 85
    right = 30
    top = 55
    bottom = 55
    panel_gap = 55
    panel_height = 320
    height = int(
        top
        + bottom
        + panel_height * len(PLOT_PANELS)
        + panel_gap * max(0, len(PLOT_PANELS) - 1)
    )
    plot_width = width - left - right

    def x_map(second: float) -> float:
        start = seconds[0]
        end = seconds[-1]
        span = end - start if end != start else 1.0
        return left + (second - start) / span * plot_width

    def y_map(value: float, ymin: float, ymax: float, panel_top: float) -> float:
        if ymax == ymin:
            return panel_top + panel_height / 2.0
        return panel_top + panel_height - (value - ymin) / (ymax - ymin) * panel_height

    def y_ticks(ymax: int) -> list[int]:
        if ymax <= 0:
            return [0]
        rounded_max = int(math.ceil(ymax / 5.0) * 5)
        step = max(1, rounded_max // 4)
        return [0, step, step * 2, step * 3, rounded_max]

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append("<style>")
    lines.append("text { font-family: Arial, Helvetica, sans-serif; fill: #111827; }")
    lines.append(".small { font-size: 12px; }")
    lines.append(".label { font-size: 13px; font-weight: 600; }")
    lines.append(".title { font-size: 22px; font-weight: 700; }")
    lines.append(".subtitle { font-size: 13px; fill: #4b5563; }")
    lines.append(".grid { stroke: #e5e7eb; stroke-width: 1; }")
    lines.append(".axis { stroke: #9ca3af; stroke-width: 1.25; }")
    lines.append("</style>")
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')

    max_second = int(seconds[-1])
    x_ticks = list(range(0, max_second + 1, 5))
    if not x_ticks or x_ticks[-1] != max_second:
        x_ticks.append(max_second)

    for panel_index, panel in enumerate(PLOT_PANELS):
        panel_top = top + panel_index * (panel_height + panel_gap)
        ymax = max(max(series[name]) for name in panel["series"])
        ticks = y_ticks(ymax)
        ymax = ticks[-1]

        for tick in ticks:
            y = y_map(tick, 0, ymax, panel_top)
            lines.append(
                f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>'
            )
            lines.append(
                f'<text class="small" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick}</text>'
            )

        for tick in x_ticks:
            x = x_map(float(tick))
            lines.append(
                f'<line class="grid" x1="{x:.2f}" y1="{panel_top:.2f}" x2="{x:.2f}" y2="{panel_top + panel_height:.2f}"/>'
            )
            lines.append(
                f'<text class="small" x="{x:.2f}" y="{panel_top + panel_height + 18:.2f}" text-anchor="middle">{tick}s</text>'
            )

        lines.append(
            f'<line class="axis" x1="{left}" y1="{panel_top + panel_height:.2f}" x2="{left + plot_width}" y2="{panel_top + panel_height:.2f}"/>'
        )
        lines.append(
            f'<line class="axis" x1="{left}" y1="{panel_top:.2f}" x2="{left}" y2="{panel_top + panel_height:.2f}"/>'
        )
        lines.append(
            f'<text class="label" x="{left}" y="{panel_top - 12:.2f}">{svg_escape(panel["title"])}</text>'
        )

        legend_x = left + 10
        legend_y = panel_top + 18
        for legend_index, name in enumerate(panel["series"]):
            y = legend_y + legend_index * 18
            color = SERIES_COLORS[name]
            lines.append(
                f'<line x1="{legend_x}" y1="{y:.2f}" x2="{legend_x + 20}" y2="{y:.2f}" stroke="{color}" stroke-width="3"/>'
            )
            lines.append(
                f'<text class="small" x="{legend_x + 28}" y="{y + 4:.2f}">{svg_escape(SERIES_LABELS[name])}</text>'
            )

        for name in panel["series"]:
            points = " ".join(
                f"{x_map(seconds[idx]):.2f},{y_map(series[name][idx], 0, ymax, panel_top):.2f}"
                for idx in range(len(seconds))
            )
            color = SERIES_COLORS[name]
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>'
            )

            peak = max(series[name])
            if peak > 0:
                peak_index = series[name].index(peak)
                peak_x = x_map(seconds[peak_index])
                peak_y = y_map(peak, 0, ymax, panel_top)
                lines.append(
                    f'<circle cx="{peak_x:.2f}" cy="{peak_y:.2f}" r="4" fill="{color}"/>'
                )
                lines.append(
                    f'<text class="small" x="{peak_x + 6:.2f}" y="{peak_y - 6:.2f}">{peak}</text>'
                )

    lines.append("</svg>")

    svg_path.write_text("\n".join(lines))


def print_summary(seconds: list[float], series: dict[str, list[int]]) -> None:
    deltas = [seconds[idx] - seconds[idx - 1] for idx in range(1, len(seconds))]
    avg_period = sum(deltas) / len(deltas) if deltas else 0.0
    print(f"buckets={len(seconds)} avg_period_s={avg_period:.6f}")
    for name in CATEGORY_PREDICATES:
        values = series[name]
        peak = max(values)
        peak_second = seconds[values.index(peak)] if peak else 0.0
        print(f"{name}: total={sum(values)} peak={peak} peak_second={peak_second:.6f}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    csv_path = (
        Path(args.csv_path) if args.csv_path else default_output_path(input_path, "csv")
    )
    svg_path = (
        Path(args.svg_path) if args.svg_path else default_output_path(input_path, "svg")
    )

    samples = parse_trace(input_path)
    seconds, series = build_series(samples)
    write_csv(csv_path, seconds, series)
    render_svg(svg_path, input_path, seconds, series)
    print_summary(seconds, series)
    print(f"wrote {csv_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
