"""One-off plot: bailian trace-replay benchmark from data/surf/bailian-nooutput.
4 systems x 3 tasks (coder/A/B), 1 repetition each. No doc_len sweep, so each
metric is a grouped bar chart: x-axis = bailian task, bars = systems.

Run from repo root:  .venv/bin/python -m plots.plot_bailian_oneoff
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common.plot_common import plain_number_formatter, save_figure

CSV_PATH = "data/node10/bailian/merged.csv"
OUT_DIR = "node10 plots"

# base_config_name -> display label
SYSTEMS = {
    "baseline_recompute": "baseline",
    "pykvcache": "py-kvcache",
    "native_offload_disk": "kv offload (cpu+disk)",
    "lmcache_cpu_disk": "lmcache (cpu+disk)",
}

BASELINE = "baseline"

# metric key -> (title, ylabel, filename, log). "speedup_vs_baseline" is derived,
# not a CSV column (see derive_speedup): baseline_query_ttft / system_query_ttft.
METRICS = [
    ("query_mean_ttft_s",  "Alibaba Query TTFT by Task",   "Query TTFT (s)",       "bailian_query_ttft.png",  False),
    ("warmup_mean_ttft_s", "Alibaba Warmup TTFT by Task",  "Warmup TTFT (s)",      "bailian_warmup_ttft.png", False),
    ("query_total_time_s", "Alibaba Query Wall Time by Task", "Query Wall Time (s)", "bailian_query_time.png", False),
    ("speedup_vs_baseline", "Alibaba Query TTFT Speedup vs Baseline", "Speedup vs Baseline (x)", "bailian_speedup.png", False),
]

CSV_COLS = {"query_mean_ttft_s", "warmup_mean_ttft_s", "query_total_time_s"}


def _f(v):
    return float(v) if v not in (None, "") else None


def load():
    """metric -> system_label -> task -> value"""
    csv.field_size_limit(10 * 1024 * 1024)
    data = defaultdict(lambda: defaultdict(dict))
    tasks = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row.get("benchmark") != "bailian":
                continue
            base = row.get("base_config_name")
            if base not in SYSTEMS:
                continue
            label = SYSTEMS[base]
            task = row.get("bailian_task") or "?"
            if task not in tasks:
                tasks.append(task)
            for col in CSV_COLS:
                val = _f(row.get(col))
                if val is not None:
                    data[col][label][task] = val
    return data, tasks


def derive_speedup(data, tasks):
    """speedup_vs_baseline = baseline query TTFT / system query TTFT (baseline = 1.0)."""
    qt = data["query_mean_ttft_s"]
    for label, by_task in qt.items():
        for task in tasks:
            base = qt.get(BASELINE, {}).get(task)
            sysv = by_task.get(task)
            if base and sysv:
                data["speedup_vs_baseline"][label][task] = base / sysv


def plot_grouped(metric, title, ylabel, filename, log, data, tasks):
    series = data[metric]
    labels = [SYSTEMS[b] for b in SYSTEMS if SYSTEMS[b] in series]
    n = len(labels)
    width = 0.8 / max(n, 1)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * n * 0.45 + 2), 5))
    for k, label in enumerate(labels):
        xs, ys = [], []
        for i, task in enumerate(tasks):
            v = series[label].get(task)
            if v is not None:
                xs.append(i + offsets[k])
                ys.append(v)
        bars = ax.bar(xs, ys, width=width * 0.9, label=label, zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xlabel("Bailian Task")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks)
    if log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(plain_number_formatter))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_figure(os.path.join(OUT_DIR, filename))

    print(f"\n── {title} ──")
    for label in labels:
        vals = "  ".join(f"{t}={series[label].get(t, float('nan')):.3f}" for t in tasks)
        print(f"  {label:<24} {vals}")


def write_comparisons(out, metric, title, unit, data, tasks):
    """Pairwise system comparison per task: raw diff, percentage, multiplier.
    For speedup (x), larger is better; for times, smaller is better."""
    series = data[metric]
    labels = [SYSTEMS[b] for b in SYSTEMS if SYSTEMS[b] in series]
    bigger_better = unit == "x"

    out.write(f"\n{'='*78}\n")
    out.write(f"  {title}  ({unit})\n")
    out.write(f"{'='*78}\n")
    for task in tasks:
        here = [l for l in labels if task in series[l]]
        if len(here) < 2:
            continue
        out.write(f"\n  task = {task}\n")
        for l in here:
            out.write(f"    {l:<24} {series[l][task]:.3f} {unit}\n")
        out.write("\n")
        for i, a in enumerate(here):
            for b in here[i+1:]:
                va, vb = series[a][task], series[b][task]
                if bigger_better:
                    win, lose = (a, b) if va >= vb else (b, a)
                else:
                    win, lose = (a, b) if va <= vb else (b, a)
                wv, lv = series[win][task], series[lose][task]
                diff = abs(wv - lv)
                ref = lv  # worse value; pct is relative to it
                pct = (diff / ref * 100) if ref else 0.0
                mult = (wv / lv) if bigger_better else (lv / wv) if wv else float("inf")
                rel = "higher" if bigger_better else "faster"
                raw = f"{diff:.3f} {unit}" if bigger_better else f"{diff:.3f} s ({diff*1000:.1f} ms)"
                out.write(f"    {win} {rel} than {lose}: "
                          f"{raw}, {pct:.1f}%, {mult:.2f}x\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data, tasks = load()
    derive_speedup(data, tasks)
    for col, title, ylabel, filename, log in METRICS:
        plot_grouped(col, title, ylabel, filename, log, data, tasks)

    with open(os.path.join(OUT_DIR, "bailian_comparisons.txt"), "w") as out:
        for col, title, ylabel, *_ in METRICS:
            unit = "x" if col == "speedup_vs_baseline" else "s"
            write_comparisons(out, col, title, unit, data, tasks)


if __name__ == "__main__":
    main()
