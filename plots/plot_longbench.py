"""One-off plot: longbench v2 replay benchmark from data/surf/longbench.
4 systems x 1 domain, 1 repetition each. No doc_len sweep, so the summary metrics
are grouped bar charts: x-axis = longbench domain, bars = systems. A separate
fresh-vs-reuse TTFT chart reads the per-request CSVs (x-axis = system).

Run from repo root:  .venv/bin/python -m plots.plot_longbench
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common.plot_common import plain_number_formatter, save_figure
from plots.plot_bench import _maybe_float, _resolve_path

CSV_PATH = "data/surf/longbench/longbench_bench_results_20260619_103341.csv"
OUT_DIR = "longbench plots"

# base_config_name -> display label
SYSTEMS = {
    "baseline": "baseline",
    "pykvcache": "py-kvcache",
    "native_offload_disk": "kv offload (cpu+disk)",
    "lmcache_cpu_disk": "lmcache (cpu+disk)",
}

BASELINE = "baseline"

# metric key -> (title, ylabel, filename, log). "speedup_vs_baseline" is derived
# in derive_speedup: baseline_query_ttft / system_query_ttft.
METRICS = [
    ("query_mean_ttft_s",   "LongBench Query TTFT by Domain",      "Query TTFT (s)",     "longbench_query_ttft.png",  False),
    ("warmup_mean_ttft_s",  "LongBench Warmup TTFT by Domain",     "Warmup TTFT (s)",    "longbench_warmup_ttft.png", False),
    ("query_total_time_s",  "LongBench Query Wall Time by Domain", "Query Wall Time (s)", "longbench_query_time.png", False),
    ("speedup_vs_baseline", "LongBench Query TTFT Speedup vs Baseline", "Speedup vs Baseline (x)", "longbench_speedup.png", False),
]

CSV_COLS = {"query_mean_ttft_s", "warmup_mean_ttft_s", "query_total_time_s"}


def _f(v):
    return float(v) if v not in (None, "") else None


def load():
    """Returns (summary, perreq, domains).

    summary: metric -> system_label -> domain -> value
    perreq:  system_label -> domain -> {"fresh": [ttft], "reuse": [ttft]}
    """
    csv.field_size_limit(10 * 1024 * 1024)
    summary = defaultdict(lambda: defaultdict(dict))
    perreq = defaultdict(lambda: defaultdict(lambda: {"fresh": [], "reuse": []}))
    domains = []
    base_dir = os.path.dirname(os.path.abspath(CSV_PATH))
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row.get("benchmark") != "longbench":
                continue
            base = row.get("base_config_name")
            if base not in SYSTEMS:
                continue
            label = SYSTEMS[base]
            domain = row.get("longbench_domain") or "?"
            if domain not in domains:
                domains.append(domain)
            for col in CSV_COLS:
                val = _f(row.get(col))
                if val is not None:
                    summary[col][label][domain] = val
            _load_perreq(row, label, domain, perreq, base_dir)
    return summary, perreq, domains


def _load_perreq(row, label, domain, perreq, base_dir):
    path = _resolve_path(row.get("per_request_csv", ""), base_dir)
    if not path or not os.path.exists(path):
        return
    with open(path, newline="") as pf:
        for req in csv.DictReader(pf):
            if req["successful"] != "True":
                continue
            bucket = "reuse" if req["is_prefix_reuse"] == "True" else "fresh"
            perreq[label][domain][bucket].append(float(req["ttft"]))


def derive_speedup(summary, domains):
    """speedup_vs_baseline = baseline query TTFT / system query TTFT (baseline = 1.0)."""
    qt = summary["query_mean_ttft_s"]
    for label, by_domain in qt.items():
        for domain in domains:
            base = qt.get(BASELINE, {}).get(domain)
            sysv = by_domain.get(domain)
            if base and sysv:
                summary["speedup_vs_baseline"][label][domain] = base / sysv


def plot_grouped(metric, title, ylabel, filename, log, summary, domains):
    series = summary[metric]
    labels = [SYSTEMS[b] for b in SYSTEMS if SYSTEMS[b] in series]
    n = len(labels)
    width = 0.8 / max(n, 1)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(max(6, len(domains) * n * 0.45 + 2), 5))
    for k, label in enumerate(labels):
        xs, ys = [], []
        for i, domain in enumerate(domains):
            v = series[label].get(domain)
            if v is not None:
                xs.append(i + offsets[k])
                ys.append(v)
        bars = ax.bar(xs, ys, width=width * 0.9, label=label, zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xlabel("LongBench Domain")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    if log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(plain_number_formatter))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_figure(os.path.join(OUT_DIR, filename))

    print(f"\n-- {title} --")
    for label in labels:
        vals = "  ".join(f"{d}={series[label].get(d, float('nan')):.3f}" for d in domains)
        print(f"  {label:<24} {vals}")


def plot_fresh_vs_reuse(perreq, domains):
    """Grouped bars per domain: x-axis = system, two bars (fresh, reuse) of mean TTFT."""
    labels = [SYSTEMS[b] for b in SYSTEMS if SYSTEMS[b] in perreq]
    for domain in domains:
        means = {
            label: {b: (float(np.mean(perreq[label][domain][b]))
                        if perreq[label][domain][b] else None)
                    for b in ("fresh", "reuse")}
            for label in labels
        }
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4 + 2), 5))
        for j, bucket in enumerate(("fresh", "reuse")):
            xs = [i + (j - 0.5) * width for i in range(len(labels))]
            ys = [means[l][bucket] or 0.0 for l in labels]
            bars = ax.bar(xs, ys, width=width * 0.95, label=bucket, zorder=3)
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

        ax.set_xlabel("System")
        ax.set_ylabel("Mean TTFT (s)")
        ax.set_title(f"LongBench Fresh vs Reuse TTFT ({domain})")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(plain_number_formatter))
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        slug = domain.lower().replace(" ", "_").replace("-", "")
        save_figure(os.path.join(OUT_DIR, f"longbench_fresh_vs_reuse_{slug}.png"))

        print(f"\n-- Fresh vs Reuse TTFT ({domain}) --")
        for l in labels:
            fr, ru = means[l]["fresh"], means[l]["reuse"]
            print(f"  {l:<24} fresh={fr if fr is None else round(fr,3)}  "
                  f"reuse={ru if ru is None else round(ru,3)}")


def write_comparisons(out, metric, title, unit, summary, domains):
    """Pairwise system comparison per domain: raw diff, percentage, multiplier.
    For speedup (x), larger is better; for times, smaller is better."""
    series = summary[metric]
    labels = [SYSTEMS[b] for b in SYSTEMS if SYSTEMS[b] in series]
    bigger_better = unit == "x"

    out.write(f"\n{'='*78}\n")
    out.write(f"  {title}  ({unit})\n")
    out.write(f"{'='*78}\n")
    for domain in domains:
        here = [l for l in labels if domain in series[l]]
        if len(here) < 2:
            continue
        out.write(f"\n  domain = {domain}\n")
        for l in here:
            out.write(f"    {l:<24} {series[l][domain]:.3f} {unit}\n")
        out.write("\n")
        for i, a in enumerate(here):
            for b in here[i+1:]:
                va, vb = series[a][domain], series[b][domain]
                if bigger_better:
                    win, lose = (a, b) if va >= vb else (b, a)
                else:
                    win, lose = (a, b) if va <= vb else (b, a)
                wv, lv = series[win][domain], series[lose][domain]
                diff = abs(wv - lv)
                pct = (diff / lv * 100) if lv else 0.0
                mult = (wv / lv) if bigger_better else (lv / wv) if wv else float("inf")
                rel = "higher" if bigger_better else "faster"
                raw = f"{diff:.3f} {unit}" if bigger_better else f"{diff:.3f} s ({diff*1000:.1f} ms)"
                out.write(f"    {win} {rel} than {lose}: "
                          f"{raw}, {pct:.1f}%, {mult:.2f}x\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary, perreq, domains = load()
    derive_speedup(summary, domains)
    for col, title, ylabel, filename, log in METRICS:
        plot_grouped(col, title, ylabel, filename, log, summary, domains)
    plot_fresh_vs_reuse(perreq, domains)

    with open(os.path.join(OUT_DIR, "longbench_comparisons.txt"), "w") as out:
        for col, title, ylabel, *_ in METRICS:
            unit = "x" if col == "speedup_vs_baseline" else "s"
            write_comparisons(out, col, title, unit, summary, domains)


if __name__ == "__main__":
    main()
