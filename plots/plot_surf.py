"""One-off plot: compare py-kvcache / native-offload / lmcache variants from the
data/surf CSVs. Produces wall-time and query-TTFT line plots vs document length,
one of each for max_concurrency 8 and 50.

Run from repo root:  python -m plots.plot_surf_oneoff
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common.plot_common import plain_number_formatter, save_figure
from plots.plot_bench import (
    _maybe_float,
    _maybe_int,
    _parse_config_params,
    _resolve_path,
)

DATA_DIR = "data/surf"

# (relative csv path, base_config_name) -> series label
SERIES = {
    ("pykvcache-total/bench_py_kvcache_results_20260616_121252.csv", "py_kvcache_preload"): "py-kvcache (cpu+disk+preload)",
    ("pykvcache-total/bench_py_kvcache_results_20260615_152301.csv", "py_kvcache_preload"): "py-kvcache (disk+preload)",
    ("pykvcache-total/bench_py_kvcache_results_20260615_152301.csv", "py_kvcache_no_preload"): "py-kvcache (disk)",
    ("nativeoffload-total/bench_native_offload_results_20260616_113721.csv", "native_offload"): "kv offload (cpu)",
    ("nativeoffload-total/bench_native_offload_results_20260616_113721.csv", "native_offload_disk_tier"): "kv offload (cpu+disk)",
    ("lmcache/bench_lmcache_results_20260616_105153.csv", "lmcache_cpu"): "lmcache (cpu)",
    ("lmcache/bench_lmcache_results_20260616_105153.csv", "lmcache_disk"): "lmcache (disk)",
    ("lmcache/bench_lmcache_results_20260616_105153.csv", "lmcache_cpu_disk"): "lmcache (cpu+disk)",
}

CONCURRENCIES = ["8", "50"]

OUT_DIR = "surf plots"

# category slug -> list of series labels (must match SERIES values)
# category slug -> title display name ("" = no category suffix)
CATEGORY_TITLES = {
    "pure_disk": "Disk only",
    "disk_dram": "",
    "pure_dram": "Cpu only",
}

CATEGORIES = {
    "pure_disk": [
        "lmcache (disk)",
        "py-kvcache (disk+preload)",
        "py-kvcache (disk)",
    ],
    "disk_dram": [
        "lmcache (cpu+disk)",
        "kv offload (cpu+disk)",
        "py-kvcache (cpu+disk+preload)",
        "py-kvcache (disk+preload)",
    ],
    "pure_dram": [
        "py-kvcache (disk+preload)",
        "py-kvcache (cpu+disk+preload)",
        "kv offload (cpu)",
        "lmcache (cpu)",
    ],
}


def load(csv_rel, base_filter, label, concurrency, query_series, wall_series):
    csv_path = os.path.join(DATA_DIR, csv_rel)
    csv.field_size_limit(10 * 1024 * 1024)
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("benchmark") != "prefix_cache":
                continue
            if row.get("base_config_name") != base_filter:
                continue
            params = _parse_config_params(row.get("config_name", ""))
            if params.get("max_concurrency") != concurrency:
                continue
            doc_len = _maybe_int(row.get("doc_len"))
            if doc_len is None:
                continue

            qt = _maybe_float(row.get("query_total_time_s"))
            if qt is not None:
                wall_series[label][doc_len].append(qt)

            per_req = _resolve_path(row.get("per_request_csv", ""), base_dir)
            if not per_req or not os.path.exists(per_req):
                continue
            with open(per_req, newline="") as pf:
                for req in csv.DictReader(pf):
                    if req["successful"] != "True":
                        continue
                    if req["is_prefix_reuse"] == "True":
                        query_series[label][doc_len].append(float(req["ttft"]))


def plot_grouped(series, labels, title, ylabel, filename, log):
    """Grouped bar chart: x-axis = doc_len, bars = series labels, value = mean.

    series: dict[label -> dict[doc_len -> list[float]]]
    """
    present = [l for l in labels if l in series and series[l]]
    doc_lens = sorted({d for l in present for d in series[l]})
    n = len(present)
    width = 0.8 / max(n, 1)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(max(8, len(doc_lens) * n * 0.3 + 2), 5))
    for k, label in enumerate(present):
        xs, ys = [], []
        for i, doc in enumerate(doc_lens):
            vals = series[label].get(doc)
            if vals:
                xs.append(i + offsets[k])
                ys.append(float(np.mean(vals)))
        bars = ax.bar(xs, ys, width=width * 0.9, label=label, zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=6)

    ax.set_xlabel("Document Length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(doc_lens)))
    ax.set_xticklabels([f"{d:,}" for d in doc_lens])
    if log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(plain_number_formatter))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_figure(filename)

    print(f"\n── {title} ──")
    for label in present:
        vals = "  ".join(
            f"{d:,}={np.mean(series[label][d]):.3f}" for d in sorted(series[label])
        )
        print(f"  {label:<28} {vals}")


def write_comparisons(out, cat, conc, metric_name, unit, series, labels):
    """Pairwise comparison per doc_len: raw diff, percentage, multiplier."""
    import numpy as np

    present = [l for l in labels if l in series and series[l]]
    if len(present) < 2:
        return
    all_doc = sorted({d for l in present for d in series[l]})
    means = {l: {d: float(np.mean(series[l][d])) for d in series[l]} for l in present}

    out.write(f"\n{'='*78}\n")
    out.write(f"  {cat} — {metric_name} (concurrency={conc})\n")
    out.write(f"{'='*78}\n")
    for doc in all_doc:
        here = [l for l in present if doc in means[l]]
        if len(here) < 2:
            continue
        out.write(f"\n  doc_len = {doc:,}\n")
        for l in here:
            out.write(f"    {l:<28} {means[l][doc]:.3f} {unit}\n")
        out.write("\n")
        for i, a in enumerate(here):
            for b in here[i+1:]:
                ma, mb = means[a][doc], means[b][doc]
                faster, slower = (a, b) if ma <= mb else (b, a)
                fv, sv = means[faster][doc], means[slower][doc]
                diff = sv - fv                       # raw, faster is smaller
                pct = (diff / sv * 100) if sv else 0.0
                mult = (sv / fv) if fv else float("inf")
                out.write(f"    {faster} faster than {slower}: "
                          f"{diff:.3f} {unit} ({diff*1000:.1f} ms), "
                          f"{pct:.1f}%, {mult:.2f}x\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cmp_file = open(os.path.join(OUT_DIR, "comparisons.txt"), "w")
    for conc in CONCURRENCIES:
        query_series = defaultdict(lambda: defaultdict(list))
        wall_series = defaultdict(lambda: defaultdict(list))

        for (csv_rel, base), label in SERIES.items():
            load(csv_rel, base, label, conc, query_series, wall_series)

        for cat, labels in CATEGORIES.items():
            q = {l: query_series[l] for l in labels if l in query_series}
            w = {l: wall_series[l] for l in labels if l in wall_series}

            disp = CATEGORY_TITLES.get(cat, cat)
            cat_sfx = f", {disp}" if disp else ""

            plot_grouped(
                q, labels,
                f"Query TTFT vs doc length{cat_sfx}, conc={conc}",
                "Query TTFT (s)", os.path.join(OUT_DIR, f"surf_{cat}_query_ttft_conc{conc}.png"),
                log=True,
            )
            plot_grouped(
                w, labels,
                f"Query wall time vs doc length{cat_sfx}, conc={conc}",
                "Query wall time (s)", os.path.join(OUT_DIR, f"surf_{cat}_wall_time_conc{conc}.png"),
                log=True,
            )
            write_comparisons(cmp_file, cat, conc, "Query TTFT", "s", q, labels)
            write_comparisons(cmp_file, cat, conc, "Query wall time", "s", w, labels)

    cmp_file.close()


if __name__ == "__main__":
    main()
