import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np



def parse_config_name(name):
    """Extract strategy, doc_len, and batch_size from config_name.

    Returns (strategy_label, doc_len, batch_size).
    strategy_label has no doc/batch info embedded (e.g. "vLLM Offloading", "LMCache chunk=64").
    """
    doc_m = re.search(r"document_length=(\d+)", name)
    doc_len = int(doc_m.group(1)) if doc_m else None

    chunk_m = re.search(r"LMCACHE_CHUNK_SIZE=(\d+)", name)
    chunk_size = int(chunk_m.group(1)) if chunk_m else None

    batch_m = re.search(r"max_num_batched_tokens=(\d+)", name)
    batch_size = int(batch_m.group(1)) if batch_m else None

    if name.startswith("offloading"):
        strategy = "vLLM Offloading"
    elif name.startswith("baseline"):
        strategy = "Baseline"
    else:
        strategy = f"LMCache chunk={chunk_size}"

    return strategy, doc_len, batch_size


def plot_ttft(series, metric_key, title, ylabel, filename, normalize=False, boxplot=True, log=True, xlabel="Document Length (tokens)", xfmt=lambda x: f"{x:,}"):
    """Line plot: x-axis = x_key values, one line per strategy.

    series: dict[strategy -> dict[x_key -> list[float]]]
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    def label_sort_key(l):
        if l.startswith("Baseline"):
            return (0, 0)
        if "Offloading" in l:
            return (1, 0)
        chunk_m = re.search(r"chunk=(\d+)", l)
        return (2, int(chunk_m.group(1)) if chunk_m else 0)

    ordered_labels = sorted(series.keys(), key=label_sort_key)

    all_doc_lens = sorted({d for s in series.values() for d in s})
    doc_len_to_idx = {d: i for i, d in enumerate(all_doc_lens)}

    if normalize:
        baseline = {d: np.mean(v) for d, v in series.get("Baseline", {}).items()}

    active_labels = [l for l in ordered_labels if not (normalize and l == "Baseline")]
    n = len(active_labels)
    offsets = np.linspace(-0.2, 0.2, n) if (boxplot and n > 1) else [0.0] * n

    for k, label in enumerate(active_labels):
        label_doc_lens = sorted(series[label].keys())
        xs = [doc_len_to_idx[d] for d in label_doc_lens]
        off = offsets[k]

        all_vals = []
        for d in label_doc_lens:
            vals = list(series[label][d])
            if normalize:
                vals = [baseline[d] / v for v in vals if d in baseline]
            all_vals.append(vals)

        mean_vals = [np.mean(v) for v in all_vals]

        line, = ax.plot([x + off for x in xs], mean_vals,
                        marker="o", linewidth=2, label=label, zorder=3)
        color = line.get_color()

        if boxplot:
            ax.boxplot(
                all_vals, positions=[x + off for x in xs], widths=0.15,
                patch_artist=True, manage_ticks=False,
                boxprops=dict(facecolor=color, alpha=0.3),
                medianprops=dict(color=color, linewidth=1.5),
                whiskerprops=dict(color=color, alpha=0.7),
                capprops=dict(color=color, alpha=0.7),
                flierprops=dict(marker="o", markerfacecolor=color, alpha=0.3, markersize=2),
                zorder=2,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speedup over Baseline (x)" if normalize else ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(all_doc_lens)))
    ax.set_xticklabels([xfmt(d) for d in all_doc_lens])
    ax.legend()
    if normalize:
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    else:
        if log:
            ax.set_yscale("log", base=10)
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1], numticks=10))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n── {title} ──")
    for label in active_labels:
        print(f"  {label}")
        for d in sorted(series[label].keys()):
            vals = list(series[label][d])
            if normalize and d in baseline:
                vals = [baseline[d] / v for v in vals]
            print(f"    doc_len={d:>6,}  n={len(vals):>4}  "
                  f"mean={np.mean(vals):.3f}  "
                  f"median={np.median(vals):.3f}  "
                  f"p25={np.percentile(vals, 25):.3f}  "
                  f"p75={np.percentile(vals, 75):.3f}  "
                  f"min={np.min(vals):.3f}  "
                  f"max={np.max(vals):.3f}")


def print_comparison(series, title):
    """Print pairwise TTFT comparisons between all strategies.

    series: dict[strategy -> dict[doc_len -> list[float]]]
    For each doc_len, prints the speedup and absolute difference of every
    strategy pair, with baseline (if present) listed first.
    """
    def sort_key(l):
        if l.startswith("Baseline"):
            return (0, 0)
        if "Offloading" in l:
            return (1, 0)
        chunk_m = re.search(r"chunk=(\d+)", l)
        return (2, int(chunk_m.group(1)) if chunk_m else 0)

    labels = sorted(series.keys(), key=sort_key)
    all_doc_lens = sorted({d for s in series.values() for d in s})

    # means[label][doc_len]
    means = {lbl: {d: np.mean(series[lbl][d]) for d in series[lbl]} for lbl in labels}

    print(f"\n{'═'*70}")
    print(f"  Comparison — {title}")
    print(f"{'═'*70}")

    for doc_len in all_doc_lens:
        present = [l for l in labels if doc_len in means[l]]
        if len(present) < 2:
            continue
        print(f"\n  doc_len = {doc_len:,}")
        # raw means
        for lbl in present:
            print(f"    {lbl:<40}  mean = {means[lbl][doc_len]:.3f}s")
        # pairwise speedups (reference → challenger)
        print()
        for i, ref in enumerate(present):
            for chal in present[i+1:]:
                ref_mean  = means[ref][doc_len]
                chal_mean = means[chal][doc_len]
                speedup   = ref_mean / chal_mean
                diff_ms   = (ref_mean - chal_mean) * 1000
                faster    = chal if speedup >= 1 else ref
                slower    = ref  if speedup >= 1 else chal
                ratio     = max(speedup, 1/speedup)
                delta_ms  = abs(diff_ms)
                print(f"    {faster:<40} vs {slower}")
                print(f"      → {ratio:.2f}× faster  ({delta_ms:+.1f} ms difference)")


def plot_grouped_bar(series, title, ylabel, filename, log=False):
    """Grouped bar plot: x-axis = doc_size groups, bars within group = batch sizes per strategy.

    series: dict[line_label -> dict[batch_size -> list[float]]]
    line_label is e.g. "vLLM Offloading doc=32768".
    Groups on x-axis are doc sizes; bars within each group are (strategy, batch_size) combos.
    """
    # Collect all doc sizes and (strategy, batch_size) bar keys
    all_doc_sizes = sorted({
        int(re.search(r"doc=(\d+)", lbl).group(1))
        for lbl in series
        if re.search(r"doc=(\d+)", lbl)
    })

    def strategy_from_label(l):
        doc_m = re.search(r" doc=\d+", l)
        return l[:doc_m.start()] if doc_m else l

    def bar_sort_key(item):
        strategy, batch = item
        if strategy.startswith("Baseline"):
            return (0, 0, batch or 0)
        if "Offloading" in strategy:
            return (1, 0, batch or 0)
        chunk_m = re.search(r"chunk=(\d+)", strategy)
        return (2, int(chunk_m.group(1)) if chunk_m else 0, batch or 0)

    # Build ordered (strategy, batch_size) pairs
    bar_keys = sorted({
        (strategy_from_label(lbl), b)
        for lbl, bdict in series.items()
        for b in bdict
    }, key=bar_sort_key)    

    n_groups = len(all_doc_sizes)
    n_bars = len(bar_keys)
    bar_width = 0.8 / max(n_bars, 1)
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width

    fig, ax = plt.subplots(figsize=(max(6, n_groups * n_bars * 0.4 + 2), 5))

    for k, (strategy, batch) in enumerate(bar_keys):
        means = []
        valid_xs = []
        for i, doc in enumerate(all_doc_sizes):
            full_lbl = f"{strategy} doc={doc}"
            vals = series.get(full_lbl, {}).get(batch, [])
            m = np.mean(vals) if vals else None
            if m is not None and m > 0:
                means.append(m)
                valid_xs.append(i)
        if not means:
            continue
        xs = np.array(valid_xs) + offsets[k]
        bar_label = f"{strategy} batch={batch}" if batch is not None else strategy
        ax.bar(xs, means, width=bar_width * 0.9, label=bar_label, zorder=3)

    ax.set_xlabel("Document Length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f"{d:,}" for d in all_doc_sizes])
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs="all", numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n── {title} ──")
    for strategy, batch in bar_keys:
        bar_label = f"{strategy} batch={batch}" if batch is not None else strategy
        print(f"  {bar_label}")
        for doc in all_doc_sizes:
            full_lbl = f"{strategy} doc={doc}"
            vals = series.get(full_lbl, {}).get(batch, [])
            if not vals:
                continue
            print(f"    doc={doc:,}  n={len(vals):>4}  mean={np.mean(vals):.3f}  "
                  f"median={np.median(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")


def load_into_series(csv_path, warmup_series, query_series,
                     to_gpu_series, from_gpu_series, exclude, prefix="", renames=None):
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            strategy, doc_len, batch_size = parse_config_name(row["config_name"])
            if renames:
                strategy = renames.get(strategy, strategy)
            if prefix:
                strategy = f"{prefix}: {strategy}"
            if doc_len in exclude:
                continue

            # GPU bar plots: series["{strategy} doc={doc_len}"][batch_size]
            gpu_label = f"{strategy} doc={doc_len}"

            # GPU transfer throughput: bytes / dur_s → GB/s, one value per kernel call
            # All individual transfers across all repetitions are pooled; mean is taken at plot time.
            transfer_csv_path = row.get("gpu_transfer_csv", "")
            if transfer_csv_path and os.path.exists(transfer_csv_path):
                to_bw, from_bw = [], []
                with open(transfer_csv_path, newline="") as tf:
                    for tr in csv.DictReader(tf):
                        dur_us = float(tr["dur_us"]) if tr["dur_us"] else 0.0
                        num_bytes = float(tr["num_bytes"]) if tr["num_bytes"] else 0.0
                        if dur_us <= 0 or num_bytes <= 0:
                            continue
                        gbps = (num_bytes / 1e9) / (dur_us / 1e6)
                        if tr["direction"] == "to_gpu":
                            to_bw.append(gbps)
                        elif tr["direction"] == "from_gpu":
                            from_bw.append(gbps)
                is_lmcache = row["config_name"].startswith("lmcache")
                if is_lmcache:
                    to_bw, from_bw = from_bw, to_bw
                to_gpu_series[gpu_label][batch_size].extend(to_bw)
                from_gpu_series[gpu_label][batch_size].extend(from_bw)

            # TTFT line plots: series[strategy][doc_len] — pool all batch sizes
            per_req_path = row.get("per_request_csv", "")
            if not per_req_path or not os.path.exists(per_req_path):
                continue
            with open(per_req_path, newline="") as pf:
                for req in csv.DictReader(pf):
                    if req["successful"] != "True":
                        continue
                    ttft = float(req["ttft"])
                    if req["round"] == "warmup":
                        warmup_series[strategy][doc_len].append(ttft)
                    else:
                        query_series[strategy][doc_len].append(ttft)


SG_METRICS = {
    "sg_mean_ttft_ms":           "Mean TTFT (ms)",
    "sg_p99_ttft_ms":            "P99 TTFT (ms)",
    "sg_request_throughput":     "Request Throughput (req/s)",
    "sg_output_throughput":      "Output Throughput (tok/s)",
    "sg_total_token_throughput": "Total Token Throughput (tok/s)",
}


def load_sharegpt_series(csv_path, sg_series, exclude_rates=None, prefix="", renames=None):
    """Load sharegpt metrics from summary CSV.

    sg_series: dict[metric -> dict[strategy -> dict[request_rate -> list[float]]]]
    """
    exclude_rates = exclude_rates or set()
    csv.field_size_limit(10 * 1024 * 1024)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row.get("sg_result_json"):
                continue
            strategy, _, _ = parse_config_name(row["config_name"])
            if renames:
                strategy = renames.get(strategy, strategy)
            if prefix:
                strategy = f"{prefix}: {strategy}"

            rate_m = re.search(r"request_rate=(\d+(?:\.\d+)?)", row["config_name"])
            request_rate = float(rate_m.group(1)) if rate_m else float(row.get("sg_request_rate") or 0)
            if request_rate in exclude_rates:
                continue

            for metric in SG_METRICS:
                val = row.get(metric, "")
                if val:
                    sg_series[metric][strategy][request_rate].append(float(val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to the summary CSV file")
    parser.add_argument("--name", type=str, default="",
                        help="Label prefix for the primary CSV (e.g. 'Run 1')")
    parser.add_argument("--extra", metavar="NAME:PATH", action="append", default=[],
                        help="Additional CSV file with a label prefix, e.g. 'Run 2:results2.csv'")
    parser.add_argument("--exclude-doc-len", type=int, nargs="+", default=[],
                        help="Document lengths to exclude (e.g. --exclude-doc-len 81920)")
    parser.add_argument("--normalize", action="store_true",
                        help="Plot speedup relative to Baseline instead of raw TTFT")
    parser.add_argument("--no-boxplot", action="store_true",
                        help="Plot only the mean line without box plots")
    parser.add_argument("--no-log", action="store_true",
                        help="Use a regular scale, not a log scale")
    parser.add_argument("--rename", metavar="OLD:NEW", action="append", default=[],
                        help="Rename a strategy label, e.g. 'vLLM Offloading:Offload'")
    args = parser.parse_args()
    exclude = set(args.exclude_doc_len)

    renames = {}
    for entry in args.rename:
        if ":" not in entry:
            parser.error(f"--rename must be in OLD:NEW format, got: {entry!r}")
        old, new = entry.split(":", 1)
        renames[old] = new

    query_series    = defaultdict(lambda: defaultdict(list))
    warmup_series   = defaultdict(lambda: defaultdict(list))
    to_gpu_series   = defaultdict(lambda: defaultdict(list))
    from_gpu_series = defaultdict(lambda: defaultdict(list))
    sg_series       = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    load_into_series(args.csv, warmup_series, query_series,
                     to_gpu_series, from_gpu_series, exclude, prefix=args.name, renames=renames)
    load_sharegpt_series(args.csv, sg_series, prefix=args.name, renames=renames)

    for entry in args.extra:
        if ":" not in entry:
            parser.error(f"--extra must be in NAME:PATH format, got: {entry!r}")
        name, path = entry.split(":", 1)
        load_into_series(path, warmup_series, query_series,
                         to_gpu_series, from_gpu_series, exclude, prefix=name, renames=renames)
        load_sharegpt_series(path, sg_series, prefix=name, renames=renames)

    plot_ttft(query_series, "query_mean_ttft_s",
              "(log scale) Query-Round TTFT vs Max Num Batched Tokens", "Query TTFT (s)",
              "query_ttft_vs_doclen.png", normalize=args.normalize, boxplot=not args.no_boxplot, log=not args.no_log)

    plot_ttft(warmup_series, "warmup_mean_ttft_s",
              "Warmup-Round TTFT vs Max Num Batched Tokens", "Warmup TTFT (s)",
              "warmup_ttft_vs_doclen.png", normalize=args.normalize, boxplot=not args.no_boxplot, log=not args.no_log)

    combined_series = defaultdict(lambda: defaultdict(list))
    for label, doc_lens in warmup_series.items():
        for doc_len, vals in doc_lens.items():
            combined_series[label][doc_len].extend(vals)
    for label, doc_lens in query_series.items():
        for doc_len, vals in doc_lens.items():
            combined_series[label][doc_len].extend(vals)

    plot_ttft(combined_series, None,
              "Combined TTFT vs Max Num Batched Tokens", "TTFT (s)",
              "combined_ttft_vs_doclen.png", normalize=args.normalize, boxplot=not args.no_boxplot, log=not args.no_log)

    print_comparison(query_series,  "Query-Round TTFT")
    print_comparison(warmup_series, "Warmup-Round TTFT")

    if to_gpu_series:
        plot_grouped_bar(to_gpu_series,
                         "CPU→GPU Transfer Throughput vs Document Length",
                         "Throughput (GB/s)", "to_gpu_vs_doclen.png", log=True)

    if from_gpu_series:
        plot_grouped_bar(from_gpu_series,
                         "GPU→CPU Transfer Throughput vs Document Length",
                         "Throughput (GB/s)", "from_gpu_vs_doclen.png", log=True)

    for metric, ylabel in SG_METRICS.items():
        if not any(sg_series[metric].values()):
            continue
        slug = metric.replace("sg_", "").replace("_", "-")
        plot_ttft(
            sg_series[metric], None,
            f"ShareGPT — {ylabel} vs Request Rate", ylabel,
            f"sharegpt_{slug}.png",
            normalize=False, boxplot=not args.no_boxplot, log=False,
            xlabel="Request Rate (req/s)",
            xfmt=lambda x: f"{x:g}",
        )
        print_comparison(sg_series[metric], f"ShareGPT {ylabel}")


if __name__ == "__main__":
    main()
