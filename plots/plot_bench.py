import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common.plot_common import plain_number_formatter, save_figure, set_log_y_axis


def _maybe_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def _maybe_float(value):
    if value in (None, ""):
        return None
    return float(value)


STRATEGY_LABELS = {
    "baseline": "Baseline",
    "offloading": "vLLM Offloading",
    "py_kvcache_no_preload": "py-kvcache (no preload)",
    "py_kvcache_preload": "py-kvcache (preload)",
    "py_kvcache_no_preload_pc": "py-kvcache (no preload, GPU PC)",
    "py_kvcache_preload_pc": "py-kvcache (preload, GPU PC)",
}


def strategy_label_from_row(row):
    base_name = row["base_config_name"]
    chunk_size = _maybe_int(row.get("chunk_size"))

    if base_name in STRATEGY_LABELS:
        return STRATEGY_LABELS[base_name]
    if chunk_size is not None:
        return f"{base_name} chunk={chunk_size}"
    return base_name


def _resolve_path(path, base_dir):
    """Summary-CSV paths (per_request_csv, gpu_transfer_csv) are written relative
    to the run directory; resolve them against the summary CSV's location."""
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(base_dir, path)


def _parse_config_params(config_name):
    """Parse the sweep params bench.py encodes into config_name, e.g.
    'py_kvcache_preload[doc_size=1024,output_len=1,prefix_reuse_pct=0.5]'."""
    params = {}
    m = re.search(r"\[(.*)\]", config_name or "")
    if m:
        for part in m.group(1).split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = v.strip()
    return params


def _scan_sweep_dimensions(csv_paths):
    """Distinct output_len / prefix_reuse_pct / max_concurrency present across
    prefix_cache rows."""
    out_lens, reuses, concurrencies = set(), set(), set()
    csv.field_size_limit(10 * 1024 * 1024)
    for path in csv_paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("benchmark") != "prefix_cache":
                    continue
                params = _parse_config_params(row.get("config_name", ""))
                if "output_len" in params:
                    out_lens.add(params["output_len"])
                if "prefix_reuse_pct" in params:
                    reuses.add(params["prefix_reuse_pct"])
                if "max_concurrency" in params:
                    concurrencies.add(params["max_concurrency"])
    return out_lens, reuses, concurrencies


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
            set_log_y_axis(ax)
        else:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(plain_number_formatter))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    save_figure(filename)

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
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(plain_number_formatter))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(plain_number_formatter))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_figure(filename)

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
                     to_gpu_series, from_gpu_series, exclude, prefix="", renames=None,
                     outlen_filter=None, reuse_filter=None, concurrency_filter=None,
                     append_reuse=False, append_concurrency=False,
                     query_time_series=None):
    # outlen_filter / reuse_filter / concurrency_filter: each a set of allowed
    # string values (as encoded in config_name) or None to allow all.
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("benchmark") != "prefix_cache":
                continue

            params = _parse_config_params(row.get("config_name", ""))
            if outlen_filter is not None and params.get("output_len") not in outlen_filter:
                continue
            if reuse_filter is not None and params.get("prefix_reuse_pct") not in reuse_filter:
                continue
            if concurrency_filter is not None and params.get("max_concurrency") not in concurrency_filter:
                continue

            strategy = strategy_label_from_row(row)
            doc_len = _maybe_int(row.get("doc_len"))
            batch_size = _maybe_int(row.get("batch_size"))
            if renames:
                strategy = renames.get(strategy, strategy)
            # Keep each line a single prefix_reuse_pct; only label it when the
            # run mixes multiple, otherwise the suffix is noise.
            if append_reuse and "prefix_reuse_pct" in params:
                strategy = f"{strategy} reuse={params['prefix_reuse_pct']}"
            if append_concurrency and "max_concurrency" in params:
                strategy = f"{strategy} conc={params['max_concurrency']}"
            if prefix:
                strategy = f"{prefix}: {strategy}"
            if doc_len is None or doc_len in exclude:
                continue

            # Query-round wall time: one value per row (pooled across repetitions).
            if query_time_series is not None:
                qt = _maybe_float(row.get("query_total_time_s"))
                if qt is not None:
                    query_time_series[strategy][doc_len].append(qt)

            # GPU bar plots: series["{strategy} doc={doc_len}"][batch_size]
            gpu_label = f"{strategy} doc={doc_len}"

            # GPU transfer throughput: bytes / dur_s → GB/s, one value per kernel call
            # All individual transfers across all repetitions are pooled; mean is taken at plot time.
            transfer_csv_path = _resolve_path(row.get("gpu_transfer_csv", ""), base_dir)
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
                is_lmcache = row["base_config_name"].startswith("lmcache")
                if is_lmcache:
                    to_bw, from_bw = from_bw, to_bw
                to_gpu_series[gpu_label][batch_size].extend(to_bw)
                from_gpu_series[gpu_label][batch_size].extend(from_bw)

            # TTFT line plots: series[strategy][doc_len], pool all batch sizes
            per_req_path = _resolve_path(row.get("per_request_csv", ""), base_dir)
            if not per_req_path or not os.path.exists(per_req_path):
                continue
            with open(per_req_path, newline="") as pf:
                for req in csv.DictReader(pf):
                    if req["successful"] != "True":
                        continue
                    ttft = float(req["ttft"])
                    if req["is_prefix_reuse"] == "True":
                        query_series[strategy][doc_len].append(ttft)
                    else:
                        warmup_series[strategy][doc_len].append(ttft)


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
            if row.get("benchmark") != "sharegpt" or not row.get("sg_result_json"):
                continue
            strategy = strategy_label_from_row(row)
            if renames:
                strategy = renames.get(strategy, strategy)
            if prefix:
                strategy = f"{prefix}: {strategy}"

            request_rate = _maybe_float(row.get("request_rate"))
            if request_rate is None or request_rate in exclude_rates:
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
    # Sweep-dimension controls. Filters keep only the listed values; --merge-*
    # pools that dimension into one series instead of splitting it out.
    parser.add_argument("--output-len", type=str, nargs="+", default=None,
                        help="Plot only these output_len values (e.g. --output-len 1)")
    parser.add_argument("--reuse", type=str, nargs="+", default=None,
                        help="Plot only these prefix_reuse_pct values (e.g. --reuse 0.75)")
    parser.add_argument("--concurrency", type=str, nargs="+", default=None,
                        help="Plot only these max_concurrency values (e.g. --concurrency 4)")
    parser.add_argument("--merge-output-len", action="store_true",
                        help="Pool all output_len values into one plot set instead of one per value")
    parser.add_argument("--merge-reuse", action="store_true",
                        help="Pool all prefix_reuse_pct values into one series")
    parser.add_argument("--merge-concurrency", action="store_true",
                        help="Pool all max_concurrency values into one series")
    args = parser.parse_args()
    exclude = set(args.exclude_doc_len)

    renames = {}
    for entry in args.rename:
        if ":" not in entry:
            parser.error(f"--rename must be in OLD:NEW format, got: {entry!r}")
        old, new = entry.split(":", 1)
        renames[old] = new

    extra = []
    for entry in args.extra:
        if ":" not in entry:
            parser.error(f"--extra must be in NAME:PATH format, got: {entry!r}")
        name, path = entry.split(":", 1)
        extra.append((name, path))

    out_lens, reuses, concurrencies = _scan_sweep_dimensions([args.csv] + [p for _, p in extra])

    # Apply value filters (intersect available values with requested ones).
    reuse_filter       = set(args.reuse) if args.reuse else None
    concurrency_filter = set(args.concurrency) if args.concurrency else None
    if args.output_len:
        out_lens = out_lens & set(args.output_len)
    if reuse_filter:
        reuses = reuses & reuse_filter
    if concurrency_filter:
        concurrencies = concurrencies & concurrency_filter

    # Split a dimension into its own series only when >1 value survives and the
    # user did not ask to merge it.
    append_reuse       = len(reuses) > 1 and not args.merge_reuse
    append_concurrency = len(concurrencies) > 1 and not args.merge_concurrency

    # output_len yields one plot set per value; --merge-output-len pools them.
    if args.merge_output_len:
        outlen_values = [None]
        outlen_allowed = out_lens or None
    else:
        outlen_values = sorted(out_lens, key=lambda x: int(x)) or [None]
        outlen_allowed = None

    for outlen in outlen_values:
        suffix    = f"_outlen{outlen}" if outlen is not None else ""
        title_sfx = f" (output_len={outlen})" if outlen is not None else ""
        # Per-pass set of allowed output_len values handed to the loader.
        outlen_set = {outlen} if outlen is not None else outlen_allowed

        query_series      = defaultdict(lambda: defaultdict(list))
        warmup_series     = defaultdict(lambda: defaultdict(list))
        to_gpu_series     = defaultdict(lambda: defaultdict(list))
        from_gpu_series   = defaultdict(lambda: defaultdict(list))
        query_time_series = defaultdict(lambda: defaultdict(list))

        load_into_series(args.csv, warmup_series, query_series,
                         to_gpu_series, from_gpu_series, exclude, prefix=args.name,
                         renames=renames, outlen_filter=outlen_set,
                         reuse_filter=reuse_filter, concurrency_filter=concurrency_filter,
                         append_reuse=append_reuse, append_concurrency=append_concurrency,
                         query_time_series=query_time_series)
        for name, path in extra:
            load_into_series(path, warmup_series, query_series,
                             to_gpu_series, from_gpu_series, exclude, prefix=name,
                             renames=renames, outlen_filter=outlen_set,
                             reuse_filter=reuse_filter, concurrency_filter=concurrency_filter,
                             append_reuse=append_reuse, append_concurrency=append_concurrency,
                             query_time_series=query_time_series)

        plot_ttft(query_series, "query_mean_ttft_s",
                  f"(log scale) Query-Round TTFT vs Document Length{title_sfx}", "Query TTFT (s)",
                  f"query_ttft_vs_doclen{suffix}.png", normalize=args.normalize, boxplot=not args.no_boxplot, log=not args.no_log)

        plot_ttft(warmup_series, "warmup_mean_ttft_s",
                  f"Warmup-Round TTFT vs Document Length{title_sfx}", "Warmup TTFT (s)",
                  f"warmup_ttft_vs_doclen{suffix}.png", normalize=args.normalize, boxplot=not args.no_boxplot, log=not args.no_log)

        combined_series = defaultdict(lambda: defaultdict(list))
        for label, doc_lens in warmup_series.items():
            for doc_len, vals in doc_lens.items():
                combined_series[label][doc_len].extend(vals)
        for label, doc_lens in query_series.items():
            for doc_len, vals in doc_lens.items():
                combined_series[label][doc_len].extend(vals)

        plot_ttft(combined_series, None,
                  f"Combined TTFT vs Document Length{title_sfx}", "TTFT (s)",
                  f"combined_ttft_vs_doclen{suffix}.png", normalize=args.normalize, boxplot=not args.no_boxplot, log=not args.no_log)

        if query_time_series:
            plot_ttft(query_time_series, None,
                      f"Query-Round Wall Time vs Document Length{title_sfx}", "Query Wall Time (s)",
                      f"query_time_vs_doclen{suffix}.png", normalize=args.normalize, boxplot=not args.no_boxplot, log=not args.no_log)
            print_comparison(query_time_series, f"Query-Round Wall Time{title_sfx}")

        print_comparison(query_series,  f"Query-Round TTFT{title_sfx}")
        print_comparison(warmup_series, f"Warmup-Round TTFT{title_sfx}")

        if to_gpu_series:
            plot_grouped_bar(to_gpu_series,
                             f"CPU→GPU Transfer Throughput vs Document Length{title_sfx}",
                             "Throughput (GB/s)", f"to_gpu_vs_doclen{suffix}.png", log=True)

        if from_gpu_series:
            plot_grouped_bar(from_gpu_series,
                             f"GPU→CPU Transfer Throughput vs Document Length{title_sfx}",
                             "Throughput (GB/s)", f"from_gpu_vs_doclen{suffix}.png", log=True)

    sg_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    load_sharegpt_series(args.csv, sg_series, prefix=args.name, renames=renames)
    for name, path in extra:
        load_sharegpt_series(path, sg_series, prefix=name, renames=renames)

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
