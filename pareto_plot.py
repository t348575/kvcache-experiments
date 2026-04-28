#!/usr/bin/env python3
"""
pareto_plot.py
──────────────
Generates Pareto frontier plots from a pareto_measure.py results CSV.

Three figures are produced:

  1. ttft_curves.png     — Measured f(N) cold-prefill and g(P) cache-hit TTFT
                           curves with p05–p95 shading.
  2. pareto_frontier.png — Speedup heatmap in (doc_size × prefix_fraction)
                           space with the break-even contour (speedup = 1).
  3. concurrency.png     — TTFT vs concurrency for cold and cache-hit,
                           one line per doc size.  Skipped if no concurrency
                           sweep data is present.

Usage
─────
  python pareto_plot.py --results pareto_measure_<ts>.csv [--output-dir plots/]
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import os
import re
import statistics
import sys
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import PchipInterpolator

from plot_common import configure_plots, save_figure, thousands_formatter, token_k_formatter

configure_plots()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────


def load_results(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    # Cast numeric columns
    for r in rows:
        for col in (
            "doc_size",
            "prefix_len",
            "concurrency",
            "n_queries",
            "n_successful",
        ):
            if r.get(col):
                r[col] = int(r[col])
        for col in (
            "ttft_mean_s",
            "ttft_median_s",
            "ttft_p05_s",
            "ttft_p95_s",
            "ttft_p99_s",
            "prefix_frac",
        ):
            if r.get(col):
                r[col] = float(r[col])
    return [r for r in rows if not r.get("error")]


def available_cache_server_configs(rows: list[dict]) -> list[str]:
    return sorted({
        r["server_config"]
        for r in rows
        if r.get("curve") == "cache_hit" and r.get("server_config")
    })


def select_cache_server_config(rows: list[dict], requested: str | None) -> str | None:
    configs = available_cache_server_configs(rows)
    if requested:
        if requested not in configs:
            available = ", ".join(configs) if configs else "none"
            raise ValueError(
                f"cache server config {requested!r} not found in results; available: {available}"
            )
        return requested

    if len(configs) <= 1:
        return configs[0] if configs else None

    print("\nAvailable cache_hit server configs:")
    for i, config in enumerate(configs, start=1):
        n_rows = sum(1 for r in rows if r.get("curve") == "cache_hit" and r.get("server_config") == config)
        print(f"  {i}) {config} ({n_rows} rows)")

    while True:
        raw = input(f"Select cache_hit server config [1-{len(configs)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(configs):
            return configs[int(raw) - 1]
        print(f"  Enter a number between 1 and {len(configs)}.")


def filter_cache_server_config(rows: list[dict], server_config: str | None) -> list[dict]:
    if server_config is None:
        return rows
    return [
        r for r in rows
        if r.get("curve") != "cache_hit" or r.get("server_config") == server_config
    ]


def cache_hit_job_numbers(csv_path: str, server_config: str | None) -> set[int] | None:
    if server_config is None:
        return None

    job_nums: set[int] = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for job_idx, row in enumerate(reader, start=1):
            if row.get("error"):
                continue
            if row.get("curve") != "cache_hit":
                continue
            if row.get("server_config") == server_config:
                job_nums.add(job_idx)
    return job_nums


def serial_curve(rows: list[dict], curve: str) -> dict[int, dict]:
    """Return {doc_size: row} for serial (concurrency=1) rows of the given curve."""
    return {
        r["doc_size"]: r for r in rows if r["curve"] == curve and r["concurrency"] == 1
    }


def concurrency_data(rows: list[dict]) -> dict[str, dict[int, dict[int, float]]]:
    """
    Return {curve: {doc_size: {concurrency: ttft_mean_s}}}
    for all concurrency-sweep rows.
    """
    result: dict[str, dict[int, dict[int, float]]] = {
        "cold_prefill": {},
        "cache_hit": {},
    }
    for r in rows:
        c = r["concurrency"]
        d = r["doc_size"]
        curve = r["curve"]
        if curve not in result:
            continue
        result[curve].setdefault(d, {})[c] = r["ttft_mean_s"]
    return result


def _find_result_dir(csv_path: str) -> str:
    """Return the result directory adjacent to a pareto_measure summary CSV."""
    csv_abs = os.path.abspath(csv_path)
    stem = os.path.splitext(os.path.basename(csv_abs))[0]
    return os.path.join(os.path.dirname(csv_abs), stem)


def load_per_run_curves(
    csv_path: str,
    cache_server_config: str | None = None,
) -> tuple[dict[int, dict] | None, dict[int, dict] | None]:
    """
    Load f(D) and g(P) from per-run CSVs in the result directory.

    Per-run CSVs contain per-request data with is_prefix_reuse flags:
      cold_prefill CSVs: all is_prefix_reuse=False (baseline, no KV cache)
      cache_hit CSVs:    is_prefix_reuse=False = STORE op (first request)
                         is_prefix_reuse=True  = LOAD op  (cache reuse)

    Returns (cold, hit) dicts in the same {doc_size: row_dict} format
    as serial_curve(), or (None, None) if no per-run CSVs found.
    """
    result_dir = _find_result_dir(csv_path)
    cold_csvs = sorted(
        globmod.glob(os.path.join(result_dir, "job_*_cold_prefill_*_c1.csv"))
    )
    cache_csvs = sorted(
        globmod.glob(os.path.join(result_dir, "job_*_cache_hit_*_c1.csv"))
    )
    selected_cache_jobs = cache_hit_job_numbers(csv_path, cache_server_config)
    if selected_cache_jobs is not None:
        cache_csvs = [
            fp for fp in cache_csvs
            if (m := re.match(r"job_(\d+)_", os.path.basename(fp)))
            and int(m.group(1)) in selected_cache_jobs
        ]

    if not cold_csvs and not cache_csvs:
        return None, None

    # Cold prefill: median of non-warmup requests (skip first)
    cold: dict[int, dict] = {}
    for fp in cold_csvs:
        with open(fp) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        doc_size = int(rows[0]["doc_tokens"])
        ttfts = [float(r["ttft"]) for r in rows if r.get("successful") == "True"]
        clean = ttfts[1:] if len(ttfts) > 1 else ttfts
        if not clean:
            continue
        med = statistics.median(clean)
        # Build a row_dict compatible with serial_curve output
        if doc_size not in cold:
            cold[doc_size] = {
                "ttft_mean_s": med,
                "ttft_p05_s": min(clean),
                "ttft_p95_s": max(clean),
            }

    # Cache hit: median of LOAD ops (is_prefix_reuse=True), filter anomalies
    hit: dict[int, dict] = {}
    for fp in cache_csvs:
        with open(fp) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        prefix_len = int(rows[0]["doc_tokens"])
        load_ttfts = [
            float(r["ttft"])
            for r in rows
            if r.get("is_prefix_reuse") == "True" and r.get("successful") == "True"
        ]
        if not load_ttfts:
            continue
        med = statistics.median(load_ttfts)
        clean = [t for t in load_ttfts if t < 2 * med]
        if not clean:
            continue
        clean_med = statistics.median(clean)
        if prefix_len not in hit:
            hit[prefix_len] = {
                "ttft_mean_s": clean_med,
                "ttft_p05_s": min(clean),
                "ttft_p95_s": max(clean),
            }

    return (cold or None), (hit or None)


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATION & SPEEDUP
# ─────────────────────────────────────────────────────────────────────────────


def build_interpolator(
    curve_dict: dict[int, dict],
    col: str = "ttft_mean_s",
    floor: float = 0.0,
) -> PchipInterpolator:
    """Monotone cubic interpolator for {tokens: ttft_s}, anchored at (0, floor)."""
    sizes = sorted(curve_dict.keys())
    vals = [curve_dict[s][col] for s in sizes]
    if sizes[0] != 0:
        sizes = [0] + sizes
        vals = [floor] + vals
    return PchipInterpolator(sizes, vals, extrapolate=True)


def speedup_grid(
    f_interp: PchipInterpolator,
    g_interp: PchipInterpolator,
    doc_sizes: np.ndarray,
    prefix_fracs: np.ndarray,
) -> np.ndarray:
    """
    speedup(D, frac) = f(D) / (g(frac*D) + f(D) - f(frac*D))
    Shape: (len(prefix_fracs), len(doc_sizes))
    """
    D, F = np.meshgrid(doc_sizes, prefix_fracs)
    cold = np.maximum(0, f_interp(D))
    cached = np.maximum(0, g_interp(F * D)) + np.maximum(0, cold - f_interp(F * D))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cached > 0, cold / cached, np.inf)


def break_even_curve_values(
    f_interp: PchipInterpolator,
    g_interp: PchipInterpolator,
    doc_sizes: list[int],
    samples: int = 1001,
) -> list[dict[str, float | int | None]]:
    """Return the smallest prefix length where cache-hit TTFT breaks even."""
    values: list[dict[str, float | int | None]] = []
    for doc_size in doc_sizes:
        prefix_tokens = np.linspace(0.0, float(doc_size), samples)
        delta = f_interp(prefix_tokens) - g_interp(prefix_tokens)

        if delta[0] >= 0:
            break_even_tokens = 0.0
        else:
            crossings = np.where(delta >= 0)[0]
            if len(crossings) == 0:
                values.append({
                    "doc_size": doc_size,
                    "prefix_tokens": None,
                    "prefix_frac": None,
                })
                continue

            hi = int(crossings[0])
            lo = max(0, hi - 1)
            x0, x1 = prefix_tokens[lo], prefix_tokens[hi]
            y0, y1 = delta[lo], delta[hi]
            if y1 == y0:
                break_even_tokens = x1
            else:
                break_even_tokens = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)

        values.append({
            "doc_size": doc_size,
            "prefix_tokens": break_even_tokens,
            "prefix_frac": break_even_tokens / doc_size if doc_size else None,
        })
    return values


def print_break_even_curve(values: list[dict[str, float | int | None]]) -> None:
    print("\nBreak-even curve values:")
    print("  doc_tokens,prefix_tokens,prefix_fraction_pct")
    for row in values:
        doc_size = int(row["doc_size"])
        prefix_tokens = row["prefix_tokens"]
        prefix_frac = row["prefix_frac"]
        if prefix_tokens is None or prefix_frac is None:
            print(f"  {doc_size},none,none")
        else:
            print(f"  {doc_size},{prefix_tokens:.1f},{prefix_frac * 100:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: TTFT CURVES
# ─────────────────────────────────────────────────────────────────────────────


def plot_ttft_curves(
    cold: dict[int, dict],
    hit: dict[int, dict],
    model: str,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Measured TTFT curves {model}", fontweight="bold")

    def _plot_curve(ax, data: dict[int, dict], title: str, color: str):
        sizes = sorted(data.keys())
        means = [data[s]["ttft_mean_s"] * 1000 for s in sizes]
        p05 = [data[s]["ttft_p05_s"] * 1000 for s in sizes]
        p95 = [data[s]["ttft_p95_s"] * 1000 for s in sizes]
        ax.fill_between(sizes, p05, p95, alpha=0.2, color=color, label="p05–p95")
        ax.plot(sizes, means, "o-", color=color, label="mean", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("TTFT (ms)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(token_k_formatter))
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_curve(axes[0], cold, "cold prefill", "steelblue")
    _plot_curve(axes[1], hit, "cache hit", "tomato")

    save_figure(out_path, dpi=150)
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: PARETO FRONTIER
# ─────────────────────────────────────────────────────────────────────────────


def plot_pareto_frontier(
    cold: dict[int, dict],
    hit: dict[int, dict],
    model: str,
    out_path: str,
) -> None:
    # X-axis: only the actually measured doc sizes
    doc_sizes_measured = sorted(cold.keys())
    doc_arr = np.array(doc_sizes_measured, dtype=float)
    prefix_fracs = np.linspace(0.0, 1.0, 60)

    # Interpolators needed to evaluate f((1-frac)*D) and g(frac*D) at
    # sub-measured token counts, but the displayed x-axis stays real data only.
    g_floor = min(hit[d]["ttft_mean_s"] for d in hit)
    f_interp = build_interpolator(cold)
    g_interp = build_interpolator(hit, floor=g_floor)
    print_break_even_curve(
        break_even_curve_values(f_interp, g_interp, doc_sizes_measured)
    )

    cold_ttft_arr = np.array([cold[d]["ttft_mean_s"] for d in doc_sizes_measured])

    # speedup_mat[i, j] = speedup at prefix_fracs[i], doc_arr[j]
    speedup_mat = np.zeros((len(prefix_fracs), len(doc_arr)))
    for i, frac in enumerate(prefix_fracs):
        cached = np.maximum(0, g_interp(frac * doc_arr)) + np.maximum(
            0, cold_ttft_arr - f_interp(frac * doc_arr)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            speedup_mat[i] = np.where(cached > 0, cold_ttft_arr / cached, np.inf)

    log_speedup = np.log2(np.clip(speedup_mat, 0.25, 4.0))

    # Use integer indices for evenly-spaced x positions; label with real sizes
    x_idx = np.arange(len(doc_sizes_measured))
    x_labels = [f"{int(d) // 1024}k" for d in doc_sizes_measured]
    # pcolormesh needs bin edges: place edges at -0.5, 0.5, 1.5, …
    x_edges = np.arange(len(doc_sizes_measured) + 1) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"KV-Cache pareto frontier ({model})", fontweight="bold")

    # ── Left: heatmap at measured doc sizes only ──────────────────────────────
    ax = axes[0]
    im = ax.pcolormesh(
        x_idx,
        prefix_fracs * 100,
        log_speedup,
        cmap="RdYlGn",
        vmin=-2,
        vmax=2,
        shading="nearest",
    )
    plt.colorbar(im, ax=ax, label="log₂(speedup)  [green = faster with cache]")

    cs = ax.contour(
        x_idx,
        prefix_fracs * 100,
        speedup_mat,
        levels=[1.0],
        colors=["black"],
        linewidths=[2],
    )
    ax.clabel(cs, fmt="break-even", fontsize=8)

    ax.set_xlabel("Document size")
    ax.set_ylabel("Prefix fraction (%)")
    ax.set_title("Speedup heatmap")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # ── Right: speedup at measured points only, for fixed prefix fracs ────────
    ax = axes[1]
    for frac, color in [
        (0.25, "royalblue"),
        (0.5, "green"),
        (0.75, "darkorange"),
        (0.9, "red"),
    ]:
        cached = np.maximum(0, g_interp(frac * doc_arr)) + np.maximum(
            0, cold_ttft_arr - f_interp(frac * doc_arr)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            speedup = np.where(cached > 0, cold_ttft_arr / cached, np.inf)
        ax.plot(x_idx, speedup, "o-", label=f"{frac * 100:.0f}% prefix", color=color)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="break-even")
    ax.set_xlabel("Document size")
    ax.set_ylabel("Speedup (cold / cached TTFT)")
    ax.set_title("Speedup vs doc size at fixed prefix fractions")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    save_figure(out_path, dpi=150)
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: CONCURRENCY SWEEP
# ─────────────────────────────────────────────────────────────────────────────


def plot_concurrency(
    conc_data: dict[str, dict[int, dict[int, float]]],
    out_path: str,
) -> None:
    # Only plot doc sizes that appear in both curves
    cold_sizes = set(conc_data["cold_prefill"].keys())
    hit_sizes = set(conc_data["cache_hit"].keys())
    doc_sizes = sorted(cold_sizes | hit_sizes)

    if not doc_sizes:
        print("  No concurrency sweep data — skipping concurrency plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Concurrency sweep", fontweight="bold")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(doc_sizes)))

    # ── Left: raw TTFT vs concurrency ─────────────────────────────────────────
    ax = axes[0]
    for d, color in zip(doc_sizes, colors):
        label = f"{d // 1024}k tokens"
        cold = conc_data["cold_prefill"].get(d, {})
        hit = conc_data["cache_hit"].get(d, {})
        if cold:
            cs = sorted(cold.keys())
            ax.plot(
                cs,
                [cold[c] * 1000 for c in cs],
                "o-",
                color=color,
                label=f"{label} cold",
            )
        if hit:
            cs = sorted(hit.keys())
            ax.plot(
                cs,
                [hit[c] * 1000 for c in cs],
                "s--",
                color=color,
                label=f"{label} cached",
                alpha=0.7,
            )

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_title("TTFT vs concurrency")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Right: speedup (cold / cached) vs concurrency ─────────────────────────
    ax = axes[1]
    for d, color in zip(doc_sizes, colors):
        cold = conc_data["cold_prefill"].get(d, {})
        hit = conc_data["cache_hit"].get(d, {})
        shared_c = sorted(set(cold) & set(hit))
        if not shared_c:
            continue
        speedups = [cold[c] / hit[c] for c in shared_c]
        ax.plot(shared_c, speedups, "o-", color=color, label=f"{d // 1024}k tokens")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="break-even")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Speedup (cold TTFT / cached TTFT)")
    ax.set_title("Cache speedup vs concurrency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    save_figure(out_path, dpi=150)
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: IO BUDGET
# ─────────────────────────────────────────────────────────────────────────────


def load_cuda_transfer_curve(
    csv_path: str,
    cache_server_config: str | None = None,
) -> dict[int, float]:
    """
    Read cache_hit profile JSONs to extract median cuda_transfer(cpu_to_gpu)
    duration per doc_size.  Returns {doc_tokens: cuda_transfer_ms}.
    """
    result_dir = _find_result_dir(csv_path)
    selected_cache_jobs = cache_hit_job_numbers(csv_path, cache_server_config)
    mapping: dict[int, int] = {}
    for fp in globmod.glob(os.path.join(result_dir, "job_*_cache_hit_*_c1.csv")):
        m = re.match(
            r"job_(\d+)_(?:cold_prefill|cache_hit)_(\d+)_c\d+\.csv",
            os.path.basename(fp),
        )
        if not m:
            continue
        job_num = int(m.group(1))
        if selected_cache_jobs is not None and job_num not in selected_cache_jobs:
            continue
        mapping[job_num] = int(m.group(2))

    to_gpu_event_markers = (
        "cuda_transfer(cpu_to_gpu",
        "cuda_staging(cpu_to_gpu",
        "VLLMPagedMemGPUConnectorV2.to_gpu.kernel",
    )

    result: dict[int, float] = {}
    for job_num, doc_tokens in sorted(mapping.items()):
        profile_path = os.path.join(result_dir, f"job_{job_num:04d}_profile.json")
        if not os.path.exists(profile_path):
            continue
        with open(profile_path) as f:
            data = json.load(f)
        events = data.get("traceEvents", data) if isinstance(data, dict) else data
        events = [e for e in events if isinstance(e, dict) and e.get("ph") == "X"]
        requests = sorted(
            [e for e in events if e["name"].startswith("request(")],
            key=lambda e: e["ts"],
        )
        cuda_ms_list: list[float] = []
        for req in requests:
            rs, re_ = req["ts"], req["ts"] + req["dur"]
            inner = [
                e
                for e in events
                if e["ts"] >= rs
                and e["ts"] < re_
                and not e["name"].startswith("request(")
            ]
            if not [e for e in inner if "load_e2e" in e["name"]]:
                continue
            cuda_evts = [
                e
                for e in inner
                if any(marker in e["name"] for marker in to_gpu_event_markers)
            ]
            if not cuda_evts:
                continue
            nb = cuda_evts[0]["args"].get("num_bytes", 0)
            if nb < 10_000_000:
                continue
            cuda_ms_list.append(cuda_evts[0]["dur"] / 1000)
        if cuda_ms_list:
            result[doc_tokens] = statistics.median(cuda_ms_list)
    return result


def plot_io_budget(
    cold: dict[int, dict],
    hit: dict[int, dict],
    cuda_transfer_ms: dict[int, float],
    kv_bpt: int,
    model: str,
    out_path: str,
) -> None:
    """
    Plot the maximum allowable IO copy time and minimum required bandwidth for
    KV caching to remain beneficial.

    With the speedup formula f(D) / (g(P) + f(D) - f(P)), break-even is:
        g(P) = f(P)

    So the maximum allowable IO copy time at prefix P is:
        max_io(P) = f(P) - (g(P) - cuda_transfer(P))
                  = f(P) - non_io_overhead(P)

    And the minimum required IO bandwidth:
        min_bw(P) = P × kv_bpt / max_io(P)
    """
    common = sorted(set(cold) & set(hit) & set(cuda_transfer_ms))
    if not common:
        print("  No common doc sizes for IO budget plot — skipping.")
        return

    tokens = np.array(common, dtype=float)
    f_ms = np.array([cold[p]["ttft_mean_s"] * 1000 for p in common])
    g_ms = np.array([hit[p]["ttft_mean_s"] * 1000 for p in common])
    ct_ms = np.array([cuda_transfer_ms[p] for p in common])

    non_io_ms = g_ms - ct_ms  # everything in g except the IO copy
    max_io_ms = f_ms - non_io_ms  # budget left for IO before caching loses
    kv_bytes = tokens * kv_bpt
    min_bw = kv_bytes / (np.maximum(max_io_ms, 0.1) / 1000) / 1e9  # GB/s

    x_labels = [f"{int(p) // 1024}k" for p in common]
    x_idx = np.arange(len(common))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Max IO copy time for KV caching to be useful ({model})",
        fontweight="bold",
    )

    # ── Left: max allowable IO time vs actual IO time ─────────────────────────
    ax1.fill_between(x_idx, max_io_ms, alpha=0.2, color="steelblue")
    ax1.plot(
        x_idx, max_io_ms, "o-", color="steelblue", linewidth=2, label="Max IO time"
    )
    ax1.plot(x_idx, ct_ms, "s--", color="tomato", linewidth=2, label="RAM IO time")
    ax1.set_xlabel("Document size")
    ax1.set_ylabel("IO copy time (ms)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 3, 5]))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Right: min required bandwidth vs actual + device references ───────────
    ax2.plot(
        x_idx, min_bw, "o-", color="tomato", linewidth=2, label="Min required bandwidth"
    )

    ref_bws = [
        (13.0, "Kioxia CM7-R (13 GB/s)"),
        (6.5, "Samsung PM9A3 (6.5 GB/s)"),
    ]
    for bw_gbps, label in ref_bws:
        ax2.axhline(bw_gbps, color="grey", linestyle=":", linewidth=1.5, alpha=0.8)
        ax2.text(
            0, bw_gbps, f"  {label}", va="bottom", ha="left", fontsize=7, color="grey"
        )

    data_top = max(max(min_bw), max(bw for bw, _ in ref_bws)) * 1.15
    ax2.set_ylim(0, data_top)
    ax2.set_xlabel("Document size")
    ax2.set_ylabel("Bandwidth (GB/s)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    save_figure(out_path, dpi=150)
    print(f"  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="Plot Pareto frontier from pareto_measure results."
    )
    p.add_argument("--results", required=True, help="pareto_measure CSV file.")
    p.add_argument("--output-dir", default=".", help="Directory to write plots into.")
    p.add_argument("--model", required=True, help="Model name")
    p.add_argument(
        "--cache-server-config",
        default=None,
        help="cache_hit server_config to plot. If omitted and multiple are available, prompt interactively.",
    )
    # Optional: model architecture for IO budget plot
    p.add_argument("--layers", type=int, default=None, help="Number of KV layers.")
    p.add_argument("--kv-heads", type=int, default=None, help="Number of KV heads.")
    p.add_argument("--head-dim", type=int, default=None, help="KV head dimension.")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp8"], default="fp16")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.results}…")
    rows = load_results(args.results)
    print(f"  {len(rows)} summary rows loaded.")

    try:
        cache_server_config = select_cache_server_config(rows, args.cache_server_config)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    if cache_server_config:
        rows = filter_cache_server_config(rows, cache_server_config)
        print(f"  Using cache_hit server_config: {cache_server_config}")

    # Prefer per-run CSVs (proper is_prefix_reuse filtering)
    cold_pr, hit_pr = load_per_run_curves(args.results, cache_server_config)
    if cold_pr and hit_pr:
        cold = cold_pr
        hit = hit_pr
        print("  Using per-run CSVs (LOAD ops only, warmup filtered).")
    else:
        cold = serial_curve(rows, "cold_prefill")
        hit = serial_curve(rows, "cache_hit")
        print("  Using summary CSV means (no per-run CSVs found).")

    if not cold:
        print("ERROR: no cold_prefill serial data found.")
        return
    if not hit:
        print("ERROR: no cache_hit serial data found.")
        return

    print(f"  cold_prefill points: {sorted(cold.keys())}")
    print(f"  cache_hit points:    {sorted(hit.keys())}")

    plot_ttft_curves(
        cold,
        hit,
        args.model,
        os.path.join(args.output_dir, "ttft_curves.png"),
    )
    plot_pareto_frontier(
        cold,
        hit,
        args.model,
        os.path.join(args.output_dir, "pareto_frontier.png"),
    )

    conc = concurrency_data(rows)
    has_conc = any(conc[c] for c in conc)
    if has_conc:
        plot_concurrency(conc, os.path.join(args.output_dir, "concurrency.png"))
    else:
        print("  No concurrency sweep data found — skipping concurrency plot.")

    if args.layers and args.kv_heads and args.head_dim:
        dtype_bytes = {"fp16": 2, "bf16": 2, "fp8": 1}[args.dtype]
        kv_bpt = 2 * args.layers * args.kv_heads * args.head_dim * dtype_bytes
        print(f"\n  KV bytes/token: {kv_bpt:,}  — generating IO budget plot…")
        cuda_transfer = load_cuda_transfer_curve(args.results, cache_server_config)
        if cuda_transfer:
            plot_io_budget(
                cold,
                hit,
                cuda_transfer,
                kv_bpt,
                args.model,
                os.path.join(args.output_dir, "io_budget.png"),
            )
        else:
            print("  No profile JSONs found — skipping IO budget plot.")


if __name__ == "__main__":
    main()
