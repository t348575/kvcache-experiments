#!/usr/bin/env python3
"""
pareto_estimate.py
──────────────────
Estimates the KV-cache Pareto frontier for a target model from:

  1. Reference model profile JSONs (baseline + cache_hit) — provides PCIe
     bandwidth, e2e overhead, decode forward time, and request overhead.
  2. Target model baseline profile JSONs — provides compute characteristics.
  3. Architecture parameters for both models — computes KV bytes per token.

Method
──────
  From reference cache_hit profiles (LOAD ops only, skipping the store request):
    - bandwidth   = num_bytes / cuda_transfer_duration   (~53 GB/s PCIe)
    - e2e_overhead = load_e2e - cuda_transfer             (disk/scheduling)
    - decode_fwd   = forward duration during LOAD ops
    - req_overhead = request_duration - load_e2e - forward (scheduling)

  For target estimation:
    target_kv_bytes    = P × kv_bytes_per_token(target)
    target_cuda_ms     = target_kv_bytes / bandwidth
    target_load_e2e_ms = target_cuda_ms + e2e_overhead(P)
    target_fwd_ms      = ref_decode_fwd(P) × compute_ratio
    g_target(P)        = target_load_e2e + target_fwd + req_overhead

  speedup(D, frac) = f(D) / (g(frac × D) + f((1 − frac) × D))

Usage
─────
  python pareto_estimate.py \\
      --ref-results   pareto_measure_ref.csv \\
      --target-results pareto_measure_target.csv \\
      --ref-layers 28   --ref-kv-heads 8  --ref-head-dim 128 \\
      --target-layers 40 --target-kv-heads 8 --target-head-dim 128
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import os
import re
import statistics
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


# ─────────────────────────────────────────────────────────────────────────────
# KV SIZE
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1}


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, dtype: str) -> int:
    return 2 * num_layers * num_kv_heads * head_dim * DTYPE_BYTES[dtype]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DIR DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def _find_result_dir(csv_path: str) -> str:
    """Return the result directory adjacent to a pareto_measure summary CSV."""
    csv_abs = os.path.abspath(csv_path)
    stem = os.path.splitext(os.path.basename(csv_abs))[0]
    return os.path.join(os.path.dirname(csv_abs), stem)


def _job_doc_tokens(result_dir: str, pattern: str) -> dict[int, int]:
    """Map job number → doc_tokens from per-run CSV filenames."""
    mapping = {}
    for fp in globmod.glob(os.path.join(result_dir, pattern)):
        fname = os.path.basename(fp)
        m = re.match(r"job_(\d+)_(?:cold_prefill|cache_hit)_(\d+)_c\d+\.csv", fname)
        if m:
            mapping[int(m.group(1))] = int(m.group(2))
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE EXTRACTION — BASELINE (COLD PREFILL)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineProfile:
    forward_ms: float   # median forward pass wall time
    request_ms: float   # median request wall time


def extract_baseline_profiles(result_dir: str) -> dict[int, BaselineProfile]:
    """
    Read cold_prefill profile JSONs. Per doc_size, extract median forward
    and request wall times (skip first request as warmup).

    Returns {doc_tokens: BaselineProfile}.
    """
    job_to_tokens = _job_doc_tokens(result_dir, "job_*_cold_prefill_*_c1.csv")
    if not job_to_tokens:
        return {}

    raw: dict[int, list[tuple[float, float]]] = {}

    for job_num, doc_tokens in sorted(job_to_tokens.items()):
        profile_path = os.path.join(result_dir, f"job_{job_num:04d}_profile.json")
        if not os.path.exists(profile_path):
            continue

        with open(profile_path) as f:
            data = json.load(f)
        events = data.get("traceEvents", [])
        x_events = [e for e in events if e.get("ph") == "X"]

        requests = sorted(
            [e for e in x_events if e["name"].startswith("request(")],
            key=lambda e: e["ts"],
        )
        forwards = sorted(
            [e for e in x_events if "forward" in e["name"]],
            key=lambda e: e["ts"],
        )

        if len(requests) < 2:
            continue

        # Skip first request (warmup), take the rest
        for req in requests[1:]:
            rs, re_ = req["ts"], req["ts"] + req["dur"]
            req_ms = req["dur"] / 1000

            # Find forward events within this request
            fwd_in_req = [e for e in forwards if e["ts"] >= rs and e["ts"] < re_]
            fwd_ms = sum(e["dur"] / 1000 for e in fwd_in_req) if fwd_in_req else 0

            raw.setdefault(doc_tokens, []).append((fwd_ms, req_ms))

    result = {}
    for doc_tokens, entries in raw.items():
        result[doc_tokens] = BaselineProfile(
            forward_ms=statistics.median([e[0] for e in entries]),
            request_ms=statistics.median([e[1] for e in entries]),
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE EXTRACTION — CACHE HIT (LOAD OPS)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheHitDecomp:
    cuda_transfer_ms: float  # raw PCIe transfer time (cpu→gpu)
    load_e2e_ms: float       # end-to-end load time (includes disk/scheduling)
    forward_ms: float        # decode forward pass
    request_ms: float        # full request wall time
    num_bytes: int           # bytes transferred in cuda_transfer


def extract_cache_hit_profiles(
    result_dir: str,
) -> tuple[float, dict[int, CacheHitDecomp]]:
    """
    Read cache_hit profile JSONs. For each profile:
      - Skip store ops (requests with no load_e2e)
      - Filter eviction artifacts (loads with num_bytes < 10MB)
      - Extract cuda_transfer, load_e2e, forward, request durations + num_bytes

    Returns (median_bandwidth_gbps, {doc_tokens: CacheHitDecomp}).
    """
    job_to_tokens = _job_doc_tokens(result_dir, "job_*_cache_hit_*_c1.csv")
    if not job_to_tokens:
        return 0.0, {}

    all_bandwidths: list[float] = []
    raw: dict[int, list[tuple[float, float, float, float, int]]] = {}

    for job_num, doc_tokens in sorted(job_to_tokens.items()):
        profile_path = os.path.join(result_dir, f"job_{job_num:04d}_profile.json")
        if not os.path.exists(profile_path):
            continue

        with open(profile_path) as f:
            data = json.load(f)
        events = data.get("traceEvents", [])
        x_events = [e for e in events if e.get("ph") == "X"]

        requests = sorted(
            [e for e in x_events if e["name"].startswith("request(")],
            key=lambda e: e["ts"],
        )

        for req in requests:
            rs, re_ = req["ts"], req["ts"] + req["dur"]
            inner = [
                e for e in x_events
                if e["ts"] >= rs and e["ts"] < re_
                and not e["name"].startswith("request(")
            ]

            # Only LOAD ops (must have load_e2e)
            load_evts = [e for e in inner if "load_e2e" in e["name"]]
            if not load_evts:
                continue

            # Find cpu_to_gpu cuda_transfer (direction is in the event name)
            cuda_evts = [
                e for e in inner
                if "cuda_transfer(cpu_to_gpu" in e["name"]
            ]
            if not cuda_evts:
                continue

            num_bytes = cuda_evts[0]["args"].get("num_bytes", 0)

            # Filter eviction artifacts (tiny loads)
            if num_bytes < 10_000_000:
                continue

            cuda_ms = cuda_evts[0]["dur"] / 1000
            le2e_ms = load_evts[0]["dur"] / 1000

            fwd_evts = [e for e in inner if "forward" in e["name"]]
            fwd_ms = fwd_evts[0]["dur"] / 1000 if fwd_evts else 0

            req_ms = req["dur"] / 1000

            # Bandwidth = bytes / seconds
            if cuda_ms > 0.01:
                bw_gbps = num_bytes / (cuda_ms / 1000) / 1e9
                all_bandwidths.append(bw_gbps)

            raw.setdefault(doc_tokens, []).append(
                (cuda_ms, le2e_ms, fwd_ms, req_ms, num_bytes)
            )

    bandwidth = statistics.median(all_bandwidths) if all_bandwidths else 0.0

    decomp = {}
    for doc_tokens, entries in raw.items():
        decomp[doc_tokens] = CacheHitDecomp(
            cuda_transfer_ms=statistics.median([e[0] for e in entries]),
            load_e2e_ms=statistics.median([e[1] for e in entries]),
            forward_ms=statistics.median([e[2] for e in entries]),
            request_ms=statistics.median([e[3] for e in entries]),
            num_bytes=int(statistics.median([e[4] for e in entries])),
        )

    return bandwidth, decomp


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT OVERHEAD (tokenization + network, scales with doc size)
# ─────────────────────────────────────────────────────────────────────────────

def extract_client_overhead(csv_path: str, baseline: dict[int, BaselineProfile]) -> dict[int, float]:
    """
    Compute client overhead per doc size from cold_prefill data:
      client_overhead(D) = csv_ttft(D) - profile_request_ms(D)

    This captures tokenization + network round-trip, which scales with input
    length and is identical for cold_prefill and cache_hit at the same doc size.

    Returns {doc_tokens: overhead_ms}.
    """
    result_dir = _find_result_dir(csv_path)
    cold_csvs = sorted(globmod.glob(os.path.join(result_dir, "job_*_cold_prefill_*_c1.csv")))

    overhead: dict[int, float] = {}
    for fp in cold_csvs:
        with open(fp) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        doc_size = int(rows[0]["doc_tokens"])
        if doc_size not in baseline:
            continue
        ttfts_ms = [float(r["ttft"]) * 1000 for r in rows if r.get("successful") == "True"]
        ttfts_ms = ttfts_ms[1:] if len(ttfts_ms) > 1 else ttfts_ms
        if not ttfts_ms:
            continue
        csv_ttft_ms = statistics.median(ttfts_ms)
        overhead[doc_size] = csv_ttft_ms - baseline[doc_size].request_ms

    return overhead


# ─────────────────────────────────────────────────────────────────────────────
# COLD PREFILL CURVE (from per-run CSVs for TTFT values)
# ─────────────────────────────────────────────────────────────────────────────

def load_cold_prefill_curve(csv_path: str) -> dict[int, float]:
    """Load f(D): doc_size → median TTFT (seconds) from baseline per-run CSVs."""
    result_dir = _find_result_dir(csv_path)
    cold_csvs = sorted(globmod.glob(os.path.join(result_dir, "job_*_cold_prefill_*_c1.csv")))

    if cold_csvs:
        curve: dict[int, list[float]] = {}
        for fp in cold_csvs:
            with open(fp) as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            doc_size = int(rows[0]["doc_tokens"])
            ttfts = [float(r["ttft"]) for r in rows if r.get("successful") == "True"]
            clean = ttfts[1:] if len(ttfts) > 1 else ttfts
            curve.setdefault(doc_size, []).extend(clean)
        return {d: statistics.median(vs) for d, vs in curve.items() if vs}

    # Fallback: summary CSV
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    result = {}
    for r in rows:
        if r.get("curve") == "cold_prefill" and int(r.get("concurrency", 1)) == 1:
            if r.get("ttft_mean_s"):
                result[int(r["doc_size"])] = float(r["ttft_mean_s"])
    return result


def load_cache_load_curve(csv_path: str) -> dict[int, float]:
    """Load g(P) from per-run CSVs (LOAD ops only). Used for verification."""
    result_dir = _find_result_dir(csv_path)
    cache_csvs = sorted(globmod.glob(os.path.join(result_dir, "job_*_cache_hit_*_c1.csv")))

    if not cache_csvs:
        return {}

    curve: dict[int, list[float]] = {}
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
        if clean:
            curve.setdefault(prefix_len, []).extend(clean)

    return {p: statistics.median(vs) for p, vs in curve.items() if vs}


# ─────────────────────────────────────────────────────────────────────────────
# g_target ESTIMATION (bandwidth-based)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_g_target(
    ref_decomp: dict[int, CacheHitDecomp],
    bandwidth_gbps: float,
    target_kv_bpt: int,
    compute_ratio: float,
    client_overhead_ms: dict[int, float],
) -> tuple[dict[int, float], float]:
    """
    Estimate target cache-load TTFT using bandwidth-based approach.

    For each doc_size P:
      target_kv_bytes    = P × target_kv_bpt
      target_cuda_ms     = target_kv_bytes / bandwidth
      target_load_e2e_ms = target_cuda_ms + e2e_overhead(P)
      target_fwd_ms      = ref_decode_fwd(P) × compute_ratio
      client_oh          = csv_ttft - profile_req from target cold_prefill (tokenization)
      g_target(P)        = target_load_e2e + target_fwd + request_overhead + client_oh

    Returns (g_target_dict, floor_s) where floor_s is the minimum TTFT
    at P=0: decode_forward + request_overhead + client_overhead (no transfer needed).
    """
    bw_bytes_per_ms = bandwidth_gbps * 1e9 / 1000  # GB/s → bytes/ms

    result = {}
    floors = []
    for p in sorted(ref_decomp):
        d = ref_decomp[p]
        e2e_overhead_ms = d.load_e2e_ms - d.cuda_transfer_ms
        req_overhead_ms = d.request_ms - d.load_e2e_ms - d.forward_ms

        target_kv_bytes = p * target_kv_bpt
        target_cuda_ms = target_kv_bytes / bw_bytes_per_ms
        target_load_e2e_ms = target_cuda_ms + e2e_overhead_ms
        target_fwd_ms = d.forward_ms * compute_ratio
        client_oh = client_overhead_ms.get(p, 0.0)
        target_ttft_ms = target_load_e2e_ms + target_fwd_ms + req_overhead_ms + client_oh

        result[p] = target_ttft_ms / 1000  # → seconds

        # Floor = non-transfer components (decode forward + request overhead + client overhead)
        floors.append((target_fwd_ms + req_overhead_ms + client_oh) / 1000)

    floor_s = statistics.median(floors) if floors else 0.0
    return result, floor_s


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATORS
# ─────────────────────────────────────────────────────────────────────────────

def build_interpolator(
    curve: dict[int, float],
    floor: float = 0.0,
) -> PchipInterpolator:
    """
    Monotone cubic interpolator for {tokens: ttft_s}.

    Anchored at (0, floor). For cold prefill curves, floor=0 (no compute at
    0 tokens). For cache-hit curves, floor is the non-transfer overhead
    (decode forward + request overhead) — the minimum cost even at P=0.
    """
    sizes = sorted(curve.keys())
    ttfts = [curve[s] for s in sizes]
    if sizes[0] != 0:
        sizes = [0] + sizes
        ttfts = [floor] + ttfts
    return PchipInterpolator(sizes, ttfts, extrapolate=True)


def eval_f(interp: PchipInterpolator, n: np.ndarray) -> np.ndarray:
    """Evaluate prefill time, clamped to ≥ 0."""
    return np.maximum(0.0, interp(n))


# ─────────────────────────────────────────────────────────────────────────────
# PARETO COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_speedup_grid(
    cold_interp: PchipInterpolator,
    g_interp: PchipInterpolator,
    doc_sizes: np.ndarray,
    prefix_fracs: np.ndarray,
    cold_time_D: np.ndarray | None = None,
) -> np.ndarray:
    """
    speedup(D, frac) = f(D) / (g(frac*D) + f(D) - f(frac*D))

    The suffix cost is f(D) - f(frac*D): the marginal prefill cost of the
    suffix given frac*D tokens are already in the KV cache.  This accounts
    for the suffix forward pass attending over the full D-token context,
    not just the suffix in isolation.

    Shape: (len(prefix_fracs), len(doc_sizes))
    """
    D, F = np.meshgrid(doc_sizes, prefix_fracs)

    if cold_time_D is not None:
        cold_time = cold_time_D[np.newaxis, :]
    else:
        cold_time = eval_f(cold_interp, D)

    cache_time = np.maximum(0, g_interp(F * D))
    suffix_time = np.maximum(0, cold_time - eval_f(cold_interp, F * D))
    cached_time = cache_time + suffix_time

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cached_time > 0, cold_time / cached_time, np.inf)


def find_frontier(
    doc_sizes: np.ndarray,
    prefix_fracs: np.ndarray,
    speedup: np.ndarray,
    min_speedup: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """For each doc_size, find min prefix_frac where speedup > min_speedup."""
    frontier_d, frontier_f = [], []
    for di, d in enumerate(doc_sizes):
        col = speedup[:, di]
        idx = np.where(col > min_speedup)[0]
        if idx.size > 0:
            frontier_d.append(d)
            frontier_f.append(prefix_fracs[idx[0]])
    return np.array(frontier_d), np.array(frontier_f)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    ref_doc_sizes: list[int],
    target_doc_sizes: list[int],
    prefix_fracs: np.ndarray,
    speedup_ref: np.ndarray,
    speedup_target: np.ndarray,
    ref_label: str,
    target_label: str,
    f_ref: PchipInterpolator,
    f_target: PchipInterpolator,
    g_ref: PchipInterpolator,
    g_target: PchipInterpolator,
    output_png: str,
) -> None:
    ref_x_idx = np.arange(len(ref_doc_sizes))
    ref_x_labels = [f"{d // 1024}k" for d in ref_doc_sizes]

    tgt_x_idx = np.arange(len(target_doc_sizes))
    tgt_x_labels = [f"{d // 1024}k" for d in target_doc_sizes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("KV-Cache Pareto Frontier Estimation", fontsize=13, fontweight="bold")

    def _heatmap(ax, x_idx, x_labels, title, doc_sizes_list, f_interp, g_func):
        hm_fracs = np.linspace(0.0, 1.0, 60)
        doc_arr = np.array(doc_sizes_list, dtype=float)
        cold_ttft_arr = np.array([float(f_interp(d)) for d in doc_arr])
        sp_mat = np.zeros((len(hm_fracs), len(doc_arr)))
        for i, frac in enumerate(hm_fracs):
            cached = (
                np.maximum(0, g_func(frac * doc_arr))
                + np.maximum(0, cold_ttft_arr - f_interp(frac * doc_arr))
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                sp_mat[i] = np.where(cached > 0, cold_ttft_arr / cached, np.inf)

        log_sp = np.log2(np.clip(sp_mat, 0.25, 4.0))
        pc = ax.pcolormesh(
            x_idx, hm_fracs * 100, log_sp,
            cmap="RdYlGn", vmin=-2, vmax=2, shading="nearest",
        )
        cs = ax.contour(
            x_idx, hm_fracs * 100, sp_mat,
            levels=[1.0], colors=["black"], linewidths=[2],
        )
        ax.clabel(cs, fmt="break-even", fontsize=8)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel("Document size")
        ax.set_ylabel("Prefix fraction (%)")
        ax.set_title(title)
        plt.colorbar(pc, ax=ax, label="log₂(speedup)  [green = caching faster]")

    # Panel 1: reference heatmap (measured)
    _heatmap(axes[0], ref_x_idx, ref_x_labels,
             f"Speedup — {ref_label} (measured)",
             ref_doc_sizes, f_ref, g_ref)

    # Panel 2: target heatmap (estimated)
    _heatmap(axes[1], tgt_x_idx, tgt_x_labels,
             f"Speedup — {target_label} (estimated)",
             target_doc_sizes, f_target, g_target)

    # Panel 3: speedup vs doc size at fixed prefix fractions
    ax = axes[2]
    doc_arr_tgt = np.array(target_doc_sizes, dtype=float)
    cold_arr_tgt = np.array([float(f_target(d)) for d in doc_arr_tgt])

    for frac, color in [(0.25, "royalblue"), (0.5, "green"), (0.75, "darkorange"), (0.9, "red")]:
        cached = (
            np.maximum(0, g_target(frac * doc_arr_tgt))
            + np.maximum(0, cold_arr_tgt - f_target(frac * doc_arr_tgt))
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            speedup = np.where(cached > 0, cold_arr_tgt / cached, np.inf)
        ax.plot(tgt_x_idx, speedup, "o-", label=f"{frac * 100:.0f}% prefix", color=color)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="break-even")
    ax.set_xlabel("Document size")
    ax.set_ylabel("Speedup (cold / cached TTFT)")
    ax.set_title(f"Speedup vs doc size — {target_label} (est.)")
    ax.set_xticks(tgt_x_idx)
    ax.set_xticklabels(tgt_x_labels, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output_png}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_frontier_csv(
    doc_sizes: np.ndarray,
    prefix_fracs: np.ndarray,
    speedup_ref: np.ndarray,
    speedup_target: np.ndarray,
    ref_label: str,
    target_label: str,
    path: str,
) -> None:
    fd_ref, ff_ref = find_frontier(doc_sizes, prefix_fracs, speedup_ref)
    fd_target, ff_target = find_frontier(doc_sizes, prefix_fracs, speedup_target)

    rows = []
    for d, f in zip(fd_ref, ff_ref):
        rows.append({"doc_size": int(d), "min_prefix_frac": round(float(f), 4),
                     "model": ref_label, "type": "measured"})
    for d, f in zip(fd_target, ff_target):
        rows.append({"doc_size": int(d), "min_prefix_frac": round(float(f), 4),
                     "model": target_label, "type": "estimated"})

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "type", "doc_size", "min_prefix_frac"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Frontier CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate the KV-cache Pareto frontier for a target model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--ref-results", required=True,
                   help="pareto_measure summary CSV for the reference model "
                        "(must have cache_hit runs with profile JSONs).")
    p.add_argument("--target-results", required=True,
                   help="pareto_measure summary CSV for the target model "
                        "(needs at least cold_prefill runs with profile JSONs).")
    p.add_argument("--verify-results", default=None,
                   help="Optional: pareto_measure CSV with measured target cache_hit "
                        "data, for comparing estimated vs measured g_target.")

    p.add_argument("--ref-layers",    type=int, required=True)
    p.add_argument("--ref-kv-heads",  type=int, required=True)
    p.add_argument("--ref-head-dim",  type=int, required=True)

    p.add_argument("--target-layers",   type=int, required=True)
    p.add_argument("--target-kv-heads", type=int, required=True)
    p.add_argument("--target-head-dim", type=int, required=True)

    p.add_argument("--dtype", choices=["fp16", "bf16", "fp8"], default="fp16")
    p.add_argument("--ref-label", default="reference")
    p.add_argument("--target-label", default="target")

    p.add_argument("--output-png", default="pareto_estimate.png")
    p.add_argument("--output-csv", default="pareto_estimate.csv")

    p.add_argument("--n-doc-sizes", type=int, default=200)
    p.add_argument("--n-prefix-fracs", type=int, default=1000)
    p.add_argument("--max-doc-size", type=int, default=81920)

    return p


def main():
    args = build_parser().parse_args()
    ref_dir = _find_result_dir(args.ref_results)
    target_dir = _find_result_dir(args.target_results)

    # ── Step 1: Reference baseline profiles ───────────────────────────────────
    print("\n[Step 1] Reading reference baseline profiles…")
    ref_baseline = extract_baseline_profiles(ref_dir)
    if not ref_baseline:
        print("ERROR: no baseline profiles found in reference directory.")
        return

    print(f"  {'D':>8}  {'forward_ms':>11}  {'request_ms':>11}")
    for d in sorted(ref_baseline):
        b = ref_baseline[d]
        print(f"  {d:>8}  {b.forward_ms:>11.1f}  {b.request_ms:>11.1f}")

    # ── Step 2: Reference cache_hit profiles ──────────────────────────────────
    print("\n[Step 2] Reading reference cache_hit profiles…")
    bandwidth_gbps, ref_cache_hit = extract_cache_hit_profiles(ref_dir)
    if not ref_cache_hit:
        print("ERROR: no cache_hit profiles found in reference directory.")
        return

    print(f"  PCIe bandwidth: {bandwidth_gbps:.1f} GB/s")
    print(f"\n  {'P':>8}  {'cuda_ms':>9}  {'le2e_ms':>9}  {'fwd_ms':>8}  {'req_ms':>8}  "
          f"{'bytes':>14}  {'bw_GB/s':>9}  {'oh_ms':>7}")
    for p in sorted(ref_cache_hit):
        d = ref_cache_hit[p]
        bw = d.num_bytes / (d.cuda_transfer_ms / 1000) / 1e9 if d.cuda_transfer_ms > 0 else 0
        oh = d.load_e2e_ms - d.cuda_transfer_ms
        print(f"  {p:>8}  {d.cuda_transfer_ms:>9.1f}  {d.load_e2e_ms:>9.1f}  "
              f"{d.forward_ms:>8.1f}  {d.request_ms:>8.1f}  "
              f"{d.num_bytes:>14,}  {bw:>9.1f}  {oh:>7.1f}")

    # ── Step 3: Target baseline profiles ──────────────────────────────────────
    print("\n[Step 3] Reading target baseline profiles…")
    target_baseline = extract_baseline_profiles(target_dir)
    if not target_baseline:
        print("ERROR: no baseline profiles found in target directory.")
        return

    print(f"  {'D':>8}  {'forward_ms':>11}  {'request_ms':>11}")
    for d in sorted(target_baseline):
        b = target_baseline[d]
        print(f"  {d:>8}  {b.forward_ms:>11.1f}  {b.request_ms:>11.1f}")

    # ── Compute ratio ─────────────────────────────────────────────────────────
    common_sizes = sorted(set(ref_baseline) & set(target_baseline))
    if not common_sizes:
        print("ERROR: no common doc sizes between ref and target baselines.")
        return

    ratios = []
    for d in common_sizes:
        if ref_baseline[d].forward_ms > 0:
            ratios.append(target_baseline[d].forward_ms / ref_baseline[d].forward_ms)
    compute_ratio = statistics.median(ratios)
    print(f"\n  Compute ratio (target/ref baseline forward): {compute_ratio:.2f}×")

    # ── KV sizes ──────────────────────────────────────────────────────────────
    ref_kv_bpt = kv_bytes_per_token(
        args.ref_layers, args.ref_kv_heads, args.ref_head_dim, args.dtype)
    target_kv_bpt = kv_bytes_per_token(
        args.target_layers, args.target_kv_heads, args.target_head_dim, args.dtype)
    kv_ratio = target_kv_bpt / ref_kv_bpt

    print(f"\n  {args.ref_label}    KV bytes/token: {ref_kv_bpt:,}")
    print(f"  {args.target_label}  KV bytes/token: {target_kv_bpt:,}")
    print(f"  KV ratio (target/ref): {kv_ratio:.2f}×")

    # ── Step 4: Estimate g_target ─────────────────────────────────────────────
    print("\n[Step 4] Estimating target cache-load TTFT (bandwidth-based)…")
    # Client overhead: tokenization + network, measured from target cold_prefill
    print("\n  Client overhead (csv_ttft - profile_req, from target cold_prefill):")
    client_overhead = extract_client_overhead(args.target_results, target_baseline)
    print(f"  {'D':>8}  {'client_oh_ms':>14}")
    for d in sorted(client_overhead):
        print(f"  {d:>8}  {client_overhead[d]:>14.1f}")

    g_target_pts, g_target_floor = estimate_g_target(
        ref_cache_hit, bandwidth_gbps, target_kv_bpt, compute_ratio, client_overhead)

    print(f"\n  g(0) floor (decode_fwd + req_overhead + client_oh): {g_target_floor*1000:.1f}ms")
    print(f"\n  {'P':>8}  {'tgt_bytes':>14}  {'tgt_cuda':>10}  {'tgt_le2e':>10}  {'tgt_fwd':>9}  "
          f"{'req_oh':>8}  {'client_oh':>10}  {'g_target':>10}")
    bw_bytes_per_ms = bandwidth_gbps * 1e9 / 1000
    for p in sorted(g_target_pts):
        d = ref_cache_hit[p]
        target_kv_bytes = p * target_kv_bpt
        target_cuda_ms = target_kv_bytes / bw_bytes_per_ms
        e2e_oh = d.load_e2e_ms - d.cuda_transfer_ms
        target_le2e = target_cuda_ms + e2e_oh
        target_fwd = d.forward_ms * compute_ratio
        req_oh = d.request_ms - d.load_e2e_ms - d.forward_ms
        client_oh = client_overhead.get(p, 0.0)
        print(f"  {p:>8}  {target_kv_bytes:>14,.0f}  {target_cuda_ms:>10.1f}  {target_le2e:>10.1f}  "
              f"{target_fwd:>9.1f}  {req_oh:>8.1f}  {client_oh:>10.1f}  {g_target_pts[p]*1000:>10.1f}")

    # ── Verification (if available) ───────────────────────────────────────────
    verify_csv = args.verify_results
    if verify_csv:
        measured_g = load_cache_load_curve(verify_csv)
        if measured_g:
            print(f"\n  Verification against measured target cache-hit:")
            print(f"  {'P':>8}  {'estimated':>10}  {'measured':>10}  {'error':>8}")
            for p in sorted(set(g_target_pts) & set(measured_g)):
                est = g_target_pts[p] * 1000
                meas = measured_g[p] * 1000
                err = (est - meas) / meas * 100
                print(f"  {p:>8}  {est:>10.1f}  {meas:>10.1f}  {err:>7.1f}%")

    # ── Load f curves from per-run CSVs ───────────────────────────────────────
    ref_cold = load_cold_prefill_curve(args.ref_results)
    ref_load = load_cache_load_curve(args.ref_results)
    target_cold = load_cold_prefill_curve(args.target_results)

    if not ref_cold or not ref_load or not target_cold:
        print("ERROR: missing cold_prefill or cache_hit CSV data.")
        return

    # ── Build interpolators ───────────────────────────────────────────────────
    # Compute ref g floor: non-transfer overhead at smallest doc size
    # (decode_fwd + req_overhead + client_overhead) — consistent with ref_load
    # which comes from CSV TTFTs (includes client overhead).
    ref_client_overhead = extract_client_overhead(args.ref_results, ref_baseline)
    smallest_ref_p = min(ref_cache_hit)
    d0 = ref_cache_hit[smallest_ref_p]
    ref_client_oh = ref_client_overhead.get(smallest_ref_p, 0.0)
    ref_g_floor = (d0.forward_ms + (d0.request_ms - d0.load_e2e_ms - d0.forward_ms) + ref_client_oh) / 1000

    f_ref = build_interpolator(ref_cold)
    g_ref = build_interpolator(ref_load, floor=ref_g_floor)
    f_target = build_interpolator(target_cold)
    g_target = build_interpolator(g_target_pts, floor=g_target_floor)

    n_ref = sorted(ref_cold.keys())
    n_target = sorted(target_cold.keys())

    # ── Evaluation grid ───────────────────────────────────────────────────────
    doc_sizes = np.linspace(1, args.max_doc_size, args.n_doc_sizes)
    prefix_fracs = np.linspace(0.0, 1.0, args.n_prefix_fracs)

    cold_time_ref_D = eval_f(f_ref, doc_sizes)
    cold_time_target_D = eval_f(f_target, doc_sizes)

    print("\nComputing speedup grids…")
    speedup_ref = compute_speedup_grid(
        f_ref, g_ref, doc_sizes, prefix_fracs, cold_time_D=cold_time_ref_D)
    speedup_target = compute_speedup_grid(
        f_target, g_target, doc_sizes, prefix_fracs, cold_time_D=cold_time_target_D)

    # ── Break-even summary ────────────────────────────────────────────────────
    # With the new formula, break-even is where g(frac*D) = f(frac*D).
    # Find numerically by scanning prefix fracs.
    def find_breakeven_frac(d: float, f_interp, g_interp) -> float:
        for frac in np.linspace(0.0, 1.0, 2000):
            p = frac * d
            if float(g_interp(p)) <= float(f_interp(p)):
                return frac * 100
        return float('nan')

    print(f"\n  Break-even prefix fraction at selected doc sizes:")
    print(f"  {'Doc size':>10}  {'Ref (measured)':>14}  {'Target (est.)':>16}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*16}")
    for d in [4096, 8192, 16384, 32768, 65536, 80000]:
        f_ref_be = find_breakeven_frac(d, f_ref, g_ref)
        f_tgt_be = find_breakeven_frac(d, f_target, g_target)
        print(f"  {d:>10,}  {f_ref_be:>13.1f}%  {f_tgt_be:>15.1f}%")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print()
    save_frontier_csv(
        doc_sizes, prefix_fracs,
        speedup_ref, speedup_target,
        args.ref_label, args.target_label,
        args.output_csv,
    )
    plot_results(
        n_ref, n_target, prefix_fracs,
        speedup_ref, speedup_target,
        args.ref_label, args.target_label,
        f_ref, f_target, g_ref, g_target,
        args.output_png,
    )


if __name__ == "__main__":
    main()
