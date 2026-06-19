"""Emit a medium-aware break-even data file for py_kvcache load-side gating.

Uses the pareto tooling (plots/pareto_plot.py) to find the prefix length P*
where cache-hit TTFT breaks even with cold-prefill recompute, for two media:

  P*_ssd: SSD load, solve f(P) = g_ssd(P) (measured cache_hit curve).
  P*_mem: RAM load, solve f(P) = g_mem(P), where g_mem = g_ssd minus the
          modelled disk-read time P*kv_bpt / ssd_bandwidth (the RAM staging
          cache pays only the cuda copy). Expected near 0.

The emitted JSON carries only the fields py_kvcache.break_even.load_break_even
reads at runtime: model_name, kv_dtype, the two break_even_*_tokens scalars, and
safety_margin_tokens.

Usage:
  python scripts/emit_break_even.py \
      --results data/surf/h100-pyoffload/pareto_measure_merged.csv \
      --model-name meta-llama/Llama-3.1-8B-Instruct \
      --kv-dtype auto \
      --kv-bytes-per-token 131072 \
      --ssd-bandwidth-gbps 13.0 \
      --output break_even_h100_llama31_8b_cm7r.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Repo root on path so `plots` / `common` packages import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plots.pareto_plot import (  # noqa: E402
    break_even_curve_values,
    build_interpolator,
    filter_cache_server_config,
    load_per_run_curves,
    load_results,
    select_cache_server_config,
    serial_curve,
    server_config_curve,
)


def _resolve_curves(results: str, cache_server_config: str | None):
    """Return (cold, hit) {doc_size: row} dicts, preferring per-run CSVs."""
    rows = load_results(results)
    if cache_server_config:
        rows = filter_cache_server_config(rows, cache_server_config)
    cold_pr, hit_pr = load_per_run_curves(results, cache_server_config)
    cold = cold_pr or serial_curve(rows, "cold_prefill")
    hit = hit_pr or server_config_curve(rows, "cache_hit", cache_server_config)
    if not cold:
        raise SystemExit("no cold_prefill data found in results")
    if not hit:
        raise SystemExit("no cache_hit data found in results")
    return cold, hit


def _g_mem_curve(
    hit: dict[int, dict],
    kv_bpt: int,
    ssd_bandwidth_gbps: float,
) -> dict[int, dict]:
    """g_mem(P) = g_ssd(P) - disk_read(P), floored at 0.

    Models a RAM-cache hit as the measured SSD-hit minus the disk read it skips
    (P·kv_bpt / bandwidth). The cuda staging->GPU copy, already in g_ssd, remains.
    """
    bw_bytes_per_s = ssd_bandwidth_gbps * 1e9
    g_mem: dict[int, dict] = {}
    for tokens, row in hit.items():
        disk_read_s = tokens * kv_bpt / bw_bytes_per_s
        g_mem[tokens] = {"ttft_mean_s": max(0.0, row["ttft_mean_s"] - disk_read_s)}
    return g_mem


def _scalar_break_even(f_interp, g_interp, doc_sizes: list[int]) -> int:
    """Single P*: the crossing at the largest reachable doc size, rounded up."""
    values = break_even_curve_values(f_interp, g_interp, doc_sizes)
    resolved = [v for v in values if v.get("prefix_tokens") is not None]
    if not resolved:
        # f >= g everywhere reachable: caching always pays -> no gate.
        return 0
    largest = max(resolved, key=lambda v: v["doc_size"])
    return int(math.ceil(float(largest["prefix_tokens"])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, help="pareto_measure merged CSV.")
    p.add_argument("--model-name", required=True, help="HF model id (validated at runtime).")
    p.add_argument("--kv-dtype", default="auto", help="KV cache dtype string (e.g. auto, fp8).")
    p.add_argument("--kv-bytes-per-token", type=int, required=True)
    p.add_argument("--ssd-bandwidth-gbps", type=float, required=True,
                   help="Sequential read bandwidth, for the g_mem disk-read model.")
    p.add_argument("--cache-server-config", default=None,
                   help="cache_hit server_config (the SSD) to use.")
    p.add_argument("--safety-margin-tokens", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cache_server_config = select_cache_server_config(
        load_results(args.results), args.cache_server_config
    )
    cold, hit = _resolve_curves(args.results, cache_server_config)
    doc_sizes = sorted(cold.keys())

    f_interp = build_interpolator(cold)
    g_ssd_interp = build_interpolator(hit, floor=min(r["ttft_mean_s"] for r in hit.values()))
    g_mem = _g_mem_curve(hit, args.kv_bytes_per_token, args.ssd_bandwidth_gbps)
    g_mem_interp = build_interpolator(g_mem, floor=min(r["ttft_mean_s"] for r in g_mem.values()))

    p_ssd = _scalar_break_even(f_interp, g_ssd_interp, doc_sizes)
    p_mem = _scalar_break_even(f_interp, g_mem_interp, doc_sizes)

    payload = {
        "model_name": args.model_name,
        "kv_dtype": args.kv_dtype,
        "break_even_ssd_tokens": p_ssd,
        "break_even_mem_tokens": p_mem,
        "safety_margin_tokens": args.safety_margin_tokens,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"P*_ssd = {p_ssd} tokens, P*_mem = {p_mem} tokens -> {args.output}")


if __name__ == "__main__":
    main()
