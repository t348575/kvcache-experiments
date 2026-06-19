"""Download SCBench config parquet files from HuggingFace into a local dir.

SCBench ships one parquet per config (subtask) under <config>/test-*.parquet.
This mirrors how the Bailian traces live under dataset/bailian/: after running
this, dataset/scbench/<config>.parquet holds each subtask, ready for
scbench_stats.py and the replay harness.

Examples:
    # All configs into the default dataset/scbench/
    python scripts/scbench_download.py

    # Just two configs
    python scripts/scbench_download.py --configs scbench_kv scbench_repoqa_and_kv
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import SCBENCH_CONFIGS, SCBENCH_REPO  # noqa: E402


def download_config(api, config, out_dir, token):
    """Fetch every test parquet shard for one config, concatenate names to disk.

    Most configs are a single shard (test-00000-of-00001.parquet); the glob keeps
    multi-shard configs working. Shard 0 is written as <config>.parquet and any
    further shards as <config>.partNN.parquet so a later reader can glob them.
    """
    from huggingface_hub import hf_hub_download

    repo_files = api.list_repo_files(SCBENCH_REPO, repo_type="dataset", token=token)
    shards = sorted(
        f for f in repo_files
        if f.startswith(f"{config}/") and f.endswith(".parquet")
    )
    if not shards:
        raise SystemExit(f"No parquet files found for config {config!r} in {SCBENCH_REPO}")

    written = []
    for idx, remote in enumerate(shards):
        local = hf_hub_download(
            repo_id=SCBENCH_REPO, repo_type="dataset", filename=remote, token=token
        )
        if idx == 0:
            dst = out_dir / f"{config}.parquet"
        else:
            dst = out_dir / f"{config}.part{idx:02d}.parquet"
        shutil.copy(local, dst)
        written.append(dst)
        print(f"  {remote} -> {dst} ({os.path.getsize(dst) // 1024} KB)")
    return written


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="dataset/scbench",
                   help="Output directory (default: dataset/scbench).")
    p.add_argument("--configs", nargs="+", default=SCBENCH_CONFIGS,
                   choices=SCBENCH_CONFIGS,
                   help="Configs to download (default: all).")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"),
                   help="HF token (defaults to $HF_TOKEN; not required for this public dataset).")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Downloading {len(args.configs)} SCBench config(s) from {SCBENCH_REPO} -> {out_dir}")
    for config in args.configs:
        print(f"[{config}]")
        download_config(api, config, out_dir, args.token)
    print("Done.")


if __name__ == "__main__":
    main()
