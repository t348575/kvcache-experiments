#!/usr/bin/env bash
# Pre-fetch the LongBench v2 dataset into the HuggingFace cache so the driver runs
# offline. The driver (scripts/longbench_v2_replay.py) loads it via
# datasets.load_dataset("THUDM/LongBench-v2"), so this is only a warm-up; on a box
# with network the driver fetches it on first use anyway.
#
# Source: https://huggingface.co/datasets/THUDM/LongBench-v2  (503 questions)
set -euo pipefail

REPO="THUDM/LongBench-v2"

echo "Pre-fetching $REPO into the HuggingFace datasets cache"
python3 - "$REPO" <<'PY'
import sys
from datasets import load_dataset

repo = sys.argv[1]
ds = load_dataset(repo, split="train")
print(f"Cached {len(ds)} rows from {repo}")
print("Fields:", list(ds.features))
PY
echo "Done."
