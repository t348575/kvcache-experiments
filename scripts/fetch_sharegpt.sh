#!/usr/bin/env bash
# Download the ShareGPT V3 dataset used by the sharegpt (vllm bench serve) configs.
# Lands at dataset/ShareGPT_V3_unfiltered_cleaned_split.json, the path the bench
# configs reference via sharegpt_dataset_path.
#
# Source: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
set -euo pipefail

DEST="${1:-dataset}"
FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/${FILE}"
mkdir -p "$DEST"

if [[ -f "$DEST/$FILE" ]]; then
  echo "have $DEST/$FILE"
  exit 0
fi

echo "Fetching $FILE (~400MB) into $DEST"
curl -fSL "$URL" -o "$DEST/$FILE" \
  || { echo "!! failed — if gated, run 'huggingface-cli login' then re-run"; exit 1; }
echo "Done. sharegpt_dataset_path = $DEST/$FILE"
