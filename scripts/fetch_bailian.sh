#!/usr/bin/env bash
# Download the anonymized Qwen/Bailian usage traces into dataset/bailian/.
# Source: https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon
#
# The .jsonl files are git-lfs tracked, so the plain raw.githubusercontent URL
# returns a small LFS *pointer*, not the data. We pull the real bytes from the
# GitHub LFS media endpoint (no git-lfs binary required). git-lfs clone fallback
# below if the media endpoint is unavailable.
#
# The four traces (each block-size 16) are:
#   qwen_traceA_blksz_16.jsonl     to-C interactive chat
#   qwen_traceB_blksz_16.jsonl     to-B API automation
#   qwen_coder_blksz_16.jsonl      code generation
#   qwen_thinking_blksz_16.jsonl   reasoning-heavy
set -euo pipefail

DEST="${1:-dataset/bailian}"
OWNER="alibaba-edu/qwen-bailian-usagetraces-anon"
MEDIA="https://media.githubusercontent.com/media/${OWNER}/main"
mkdir -p "$DEST"

FILES=(
  qwen_traceA_blksz_16.jsonl
  qwen_traceB_blksz_16.jsonl
  qwen_coder_blksz_16.jsonl
  qwen_thinking_blksz_16.jsonl
)

echo "Downloading Bailian traces (git-lfs media endpoint) into $DEST"
for f in "${FILES[@]}"; do
  if [[ -s "$DEST/$f" ]] && ! head -c 64 "$DEST/$f" | grep -q "git-lfs"; then
    echo "  have $f"
    continue
  fi
  echo "  fetching $f"
  curl -fSL "$MEDIA/$f" -o "$DEST/$f" \
    || echo "  !! failed $f"
  # Guard against a silently-saved LFS pointer.
  if head -c 64 "$DEST/$f" 2>/dev/null | grep -q "git-lfs"; then
    echo "  !! $f is an LFS pointer, not data — fall back to git-lfs clone (below)"
    rm -f "$DEST/$f"
  fi
done

if ! ls "$DEST"/qwen_*_blksz_16.jsonl >/dev/null 2>&1; then
  echo
  echo "Media download failed. git-lfs clone fallback:"
  echo "  git lfs install"
  echo "  git clone https://github.com/${OWNER} $DEST"
  exit 1
fi
echo "Done. Point bailian_trace_path at $DEST"
