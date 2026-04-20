#!/bin/bash
#SBATCH --job-name=bench-array
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --constraint=scratch-node
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --error=slurm-logs/%x-%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p slurm-logs

source load.sh

CONFIG_FILES=(
  "bench_configs/llmd.json"
  "bench_configs/py-kvcache.json"
  "bench_configs/native_offload.json"
  "bench_configs/baseline.json"
)

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

for CONFIG_PATH in "${CONFIG_FILES[@]}"; do
  echo "Running config: $CONFIG_PATH"
  python3 bench.py --config "$CONFIG_PATH"
done
