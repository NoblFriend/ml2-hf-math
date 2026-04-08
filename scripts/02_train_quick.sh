#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common_env.sh

mkdir -p artifacts artifacts/tb_logs

echo "[Step 2] Quick train (MPS if available, else CPU)..."
uv run python train.py \
  --artifacts-dir artifacts \
  --tb-log-dir artifacts/tb_logs \
  --epochs 5 \
  --batch-size 64 \
  --lr 2e-5 \
  --max-length 320

echo "Train complete."
echo "To watch learning curves:"
echo "  uv run tensorboard --logdir artifacts/tb_logs --port 6006"
