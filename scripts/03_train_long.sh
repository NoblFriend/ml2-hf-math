#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common_env.sh

mkdir -p artifacts artifacts/tb_logs

echo "[Step 3] Longer training run for better quality..."
uv run python train.py \
  --artifacts-dir artifacts \
  --tb-log-dir artifacts/tb_logs \
  --epochs 10 \
  --patience 4 \
  --batch-size 8 \
  --lr 2e-5 \
  --max-length 256

echo "Long train complete."
