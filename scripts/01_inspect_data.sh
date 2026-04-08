#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common_env.sh

mkdir -p artifacts

echo "[Step 1] Loading and inspecting MATH dataset..."
uv run python -m tools.inspect_data \
  --sample-rows 8 \
  --save-csv artifacts/data_preview.csv

echo "Done. Preview saved to artifacts/data_preview.csv"
