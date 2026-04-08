#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common_env.sh

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/04_infer_cli.sh \"Your math problem text\""
  exit 1
fi

echo "[Step 4] Running local CLI inference..."
uv run python inference.py --artifacts-dir artifacts --text "$1"
