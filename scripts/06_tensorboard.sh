#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common_env.sh

echo "Starting TensorBoard on http://localhost:6006"
uv run tensorboard --logdir artifacts/tb_logs --port 6006
