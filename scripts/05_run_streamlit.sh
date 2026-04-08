#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common_env.sh

echo "[Step 5] Starting Streamlit locally..."
uv run streamlit run app.py --server.fileWatcherType none
