#!/usr/bin/env bash
set -euo pipefail

# Load optional workspace-level .env (one level above project root).
# Read only HF_TOKEN to avoid breaking on unrelated .env syntax.
if [[ -f ../.env ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  hf_line="$(grep -E '^HF_TOKEN=' ../.env | tail -n 1 || true)"
  if [[ -n "$hf_line" ]]; then
    hf_value="${hf_line#HF_TOKEN=}"
    hf_value="${hf_value%$'\r'}"
    hf_value="${hf_value%\"}"
    hf_value="${hf_value#\"}"
    export HF_TOKEN="$hf_value"
  fi
fi

# Make HF download path more robust on some networks.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
