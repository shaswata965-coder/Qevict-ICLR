#!/usr/bin/env bash
# Faithfulness metrics from a parity base/ours pair (no model load). Writes outputs/faithfulness_results.npz.
#
#   scripts/run_faithfulness.sh [--override key=value ...]
#
# Config: configs/llama/eval_faithfulness.yaml (edit it, or pass --override; see main.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --config configs/llama/eval_faithfulness.yaml "$@"
