#!/usr/bin/env bash
# Figures from the parity / faithfulness / perf outputs (no model load). Writes outputs/figures/.
#
#   scripts/run_visualize.sh [--override key=value ...]
#
# Config: configs/llama/eval_visualize.yaml (edit it, or pass --override; see main.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --config configs/llama/eval_visualize.yaml "$@"
