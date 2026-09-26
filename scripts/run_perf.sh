#!/usr/bin/env bash
# Perf: latency factorial (FullKV vs the windowed cache at three budgets), batch 1. Print with scripts/print_perf_table.py --npz-dir outputs.
#
#   scripts/run_perf.sh [--override key=value ...]
#
# Config: configs/llama/eval_perf.yaml (edit it, or pass --override; see main.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --config configs/llama/eval_perf.yaml "$@"
