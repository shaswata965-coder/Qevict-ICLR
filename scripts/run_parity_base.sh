#!/usr/bin/env bash
# Parity, base run: FullKV greedy decode that the ours runs are teacher-forced on. Writes outputs/parity_base_wikitext-103.npz.
#
#   scripts/run_parity_base.sh [--override key=value ...]
#
# Config: configs/llama/eval_parity_base.yaml (edit it, or pass --override; see main.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --config configs/llama/eval_parity_base.yaml "$@"
