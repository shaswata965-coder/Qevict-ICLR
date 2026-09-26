#!/usr/bin/env bash
# Parity, ours (flash_attn backend), teacher-forced on the base run. Run run_parity_base.sh first.
#
#   scripts/run_parity_ours_flash.sh [--override key=value ...]
#
# Config: configs/llama/eval_parity_ours_flash.yaml (edit it, or pass --override; see main.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main.py --config configs/llama/eval_parity_ours_flash.yaml "$@"
