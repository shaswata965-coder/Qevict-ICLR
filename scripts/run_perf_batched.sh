#!/usr/bin/env bash
# Perf: batched decode throughput and the max-batch search (FullKV vs RecoverKV).
# Every config runs at every rung of the batch ladder; an OOM ends that config's
# ladder, so a config's max-B is the largest batch it completed.
#
#   scripts/run_perf_batched.sh [--override key=value ...]
#
# Config: configs/llama/eval_perf_batched.yaml. Writes outputs/perf_batched/ and
# prints the summary (decode tok/s, e2e tok/s, TTFT, peak memory, max-B).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT_DIR=outputs/perf_batched
mkdir -p "$OUT_DIR"

python main.py --config configs/llama/eval_perf_batched.yaml "$@"
python scripts/print_perf_batched.py --npz-dir "$OUT_DIR" --out "$OUT_DIR/summary.txt"
