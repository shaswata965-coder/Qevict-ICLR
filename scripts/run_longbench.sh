#!/usr/bin/env bash
# LongBench generation + scoring for one (model, method).
#
#   scripts/run_longbench.sh <model> <method> [--override key=value ...]
#
#   model   llama | mistral            -> configs/<model>/
#   method  full_cache | ours_flash_attn
#
# Examples:
#   scripts/run_longbench.sh llama full_cache
#   scripts/run_longbench.sh mistral ours_flash_attn
#   # one dataset (e.g. to spread datasets over GPUs), a 10% budget:
#   CUDA_VISIBLE_DEVICES=3 scripts/run_longbench.sh llama ours_flash_attn \
#       --override longbench.datasets=[qasper] cache.cache_budget=0.10 \
#                  longbench.output_dir=outputs/longbench/ours_b010
#   # a local checkpoint instead of the hub id:
#   scripts/run_longbench.sh llama ours_flash_attn --override model.name=/path/to/ckpt
#
# Data: data/longbench/<dataset>.jsonl if present (or LONGBENCH_LOCAL_DIR=/dir),
# else the HF hub copy of THUDM/LongBench (needs datasets<3, a script dataset).
# Afterwards every run under outputs/longbench/ is scored (<run>/scores.csv) and
# outputs/longbench/comparison.csv is rebuilt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ $# -lt 2 ]]; then
  sed -n '2,12p' "$0"; exit 2
fi
MODEL="$1"; METHOD="$2"; shift 2
CONFIG="configs/$MODEL/longbench_$METHOD.yaml"
if [[ ! -f "$CONFIG" ]]; then
  echo "error: no config $CONFIG" >&2
  echo "available:" >&2; ls configs/*/longbench_*.yaml >&2; exit 2
fi

python main.py --config "$CONFIG" "$@"

# Score every run under outputs/longbench/ (scores.csv per run) and rebuild
# outputs/longbench/comparison.csv.
python main.py --config "$CONFIG" --override run.mode=longbench_score
