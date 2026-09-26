#!/usr/bin/env bash
# RULER generation + scoring (all 13 tasks of one context-length split).
#
#   DATA_DIR=/path/to/ruler/32768 scripts/run_ruler.sh [--override key=value ...]
#
# DATA_DIR is a HF `datasets` directory written with save_to_disk (13 tasks x
# 500 examples), e.g. the ruler/<len> folders of the DefensiveKV release; this
# repo does not download it.
#
#   CONFIG   default configs/llama/ruler_ours_flash_attn.yaml
#            (configs/llama/ruler_full_cache.yaml for the FullKV baseline)
#   OUT_DIR  default outputs/ruler/<config name>
#
# Examples:
#   DATA_DIR=... scripts/run_ruler.sh --override ruler.tasks=[niah_multikey_3]
#   DATA_DIR=... OUT_DIR=outputs/ruler/ws32 scripts/run_ruler.sh \
#       --override cache.window_size=32 window.window_size=32
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIG="${CONFIG:-configs/llama/ruler_ours_flash_attn.yaml}"
OUT_DIR="${OUT_DIR:-outputs/ruler/$(basename "$CONFIG" .yaml)}"

OVERRIDES=(ruler.output_dir="$OUT_DIR")
if [ -n "${DATA_DIR:-}" ]; then OVERRIDES+=(ruler.data_dir="$DATA_DIR"); fi

# Repeated --override flags accumulate (main.py), so the caller's are kept.
python main.py --config "$CONFIG" --override "${OVERRIDES[@]}" "$@"

python -m modules.evaluation.ruler_scoring \
    --predictions_dir "$OUT_DIR" \
    --out_csv "$OUT_DIR/scores.csv"
