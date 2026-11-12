#!/usr/bin/env bash
# scripts/run_longgenbench.sh
#
# ============================================================================
# LongGenBench (Wu et al., ICLR'25) end to end: data -> generate -> judge -> score
# ============================================================================
#   bash scripts/run_longgenbench.sh                                  # ours, 16K, budget 0.10
#   ARMS="fullkv ours control" bash scripts/run_longgenbench.sh       # + full KV, + gate control
#   N_PER_TYPE=5 bash scripts/run_longgenbench.sh                     # first 5 documents per type
#   JUDGE_PY=judge_env/bin/python bash scripts/run_longgenbench.sh    # also judge and score
#
# Arms (each one main.py run over the same documents):
#   ours     configs/llama/longgenbench_ours_flash_attn.yaml at cache_budget=BUDGET
#            (the operating point otherwise: window 8, sink 5, local 128,
#            quant_ratio 0.70 in bytes, read ratio 0.25)
#   control  ours with quant_gate_ratio=1.0 (every int2 window read: the gate's
#            control arm)
#   fullkv   configs/llama/longgenbench_full_cache.yaml (no compression)
#
# Generation decodes the paper's reference sampling (T=0.95, top_p=0.95, a fixed
# seed per document) in equal-length batches of 16, falling back to 8 then 6 on
# out-of-memory. One A100-80GB; the whole 16K benchmark is 400 documents of up to
# 16,000 tokens per arm.
#
# The judge (Llama-3.1-8B-Instruct on vLLM) runs in its OWN environment, because
# vLLM pins a different torch / transformers from the generation stack:
#   uv venv --python 3.12 judge_env
#   uv pip install --python judge_env/bin/python -r longgenbench_eval/requirements-judge.lock.txt
# then pass JUDGE_PY=judge_env/bin/python. Without JUDGE_PY the script stops
# after generation. Scoring needs only the standard library.
# See longgenbench_eval/README.md for what each step reproduces.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PY="${PY:-python}"
JUDGE_PY="${JUDGE_PY:-}"
SETTING="${SETTING:-16K}"                  # 16K | 32K
BUDGET="${BUDGET:-0.10}"
ARMS="${ARMS:-ours}"                       # any of: fullkv ours control
N_PER_TYPE="${N_PER_TYPE:-}"               # empty = every document
MODEL="${MODEL:-}"
DATA_DIR="${DATA_DIR:-data/longgenbench}"
OUT="${OUT:-outputs/longgenbench/${SETTING}_b${BUDGET}}"

# ---- 1. the pinned dataset ------------------------------------------------------
"$PY" -m longgenbench_eval.data --out "$DATA_DIR" --setting "$SETTING"

# ---- 2. generation -----------------------------------------------------------------
COMMON=(longgenbench.data_dir="$DATA_DIR" longgenbench.setting="$SETTING")
if [ -n "$N_PER_TYPE" ]; then COMMON+=(longgenbench.end="$N_PER_TYPE"); fi
if [ -n "$MODEL" ]; then COMMON+=(model.name="$MODEL"); fi

for arm in $ARMS; do
  dir="$OUT/$arm"
  if [ -f "$dir/generations.meta.json" ]; then echo "[$arm] done -- skipping ($dir)"; continue; fi
  mkdir -p "$dir"
  case "$arm" in
    fullkv)  cfg=configs/llama/longgenbench_full_cache.yaml; extra=() ;;
    ours)    cfg=configs/llama/longgenbench_ours_flash_attn.yaml
             extra=(cache.cache_budget="$BUDGET" cache.quant_gate_ratio=0.25) ;;
    control) cfg=configs/llama/longgenbench_ours_flash_attn.yaml
             extra=(cache.cache_budget="$BUDGET" cache.quant_gate_ratio=1.0) ;;
    *) echo "unknown arm: $arm (fullkv | ours | control)" >&2; exit 2 ;;
  esac
  echo "[$arm] generating -> $dir"
  "$PY" main.py --config "$cfg" --override "${COMMON[@]}" "${extra[@]}" \
      longgenbench.output_dir="$dir" 2>&1 | tee "$dir/run.log"
done

# ---- 3. judge (separate vLLM environment) ----------------------------------------------
if [ -z "$JUDGE_PY" ]; then
  echo "Generation done. Set JUDGE_PY=<judge env python> to judge and score (see the header)."
  exit 0
fi
GENS=()
for arm in $ARMS; do GENS+=("$OUT/$arm/generations.jsonl"); done
"$JUDGE_PY" -m longgenbench_eval.judge --setting "$SETTING" --data "$DATA_DIR" --generations "${GENS[@]}"

# ---- 4. score --------------------------------------------------------------------------
RUNS=()
for arm in $ARMS; do RUNS+=(--run "$arm" "$OUT/$arm/generations.jsonl"); done
"$PY" -m longgenbench_eval.score --setting "$SETTING" --data "$DATA_DIR" "${RUNS[@]}" \
    --json "$OUT/scores.json"
echo "scores: $OUT/scores.json"
