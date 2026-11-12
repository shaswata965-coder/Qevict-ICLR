#!/usr/bin/env bash
# scripts/run_completion_ablation.sh
#
# ============================================================================
# The completion-term ablation: how much does quality depend on the calibrated
# window-level completion that stands in for the int2 windows the read gate
# does NOT open?
# ============================================================================
#   bash scripts/run_completion_ablation.sh                  # core arms
#   STAGE=extended bash scripts/run_completion_ablation.sh   # + ceiling, stress, scales, coupled
#   STAGE=collect bash scripts/run_completion_ablation.sh    # Experiment 1 (matched states)
#   SIGMA=0.31 ARMS="noise1x_s0" bash scripts/run_completion_ablation.sh   # a noise arm
#
# Setting (every arm): Llama-3.1-8B-Instruct, cache_budget 0.10, window 8,
# quant_ratio 0.70 in bytes mode, read ratio 0.25 unless the arm says otherwise,
# greedy. LongBench's three English tasks with 512-token generations --
# gov_report, qmsum, multi_news -- 50 examples each: the first 50 at least
# PROMPT_TOKENS long, every prompt middle-truncated to exactly that many tokens
# (6,144 / 6,144 / 2,048) so that rows of a task share a length and decode as one
# batch (the windowed cache takes no padding). gov_report and qmsum share
# batches. The ground truth on these prompts is `fullkv`, the uncompressed model
# on the SAME truncated prompts.
#
# The knob (cache.quant_completion, utils/completion_spec.py):
#   null          the shipped kernel, untouched (arm `cal`)
#   calibrated    the shipped completion recomputed by a torch epilogue around
#                 the unchanged kernel, with a common-scale error (scale) and/or
#                 total-preserving per-window noise (sigma, seed)
#   uncalibrated  the card estimate without the read windows' correction
#   off           no completion: unread int2 windows contribute nothing
# quant_completion_scores=coupled also feeds the variant into the eviction
# scores (default `output`: the output only). Every sidecar records the knob
# and read_gate.completion_applied, which must equal `gated` on a variant arm
# and 0 on a native one.
#
# Arms (ARMS overrides the STAGE's list):
#   core      cal fullkv off uncal scale2 cal_repeat cal_ep
#   extended  ceiling_g1.0 stress_g0.10_cal stress_g0.10_off scale0.5 scale0.8
#             scale1.25 scale4 coupled_scale0.5 coupled_scale2
#             rec_g0.25 rec_g0.10   (cal / stress_g0.10_cal with the matched-
#                                    state recorder on: longbench.completion_rungs)
#   noise<m>x_s<seed>   calibrated with sigma = m x SIGMA (SIGMA = the median
#                       residual spread sigma_hat_U from Experiment 1)
#   collect   Experiment 1: per task and read ratio (0.25, 0.10), the
#             observation collector with --completion-rungs computes every
#             variant on the SAME decode states (4 documents, 128 steps).
#
# Outputs: $OUT/<arm>/<task>.{jsonl,meta.json} and scores.csv per arm, and
# $OUT/comparison.csv (every arm against fullkv). Finished tasks are skipped.
# Sized for one A100-80GB (BATCH_TOKENS=212000: ~34 rows of 6,144 tokens, ~75 GB
# peak); use BATCH_TOKENS=64000 on a 40 GB card.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PY="${PY:-python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-data/longbench}"
OUT="${OUT:-outputs/completion_ablation}"
STAGE="${STAGE:-core}"                 # core | extended | collect
BUDGET=0.10; GATE=0.25; STRESS_GATE=0.10
TASKS=(gov_report qmsum multi_news)
NUM_SAMPLES="${NUM_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BATCH_TOKENS="${BATCH_TOKENS:-212000}"
RECORD_STRIDE="${RECORD_STRIDE:-8}"
SIGMA="${SIGMA:-}"
SKIP_OOM="${SKIP_OOM:-false}"

case "$STAGE" in
  core)     DEFAULT_ARMS="cal fullkv off uncal scale2 cal_repeat cal_ep" ;;
  extended) DEFAULT_ARMS="ceiling_g1.0 stress_g0.10_cal stress_g0.10_off scale0.5 scale0.8 scale1.25 scale4 coupled_scale0.5 coupled_scale2 rec_g0.25 rec_g0.10" ;;
  collect)  DEFAULT_ARMS="" ;;
  *) echo "STAGE must be core, extended or collect" >&2; exit 2 ;;
esac
ARMS="${ARMS:-$DEFAULT_ARMS}"

"$PY" scripts/fetch_longbench.py --out "$LONGBENCH_LOCAL_DIR" "${TASKS[@]}"

# ---- Experiment 1: every variant on the same decode states -------------------------
if [ "$STAGE" = collect ]; then
  declare -A PREFILL=([gov_report]=6144 [qmsum]=6144 [multi_news]=2048)
  for g in 0.25 0.10; do
    for t in "${TASKS[@]}"; do
      zp="$OUT/exp1/g${g}/${t}_p${PREFILL[$t]}_n128_k4.zip"
      if [ -f "$zp" ]; then echo "[exp1 g$g $t] exists -- skipping"; continue; fi
      mkdir -p "$(dirname "$zp")"
      bash scripts/collect_observations.sh --model-path "$MODEL" --dataset "$t" \
          --window-size 8 --cache-budget "$BUDGET" --gate-ratio "$g" --quant-ratio 0.70 \
          --quant-budget-mode bytes --prefill "${PREFILL[$t]}" --gen 128 --num-samples 4 \
          --completion-rungs --out "$zp"
    done
  done
  echo "Experiment 1 zips: $OUT/exp1 (summarise with scripts/analyze_observation_zip.py)"
  exit 0
fi

# ---- the LongBench arms -------------------------------------------------------------
COMMON=(model.name="$MODEL" "longbench.datasets=[gov_report,qmsum,multi_news]"
        longbench.num_samples="$NUM_SAMPLES" longbench.resume=false longbench.skip_oom="$SKIP_OOM"
        longbench.batch_size="$BATCH_SIZE" longbench.batch_tokens="$BATCH_TOKENS"
        longbench.pool_datasets=true longbench.aggressive_cache_clear=true
        longbench.prompt_tokens.gov_report=6144 longbench.prompt_tokens.qmsum=6144
        longbench.prompt_tokens.multi_news=2048)

completion() {  # mode scale sigma seed scores
  echo "cache.quant_completion=$1 cache.quant_completion_scale=$2 cache.quant_completion_sigma=$3 cache.quant_completion_seed=$4 cache.quant_completion_scores=$5"
}
arm_overrides() {  # echoes the arm's cache overrides; "FULLKV" for the full-cache arm
  local a="$1" gate="$GATE" comp=""
  case "$a" in
    fullkv)            echo FULLKV; return ;;
    cal|cal_repeat)    ;;
    cal_ep)            comp="$(completion calibrated 1.0 0.0 0 output)" ;;
    off)               comp="$(completion off 1.0 0.0 0 output)" ;;
    uncal)             comp="$(completion uncalibrated 1.0 0.0 0 output)" ;;
    ceiling_g1.0)      gate=1.0 ;;
    stress_g0.10_cal)  gate=$STRESS_GATE ;;
    stress_g0.10_off)  gate=$STRESS_GATE; comp="$(completion off 1.0 0.0 0 output)" ;;
    rec_g0.25)         comp="longbench.completion_rungs=true longbench.completion_rungs_stride=$RECORD_STRIDE" ;;
    rec_g0.10)         gate=$STRESS_GATE
                       comp="longbench.completion_rungs=true longbench.completion_rungs_stride=$RECORD_STRIDE" ;;
    coupled_scale*)    comp="$(completion calibrated "${a#coupled_scale}" 0.0 0 coupled)" ;;
    scale*)            comp="$(completion calibrated "${a#scale}" 0.0 0 output)" ;;
    noise*x_s*)        [ -n "$SIGMA" ] || { echo "arm $a needs SIGMA (sigma_hat_U from Experiment 1)" >&2; exit 2; }
                       local m="${a#noise}"; m="${m%%x_s*}"; local s="${a##*_s}"
                       comp="$(completion calibrated 1.0 "$(awk -v m="$m" -v s="$SIGMA" 'BEGIN{print m*s}')" "$s" output)" ;;
    *) echo "unknown arm: $a" >&2; exit 2 ;;
  esac
  echo "cache.cache_budget=$BUDGET cache.quant_gate_ratio=$gate $comp"
}

for a in $ARMS; do
  dir="$OUT/$a"
  done_all=1
  for t in "${TASKS[@]}"; do [ -f "$dir/$t.meta.json" ] || done_all=0; done
  if [ "$done_all" = 1 ]; then echo "[$a] done -- skipping"; continue; fi
  ov="$(arm_overrides "$a")"
  mkdir -p "$dir"
  if [ "$ov" = FULLKV ]; then
    cfg=configs/llama/longbench_full_cache.yaml; extra=()
  else
    cfg=configs/llama/longbench_ours_flash_attn.yaml; read -r -a extra <<< "$ov"
  fi
  echo "[$a] $cfg ${extra[*]}"
  "$PY" main.py --config "$cfg" --override "${COMMON[@]}" "${extra[@]}" \
      longbench.output_dir="$dir" 2>&1 | tee "$dir/run.log"
  "$PY" -m modules.evaluation.longbench_scoring --predictions_dir "$dir" --out_csv "$dir/scores.csv"
done

# Every compressed arm's sidecars must prove a gated run.
GATE_DIRS=()
for a in $ARMS; do [ "$a" = fullkv ] || GATE_DIRS+=("$OUT/$a"); done
STATUS=0
if [ "${#GATE_DIRS[@]}" -gt 0 ]; then "$PY" scripts/check_read_gate.py "${GATE_DIRS[@]}" || STATUS=$?; fi

if [ -f "$OUT/fullkv/scores.csv" ]; then
  VARIANTS=()
  for a in $ARMS; do [ "$a" != fullkv ] && [ -f "$OUT/$a/scores.csv" ] && VARIANTS+=("$OUT/$a/scores.csv"); done
  if [ "${#VARIANTS[@]}" -gt 0 ]; then
    "$PY" -m modules.evaluation.longbench_scoring --out "$OUT/comparison.csv" \
        --baseline "$OUT/fullkv/scores.csv" --variants "${VARIANTS[@]}"
  fi
fi
exit "$STATUS"
