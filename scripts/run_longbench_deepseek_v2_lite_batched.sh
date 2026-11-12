#!/usr/bin/env bash
# scripts/run_longbench_deepseek_v2_lite_batched.sh
#
# ============================================================================
# DeepSeek-V2-Lite-Chat on LongBench at a 10% KV budget, under the protocol and
# batching of the DeepSeek-V2 SnapKV / KVQuant baselines (DefensiveKV's batched
# kvpress script), so the rows compare with theirs.
# ============================================================================
#   bash scripts/run_longbench_deepseek_v2_lite_batched.sh
#   TASKS="qasper trec" bash scripts/run_longbench_deepseek_v2_lite_batched.sh
#   MODEL=/path/to/DeepSeek-V2-Lite-Chat bash scripts/run_longbench_deepseek_v2_lite_batched.sh
#   LIMIT=8 bash scripts/run_longbench_deepseek_v2_lite_batched.sh          # smoke run
#
# * Protocol: DefensiveKV's context / question templates and max_new_tokens,
#   its kvpress `preprocess`, and its `calculate_metrics`, all read from a
#   checkout of DefensiveKV pinned below (cloned into $DKV_DIR if missing). The
#   context is prefilled and compressed, the question appended, the answer
#   decoded greedily. Harness: modules/evaluation/longbench_qevict_batched.py.
# * Batching (the baseline's defaults): longest contexts first, at most 8 rows
#   and 140,000 padded prefill tokens per batch, no truncation. Each row has its
#   own cache (models/deepseek_v2/row_batch.py).
# * Method: configs/deepseek/longbench_ours_flash_attn.yaml (the operating point:
#   window 8, sink 5, local 128, quant_ratio 0.70 in bytes, read ratio 0.25) at
#   cache_budget = BUDGET x (context + question + max_new_tokens) per row.
#
# One task per process, as the baseline script runs. A finished task (its
# <task>_qevict_b<budget>_bs<bs>.meta.json exists) is skipped, so re-running
# resumes. Every sidecar must report read_gate "gated" and every example.
# Needs one 80 GB GPU (A100), flash-attn and Triton.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PY="${PY:-python}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
REVISION="${REVISION:-85864749cd611b4353ce1decdb286193298f64c7}"   # models.deepseek_v2.loader.PINNED_REVISIONS
CONFIG="${CONFIG:-configs/deepseek/longbench_ours_flash_attn.yaml}"
BUDGET="${BUDGET:-0.10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-140000}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-0}"     # 0 = no truncation
LIMIT="${LIMIT:-0}"                             # 0 = every example
LB_DIR="${LONGBENCH_LOCAL_DIR:-data/longbench}"
OUT="${OUT:-outputs/longbench_deepseek_v2_lite/qevict_b${BUDGET}}"
DKV_DIR="${DKV_DIR:-third_party/DefensiveKV}"
DKV_URL="https://github.com/FFY0/DefensiveKV.git"
DKV_COMMIT="9251b95e7e282a6c18dd98ef6318dd489e639a58"
# The paper's LongBench table, without NarrativeQA.
TASKS="${TASKS:-qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum}"

# ---- DefensiveKV at the pinned commit (the protocol is read from it) ----------
if [ ! -d "$DKV_DIR/.git" ]; then
  mkdir -p "$(dirname "$DKV_DIR")"
  git clone -q "$DKV_URL" "$DKV_DIR"
fi
git -C "$DKV_DIR" checkout -q --force "$DKV_COMMIT"
echo "DefensiveKV $(git -C "$DKV_DIR" rev-parse HEAD)"

# ---- LongBench files ------------------------------------------------------------
# shellcheck disable=SC2086
"$PY" scripts/fetch_longbench.py --out "$LB_DIR" $TASKS

# ---- one process per task ---------------------------------------------------------
mkdir -p "$OUT"
for t in $TASKS; do
  echo "=== $t"
  "$PY" -m modules.evaluation.longbench_qevict_batched \
      --model "$MODEL" --revision "$REVISION" --task "$t" \
      --data_dir "$LB_DIR" --defensivekv_dir "$DKV_DIR" --config "$CONFIG" \
      --cache_budget "$BUDGET" --batch_size "$BATCH_SIZE" \
      --max_batch_tokens "$MAX_BATCH_TOKENS" --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
      --limit "$LIMIT" --out_dir "$OUT" 2>&1 | tee -a "$OUT/run.log"
done

# ---- verify every sidecar and print the table ---------------------------------------
OUT="$OUT" LB_DIR="$LB_DIR" TASKS="$TASKS" LIMIT="$LIMIT" BUDGET="$BUDGET" \
BATCH_SIZE="$BATCH_SIZE" MAX_PROMPT_TOKENS="$MAX_PROMPT_TOKENS" "$PY" - <<'PY'
import json, os, statistics, sys
from pathlib import Path
out, lb = Path(os.environ["OUT"]), Path(os.environ["LB_DIR"])
tasks, limit = os.environ["TASKS"].split(), int(os.environ["LIMIT"])
# The runner's own file stem (longbench_qevict_batched.main), float formatting included.
budget, bs, trunc = float(os.environ["BUDGET"]), int(os.environ["BATCH_SIZE"]), int(os.environ["MAX_PROMPT_TOKENS"])
suffix = f"_qevict_b{budget}_bs{bs}" + (f"_trunc{trunc}" if trunc else "")
bad, scores = [], []
for t in tasks:
    meta = json.loads((out / f"{t}{suffix}.meta.json").read_text())
    n_all = sum(1 for line in open(lb / f"{t}.jsonl", encoding="utf-8") if line.strip())
    want = min(limit, n_all) if limit else n_all
    verdict = (meta.get("read_gate") or {}).get("verdict")
    if verdict != "gated" or meta["n"] != want:
        bad.append(f"{t}: read_gate={verdict!r}, n={meta['n']} (expected 'gated', {want})")
    scores.append(meta["score"])
    print(f"{t:16s} n={meta['n']:4d}  score {meta['score']:6.2f}  read_gate={verdict}  "
          f"{meta['n_batches']} batches  {meta['seconds'] / 60:.0f} min  peak {meta['peak_gb']} GB")
print(f"\naverage over {len(tasks)} tasks: {statistics.mean(scores):.2f}")
if bad:
    print("\nREFUSED:\n  " + "\n  ".join(bad))
    sys.exit(1)
PY
