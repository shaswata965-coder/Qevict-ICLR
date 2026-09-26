#!/usr/bin/env bash
# Score every LongBench run under outputs/longbench/ and build one comparison
# table per model against that model's FullKV run.
#
#   scripts/score_longbench.sh
#
# Writes <run>/scores.csv for every run directory, then
#   outputs/longbench/comparison_llama.csv    full_cache         vs ours_*
#   outputs/longbench/comparison_mistral.csv  mistral_full_cache vs mistral_ours_*
# (run directories are the longbench.output_dir values in configs/<model>/).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONHASHSEED=0

ROOT=outputs/longbench

for run_dir in "$ROOT"/*/; do
    [ -d "$run_dir" ] || continue
    ls "$run_dir"*.jsonl >/dev/null 2>&1 || continue
    echo "Scoring: $run_dir"
    python -m modules.evaluation.longbench_scoring \
        --predictions_dir "$run_dir" --out_csv "${run_dir}scores.csv"
done

compare() {  # $1 = model tag, $2 = baseline dir, $3 = variant glob
    local base="$ROOT/$2/scores.csv"
    [ -f "$base" ] || { echo "(no $base; skipping the $1 comparison)"; return; }
    local variants=()
    for v in $ROOT/$3/scores.csv; do [ -f "$v" ] && variants+=("$v"); done
    [ ${#variants[@]} -gt 0 ] || { echo "(no $1 variants to compare)"; return; }
    echo "=== $1: $2 vs ${#variants[@]} run(s) ==="
    python -m modules.evaluation.longbench_scoring \
        --baseline "$base" --variants "${variants[@]}" \
        --out "$ROOT/comparison_$1.csv"
}
compare llama full_cache "ours_*"
compare mistral mistral_full_cache "mistral_ours_*"
