#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# One run -> one zip of per-step observation data.
# ============================================================================
#   scripts/collect_observations.sh \
#       --model-path meta-llama/Llama-3.1-8B-Instruct \
#       --dataset hotpotqa \
#       --window-size 8 --cache-budget 0.20 --gate-ratio 0.25 --quant-ratio 0.70
#
# Runs --num-samples prompts (default 8), each --prefill tokens of a document
# (default 512) plus --gen greedy decode steps (default 512) through the real
# cache, and writes
#   outputs/obs_data/<dataset>_ws<ws>_b<budget>_g<gate>_q<q>_p512_n512.zip
# (or --out PATH): meta.json, SCHEMA.md, sample_NNN.npz. Every array is per
# step and per layer; SCHEMA.md lists them and which observation each feeds.
#
# --dataset: a LongBench name (reads $LONGBENCH_LOCAL_DIR/<name>.jsonl),
# ruler:<task> (reads $RULER_DATA), wikitext-103, pg19, or a path. Each sample
# is one ARTICLE, and a .txt file is exactly one article -- a whole-corpus .txt
# yields one sample however large it is. Pass a .jsonl (one article per line)
# or a directory of .txt files for more.
# Other flags: --sink 5 --local 128 --quant-budget-mode bytes --card-bits
# --promotion bidir|oneway --promote-source dequant|original --token-heads
# --completion-rungs --article-index --seed.
#
# Summarise a zip (per-observation tables):
#   python scripts/analyze_observation_zip.py outputs/obs_data/<run>.zip --out outputs/<run>.md
#
# Turn a zip into the parity pair the observation suite reads:
#   python -m modules.evaluation.observation_collector export \
#       --zip outputs/obs_data/<run>.zip --out-dir outputs/obs_data/<run>
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

export LONGBENCH_LOCAL_DIR="${LONGBENCH_LOCAL_DIR:-$PWD/data/longbench}"   # <dataset>.jsonl
export RULER_DATA="${RULER_DATA:-$PWD/data/ruler/32768}"                 # RULER save_to_disk dir

python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("error: torch not importable -- activate the environment from environment.yml")
if not torch.cuda.is_available():
    sys.exit("error: no CUDA device -- the read gate only exists on the CUDA flash path.")
try:
    import flash_attn  # noqa: F401
except ImportError:
    sys.exit("error: flash_attn not importable; the gated path needs flash_attention_2.")
from utils.cache_factory import assert_transformers_version_supported
assert_transformers_version_supported()
PY

exec python -m modules.evaluation.observation_collector "$@"
