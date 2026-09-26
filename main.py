"""RecoverKV -- the single entry point for every experiment in this repository.

    python main.py --config <configs/MODEL/SUITE.yaml> [--override key=value ...]

A YAML config names the suite to run (``run.mode``) and everything it needs.
This file maps that mode to its runner (``SUITES`` below), loads the config,
applies the overrides, seeds everything and runs. It holds no experiment logic.
Every script in ``scripts/`` is a thin wrapper around this command.

Suites (``run.mode`` -> config -> what it produces)
--------------------------------------------------
Accuracy
  longbench        configs/{llama,mistral}/longbench_{full_cache,ours_flash_attn}.yaml
                   16 English LongBench datasets -> outputs/longbench/<run>/<dataset>.jsonl
  longbench_score  any LongBench config + --override run.mode=longbench_score
                   scores every run under outputs/longbench/ -> <run>/scores.csv, comparison.csv
  ruler            configs/llama/ruler_{full_cache,ours_flash_attn}.yaml (set ruler.data_dir)
                   13 RULER tasks at one context length -> outputs/ruler/<run>/<task>.jsonl
  ruler_score      any RULER config + --override run.mode=ruler_score
  gsm8k            configs/llama/gsm8k_{full_cache,budget20,budget80}.yaml
                   8-shot chain-of-thought GSM8K -> outputs/gsm8k/<run>/predictions.jsonl
  gsm8k_score      any GSM8K config + --override run.mode=gsm8k_score

Speed and memory
  perf             configs/llama/eval_perf.yaml          latency factorial, batch 1
                   configs/llama/eval_perf_batched.yaml  throughput and max batch size
                   configs/llama/eval_efficiency.yaml    prompt-compression protocol (24k-40k)
                   scripts/run_perf_table.sh             the decode table (generates its config)
                   -> perf_prefill{P}_gen{G}_bs{B}.npz; print with scripts/print_perf_*.py

Analysis (teacher-forced parity runs, then post-processing without a model)
  parity_base          configs/llama/eval_parity_base.yaml        FullKV reference decode
  parity_ours          configs/llama/eval_parity_ours_flash.yaml
  faithfulness         configs/llama/eval_faithfulness.yaml
  qevict_observations  configs/llama/eval_qevict.yaml
  tier_study           configs/llama/eval_tier_study.yaml
  visualize            configs/llama/eval_visualize.yaml

Changing the configuration
--------------------------
1. Pick the model by picking the config directory: ``configs/llama/`` or
   ``configs/mistral/``. What differs between models (checkpoint, LongBench
   truncation length, and the three attention-interface adaptations) is listed
   in ``models/__init__.py``.
2. Edit the YAML, or override any field from the command line with dotted keys.
   Values are parsed as int / float / bool / None / list, else string; lists
   use brackets. ``--override`` may be repeated; later values win.

       python main.py --config configs/llama/longbench_ours_flash_attn.yaml \\
           --override cache.cache_budget=0.10 longbench.datasets=[qasper,trec] \\
                      longbench.output_dir=outputs/longbench/ours_b010
       python main.py --config configs/llama/longbench_ours_flash_attn.yaml \\
           --override model.name=/path/to/local/Llama-3.1-8B-Instruct

3. The fields that define the method live in the ``cache:`` block:

       cache_budget        fraction of the full KV cache kept (0.20)
       window_size         eviction granularity in tokens; a multiple of 4 when
                           quant_ratio > 0 (8)
       num_sink_tokens     leading tokens kept in fp16, never evicted (5)
       local_window_size   most recent tokens kept in fp16, never evicted (128)
       quant_ratio         share of the evictable budget held in the int2 tier;
                           0 = fp16 tier only (0.70)
       quant_budget_mode   bytes: quant_ratio splits the byte budget (memory-
                           matched); tokens: it splits the window count (bytes)
       quant_gate_ratio    read gate: fraction of int2 windows dequantized per
                           decode step; 1.0 opens every window (0.25)
       first_eviction_step decode step of the first eviction; 0 compacts the
                           prompt before the second token (0)
       backend_package     flash_attn, paired with model.attn_implementation
                           flash_attention_2

   Values in brackets are the operating point every headline number uses
   (``utils.config.OPERATING_POINT``). The parity and perf suites read the same
   geometry from the ``window:`` block, and the LongBench / RULER runners warn
   when ``cache.*`` and ``window.*`` disagree, so change both together.

4. Every field of every block, with its default, is in the dataclasses of
   ``utils/config.py``. ``configs/base.yaml`` holds the shared defaults each
   config inherits through ``_base_``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Dict, List

from utils.config import load_config
from utils.logger import get_logger
from utils.seed import seed_everything

log = get_logger("main")

#: run.mode -> "module.path.RunnerClass". Each runner takes the loaded
#: ExperimentConfig in its constructor and does all of its work in run().
SUITES: Dict[str, str] = {
    # accuracy
    "longbench": "modules.evaluation.longbench_runner.LongBenchRunner",
    "longbench_score": "modules.evaluation.longbench_scoring.LongBenchScorer",
    "ruler": "modules.evaluation.ruler_runner.RulerRunner",
    "ruler_score": "modules.evaluation.ruler_scoring.RulerScorer",
    "gsm8k": "modules.evaluation.gsm8k_runner.GSM8KRunner",
    "gsm8k_score": "modules.evaluation.gsm8k_scoring.GSM8KScorer",
    # speed and memory
    "perf": "modules.evaluation.perf_runner.PerfRunner",
    # analysis
    "parity_base": "modules.evaluation.base_parity_runner.BaseParityRunner",
    "parity_ours": "modules.evaluation.ours_parity_runner.OursParityRunner",
    "faithfulness": "modules.evaluation.faithfulness_runner.FaithfulnessRunner",
    "qevict_observations": "modules.evaluation.qevict_observations.QEvictObservationRunner",
    "tier_study": "modules.evaluation.tier_study.combined.TierStudyRunner",
    "visualize": "modules.evaluation.visualize.VisualizeRunner",
}


def import_runner(mode: str):
    """The runner class registered for ``mode``; exits with the list if unknown."""
    if mode not in SUITES:
        log.error("Unknown run.mode %r. Available: %s", mode, ", ".join(sorted(SUITES)))
        sys.exit(1)
    module_path, class_name = SUITES[mode].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def parse_value(val: str) -> Any:
    """Parse one override value: ``[a,b]`` list, bool, None, int, float, else str.

    Lists need brackets, so a string that contains a comma is never split.
    """
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        return [parse_value(item) for item in inner.split(",")] if inner else []
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


def parse_overrides(items: List[str]) -> Dict[str, Any]:
    """``["a.b=1", "c=x"]`` -> ``{"a": {"b": 1}, "c": "x"}``."""
    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            log.error("Override must be key=value, got: %r", item)
            sys.exit(1)
        key, val = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = overrides
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = parse_value(val)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RecoverKV experiments. See the module docstring for every suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="YAML config, e.g. configs/llama/longbench_ours_flash_attn.yaml")
    parser.add_argument(
        "--override", nargs="*", action="extend", default=[],
        help="key=value overrides with dotted keys, e.g. cache.cache_budget=0.10 "
             "longbench.datasets=[qasper]. May be repeated.")
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides=overrides or None)

    seed_everything(config.run.seed)
    log.info("Seed set to %d", config.run.seed)

    runner_cls = import_runner(config.run.mode)
    log.info("Running mode: %s", config.run.mode)
    runner_cls(config).run()


if __name__ == "__main__":
    main()
