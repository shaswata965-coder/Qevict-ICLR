# RecoverKV: a two-tier evicting KV cache with a rank-1 read gate

RecoverKV keeps a small fp16 tier of the KV cache exactly, holds a larger int2
tier approximately, and on each decode step dequantizes only the int2 windows a
per-window summary ("card") says the current query needs. The goal is more batch
at the same memory, with decode latency at or below a full cache.

This repository contains the implementation (a HuggingFace `Cache` subclass,
score hooks and Triton kernels) and every experiment behind the paper, driven
from one entry point, `main.py`.

Supported models: **Llama-3.1-8B-Instruct** / Llama-3-8B-Instruct and
**Mistral-7B-Instruct-v0.2** (see [Models](#models)).

---

## Method

The context is cut into **windows** of `window_size` tokens. Each window carries
a cumulative attention score, and at every eviction the windows are ranked and
split into bands:

| band | storage | rule |
|---|---|---|
| sink + local | fp16, never evicted | first `num_sink_tokens`, last `local_window_size` tokens |
| fp tier | fp16, exact | top evictable windows by score |
| int2 tier | 2-bit codes, one-byte grid, and a card | the next windows |
| evicted | none | everything below |

`cache_budget` is the fraction of the full cache kept, and `quant_ratio` splits
the evictable part of it between the fp16 and int2 tiers. Under
`quant_budget_mode: bytes` the split is by bytes, so the cache costs
`cache_budget` of the full cache at every `quant_ratio`, and the cheaper int2
windows buy more retained context. Eviction runs every `window_size` decode
steps for all layers in one batched, compiled pass. Positions are never
renumbered, so a demoted window's encoding is written once and frozen.

**The card.** Every int2 window gets a card when it is demoted: a rank-1 model of
its post-RoPE keys, `k_i ≈ mu + t_i · v` (with `v` the direction of the token
furthest from the mean), plus the mean of its values `vbar`. `mu` and `vbar` are
int4, and `v` and `t` are int8, each with an fp16 scale: 272 B per KV head.
`mu`/`vbar` are stored as residuals from a frozen per-(layer, head) anchor.

**One decode step.**
1. *Score every card*: two `D`-length dot products per (window, head) estimate
   the window's attention mass for each query head without touching its codes.
2. *Select*, per KV head, the `quant_gate_ratio` fraction of int2 windows with
   the largest share of their own int2 mass under any query head of the GQA
   group (shares are normalised per head before the max).
3. *Attend* over `[sink | fp tier | selected int2 windows]` in one fused Triton
   kernel. The int2 unpack, dequantization and RoPE all happen in registers.
4. *Credit the skipped windows*: the read windows measure how far the cards were
   off for this query (the mean log ratio of exact to estimated mass), and each
   skipped window is credited with its corrected estimate. The credit goes both
   to its eviction score and to the output through its value centroid.

`quant_gate_ratio = 1.0` opens every window through the same code path and is
the control arm for the gate.

## Installation

```bash
conda env create -f environment.yml && conda activate recoverkv   # or: pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Requires a CUDA GPU (Ampere or newer; every run uses one A100-80GB) and
`transformers` 4.47.x. The cache targets the 4.47 attention
signatures and refuses other versions at start-up.

## Data

| benchmark | source |
|---|---|
| LongBench | `THUDM/LongBench` from the HF hub (needs `datasets<3`), or `<dataset>.jsonl` files in `data/longbench/` (or `$LONGBENCH_LOCAL_DIR`) |
| RULER | pre-generated splits saved with `datasets.save_to_disk`, one directory per context length (e.g. the `ruler/32768` folder of the DefensiveKV release); pass it as `ruler.data_dir` |
| GSM8K | built by `scripts/run_gsm8k_e2e.sh` into `data/gsm8k_cot/` |
| WikiText-103 | from the HF hub (parity / analysis suites and the perf prompts) |

## Quick start

```bash
# LongBench, RecoverKV at the operating point, one dataset
python main.py --config configs/llama/longbench_ours_flash_attn.yaml \
    --override longbench.datasets=[qasper]

# the same on Mistral
python main.py --config configs/mistral/longbench_ours_flash_attn.yaml \
    --override longbench.datasets=[qasper]
```

`main.py` is the only entry point; its docstring lists every suite, its config,
and how to change configuration. Every script under `scripts/` wraps it.

## Reproducing the experiments

All commands run from the repository root on one GPU. Outputs go to `outputs/`.

**LongBench** (16 English datasets; FullKV and RecoverKV, both models)
```bash
scripts/run_longbench.sh llama   full_cache
scripts/run_longbench.sh llama   ours_flash_attn
scripts/run_longbench.sh mistral full_cache
scripts/run_longbench.sh mistral ours_flash_attn
scripts/score_longbench.sh          # comparison_{llama,mistral}.csv
```
Budget sweep (10% / 5%), or one dataset per GPU:
```bash
scripts/run_longbench.sh llama ours_flash_attn --override cache.cache_budget=0.10 \
    longbench.output_dir=outputs/longbench/ours_b010
CUDA_VISIBLE_DEVICES=3 scripts/run_longbench.sh llama ours_flash_attn \
    --override longbench.datasets=[gov_report]
```

**RULER** (13 tasks at 32k)
```bash
DATA_DIR=/path/to/ruler/32768 scripts/run_ruler.sh
DATA_DIR=/path/to/ruler/32768 CONFIG=configs/llama/ruler_full_cache.yaml scripts/run_ruler.sh
# window-size ablation (set both blocks)
DATA_DIR=/path/to/ruler/32768 OUT_DIR=outputs/ruler/ws32 scripts/run_ruler.sh \
    --override cache.window_size=32 window.window_size=32
```

**GSM8K** (8-shot chain-of-thought; FullKV, 80% and 20% budgets, then the comparison)
```bash
scripts/run_gsm8k_e2e.sh
```

**Decode throughput and memory**
```bash
scripts/run_perf_table.sh --model meta-llama/Llama-3.1-8B-Instruct   # decode table, RecoverKV
scripts/run_perf_table.sh --model meta-llama/Llama-3.1-8B-Instruct --gate-ratio 1.0   # gate control
scripts/run_perf_sweep.sh --model meta-llama/Llama-3.1-8B-Instruct --dry-run   # knob sweep, one dir per config
scripts/run_perf_batched.sh      # FullKV vs RecoverKV, batch ladder -> max batch size
scripts/run_efficiency.sh        # prompt-compression protocol, 24k-40k context, batch 1
scripts/run_perf.sh              # latency factorial, batch 1
```

**Analysis** (teacher-forced parity runs, then post-processing)
```bash
scripts/run_parity_base.sh
scripts/run_parity_ours_flash.sh
scripts/run_faithfulness.sh
scripts/run_qevict_observations.sh
python scripts/run_tier_study_local.py --help
scripts/run_visualize.sh
```

### Reading the timing columns

The perf runner synchronizes the GPU before and after each timed region and
records every measured run; the table printers report the median by default:

| field | definition |
|---|---|
| `TTFT` | the prefill forward |
| `step0` | decode step 0. RecoverKV scores and compacts the whole prompt here, so compare `TTFT + step0` with a method that compresses inside prefill |
| `TPOT_steady` | mean latency of decode steps 1..n, eviction steps included |
| `dec tok/s` | `batch / TPOT_steady` |
| `e2e tok/s` | `batch * gen_len / (end of decode - start of prefill)` |

The `throughput_tokps` field kept in the npz files for backward compatibility
leaves out the prefill-to-decode interval; do not quote it.

## Configuration

A run is `python main.py --config <yaml> [--override key=value ...]`. Configs
inherit `configs/base.yaml`, and every field, with its default, is defined in
`utils/config.py`. The method is set in the `cache:` block:

| key | operating point | meaning |
|---|---|---|
| `cache_budget` | 0.20 | fraction of the full KV cache kept |
| `window_size` | 8 | tokens per eviction window (a multiple of 4 when `quant_ratio > 0`) |
| `num_sink_tokens` | 5 | leading tokens kept in fp16 |
| `local_window_size` | 128 | most recent tokens kept in fp16 |
| `quant_ratio` | 0.70 | share of the evictable budget in the int2 tier (0 = fp16 only) |
| `quant_budget_mode` | bytes | `bytes`: split the byte budget; `tokens`: split the window count |
| `quant_gate_ratio` | 0.25 | fraction of int2 windows dequantized per decode step |
| `first_eviction_step` | 0 | compact the prompt on the first decode step |
| `backend_package` | flash_attn | paired with `model.attn_implementation: flash_attention_2` |

The parity and perf suites read the window geometry from the `window:` block,
so set both blocks when changing it.

## Models

All models share one implementation. What differs between them is listed in
`models/__init__.py`, and each model has its own config directory:

| | Llama-3.1-8B-Instruct | Mistral-7B-Instruct-v0.2 |
|---|---|---|
| configs | `configs/llama/` | `configs/mistral/` |
| hooked attention class | `LlamaAttention` | `MistralAttention` |
| LongBench truncation | none (128K context) | 31,500 tokens (32K context) |
| RoPE for the score hook | `position_embeddings` argument | recomputed from `module.rotary_emb` |
| `cache_position` | passed by the model | derived from the cache's token count |
| K/V handed to flash | `H_kv` heads | repeated to `H_q` by the model, collapsed back to `H_kv` in the fused decode |

To add a model: add an entry to `models.MODEL_FAMILIES`, add
`configs/<model>/`, and check the three adaptations above against its attention
forward.

## Repository layout

```
main.py                     entry point: run.mode -> runner (every suite)
configs/
  base.yaml                 shared defaults
  llama/                    Llama-3.x-8B-Instruct: LongBench, RULER, GSM8K, perf, parity/analysis
  mistral/                  Mistral-7B-Instruct-v0.2: LongBench
models/                     per-model registry (attention classes, adaptations)
modules/
  windowed_cache/           the cache: eviction policy, score hooks, Triton kernels
    cache.py                Cache.update(), layer-major eviction, fused decode hand-off
    decode_kernel.py        fused two-tier decode (fp tier + gated int2 tier) + CPU reference
    gate_kernel.py          card scan and per-KV-head selection
    score_kernel.py         fused prefill scoring
    flash_decode.py         the flash_attn_func seam where the gate runs
    policy.py state.py scorer.py config.py hooks.py
  quant/                    the int2 tier: quantizer, slot table, cards (sketch.py)
  evaluation/               one runner per suite, scoring, metrics
data/                       dataset loaders; LongBench prompt/metric configs
utils/                      config schema, cache factory, memory probe, logging
scripts/                    per-suite wrappers around main.py and table printers
```

## Citation

Withheld for anonymous review.
