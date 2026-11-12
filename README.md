# RecoverKV: a two-tier evicting KV cache with a rank-1 read gate

RecoverKV keeps a small fp16 tier of the KV cache exactly, holds a larger int2
tier approximately, and on each decode step dequantizes only the int2 windows a
per-window summary ("card") says the current query needs. The goal is more batch
at the same memory, with decode latency at or below a full cache.

This repository contains the implementation (a HuggingFace `Cache` subclass,
score hooks and Triton kernels) and every experiment behind the paper, driven
from one entry point, `main.py`.

Supported models: **Llama-3.1-8B-Instruct** / Llama-3-8B-Instruct,
**Mistral-7B-Instruct-v0.2** and **DeepSeek-V2-Lite-Chat** (multi-head latent
attention; see [Models](#models)).

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
signatures and refuses other versions at start-up. The recorded runs used
torch 2.6.0 (cu124), triton 3.2.0, transformers 4.47.1 and flash-attn 2.7.4;
on torch 2.7.1 the compiled eviction fails at decode step 0, so stay on 2.6.

DeepSeek-V2-Lite-Chat needs nothing extra: its remote code is loaded with
`trust_remote_code` at a pinned revision, and a copy of that code is vendored
under `models/deepseek_v2/upstream/` as the reference the owned attention
forward is checked against. The LongGenBench judge (vLLM) runs in a separate
environment; see [LongGenBench](#longgenbench).

## Data

| benchmark | source |
|---|---|
| LongBench | `THUDM/LongBench` from the HF hub (needs `datasets<3`), or `<dataset>.jsonl` files in `data/longbench/` (or `$LONGBENCH_LOCAL_DIR`); `python scripts/fetch_longbench.py` stages them |
| RULER | pre-generated splits saved with `datasets.save_to_disk`, one directory per context length (e.g. the `ruler/32768` folder of the DefensiveKV release); pass it as `ruler.data_dir` |
| GSM8K | built by `scripts/run_gsm8k_e2e.sh` into `data/gsm8k_cot/` |
| WikiText-103 | from the HF hub (parity / analysis suites and the perf prompts) |
| LongGenBench | `mozhu/LongGenBench` at a pinned revision, SHA-256 checked: `python -m longgenbench_eval.data --out data/longgenbench` |

## Quick start

```bash
# LongBench, RecoverKV at the operating point, one dataset
python main.py --config configs/llama/longbench_ours_flash_attn.yaml \
    --override longbench.datasets=[qasper]

# the same on Mistral
python main.py --config configs/mistral/longbench_ours_flash_attn.yaml \
    --override longbench.datasets=[qasper]

# and on DeepSeek-V2-Lite-Chat
python main.py --config configs/deepseek/longbench_ours_flash_attn.yaml \
    --override longbench.datasets=[qasper]
```

`main.py` is the entry point for every suite; its docstring lists each suite,
its config, and how to change configuration. The scripts under `scripts/` wrap
it (the few that run their own module say so in their header).

## Reproducing the experiments

All commands run from the repository root on one GPU. Outputs go to `outputs/`.

**LongBench** (16 English datasets; FullKV and RecoverKV, every model)
```bash
scripts/run_longbench.sh llama    full_cache
scripts/run_longbench.sh llama    ours_flash_attn
scripts/run_longbench.sh mistral  full_cache
scripts/run_longbench.sh mistral  ours_flash_attn
scripts/run_longbench.sh deepseek full_cache
scripts/run_longbench.sh deepseek ours_flash_attn
scripts/score_longbench.sh          # comparison_{llama,mistral,deepseek}.csv
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
FAMILY=deepseek ARMS=budget20 BASELINE_MIN=60 scripts/run_gsm8k_e2e.sh   # DeepSeek-V2-Lite-Chat
```

**DeepSeek-V2-Lite-Chat** (MLA; the cache holds the compressed latent)
```bash
python scripts/deepseek_gpu_preflight.py --model-dir /path/to/DeepSeek-V2-Lite-Chat --budget 0.10
scripts/run_longbench_deepseek_v2_lite_batched.sh   # 10% budget, the SnapKV/KVQuant baselines' protocol
scripts/run_longbench_deepseek_v2_lite.sh           # FullKV + RecoverKV through main.py, scored
```
The preflight checks the GPU-only paths (FA2 prefill, the score kernel, the
fused MLA decode, the gate at the run's ratio, the compiled eviction) on a
small model with V2-Lite's real attention geometry, against the CPU reference.
The batched script reads the LongBench protocol (prompt templates,
preprocessing, metrics) from a pinned checkout of DefensiveKV, which it clones
into `third_party/`, so its rows compare with those baselines' published ones.

<a id="longgenbench"></a>
**LongGenBench-16K** (long generation: up to 16,000 new tokens per document)
```bash
scripts/run_longgenbench.sh                                   # RecoverKV at budget 0.10
ARMS="fullkv ours control" scripts/run_longgenbench.sh        # + full KV, + the gate's control arm
# judge (Llama-3.1-8B-Instruct on vLLM) in its own environment, then score:
uv venv --python 3.12 judge_env
uv pip install --python judge_env/bin/python -r longgenbench_eval/requirements-judge.lock.txt
JUDGE_PY=judge_env/bin/python scripts/run_longgenbench.sh
```
`longgenbench_eval/README.md` lists what each step reproduces from the
benchmark's own code.

**Decode throughput and memory**
```bash
scripts/run_perf_table.sh --model meta-llama/Llama-3.1-8B-Instruct   # decode table, RecoverKV
scripts/run_perf_table.sh --model meta-llama/Llama-3.1-8B-Instruct --gate-ratio 1.0   # gate control
scripts/run_perf_sweep.sh --model meta-llama/Llama-3.1-8B-Instruct --dry-run   # knob sweep, one dir per config
scripts/run_perf_batched.sh      # FullKV vs RecoverKV, batch ladder -> max batch size
scripts/run_efficiency.sh        # prompt-compression protocol, 24k-40k context, batch 1
scripts/run_perf.sh              # latency factorial, batch 1
```

**Ablations**

*Completion term.* How much quality depends on the calibrated window-level
completion that stands in for the int2 windows the gate does not open
(`cache.quant_completion`: `off`, `uncalibrated`, or `calibrated` with a
common-scale error and/or per-window noise), on LongBench's three 512-token
generation tasks at budget 0.10, against full KV on the same prompts:
```bash
scripts/run_completion_ablation.sh                    # cal, fullkv, off, uncal, scale2, run-to-run floors
STAGE=extended scripts/run_completion_ablation.sh     # read ratio 1.0 / 0.10, scales, coupled scores
STAGE=collect scripts/run_completion_ablation.sh      # every variant on the same decode states
```
*Read ratio, card width, promotion.* One queue over all GPUs runs the
observation suite and three accuracy experiments: read ratios
(`quant_gate_ratio`), bidirectional vs one-way promotion
(`quant_promotion: oneway`), and promotion from the exact fp window
(`quant_promote_source: original`, an evaluation-only oracle held outside the
budget), then writes one report:
```bash
DRY_RUN=1 scripts/run_recoverkv_all.sh                # the job list
SMOKE=1 scripts/run_recoverkv_all.sh                  # a trimmed grid (~1 h)
scripts/run_recoverkv_all.sh                          # SUITES="obs e1 e2 e3"
```
It reads LongBench from `$LONGBENCH_LOCAL_DIR` (default `data/longbench/`, see
`scripts/fetch_longbench.py`) and RULER 32k from `$RULER_DATA` (default
`data/ruler/32768`), and uses every GPU in `CUDA_VISIBLE_DEVICES`.
The card's field widths are `cache.quant_card_bits` (`null` = int4 `mu`/`vbar`,
int8 `v`/`t`; or one width for all fields, e.g. `4`, or per field,
`mu=4,v=4,t=2,vm=2`). Under `quant_budget_mode: bytes` a narrower card buys
more int2 windows, so it is a quality change; every sidecar records it.
Per-step observation data for one run (the arrays behind Observations I-V):
```bash
scripts/collect_observations.sh --model-path meta-llama/Llama-3.1-8B-Instruct \
    --dataset hotpotqa --window-size 8 --cache-budget 0.20 --gate-ratio 0.25 --quant-ratio 0.70
python scripts/analyze_observation_zip.py outputs/obs_data/<run>.zip --out outputs/<run>.md
```

Every quality sidecar records the read gate's verdict (`read_gate`): `gated`
for a RecoverKV run, `not-expected` for full KV. `python scripts/check_read_gate.py
<output_dir> ...` refuses a run directory whose sidecars do not prove it.

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

## Tests

```bash
python -m pytest tests            # CPU only, no model download; add -n 4 with pytest-xdist
```

The suite checks the method's arithmetic, not GPU speed. Every Triton kernel
has a CPU reference that mirrors its tiling and masking, and the tests pin the
kernels to those references: under `TRITON_INTERPRET=1` when Triton is
installed, otherwise against the references directly. The cache, eviction,
read gate, card widths, promotion and completion knobs, the DeepSeek-V2 latent
port and the runners' batching are all exercised end to end on small random
models. Tests that need CUDA or flash-attn skip without them; on a GPU,
`python scripts/deepseek_gpu_preflight.py` checks the GPU-only paths.
`tests/conftest.py` sets two CPU-only options and says why; neither touches a
GPU run.

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

Ablation knobs (off at their defaults; each one is recorded in every sidecar):

| key | default | meaning |
|---|---|---|
| `quant_card_bits` | null | card field widths (8, 4 or 2 bits each); null = `mu=4,v=8,t=8,vm=4` |
| `quant_promotion` | bidir | `oneway`: no int2 -> fp16 promotion |
| `quant_promote_source` | dequant | `original`: promote the exact fp16 window (evaluation-only oracle) |
| `quant_completion` | null | `off` / `uncalibrated` / `calibrated`: replace the completion term for unread int2 windows |
| `quant_completion_scale`, `_sigma`, `_seed` | 1.0, 0.0, 0 | calibrated only: common-scale error, per-window noise (nats) |
| `quant_completion_scores` | output | `coupled`: the variant also feeds the eviction scores |

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

**DeepSeek-V2-Lite-Chat** (`configs/deepseek/`, `models/deepseek_v2/`) uses
multi-head latent attention, so its KV cache is a compressed latent rather than
per-head keys and values. The port maps that latent onto the same cache, read
gate and eviction; the method and the operating point are unchanged:

| | DeepSeek-V2-Lite-Chat |
|---|---|
| loading | `trust_remote_code` at a pinned revision (`models.deepseek_v2.loader.PINNED_REVISIONS`), bf16 |
| attention | an owned MLA forward replaces the remote code's on every layer (`models/deepseek_v2/attention.py`); it writes the eviction scores itself, so no score hooks are installed |
| cached tensor | the 576-wide latent (512 compressed + 64 RoPE lanes) as one KV head whose value is its latent lanes: 1,152 B/token per layer at bf16 |
| RoPE | at absolute positions on the 64 RoPE lanes only (`PackedRope`) |
| decode | the same gate, then `modules/windowed_cache/mla_decode_kernel.py` (absorbed attention over the latent) |
| full-KV baseline | the same latent in a `DynamicCache`, decoded with the same absorbed math |
| LongBench truncation | 31,500 tokens (32K context) |
| padded batches | one cache per row (`models/deepseek_v2/row_batch.py`) |

## Repository layout

```
main.py                     entry point: run.mode -> runner (every suite)
configs/
  base.yaml                 shared defaults
  llama/                    Llama-3.x-8B-Instruct: LongBench, RULER, GSM8K, LongGenBench, perf, parity/analysis
  mistral/                  Mistral-7B-Instruct-v0.2: LongBench
  deepseek/                 DeepSeek-V2-Lite-Chat: LongBench, GSM8K
models/                     per-model registry (attention classes, adaptations)
  deepseek_v2/              MLA port: latent geometry, RoPE, owned attention, loader, per-row batching
    upstream/               the pinned DeepSeek-V2 remote code, vendored as the reference
modules/
  windowed_cache/           the cache: eviction policy, score hooks, Triton kernels
    cache.py                Cache.update(), layer-major eviction, fused decode hand-off
    decode_kernel.py        fused two-tier decode (fp tier + gated int2 tier) + CPU reference
    gate_kernel.py          card scan and per-KV-head selection
    score_kernel.py         fused prefill scoring
    flash_decode.py         the flash_attn_func seam where the gate runs
    mla_decode_kernel.py    the fused decode for an MLA latent (DeepSeek-V2)
    completion.py           the completion-term ablation's epilogue (off unless quant_completion is set)
    policy.py state.py scorer.py config.py hooks.py
  quant/                    the int2 tier: quantizer, slot table, cards (sketch.py)
  evaluation/               one runner per suite, scoring, metrics
data/                       dataset loaders; LongBench prompt/metric configs
longgenbench_eval/          LongGenBench data, vLLM judge and scoring (self-contained)
utils/                      config schema, cache factory, memory probe, logging
scripts/                    per-suite wrappers around main.py and table printers
tests/                      CPU test suite (python -m pytest tests)
```

## Citation

Withheld for anonymous review.
