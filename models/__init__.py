"""Per-model support: what differs between the model families RecoverKV runs on.

Every model goes through ONE code path. The windowed cache replaces
``past_key_values``, the score hooks attach to each attention module, and the
fused decode kernel wraps ``flash_attn_func``. Nothing is re-implemented per
model. What does differ between families is small, and it is all listed here:

* ``attention_class`` -- the HuggingFace attention module the score hooks attach
  to (``modules/windowed_cache/hooks.py`` reads it through
  :func:`attention_classes`). A family missing from this table gets no hooks, so
  its cache would evict without scores; the hook installer raises instead.
  Empty for a family whose attention forward is *owned* by this repository
  (DeepSeek-V2, below): that forward writes the scores itself.
* ``configs`` -- the directory holding that model's experiment configs.
* ``adaptations`` -- the model-specific behaviour the shared path handles, and
  where. Each one is a no-op on Llama.

To add a model: add an entry here, add ``configs/<family>/`` with the model's
``model.name`` and ``longbench.max_length``, and check the three Mistral
adaptations below against its attention forward.

DeepSeek-V2 (multi-head latent attention) is the one family that is not a
GQA model, and the one that needed more than an adaptation: its KV cache is a
compressed latent, not per-head keys and values. ``models/deepseek_v2/`` maps
that latent onto the same cache, the same read gate and the same eviction
without changing the method; its entry below lists exactly what differs.

Tested with ``transformers>=4.47,<4.48`` (``utils/cache_factory.py`` refuses
other versions: the attention signatures below are specific to it).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelFamily:
    attention_class: str                    # "module.path:ClassName"
    checkpoints: Tuple[str, ...]            # HF hub ids the configs are written for
    configs: str                            # config directory for this family
    adaptations: Tuple[str, ...] = field(default_factory=tuple)


MODEL_FAMILIES: Dict[str, ModelFamily] = {
    "llama": ModelFamily(
        attention_class="transformers.models.llama.modeling_llama:LlamaAttention",
        checkpoints=(
            "meta-llama/Llama-3.1-8B-Instruct",       # LongBench / RULER / GSM8K / perf
            "meta-llama/Meta-Llama-3-8B-Instruct",    # parity / efficiency suites
        ),
        configs="configs/llama",
        adaptations=(),                               # the reference path
    ),
    "mistral": ModelFamily(
        attention_class="transformers.models.mistral.modeling_mistral:MistralAttention",
        checkpoints=("mistralai/Mistral-7B-Instruct-v0.2",),
        configs="configs/mistral",
        adaptations=(
            # hooks.py, install_score_hooks: the layer receives no
            # position_embeddings argument, so the score hook recomputes
            # (cos, sin) from module.rotary_emb.
            "RoPE (cos, sin) recomputed from module.rotary_emb",
            # cache.py, _resolve_cache_position: MistralFlashAttention2 passes
            # only {sin, cos} to cache.update(), so the absolute position is
            # taken from the cache's own monotonic token count.
            "cache_position derived from the cache's token count",
            # flash_decode.py, _undo_repeat_kv: MistralFlashAttention2 calls
            # repeat_kv (H_kv -> H_q) before flash; the fused decode collapses
            # it back to H_kv, losslessly, so K/V match the per-KV-head store.
            "repeat_kv before flash undone in the fused decode",
        ),
    ),
    "deepseek": ModelFamily(
        # The remote-code attention classes (DeepseekV2Attention /
        # DeepseekV2FlashAttention2) live in a dynamically named module, so they
        # are matched by name in models.deepseek_v2.loader, not imported here.
        attention_class="",
        checkpoints=("deepseek-ai/DeepSeek-V2-Lite-Chat",),
        configs="configs/deepseek",
        adaptations=(
            # utils/model_loading.load_causal_lm: trust_remote_code at the
            # revision pinned in models.deepseek_v2.loader.PINNED_REVISIONS, then
            # the owned MLA forward is installed on every layer.
            "owned MLA attention forward (models/deepseek_v2/attention.py)",
            # models/deepseek_v2/geometry.py: the cache holds the 576-wide latent
            # (kv_lora_rank 512 + RoPE 64) as ONE KV head whose value is its
            # first 512 lanes -- 1,152 B/token at bf16 -- not the decompressed
            # per-head K/V the remote code caches.
            "latent cache: one KV head of width 576, value aliased to the latent",
            # models/deepseek_v2/rope.py: RoPE at absolute positions on the 64
            # RoPE lanes only; the latent lanes are never rotated.
            "RoPE on the RoPE lanes only (PackedRope)",
            # hooks.py, _install_mla: no score hooks -- the owned forward writes
            # the eviction scores itself; install_score_hooks only arms the
            # fused decode, which routes to mla_decode_kernel.py.
            "fused decode via modules/windowed_cache/mla_decode_kernel.py",
            # models/deepseek_v2/row_batch.py: the windowed cache takes no
            # padding, so a left-padded batch runs attention one row (and one
            # cache) at a time (modules/evaluation/longbench_qevict_batched.py).
            "left-padded batches: one cache per row",
        ),
    ),
    "qwen2": ModelFamily(
        attention_class="transformers.models.qwen2.modeling_qwen2:Qwen2Attention",
        checkpoints=(),                               # hooks only; no configs shipped
        configs="",
        adaptations=(),
    ),
}


def _import_class(spec: str):
    module, name = spec.split(":")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None


def attention_classes() -> Tuple[type, ...]:
    """The attention classes the score hooks attach to, for every family whose
    class is importable on the installed transformers."""
    classes = []
    for fam in MODEL_FAMILIES.values():
        if not fam.attention_class:          # owned forward; nothing to hook
            continue
        cls = _import_class(fam.attention_class)
        if cls is not None and cls not in classes:
            classes.append(cls)
    return tuple(classes)


def family_of(model_name: str) -> str:
    """Best-effort family name for a hub id or local path (``"llama"``,
    ``"mistral"``, ...), or ``"unknown"``. Used for logging only."""
    low = model_name.lower()
    for fam in MODEL_FAMILIES:
        if fam in low:
            return fam
    return "unknown"
