"""Per-key attention mass for the FullKV reference run, under any attention backend.

The parity base run needs, per layer and decode step, how much softmax attention
each key receives from the step's query rows -- ``softmax(q k^T * scale)`` summed
over queries, shape ``[B, H_q, S]``. :class:`AttentionMassRecorder` computes it
from the attention module's own inputs in a forward hook: the post-RoPE query is
rebuilt from ``hidden_states`` with the module's ``q_proj`` and RoPE, and the
keys are the ones the ``DynamicCache`` holds for that layer after the forward
(already RoPE'd and appended). Nothing depends on how the model itself computed
attention, so the model runs ``flash_attention_2`` like every other suite.

The arithmetic follows the reference attention in transformers 4.47: logits in
the query dtype, softmax in fp32, probabilities cast back to the query dtype,
then summed over the query rows in fp32. The sum is returned in fp32 and NOT
rounded back to the query dtype: the parity runner accumulates it across decode
steps, and an fp16 step sum would round small steps away. Causal masking uses
the absolute positions of this step's queries (the last ``T`` of ``S`` keys).
Query rows are processed in chunks so the ``[B, H_q, chunk, S]`` block stays
bounded at long prefills.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

#: Query rows per chunk when forming the ``[B, H_q, chunk, S]`` probability block.
QUERY_CHUNK = 512


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """``[B, H_kv, S, D] -> [B, H_kv * n_rep, S, D]`` (each head repeated in place)."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """RoPE on ``q`` ``[B, H, T, D]`` with ``cos``/``sin`` ``[B, T, D]``."""
    return q * cos.unsqueeze(1) + _rotate_half(q) * sin.unsqueeze(1)


class AttentionMassRecorder:
    """Forward hooks that record per-key attention mass for every attention layer.

    Usage::

        rec = AttentionMassRecorder(model)
        rec.bind(cache)                      # the DynamicCache passed to forward
        out = model(..., past_key_values=cache)
        mass = rec.pop()                     # list over layers of [B, H_q, S]
        rec.remove()
    """

    def __init__(self, model: nn.Module) -> None:
        from models import attention_classes

        classes = attention_classes()
        self._cache: Any = None
        self._mass: Dict[int, Tensor] = {}
        self._handles: List[Any] = []
        self._n_layers = 0
        for module in model.modules():
            if isinstance(module, classes):
                self._handles.append(
                    module.register_forward_hook(self._hook, with_kwargs=True))
                self._n_layers += 1
        if self._n_layers == 0:
            raise RuntimeError(
                "AttentionMassRecorder found no attention module of a family in "
                "models.MODEL_FAMILIES; cannot record attention mass.")

    def bind(self, cache: Any) -> None:
        """The cache the next forward passes use (its keys are read after update)."""
        self._cache = cache

    def pop(self) -> List[Tensor]:
        """Per-layer ``[B, H_q, S]`` fp32 mass from the last forward, in layer order."""
        if len(self._mass) != self._n_layers:
            raise RuntimeError(
                f"attention mass recorded for {len(self._mass)} of "
                f"{self._n_layers} layers in the last forward")
        out = [self._mass[i] for i in sorted(self._mass)]
        self._mass = {}
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- hook ---------------------------------------------------------------
    def _hook(self, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None:
            raise RuntimeError("attention forward called without hidden_states")
        pe: Optional[tuple] = kwargs.get("position_embeddings")
        if pe is None:
            # Mistral's attention (transformers 4.47) holds its own rotary_emb
            # and is not handed position_embeddings; rebuild them the same way.
            position_ids = kwargs.get("position_ids")
            if position_ids is None or not hasattr(module, "rotary_emb"):
                raise RuntimeError(
                    "attention forward has neither position_embeddings nor "
                    "position_ids + rotary_emb; cannot rebuild the query")
            pe = module.rotary_emb(hidden, position_ids)
        cos, sin = pe

        layer = module.layer_idx
        keys = self._cache.key_cache[layer]                  # [B, H_kv, S, D], post-RoPE
        head_dim = module.head_dim
        b, t = hidden.shape[:2]
        q = module.q_proj(hidden).view(b, t, -1, head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)                          # [B, H_q, T, D]
        k = _repeat_kv(keys, q.shape[1] // keys.shape[1]).to(q.dtype)
        s = k.shape[2]
        scale = getattr(module, "scaling", head_dim ** -0.5)

        # fp32 accumulator over chunks: the same result as one fp32 reduction
        # over all query rows of the query-dtype probabilities.
        mass = torch.zeros(b, q.shape[1], s, dtype=torch.float32, device=q.device)
        key_pos = torch.arange(s, device=q.device)
        for c0 in range(0, t, QUERY_CHUNK):
            qc = q[:, :, c0:c0 + QUERY_CHUNK]
            logits = torch.matmul(qc, k.transpose(2, 3)) * scale   # [B, H, c, S]
            if t > 1:
                q_pos = (s - t) + torch.arange(c0, c0 + qc.shape[2], device=q.device)
                future = key_pos[None, :] > q_pos[:, None]         # [c, S]
                logits = logits.masked_fill(future, torch.finfo(logits.dtype).min)
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
            mass += probs.float().sum(dim=-2)
        self._mass[layer] = mass
