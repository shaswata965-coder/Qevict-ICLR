"""Speed and quality must be measured at the same operating point.

This is the guard for the one defect that a paper cannot survive: the
throughput table and the accuracy table describing different caches.

Before ``utils.config.OPERATING_POINT`` existed they did. ``run_perf_table.sh``
timed the method at ``quant_ratio 0.70`` in ``quant_budget_mode tokens``;
``configs/llama/longbench_ours_flash_attn.yaml`` scored it at ``quant_ratio 0.0`` --
no int2 tier, no read gate, pure fp16 eviction. Nothing compared them and
nothing printed the difference. Every reported RecoverKV config, on every
model, is checked here against the one operating point.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]



@pytest.mark.parametrize("name", [
    "llama/longbench_ours_flash_attn.yaml",
    "llama/gsm8k_budget20.yaml",
    "llama/longgenbench_ours_flash_attn.yaml",
    "llama/ruler_ours_flash_attn.yaml",
    "mistral/longbench_ours_flash_attn.yaml",
    "deepseek/longbench_ours_flash_attn.yaml",
    "deepseek/gsm8k_budget20.yaml",
])
def test_the_quality_configs_resolve_the_same_cache(name):
    """The flash and eager arms are meant to be the same method by two routes,
    and GSM8K the same method on another dataset. If they differ, the three
    numbers are not comparable to each other either."""
    import yaml
    from utils.config import OPERATING_POINT

    block = yaml.safe_load((ROOT / "configs" / name).read_text())["cache"]
    for field, want in OPERATING_POINT.items():
        got = block.get(field)
        if want is None:
            # A `None` reference means "derive it" -- the knob is a tri-state ARM
            # whose shipped value is the derivation, not a number. Requiring
            # every config to spell `null` would not make anything safer: a
            # config that sets it to anything OTHER than the derivation is still
            # caught by the `got == want` line below.
            assert got is None or got == want, (
                f"{name}: {field}={got!r}, operating point is {want!r} (derive)")
            continue
        assert got is not None, f"{name} does not set {field}"
        assert got == want, f"{name}: {field}={got!r}, operating point is {want!r}"


