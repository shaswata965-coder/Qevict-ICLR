"""Cache factory -- the one place the windowed cache is constructed from.

- ``get_cache_classes(backend)`` -- returns the
  ``(WindowedCache, WindowedCacheConfig, install_score_hooks)`` trio. The only
  backend is ``'flash_attn'``.
- ``validate_backend_attn_pairing(backend, attn_implementation)`` -- raises
  ``ConfigValidationError`` unless the pair is ``flash_attn`` /
  ``flash_attention_2``.
- ``quant_budget_mode_kwargs`` / ``quant_gate_ratio_kwargs`` -- thread the two
  quantization knobs into the cache config, raising if the config class ever
  stops accepting them.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Type

from utils.config import ConfigValidationError

__all__ = [
    "ConfigValidationError",
    "get_cache_classes",
    "quant_budget_mode_kwargs",
    "validate_backend_attn_pairing",
    "assert_transformers_version_supported",
    "is_transformers_version_supported",
    "MAX_SUPPORTED_TRANSFORMERS",
]


_BACKEND_TO_ATTN_IMPL = {
    "flash_attn": ("flash_attention_2",),
}

# Highest transformers version the windowed KV cache is known-correct on.
# The cache keeps surviving keys at their ORIGINAL RoPE positions after an
# eviction (rerotate_on_evict defaults False) and relies on HF ``generate``
# advancing the query's ``cache_position`` MONOTONICALLY — the transformers
# <= 4.47 behaviour. Newer transformers re-derive ``cache_position`` from the
# (now shrunken) cache length each step, so after the first eviction the query
# and the retained keys disagree on absolute position → corrupted RoPE phase
# and silently degraded results. Until the rerotation path is reworked, refuse
# to run on a newer version rather than emit wrong numbers.
MAX_SUPPORTED_TRANSFORMERS: Tuple[int, int, int] = (4, 47, 1)


def _parse_version(version: str) -> Tuple[int, int, int]:
    """Parse a version string to a ``(major, minor, patch)`` int tuple.

    Tolerant of suffixes like ``4.47.1.dev0`` / ``4.47.1+cu121`` — only the
    leading numeric ``major.minor.patch`` is used.
    """
    parts = []
    for chunk in version.split(".")[:3]:
        # Take only the LEADING run of digits, so "1+cu121" → 1 and
        # "1rc2" → 1 (not "1121" / "12").
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def is_transformers_version_supported(
    version: str,
    max_version: Tuple[int, int, int] = MAX_SUPPORTED_TRANSFORMERS,
) -> bool:
    """Return ``True`` iff *version* is <= *max_version* (inclusive)."""
    return _parse_version(version) <= max_version


def assert_transformers_version_supported(version: Optional[str] = None) -> None:
    """Raise ``ConfigValidationError`` if transformers is newer than supported.

    Called at the start of any *real* windowed-cache run (model-backed). Not
    invoked from pure-logic unit tests, which must stay runnable on dev boxes
    that have a newer transformers installed.

    Parameters
    ----------
    version : str, optional
        Version string to check. Defaults to the installed
        ``transformers.__version__``.
    """
    if version is None:
        import transformers

        version = transformers.__version__

    if not is_transformers_version_supported(version):
        supported = ".".join(str(p) for p in MAX_SUPPORTED_TRANSFORMERS)
        raise ConfigValidationError(
            f"transformers {version} is newer than the supported {supported}. "
            "The windowed KV cache keeps evicted-survivor keys at their original "
            "RoPE positions and relies on HF generate advancing cache_position "
            "monotonically (transformers <= 4.47). Newer versions re-derive "
            "cache_position from the compacted cache length after eviction, "
            "corrupting RoPE phase and silently degrading results. "
            f"Pin transformers=={supported} (see environment.yml), or rework the "
            "rerotation path and raise MAX_SUPPORTED_TRANSFORMERS before bumping."
        )


def get_cache_classes(backend: str) -> Tuple[Type, Type, Callable]:
    """Return ``(WindowedCache, WindowedCacheConfig, install_score_hooks)``.

    ``backend`` must be ``'flash_attn'``: eviction scores come from the fused
    Triton score kernel (prefill) and the fused decode kernel (decode), both of
    which run on CUDA with flash-attn installed.

    Raises
    ------
    ConfigValidationError
        If *backend* is not ``'flash_attn'``.
    """
    if backend != "flash_attn":
        raise ConfigValidationError(
            f"Unknown cache backend: {backend!r}. The only backend is 'flash_attn'."
        )
    from modules.windowed_cache import (  # type: ignore[attr-defined]
        WindowedCache,
        WindowedCacheConfig,
        install_score_hooks,
    )
    return WindowedCache, WindowedCacheConfig, install_score_hooks


def quant_budget_mode_kwargs(cache_config_cls: Type, requested: str) -> dict:
    """``{"quant_budget_mode": requested}`` for the cache config constructor.

    Routed through here so every runner threads the knob the same way: a runner
    that forgot it would silently run the dataclass default instead of the
    configured split. Raises if the config class has no such field.
    """
    if "quant_budget_mode" not in getattr(cache_config_cls, "__dataclass_fields__", {}):
        raise ConfigValidationError(
            f"{cache_config_cls.__module__}.{cache_config_cls.__name__} has no "
            "quant_budget_mode field."
        )
    return {"quant_budget_mode": requested}


def validate_backend_attn_pairing(
    backend: str, attn_implementation: str
) -> None:
    """Check the cache backend / attention implementation pair before model load.

    The only valid pair is ``flash_attn`` with ``flash_attention_2``.

    Raises
    ------
    ConfigValidationError
        On any other pair.
    """
    if backend not in _BACKEND_TO_ATTN_IMPL:
        raise ConfigValidationError(
            f"Unknown cache backend: {backend!r}. The only backend is 'flash_attn'."
        )

    allowed = _BACKEND_TO_ATTN_IMPL[backend]
    if attn_implementation not in allowed:
        raise ConfigValidationError(
            f"Cache backend {backend!r} requires "
            f"attn_implementation in {allowed!r}, but got "
            f"{attn_implementation!r}.  Fix your config."
        )


def quant_gate_ratio_kwargs(cache_config_cls: Type, requested: float) -> dict:
    """``{"quant_gate_ratio": requested}`` for the cache config constructor.

    Routed through here for the same reason as :func:`quant_budget_mode_kwargs`:
    a runner that forgot to thread the ratio would run the dataclass default
    under the config's name. Raises if the config class has no such field.
    """
    if "quant_gate_ratio" not in getattr(cache_config_cls, "__dataclass_fields__", {}):
        raise ConfigValidationError(
            f"{cache_config_cls.__module__}.{cache_config_cls.__name__} has no "
            "quant_gate_ratio field."
        )
    return {"quant_gate_ratio": requested}
