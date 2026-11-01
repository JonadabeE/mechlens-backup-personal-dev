"""Issue #1126 ("ActivationCache repr is very slow for large caches").

``ActivationCache.__repr__`` formats key names only; it must never touch the
cached tensors, whose own repr (torch summarises values) is what actually takes
seconds on a large model. These tests pin that contract without loading a model:
values are stand-ins whose repr raises, and a 96-layer key set must format fast.
"""

import time

import pytest

from transformer_lens.ActivationCache import ActivationCache

_HOOKS = [
    "hook_resid_pre",
    "ln1.hook_scale",
    "ln1.hook_normalized",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "attn.hook_attn_scores",
    "attn.hook_pattern",
    "attn.hook_z",
    "hook_attn_out",
    "hook_resid_mid",
    "ln2.hook_scale",
    "ln2.hook_normalized",
    "mlp.hook_pre",
    "mlp.hook_post",
    "hook_mlp_out",
    "hook_resid_post",
]


class _NeverRepr:
    """A cached value that must not be stringified by the cache's repr."""

    def __repr__(self) -> str:  # pragma: no cover - reaching this is the failure
        raise AssertionError("ActivationCache.__repr__ touched a cached value")

    __str__ = __repr__


def _large_cache(n_layers: int = 96) -> ActivationCache:
    keys = ["hook_embed", "hook_pos_embed"]
    keys += [f"blocks.{layer}.{hook}" for layer in range(n_layers) for hook in _HOOKS]
    keys += ["ln_final.hook_scale", "ln_final.hook_normalized"]
    return ActivationCache({k: _NeverRepr() for k in keys}, model=None)


def test_repr_never_materialises_cached_values() -> None:
    cache = _large_cache()
    text = repr(cache)
    assert text.startswith("ActivationCache")
    assert "blocks.95.hook_resid_post" in text


def test_repr_is_fast_for_a_96_layer_cache() -> None:
    cache = _large_cache()
    start = time.perf_counter()
    for _ in range(20):
        repr(cache)
    per_call_ms = (time.perf_counter() - start) / 20 * 1000
    assert per_call_ms < 50, f"repr took {per_call_ms:.1f} ms per call"


def test_str_matches_repr() -> None:
    cache = _large_cache(n_layers=2)
    assert str(cache) == repr(cache)


@pytest.mark.parametrize("n_layers", [1, 12, 96])
def test_repr_mentions_every_key(n_layers: int) -> None:
    cache = _large_cache(n_layers=n_layers)
    text = repr(cache)
    for key in cache.cache_dict:
        assert key in text
