"""ActivationCache stacking helpers must follow the cache, not the model.

Project notes #1096 / #1099 (and upstream TransformerLens #1925): a cache moved to
CPU after a CUDA run tripped device mismatches in the stacking helpers because the
layer-0 "empty stack" tensors were built on ``model.cfg.device``. Same bug, other
axis: a cache cast to bf16 got float32 empties. CI has no CUDA, so the device half
is exercised by pointing ``model.cfg.device`` at ``meta`` — a tensor built there can
never concatenate with a CPU cache — and the dtype half by casting the cache.
"""

from contextlib import contextmanager
from typing import Iterator

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

MODEL = "roneneldan/TinyStories-1M"


@pytest.fixture(scope="module")
def model() -> HookedTransformer:
    return HookedTransformer.from_pretrained(MODEL, device="cpu")


@pytest.fixture(scope="module")
def cache(model: HookedTransformer) -> ActivationCache:
    _, cache = model.run_with_cache("The quick brown fox", remove_batch_dim=False)
    return cache


@contextmanager
def _model_on(model: HookedTransformer, device: str) -> Iterator[None]:
    """Pretend the model lives on another device without moving any weights."""
    original = model.cfg.device
    model.cfg.device = device
    try:
        yield
    finally:
        model.cfg.device = original


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_layer_zero_stacks_follow_the_cache_dtype(
    cache: ActivationCache, dtype: torch.dtype
) -> None:
    cast = ActivationCache(
        {name: value.to(dtype) for name, value in cache.cache_dict.items()}, cache.model
    )
    embed = cast["hook_embed"]
    for stacked in (cast.stack_head_results(0), cast.stack_neuron_results(0)):
        assert stacked.shape[0] == 0
        assert stacked.dtype == embed.dtype
        assert stacked.device == embed.device

    direction = torch.randn(embed.shape[-1])
    projected = [
        cast.stack_neuron_results(layer, apply_ln=True, project_output_onto=direction)
        for layer in (0, 1)
    ]
    assert projected[0].shape[0] == 0
    assert projected[0].dtype == projected[1].dtype


def test_layer_zero_stacks_follow_the_cache_device(
    model: HookedTransformer, cache: ActivationCache
) -> None:
    """The #1096 shape of the bug: cache on CPU, model (nominally) elsewhere."""
    with _model_on(model, "meta"):
        heads = cache.stack_head_results(0)
        neurons = cache.stack_neuron_results(0)
        direction = torch.randn(cache["hook_embed"].shape[-1])
        projected = cache.stack_neuron_results(0, apply_ln=True, project_output_onto=direction)
    for stacked in (heads, neurons, projected):
        assert stacked.shape[0] == 0
        assert stacked.device.type == "cpu"


def test_nonempty_stacks_are_unaffected_by_model_device_label(
    model: HookedTransformer, cache: ActivationCache
) -> None:
    # Layers > 0 never used model.cfg.device; pin that they still don't.
    reference = cache.stack_head_results(-1, pos_slice=-1)
    with _model_on(model, "meta"):
        under_label = cache.stack_head_results(-1, pos_slice=-1)
    torch.testing.assert_close(reference, under_label)
