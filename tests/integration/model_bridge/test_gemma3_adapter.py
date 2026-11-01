"""Integration tests for the Gemma-3 text adapter (Gemma3ForCausalLM).

Built from a tiny random Gemma3TextConfig (no Hub checkpoint), following the
DeepSeek integration files. The main target is the embedding double-scale
regression (upstream TransformerLens #1315): transformers >= 4.54 applies the
sqrt(d_model) scale inside Gemma3TextScaledWordEmbedding.forward, so any TL-side
scaling on top of it doubles the embedding and shifts every logit. Gemma-2 pins
this with a Hub-model test that is skipped in CI; Gemma-3 text had no parity
test at all.
"""

import copy
import tempfile

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig

from transformer_lens.model_bridge.bridge import TransformerBridge


def _tiny_config() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=16,
        query_pre_attn_scalar=16,
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def gemma3_pair():
    """(bridge, untouched HF reference) over the same tiny random weights."""
    torch.manual_seed(0)
    hf_model = Gemma3ForCausalLM(_tiny_config()).eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_model.save_pretrained(tmpdir)
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.save_pretrained(tmpdir)
        bridge = TransformerBridge.boot_transformers(tmpdir, device="cpu", dtype=torch.float32)
        yield bridge, hf_model


@pytest.fixture(scope="module")
def gemma3_compat(gemma3_pair) -> TransformerBridge:
    _, hf_ref = gemma3_pair
    tok = AutoTokenizer.from_pretrained("gpt2")
    bridge = TransformerBridge.boot_transformers(
        "gemma3-tiny-compat",
        hf_model=copy.deepcopy(hf_ref),
        tokenizer=tok,
        device="cpu",
        dtype=torch.float32,
    )
    bridge.enable_compatibility_mode(disable_warnings=True)
    return bridge


TOKENS = torch.tensor([[2, 5, 17, 42, 99, 7]])


class TestGemma3BridgeCreation:
    def test_block_count(self, gemma3_pair) -> None:
        bridge, _ = gemma3_pair
        assert len(bridge.blocks) == 2

    def test_rmsnorm_offset_flag(self, gemma3_pair) -> None:
        bridge, _ = gemma3_pair
        assert bridge.cfg.rmsnorm_uses_offset is True

    def test_embed_wraps_scaled_word_embedding(self, gemma3_pair) -> None:
        bridge, _ = gemma3_pair
        assert type(bridge.embed.original_component).__name__ == "Gemma3TextScaledWordEmbedding"


class TestGemma3EmbedScaledOnce:
    """The double-scale regression, end to end."""

    def test_no_hook_conversion_on_embed(self, gemma3_pair) -> None:
        bridge, _ = gemma3_pair
        assert bridge.embed.hook_out.hook_conversion is None

    def test_embed_weight_is_raw(self, gemma3_pair) -> None:
        bridge, hf_ref = gemma3_pair
        assert torch.equal(bridge.embed.W_E, hf_ref.model.embed_tokens.weight.detach())

    def test_embed_hook_out_is_scaled_exactly_once(self, gemma3_pair) -> None:
        bridge, _ = gemma3_pair
        raw = F.embedding(TOKENS, bridge.embed.original_component.weight)
        scaled_direct = bridge.embed.original_component(TOKENS)
        _, cache = bridge.run_with_cache(TOKENS)
        assert torch.allclose(cache["embed.hook_out"], scaled_direct, atol=1e-6)
        ratio = cache["embed.hook_out"].abs().max().item() / raw.abs().max().item()
        # sqrt(d_model), not d_model (double) and not 1 (missing).
        assert ratio == pytest.approx(bridge.cfg.d_model**0.5, rel=1e-4)


class TestGemma3ForwardEquivalence:
    def test_raw_forward_matches_hf(self, gemma3_pair) -> None:
        bridge, hf_ref = gemma3_pair
        with torch.no_grad():
            bridge_out = bridge(TOKENS)
            hf_out = hf_ref(TOKENS).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"raw bridge vs HF: {max_diff:.6f}"

    def test_compat_mode_log_probs_match_hf(self, gemma3_compat, gemma3_pair) -> None:
        """Phase 3 of verify_models: fold_ln (with RMSNorm +1 offsets and the
        pre/post sandwich norms) and centering must leave the distribution intact."""
        _, hf_ref = gemma3_pair
        with torch.no_grad():
            bridge_lp = torch.log_softmax(gemma3_compat(TOKENS), dim=-1)
            hf_lp = torch.log_softmax(hf_ref(TOKENS).logits, dim=-1)
        max_diff = (bridge_lp - hf_lp).abs().max().item()
        assert max_diff < 1e-3, f"compat-mode log-probs vs HF: {max_diff:.6f}"

    def test_compat_mode_embed_still_scaled_once(self, gemma3_compat) -> None:
        # Weight processing must not fold a second sqrt(d_model) into embed.
        raw = F.embedding(TOKENS, gemma3_compat.embed.original_component.weight)
        _, cache = gemma3_compat.run_with_cache(TOKENS)
        ratio = cache["embed.hook_out"].abs().max().item() / raw.abs().max().item()
        assert ratio == pytest.approx(gemma3_compat.cfg.d_model**0.5, rel=1e-4)


class TestGemma3ComponentParity:
    def test_all_components_match_hf(self, gemma3_pair) -> None:
        from transformer_lens.benchmarks.component_benchmark import benchmark_all_components

        bridge, _ = gemma3_pair
        result = benchmark_all_components(bridge, bridge.original_model)
        failures = (result.details or {}).get("failures", {})
        report = "\n".join(f"  {path}: {info}" for path, info in failures.items())
        assert result.passed, f"{result.message}\n{report}"


class TestGemma3NormEpsPlacement:
    """Project-notes regression (internal v2.16.1 hotfix, #1121): Gemma-3 RMSNorm
    epsilon must sit inside the rsqrt, ``x * rsqrt(mean(x^2) + eps)``, not outside.

    On ordinary activations the two placements agree to ~1e-6 and a logit-parity
    test cannot tell them apart. On near-zero input (variance << eps) eps dominates
    and the placements differ by orders of magnitude, so this input isolates it.
    """

    def _eps_dominated_input(self, bridge: TransformerBridge) -> torch.Tensor:
        torch.manual_seed(1)
        # variance ~ 1e-10 vs rms_norm_eps 1e-6: eps placement decides the output.
        return torch.randn(1, 4, bridge.cfg.d_model) * 1e-5

    def test_cfg_eps_matches_hf(self, gemma3_pair) -> None:
        bridge, hf_ref = gemma3_pair
        assert bridge.cfg.eps == pytest.approx(hf_ref.config.rms_norm_eps)
        assert bridge.blocks[0].ln1.original_component.eps == pytest.approx(
            hf_ref.config.rms_norm_eps
        )

    @pytest.mark.parametrize("norm_path", ["blocks.0.ln1", "blocks.0.ln2", "ln_final"])
    def test_norm_matches_hf_when_eps_dominates(self, gemma3_pair, norm_path: str) -> None:
        bridge, hf_ref = gemma3_pair
        x = self._eps_dominated_input(bridge)
        bridge_norm: torch.nn.Module = bridge
        for part in norm_path.split("."):
            bridge_norm = getattr(bridge_norm, part)
        hf_norm = bridge_norm.original_component
        with torch.no_grad():
            bridge_out = bridge_norm(x)
            hf_out = hf_norm(x)
            # The wrong placement, for reference: x * rsqrt(var) + eps-ish terms blow
            # up when var << eps; make sure the bridge is not on that curve.
            wrong = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) * (1.0 + hf_norm.weight)
        assert torch.allclose(bridge_out, hf_out, atol=1e-6), f"{norm_path} eps placement"
        assert not torch.allclose(bridge_out, wrong, atol=1e-3)

    def test_hook_scale_includes_eps(self, gemma3_pair) -> None:
        # hook_scale is sqrt(var + eps): with var << eps it must be ~sqrt(eps), not ~0.
        bridge, hf_ref = gemma3_pair
        x = self._eps_dominated_input(bridge)
        captured = {}
        handle = bridge.blocks[0].ln1.hook_scale.register_forward_hook(
            lambda m, i, o: captured.__setitem__("scale", o.detach().clone())
        )
        try:
            with torch.no_grad():
                bridge.blocks[0].ln1(x)
        finally:
            handle.remove()
        expected = (x.pow(2).mean(-1, keepdim=True) + hf_ref.config.rms_norm_eps).sqrt()
        assert torch.allclose(captured["scale"], expected, rtol=1e-4)
