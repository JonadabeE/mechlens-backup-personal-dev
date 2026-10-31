"""Integration tests for Cohere architecture adapter (CohereForCausalLM).

Model: trl-internal-testing/tiny-CohereForCausalLM
  - 2 layers, ~8M params, CPU-safe, no gating required
  - tie_word_embeddings=True by default
  - logit_scale=0.125 (canonical Command-R is 0.0625; tiny diverges so
    regression tests catch silent-fallback bugs in the passthrough)

NOTE: The tiny model has use_qk_norm=False, so QK-norm is not exercised here.
Cohere's QK-norm is a per-head LayerNorm inside CohereAttention.forward; it is
handled via HF delegation (PositionEmbeddingsAttentionBridge calls the original
CohereAttention.forward directly), so functional correctness for that path relies
on the same delegation mechanism verified in test_forward_matches_hf.
"""

import copy
from typing import Any

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"


# boot_transformers loads the HF model with attn_implementation="eager" itself.
_BOOT_KWARGS: dict[str, Any] = {"device": "cpu", "dtype": torch.float32}


def _fold_only_process_weights(bridge: TransformerBridge) -> None:
    """process_weights with every ProcessWeights option off: only the adapter's
    preprocess_weights (logit_scale fold + untie) runs."""
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )


@pytest.fixture(scope="module")
def cohere_bridge():
    """Load tiny Cohere bridge once per module (no weight processing)."""
    return TransformerBridge.boot_transformers(MODEL, **_BOOT_KWARGS)


@pytest.fixture(scope="module")
def cohere_bridge_processed():
    """Bridge with preprocess_weights applied (fold only, no centering).

    process_weights must be called explicitly — boot_transformers does not call
    it automatically.
    """
    bridge = TransformerBridge.boot_transformers(MODEL, **_BOOT_KWARGS)
    _fold_only_process_weights(bridge)
    return bridge


@pytest.fixture(scope="module")
def cohere_bridge_compat():
    """Bridge in full compatibility mode (fold_ln + centering, HT-equivalent)."""
    bridge = TransformerBridge.boot_transformers(MODEL, **_BOOT_KWARGS)
    bridge.enable_compatibility_mode(disable_warnings=True)
    return bridge


@pytest.fixture(scope="module")
def cohere_hf() -> Any:
    """Load the raw HF model for side-by-side comparisons."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()


# ---------------------------------------------------------------------------
# 1. Bridge creation — exercises all 4 registration points + component_mapping
# ---------------------------------------------------------------------------


class TestCohereBridgeCreation:
    """Verify the bridge loads cleanly and exposes the expected structure."""

    def test_boot_transformers_succeeds(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge is not None

    def test_block_count(self, cohere_bridge: TransformerBridge) -> None:
        # tiny model has 2 layers
        assert len(cohere_bridge.blocks) == 2

    def test_has_core_components(self, cohere_bridge: TransformerBridge) -> None:
        assert hasattr(cohere_bridge, "embed")
        assert hasattr(cohere_bridge, "unembed")
        assert hasattr(cohere_bridge, "ln_final")
        assert hasattr(cohere_bridge, "rotary_emb")

    def test_no_ln2_in_blocks(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block: no post_attention_layernorm
        for block in cohere_bridge.blocks:
            assert not hasattr(block, "ln2"), "Parallel block must not have ln2"

    def test_ln1_subtracts_mean_like_hf_layernorm(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """uses_rms_norm=False must make ln1 mean-subtract (true LayerNorm), not RMS.

        Drives blocks[0].ln1 on a deliberately non-zero-mean input and compares
        against the raw HF CohereLayerNorm (which subtracts the mean). If
        uses_rms_norm regressed to True, the bridge would skip mean subtraction
        and diverge from HF by O(1), tripping this assertion.
        """
        ln1 = cohere_bridge.blocks[0].ln1
        hf_ln1 = cohere_hf.model.layers[0].input_layernorm
        d_model = cohere_bridge.cfg.d_model
        torch.manual_seed(0)
        # +5.0 shift guarantees a large mean so LN and RMS outputs differ markedly.
        x = torch.randn(1, 4, d_model) + 5.0
        with torch.no_grad():
            bridge_out = ln1(x)
            hf_out = hf_ln1(x)
        ln_diff = (bridge_out - hf_out).abs().max().item()
        assert ln_diff < 1e-4, f"ln1 does not match HF CohereLayerNorm: max_diff={ln_diff:.6f}"
        # And it must NOT match an RMS-only (no mean subtraction) normalization.
        variance = x.float().pow(2).mean(-1, keepdim=True)
        rms_only = (
            x.float()
            * torch.rsqrt(variance + ln1.original_component.variance_epsilon)
            * ln1.original_component.weight.float()
        ).to(bridge_out.dtype)
        rms_diff = (bridge_out - rms_only).abs().max().item()
        assert rms_diff > 1e-2, (
            "ln1 output matches RMS-only normalization; uses_rms_norm must be False "
            f"so the mean is subtracted (rms_diff={rms_diff:.6f})"
        )

    def test_cfg_logit_scale_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: logit_scale must propagate from HF (not silently fall back to 0.0625)."""
        bridge_scale = getattr(cohere_bridge.cfg, "logit_scale")
        assert bridge_scale == cohere_hf.config.logit_scale
        # Anchor 0.125 so a passthrough regression that defaults to 0.0625 also trips here.
        assert bridge_scale == pytest.approx(0.125)

    def test_cfg_rope_parameters_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: rope_parameters must propagate from HF (same passthrough trap as logit_scale)."""
        assert getattr(cohere_bridge.cfg, "rope_parameters") == cohere_hf.config.rope_parameters


# ---------------------------------------------------------------------------
# 2. Forward equivalence — HF logits ≈ bridge logits
# ---------------------------------------------------------------------------


class TestCohereForwardEquivalence:
    """Verify bridge and HF produce identical logits for the same input."""

    def test_forward_returns_logits(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = cohere_bridge(tokens)
        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_matches_hf(self, cohere_bridge: TransformerBridge, cohere_hf: Any) -> None:
        """Bridge delegates to HF native forward — logits should be identical."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff:.6f}"


# ---------------------------------------------------------------------------
# 3. Logit scale applied end-to-end
# ---------------------------------------------------------------------------


class TestCohereLogitScaleEndToEnd:
    """Verify logit_scale is correctly folded into the loaded model.

    process_weights must be called before the fold takes effect — boot_transformers
    alone does NOT call process_weights. cohere_bridge_processed uses a fixture that
    calls process_weights with all standard options disabled so only preprocess_weights
    (the logit_scale fold) runs.
    """

    def test_unembed_weight_is_scaled_relative_to_hf(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After preprocess_weights, lm_head.weight inside the bridge should equal
        # HF lm_head.weight * logit_scale (both [d_vocab, d_model]).
        logit_scale = getattr(cohere_bridge_processed.cfg, "logit_scale")
        tl_weight = cohere_bridge_processed.unembed.original_component.weight  # [d_vocab, d_model]
        hf_weight = cohere_hf.lm_head.weight.detach()  # [d_vocab, d_model]
        expected = hf_weight * logit_scale
        max_diff = (tl_weight - expected).abs().max().item()
        assert max_diff < 1e-5, (
            f"unembed.weight not correctly scaled: max_diff={max_diff:.6f}, "
            f"logit_scale={logit_scale}"
        )

    def test_unprocessed_bridge_leaves_hf_multiplier_alone(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        # Before process_weights nothing is folded, so HF's own multiply must stay live.
        assert cohere_bridge.original_model.logit_scale == cohere_hf.config.logit_scale

    def test_processed_bridge_disables_hf_runtime_multiplier(
        self, cohere_bridge_processed: TransformerBridge
    ) -> None:
        # The scale now lives in W_U; CohereForCausalLM.forward must multiply by 1.
        assert cohere_bridge_processed.original_model.logit_scale == 1.0

    def test_logit_scale_folded_not_applied_twice(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: after the fold, forward must still match HF.

        The bridge forwards through CohereForCausalLM.forward, which multiplies by
        ``self.logit_scale``. With the scale already folded into W_U, a live HF
        multiplier would scale logits twice (0.125^2 here) and trip this assertion.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge_processed(tokens)
            hf_out = cohere_hf(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"logit_scale applied twice after fold; diff={max_diff:.6f}"

    def test_compat_mode_forward_matches_hf(
        self, cohere_bridge_compat: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Full compatibility mode (fold_ln + centering) must preserve HF logits.

        Centering the unembed shifts logits by a per-position constant, which
        leaves log-probs unchanged, so compare log-softmax rather than raw logits.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_lp = torch.log_softmax(cohere_bridge_compat(tokens), dim=-1)
            hf_lp = torch.log_softmax(cohere_hf(tokens).logits, dim=-1)
        max_diff = (bridge_lp - hf_lp).abs().max().item()
        assert max_diff < 1e-3, f"Compat-mode log-probs diverge from HF: max_diff={max_diff:.6f}"

    def test_process_weights_twice_does_not_rescale(self, cohere_hf: Any) -> None:
        # enable_compatibility_mode after an explicit process_weights re-enters
        # preprocess_weights; W_U must carry logit_scale exactly once.
        bridge = TransformerBridge.boot_transformers(MODEL, **_BOOT_KWARGS)
        _fold_only_process_weights(bridge)
        bridge.enable_compatibility_mode(
            disable_warnings=True,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )
        logit_scale = getattr(bridge.cfg, "logit_scale")
        tl_weight = bridge.unembed.original_component.weight
        expected = cohere_hf.lm_head.weight.detach() * logit_scale
        max_diff = (tl_weight - expected).abs().max().item()
        assert max_diff < 1e-5, f"W_U scaled more than once: max_diff={max_diff:.6f}"


# ---------------------------------------------------------------------------
# 4. Tied embedding preserved — embed.W_E must NOT be scaled
# ---------------------------------------------------------------------------


class TestCohereTiedEmbedding:
    """Verify preprocess_weights does not corrupt embed.W_E in the tied case.

    Both tests use cohere_bridge_processed (process_weights called with fold-only
    options) because the untie + fold only happens inside process_weights.
    """

    def test_embed_weight_equals_hf_embed_tokens(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After the fold, embed.W_E must still equal HF's unscaled embed_tokens.weight.
        # If the fold corrupted embed (in-place on the shared tensor), this fails.
        hf_embed = cohere_hf.model.embed_tokens.weight.detach()  # [d_vocab, d_model]
        tl_embed = cohere_bridge_processed.embed.W_E  # [d_vocab, d_model]
        max_diff = (tl_embed - hf_embed).abs().max().item()
        assert (
            max_diff < 1e-6
        ), f"embed.W_E was corrupted (possibly by logit_scale fold): max_diff={max_diff:.6f}"

    @pytest.mark.parametrize("logit_scale", [0.0625, 1.0])
    def test_embed_and_unembed_weights_differ(self, logit_scale: float) -> None:
        # After the logit_scale fold, embed.W_E and unembed.weight must NOT be
        # identical for a non-trivial scale. logit_scale=1.0 is kept as a regression
        # guard for the no-op case, where the two weights stay tied.
        #
        # cfg.logit_scale is set before process_weights so the fold (which reads it
        # inside preprocess_weights) runs with the parametrized value.
        bridge = TransformerBridge.boot_transformers(MODEL, **_BOOT_KWARGS)
        setattr(bridge.cfg, "logit_scale", logit_scale)
        _fold_only_process_weights(bridge)
        tl_embed = bridge.embed.W_E
        tl_unembed = bridge.unembed.original_component.weight
        weights_identical = torch.allclose(tl_embed, tl_unembed)
        if logit_scale == 1.0:
            assert weights_identical, (
                "embed.W_E and unembed.weight should remain tied when logit_scale=1.0 "
                "(the fold is a no-op)"
            )
        else:
            assert not weights_identical, (
                "embed.W_E and unembed.weight are identical — "
                "logit_scale fold may not have been applied or untied correctly"
            )


# ---------------------------------------------------------------------------
# 5. HF delegation — RoPE, attention, normalization go through HF modules
# ---------------------------------------------------------------------------


class TestCohereHFDelegation:
    """Spot-check that bridge submodules delegate to actual HF modules.

    This confirms setup_component_testing wired rotary_emb correctly and that
    NormalizationBridge and PositionEmbeddingsAttentionBridge hold live HF objects.
    """

    def test_ln1_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        # bridge.blocks[0].ln1.original_component should be the live HF CohereLayerNorm
        ln1 = cohere_bridge.blocks[0].ln1
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.original_component is not None
        # CohereLayerNorm has variance_epsilon (not eps)
        assert hasattr(
            ln1.original_component, "variance_epsilon"
        ), "original_component is not CohereLayerNorm (missing variance_epsilon)"

    def test_attn_original_component_is_hf_attention(
        self, cohere_bridge: TransformerBridge
    ) -> None:
        attn = cohere_bridge.blocks[0].attn
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.original_component is not None
        # CohereAttention has q_proj
        assert hasattr(
            attn.original_component, "q_proj"
        ), "original_component is not CohereAttention (missing q_proj)"

    def test_ln_final_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.ln_final.original_component is not None
        assert hasattr(cohere_bridge.ln_final.original_component, "variance_epsilon")


# ---------------------------------------------------------------------------
# 6. Parallel-attn hooks fire correctly
# ---------------------------------------------------------------------------


class TestCohereParallelHooks:
    """Verify hook placement for the parallel attention+MLP block."""

    def test_no_hook_resid_mid(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block has no intermediate residual stream
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        assert not any("hook_resid_mid" in k for k in cache.keys())

    def test_attn_and_mlp_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_residual_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache


# ---------------------------------------------------------------------------
# 7. Tokenizer policy — the adapter hardcodes default_prepend_bos=True on the
#    strength of runtime behaviour, not tokenizer_config.json. Pin that claim.
# ---------------------------------------------------------------------------


class TestCohereTokenizerPolicy:
    """default_prepend_bos=True must agree with what the tokenizer actually does."""

    def test_tokenizer_prepends_bos_at_runtime(self, cohere_bridge: TransformerBridge) -> None:
        tok = cohere_bridge.tokenizer
        assert tok is not None
        ids = tok("hello").input_ids
        assert ids[0] == tok.bos_token_id, "CohereTokenizerFast no longer prepends BOS"

    def test_cfg_bos_flags_match_runtime(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.cfg.default_prepend_bos is True
        assert cohere_bridge.cfg.tokenizer_prepends_bos is True


# ---------------------------------------------------------------------------
# 8. Processed-weight edits reach the forward pass (mirrors the verify_models
#    Phase 3 "weight_modification" check that was failing for this model).
# ---------------------------------------------------------------------------


class TestCohereWeightModificationPropagates:
    """In compatibility mode, editing TL-facing weights must change the loss."""

    TOKENS = torch.tensor([[5, 17, 42, 99, 7, 3, 25, 11]])

    def test_zeroing_w_v_head_changes_loss(self, cohere_bridge_compat: TransformerBridge) -> None:
        bridge = cohere_bridge_compat
        attn = bridge.blocks[0].attn
        with torch.no_grad():
            base = bridge(self.TOKENS, return_type="loss").item()
            # W_V is a view onto v_proj.weight; an in-place zero must reach HF's forward.
            w_v = attn.W_V
            saved = w_v.clone()
            w_v[0] = 0
            modified = bridge(self.TOKENS, return_type="loss").item()
            w_v.copy_(saved)
            restored = bridge(self.TOKENS, return_type="loss").item()
        assert abs(modified - base) > 1e-6, "Zeroing W_V head 0 did not change the loss"
        assert abs(restored - base) < 1e-6, "Restoring W_V did not restore the loss"

    def test_zeroing_mlp_out_row_changes_loss(
        self, cohere_bridge_compat: TransformerBridge
    ) -> None:
        bridge = cohere_bridge_compat
        out_w = bridge.blocks[0].mlp.out.weight
        with torch.no_grad():
            base = bridge(self.TOKENS, return_type="loss").item()
            saved = out_w.clone()
            out_w[0, :] = 0
            modified = bridge(self.TOKENS, return_type="loss").item()
            out_w.copy_(saved)
        assert abs(modified - base) > 1e-6, "Zeroing an MLP output row did not change the loss"


# ---------------------------------------------------------------------------
# 9. HT-facing W_U carries logit_scale — this is what logit lens / DLA read.
# ---------------------------------------------------------------------------


class TestCohereUnembedView:
    def test_w_u_is_scaled_hf_lm_head(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        logit_scale = getattr(cohere_bridge_processed.cfg, "logit_scale")
        w_u = cohere_bridge_processed.unembed.W_U  # [d_model, d_vocab]
        expected = cohere_hf.lm_head.weight.detach().T * logit_scale
        assert w_u.shape == expected.shape
        max_diff = (w_u - expected).abs().max().item()
        assert max_diff < 1e-5, f"W_U does not equal HF lm_head.T * logit_scale: {max_diff:.6f}"

    def test_w_u_unscaled_before_processing(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        # Raw bridge: W_U is HF's raw lm_head; the scale is still applied by HF at runtime.
        w_u = cohere_bridge.unembed.W_U
        expected = cohere_hf.lm_head.weight.detach().T
        max_diff = (w_u - expected).abs().max().item()
        assert max_diff < 1e-6


# ---------------------------------------------------------------------------
# 10. QK-norm — canonical Command-R ships use_qk_norm=True; the tiny test
#     checkpoint has it off. Build the variant from the cached config so the
#     path is covered without a gated download.
# ---------------------------------------------------------------------------


def _make_qk_norm_reference() -> Any:
    """Tiny CohereForCausalLM with use_qk_norm=True, seeded random init, eager attention."""
    config = AutoConfig.from_pretrained(MODEL)
    config.use_qk_norm = True
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    return model.float().eval()


@pytest.fixture(scope="module")
def qk_norm_hf() -> Any:
    """Reference HF model, never touched by the bridge."""
    return _make_qk_norm_reference()


@pytest.fixture(scope="module")
def qk_norm_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def qk_norm_bridge(qk_norm_hf: Any, qk_norm_tokenizer: Any) -> TransformerBridge:
    """Raw bridge over a deep copy of the reference (bridge boot mutates its model)."""
    return TransformerBridge.boot_transformers(
        MODEL, hf_model=copy.deepcopy(qk_norm_hf), tokenizer=qk_norm_tokenizer, **_BOOT_KWARGS
    )


@pytest.fixture(scope="module")
def qk_norm_bridge_compat(qk_norm_hf: Any, qk_norm_tokenizer: Any) -> TransformerBridge:
    """Compatibility-mode bridge over another deep copy of the reference."""
    bridge = TransformerBridge.boot_transformers(
        MODEL, hf_model=copy.deepcopy(qk_norm_hf), tokenizer=qk_norm_tokenizer, **_BOOT_KWARGS
    )
    bridge.enable_compatibility_mode(disable_warnings=True)
    return bridge


class TestCohereQKNorm:
    """use_qk_norm=True is HF-delegated inside CohereAttention; verify that delegation
    survives both the raw path and compatibility-mode weight processing."""

    TOKENS = torch.tensor([[1, 2, 3, 4, 5, 6]])

    def test_reference_has_qk_norm(self, qk_norm_hf: Any) -> None:
        assert qk_norm_hf.config.use_qk_norm is True
        assert hasattr(qk_norm_hf.model.layers[0].self_attn, "q_norm")

    def test_bridge_attn_delegates_to_qk_norm_module(
        self, qk_norm_bridge: TransformerBridge
    ) -> None:
        attn = qk_norm_bridge.blocks[0].attn
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert hasattr(attn.original_component, "q_norm")
        assert hasattr(attn.original_component, "k_norm")

    def test_raw_forward_matches_hf(
        self, qk_norm_bridge: TransformerBridge, qk_norm_hf: Any
    ) -> None:
        with torch.no_grad():
            bridge_out = qk_norm_bridge(self.TOKENS)
            hf_out = qk_norm_hf(self.TOKENS).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"QK-norm raw forward diverges from HF: {max_diff:.6f}"

    def test_compat_mode_log_probs_match_hf(
        self, qk_norm_bridge_compat: TransformerBridge, qk_norm_hf: Any
    ) -> None:
        # fold_ln must not touch q_norm/k_norm (they normalise per-head q/k, not the
        # residual stream) and the logit_scale fold must still be applied once.
        with torch.no_grad():
            bridge_lp = torch.log_softmax(qk_norm_bridge_compat(self.TOKENS), dim=-1)
            hf_lp = torch.log_softmax(qk_norm_hf(self.TOKENS).logits, dim=-1)
        max_diff = (bridge_lp - hf_lp).abs().max().item()
        assert max_diff < 1e-3, f"QK-norm compat-mode log-probs diverge from HF: {max_diff:.6f}"

    def test_compat_mode_hf_multiplier_disabled(
        self, qk_norm_bridge_compat: TransformerBridge
    ) -> None:
        assert qk_norm_bridge_compat.original_model.logit_scale == 1.0
