"""Unit tests for MLAAttentionBridge (DeepSeek Multi-Head Latent Attention)."""

import pytest
import torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

from transformer_lens.model_bridge.generalized_components.mla_attention import (
    MLAAttentionBridge,
)


@pytest.fixture(scope="module")
def tiny_config():
    return DeepseekV3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=1,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=1000,
        first_k_dense_replace=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
        moe_intermediate_size=256,
        # Eager: SDPA applies an implicit causal mask when attention_mask=None,
        # the bridge does not — pin the backend so parity tests compare like with like.
        attn_implementation="eager",
    )


def _causal_mask(batch: int, seq: int) -> torch.Tensor:
    """Additive [batch, 1, seq, seq] causal mask, as HF passes to attention."""
    mask = torch.full((seq, seq), float("-inf")).triu(1)
    return mask[None, None].expand(batch, 1, seq, seq)


@pytest.fixture(scope="module")
def tiny_model(tiny_config):
    return DeepseekV3ForCausalLM(tiny_config)


@pytest.fixture(scope="module")
def hf_attn(tiny_model):
    return tiny_model.model.layers[0].self_attn


@pytest.fixture(scope="module")
def mla_bridge(tiny_config, hf_attn, tiny_model):
    bridge = MLAAttentionBridge(name="self_attn", config=tiny_config, submodules={})
    bridge.set_original_component(hf_attn)
    bridge.set_rotary_emb(tiny_model.model.rotary_emb)
    return bridge


class TestMLAAttentionBridgeHooks:
    def test_all_expected_hooks_exist(self, mla_bridge):
        for hook_name in [
            "hook_in",
            "hook_out",
            "hook_q_latent",
            "hook_kv_latent",
            "hook_q",
            "hook_k",
            "hook_v",
            "hook_rot_q",
            "hook_rot_k",
            "hook_attn_scores",
            "hook_pattern",
            "hook_cos",
            "hook_sin",
        ]:
            assert hasattr(mla_bridge, hook_name), f"Missing hook: {hook_name}"

    def test_W_Q_raises_not_implemented(self, mla_bridge):
        with pytest.raises(NotImplementedError, match="not available on MLA"):
            _ = mla_bridge.W_Q

    def test_W_K_raises_not_implemented(self, mla_bridge):
        with pytest.raises(NotImplementedError, match="not available on MLA"):
            _ = mla_bridge.W_K

    def test_missing_weights_are_hasattr_safe(self, mla_bridge):
        # verify_models probes b_V / W_V with hasattr/getattr; a bare NotImplementedError
        # escaped both and failed value_bias_folding + weight_modification on every
        # DeepSeek model. The error must also be an AttributeError.
        # (W_O / b_O resolve through the real `o` submodule and are not probed here.)
        for name in ("W_Q", "W_K", "W_V", "b_Q", "b_K", "b_V"):
            assert not hasattr(mla_bridge, name), name
            assert getattr(mla_bridge, name, None) is None, name
        with pytest.raises(AttributeError):
            _ = mla_bridge.b_V


class TestMLAAttentionBridgeForward:
    @pytest.fixture
    def sample_inputs(self, tiny_config, tiny_model):
        batch, seq = 2, 8
        hidden_states = torch.randn(batch, seq, tiny_config.hidden_size)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = tiny_model.model.rotary_emb(hidden_states, position_ids)
        return hidden_states, (cos, sin)

    def test_output_matches_hf(self, mla_bridge, hf_attn, sample_inputs, tiny_model):
        """Bridge MLA must reproduce HF's attention output at fp32 precision.

        The old 0.15 tolerance was hiding a real mismatch: DeepseekV3Config defaults
        to ``rope_interleave=True`` and the bridge always applied rotate-half RoPE.
        """
        hidden_states, position_embeddings = sample_inputs
        mask = _causal_mask(*hidden_states.shape[:2])

        with torch.no_grad():
            hf_attn_out = hf_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
            )[0]
            bridge_attn_out = mla_bridge(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
            )[0]

        max_diff = (hf_attn_out - bridge_attn_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge MLA diverges from HF: max diff = {max_diff}"

    def test_hooks_fire_and_have_correct_shapes(self, mla_bridge, sample_inputs, tiny_config):
        hidden_states, position_embeddings = sample_inputs
        batch, seq = hidden_states.shape[:2]
        captured = {}

        hooks_to_check = [
            "hook_q_latent",
            "hook_kv_latent",
            "hook_q",
            "hook_k",
            "hook_v",
            "hook_rot_q",
            "hook_rot_k",
            "hook_attn_scores",
            "hook_pattern",
        ]

        handles = []
        for name in hooks_to_check:

            def make_capture(n):
                def hook_fn(module, input, output):
                    captured[n] = output.shape

                return hook_fn

            handles.append(getattr(mla_bridge, name).register_forward_hook(make_capture(name)))

        try:
            with torch.no_grad():
                mla_bridge(
                    hidden_states, position_embeddings=position_embeddings, attention_mask=None
                )
        finally:
            for h in handles:
                h.remove()

        n_heads = tiny_config.num_attention_heads
        qk_head_dim = tiny_config.qk_nope_head_dim + tiny_config.qk_rope_head_dim

        for name in hooks_to_check:
            assert name in captured, f"Hook {name} did not fire"

        assert captured["hook_q_latent"] == (batch, seq, tiny_config.q_lora_rank)
        assert captured["hook_kv_latent"] == (batch, seq, tiny_config.kv_lora_rank)
        assert captured["hook_q"] == (batch, n_heads, seq, qk_head_dim)
        assert captured["hook_k"] == (batch, n_heads, seq, qk_head_dim)
        assert captured["hook_v"] == (batch, n_heads, seq, tiny_config.v_head_dim)
        assert captured["hook_attn_scores"] == (batch, n_heads, seq, seq)
        assert captured["hook_pattern"] == (batch, n_heads, seq, seq)

    def test_hook_q_is_post_rope(self, mla_bridge, sample_inputs):
        """hook_q's rope portion should match hook_rot_q."""
        hidden_states, position_embeddings = sample_inputs
        q_values: list[torch.Tensor] = []
        rot_q_values: list[torch.Tensor] = []

        h1 = mla_bridge.hook_q.register_forward_hook(lambda m, i, o: q_values.append(o.clone()))
        h2 = mla_bridge.hook_rot_q.register_forward_hook(
            lambda m, i, o: rot_q_values.append(o.clone())
        )

        try:
            with torch.no_grad():
                mla_bridge(
                    hidden_states, position_embeddings=position_embeddings, attention_mask=None
                )
        finally:
            h1.remove()
            h2.remove()

        qk_rope_dim = mla_bridge._qk_rope_head_dim
        assert torch.allclose(q_values[0][..., -qk_rope_dim:], rot_q_values[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Rotary layout and YaRN scaling — the two places a real DeepSeek checkpoint
# differs from the tiny default config above.
# ---------------------------------------------------------------------------


def _config_variant(**overrides):
    base = dict(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=1,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=1000,
        first_k_dense_replace=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
        moe_intermediate_size=256,
        attn_implementation="eager",
    )
    base.update(overrides)
    return DeepseekV3Config(**base)


# All released DeepSeek V2 / V3 / R1 checkpoints use YaRN with mscale_all_dim=1,
# which HF folds into attention.scaling as mscale**2 (~1.87x at factor 40).
_YARN = {
    "rope_type": "yarn",
    "rope_theta": 10000.0,
    "factor": 40.0,
    "original_max_position_embeddings": 128,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


def _attn_parity(config) -> tuple[float, "MLAAttentionBridge", object]:
    torch.manual_seed(0)
    model = DeepseekV3ForCausalLM(config).eval()
    hf_attn = model.model.layers[0].self_attn
    bridge = MLAAttentionBridge(name="self_attn", config=config, submodules={})
    bridge.set_original_component(hf_attn)
    bridge.set_rotary_emb(model.model.rotary_emb)
    hidden = torch.randn(2, 8, config.hidden_size)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    cos_sin = model.model.rotary_emb(hidden, pos)
    mask = _causal_mask(2, 8)
    with torch.no_grad():
        hf_out = hf_attn(hidden, position_embeddings=cos_sin, attention_mask=mask)[0]
        br_out = bridge(hidden, position_embeddings=cos_sin, attention_mask=mask)[0]
    return (hf_out - br_out).abs().max().item(), bridge, hf_attn


class TestMLARotaryLayoutAndScaling:
    def test_interleaved_rope_matches_hf(self) -> None:
        max_diff, bridge, _ = _attn_parity(_config_variant(rope_interleave=True))
        assert bridge._rope_interleave is True
        assert max_diff < 1e-5, f"interleaved RoPE diverges: {max_diff}"

    def test_rotate_half_rope_matches_hf(self) -> None:
        max_diff, bridge, _ = _attn_parity(_config_variant(rope_interleave=False))
        assert bridge._rope_interleave is False
        assert max_diff < 1e-5, f"rotate-half RoPE diverges: {max_diff}"

    def test_yarn_scaling_matches_hf(self) -> None:
        config = _config_variant(rope_interleave=True, rope_parameters=dict(_YARN))
        max_diff, bridge, hf_attn = _attn_parity(config)
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        # Sanity: this config really does change the scale, so the parity below is meaningful.
        assert hf_attn.scaling > 1.5 * qk_head_dim ** (-0.5)
        assert bridge._scaling == pytest.approx(hf_attn.scaling)
        assert max_diff < 1e-5, f"YaRN-scaled attention diverges: {max_diff}"

    def test_scaling_falls_back_without_hf_attr(self) -> None:
        config = _config_variant()
        _, bridge, hf_attn = _attn_parity(config)
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        assert hf_attn.scaling == pytest.approx(qk_head_dim ** (-0.5))
        assert bridge._scaling == pytest.approx(qk_head_dim ** (-0.5))
