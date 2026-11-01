"""Acceptance tests for the Cohere (Command-R) adapter in TransformerBridge.

Two layers:

1. The registry's own gate. ``run_benchmark_suite`` is what ``verify_models``
   scores; running Phases 1–3 here turns a Phase 3 regression (the double
   ``logit_scale`` application in compatibility mode) into a failing test
   instead of a note in supported_models.json. Phase 4 needs the GPT-2 scoring
   model and a legible checkpoint, so it stays with ``verify_models``.
2. End-to-end user workflows on a compatibility-mode bridge — loss, greedy
   generation through the KV cache, batched caching, HT-alias hook shapes —
   which the integration tier (single forward passes) does not exercise.

Model: trl-internal-testing/tiny-CohereForCausalLM (2 layers, ungated,
logit_scale=0.125, use_qk_norm=False). No HookedTransformer comparison is
possible: Cohere is Bridge-only.
"""

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.benchmarks import run_benchmark_suite
from transformer_lens.benchmarks.utils import BenchmarkSeverity
from transformer_lens.model_bridge.bridge import TransformerBridge

# Deliberately NOT marked slow: `make acceptance-test` deselects slow, and this
# is the CI gate for the Cohere family. The checkpoint is ~8M params (same one
# tests/integration/model_bridge/test_cohere_adapter.py already loads unmarked).

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"
TEXT = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def cohere_hf() -> Any:
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()


@pytest.fixture(scope="module")
def cohere_compat() -> TransformerBridge:
    """Compatibility-mode bridge: the configuration a mech-interp session uses."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    bridge.enable_compatibility_mode(disable_warnings=True)
    return bridge


# ---------------------------------------------------------------------------
# 1. verify_models Phases 1–3 as a test
# ---------------------------------------------------------------------------


class TestCohereBenchmarkSuite:
    def test_phases_1_to_3_have_no_failures(self) -> None:
        """Every non-skipped benchmark in Phases 1–3 must pass.

        Phase 1 compares the raw bridge to HF. Phase 2 has no HookedTransformer
        reference for Cohere and is expected to skip. Phase 3 enables
        compatibility mode and is where a logit_scale fold that leaves HF's
        runtime multiply live shows up (loss/logits off by 1/logit_scale).
        """
        results = run_benchmark_suite(
            MODEL,
            device="cpu",
            dtype=torch.float32,
            use_hf_reference=True,
            use_ht_reference=False,
            enable_compatibility_mode=True,
            phases=[1, 2, 3],
            verbose=False,
        )
        assert results, "Benchmark suite returned no results"
        failures = [
            r
            for r in results
            if r.severity != BenchmarkSeverity.SKIPPED
            and (r.severity in (BenchmarkSeverity.DANGER, BenchmarkSeverity.ERROR) or not r.passed)
        ]
        report = "\n".join(f"  [phase {r.phase}] {r.name}: {r.message}" for r in failures)
        assert not failures, f"verify_models-style failures for {MODEL}:\n{report}"

    def test_phase_3_actually_ran(self) -> None:
        # Guard against the gate silently passing because Phase 3 was skipped.
        results = run_benchmark_suite(
            MODEL,
            device="cpu",
            use_ht_reference=False,
            enable_compatibility_mode=True,
            phases=[3],
            verbose=False,
        )
        ran = [r for r in results if r.phase == 3 and r.severity != BenchmarkSeverity.SKIPPED]
        assert ran, "No Phase 3 benchmark executed"


# ---------------------------------------------------------------------------
# 2. Compatibility-mode workflows
# ---------------------------------------------------------------------------


class TestCohereCompatWorkflows:
    def test_loss_matches_hf(self, cohere_compat: TransformerBridge, cohere_hf: Any) -> None:
        """Cross-entropy is scale-sensitive: a doubled logit_scale inflates it toward log(d_vocab)."""
        tokens = cohere_compat.to_tokens(TEXT)
        with torch.no_grad():
            bridge_loss = cohere_compat(tokens, return_type="loss").item()
            hf_loss = cohere_hf(tokens, labels=tokens).loss.item()
        assert abs(bridge_loss - hf_loss) < 1e-3, f"loss bridge={bridge_loss:.6f} hf={hf_loss:.6f}"

    def test_greedy_generation_matches_hf(
        self, cohere_compat: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Greedy decoding through the bridge's KV-cache path must track HF token for token."""
        tokens = cohere_compat.to_tokens(TEXT)
        n_new = 8
        with torch.no_grad():
            bridge_out = cohere_compat.generate(
                tokens,
                max_new_tokens=n_new,
                do_sample=False,
                stop_at_eos=False,
                use_past_kv_cache=True,
                verbose=False,
            )
            hf_out = cohere_hf.generate(
                tokens,
                max_new_tokens=n_new,
                min_new_tokens=n_new,
                do_sample=False,
                pad_token_id=cohere_hf.config.pad_token_id or cohere_hf.config.eos_token_id,
            )
        assert isinstance(bridge_out, torch.Tensor)
        assert bridge_out.shape == hf_out.shape, f"{bridge_out.shape} vs {hf_out.shape}"
        assert torch.equal(bridge_out, hf_out), (
            f"Greedy continuation diverged:\n  bridge={bridge_out[0, -n_new:].tolist()}\n"
            f"  hf={hf_out[0, -n_new:].tolist()}"
        )

    def test_kv_cache_generation_matches_no_cache(self, cohere_compat: TransformerBridge) -> None:
        tokens = cohere_compat.to_tokens(TEXT)
        with torch.no_grad():
            with_cache = cohere_compat.generate(
                tokens,
                max_new_tokens=6,
                do_sample=False,
                stop_at_eos=False,
                use_past_kv_cache=True,
                verbose=False,
            )
            without_cache = cohere_compat.generate(
                tokens,
                max_new_tokens=6,
                do_sample=False,
                stop_at_eos=False,
                use_past_kv_cache=False,
                verbose=False,
            )
        assert torch.equal(with_cache, without_cache)

    def test_batched_run_with_cache_matches_individual(
        self, cohere_compat: TransformerBridge
    ) -> None:
        """Uneven-length batch: last-token logits must match per-prompt runs (padding masked)."""
        prompts = ["Hello, my dog is cute", "This is a much longer text. Hello, my cat is cute"]
        individual = []
        for p in prompts:
            logits, _ = cohere_compat.run_with_cache(p)
            individual.append(logits[0, -1, :])
        batched, _ = cohere_compat.run_with_cache(prompts)
        for i in range(len(prompts)):
            assert torch.allclose(
                individual[i], batched[i, -1, :], atol=1e-4
            ), f"Prompt {i}: batched last-token logits differ from individual run"

    def test_ht_alias_hook_shapes(self, cohere_compat: TransformerBridge) -> None:
        """HT-style aliases resolve and carry HT shapes, including GQA-sized K/V."""
        cfg = cohere_compat.cfg
        tokens = cohere_compat.to_tokens(TEXT)
        n_pos = tokens.shape[1]
        n_kv = getattr(cfg, "n_key_value_heads", None) or cfg.n_heads
        _, cache = cohere_compat.run_with_cache(tokens)
        assert cache["blocks.0.attn.hook_q"].shape == (1, n_pos, cfg.n_heads, cfg.d_head)
        assert cache["blocks.0.attn.hook_k"].shape == (1, n_pos, n_kv, cfg.d_head)
        assert cache["blocks.0.attn.hook_v"].shape == (1, n_pos, n_kv, cfg.d_head)
        assert cache["blocks.0.attn.hook_z"].shape == (1, n_pos, cfg.n_heads, cfg.d_head)
        assert cache["blocks.0.attn.hook_pattern"].shape == (1, cfg.n_heads, n_pos, n_pos)
        assert cache["blocks.0.hook_resid_pre"].shape == (1, n_pos, cfg.d_model)
        assert cache["blocks.0.hook_resid_post"].shape == (1, n_pos, cfg.d_model)
        if "ln_final.hook_normalized" in cohere_compat.hook_dict:
            assert cache["ln_final.hook_normalized"].shape == (1, n_pos, cfg.d_model)
        # Parallel block: attention and MLP read the same pre-norm residual.
        assert "blocks.0.hook_resid_mid" not in cache

    def test_attention_pattern_is_causal_and_normalized(
        self, cohere_compat: TransformerBridge
    ) -> None:
        tokens = cohere_compat.to_tokens(TEXT)
        _, cache = cohere_compat.run_with_cache(tokens)
        pattern = cache["blocks.0.attn.hook_pattern"][0]  # [n_heads, pos, pos]
        upper = torch.triu(torch.ones_like(pattern[0]), diagonal=1).bool()
        assert torch.all(pattern[:, upper] == 0), "Attention leaks to future positions"
        row_sums = pattern.sum(-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_head_ablation_changes_logits(self, cohere_compat: TransformerBridge) -> None:
        """The canonical intervention — zero one head's hook_z — must reach the output."""
        tokens = cohere_compat.to_tokens(TEXT)

        def zero_head_0(z: torch.Tensor, hook: Any) -> torch.Tensor:
            z = z.clone()
            z[:, :, 0, :] = 0
            return z

        with torch.no_grad():
            base = cohere_compat(tokens)
            ablated = cohere_compat.run_with_hooks(
                tokens, fwd_hooks=[("blocks.0.attn.hook_z", zero_head_0)]
            )
            restored = cohere_compat(tokens)
        assert not torch.allclose(base, ablated), "Ablating head 0 left the logits unchanged"
        assert torch.allclose(base, restored), "Hook was not removed after run_with_hooks"
