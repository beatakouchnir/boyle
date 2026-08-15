# SPDX-License-Identifier: Apache-2.0
"""Local integration: bit-identity against the resident model.

Needs cached checkpoints and a Metal device — not CI. Run with:

    BOYLE_LOCAL_MODELS=1 uv run pytest tests/test_integration_local.py -v

Contract under test (the runtime docstring's claim, measured in the research
record): with prompt_tokens x top_k <= capacity the entire generation takes
the bit-identical path, so token ids must match the resident model exactly.
The long-prompt case exercises the expert-major prefill, which promises
rounding-equivalence; empirically that has meant identical text on every
model measured, so text equality is asserted — if it ever fails, that is
contract-relevant news, not flake.
"""

import gc
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BOYLE_LOCAL_MODELS") != "1",
    reason="local-model integration (set BOYLE_LOCAL_MODELS=1)",
)

OLMOE = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
GEMMA = "mlx-community/gemma-4-26b-a4b-it-4bit"
SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = (
    "Explain, in three careful paragraphs, why the sky appears blue during "
    "the day and red at sunset, mentioning Rayleigh scattering, the "
    "wavelength dependence of scattering intensity, and the longer optical "
    "path length near the horizon. Then summarize the whole explanation in "
    "one sentence a child could understand."
)


def _generate_resident(model_id, prompt, max_tokens):
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    model, tokenizer = load(model_id)
    ids = [r.token for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens)]
    text = tokenizer.decode(ids)
    del model
    gc.collect()
    mx.clear_cache()
    return ids, text


def _generate_budgeted(model_id, prompt, max_tokens, **load_kwargs):
    import mlx.core as mx

    from boyle.loader import load

    m = load(model_id, **load_kwargs)
    ids = [r.token for r in m.generate(prompt, max_tokens=max_tokens)]
    text = m.tokenizer.decode(ids)
    stats = m.stats()
    plan = m.plan
    del m
    gc.collect()
    mx.clear_cache()
    return ids, text, stats, plan


def test_olmoe_decode_bit_identity():
    """Short prompt: whole run on the bit-identical path."""
    ref_ids, ref_text = _generate_resident(OLMOE, SHORT_PROMPT, 64)
    ids, text, stats, plan = _generate_budgeted(
        OLMOE, SHORT_PROMPT, 64,
        budget="4.8GB", max_context=2048, headroom="1GB",
    )
    # Guard the guarantee's precondition before asserting its consequence.
    assert plan.capacities[0] >= 56, plan.capacities[0]
    assert stats["layers"] == 16
    assert 0 < plan.fraction < 1  # genuinely partial, not accidentally resident
    assert stats["misses"] > 0, stats  # the offload mechanism actually ran
    assert ids == ref_ids, f"token divergence: {text!r} vs {ref_text!r}"


def test_gemma_expert_major_prefill_text_identity():
    """Long prompt at low fraction: expert-major prefill, decode after."""
    ref_ids, ref_text = _generate_resident(GEMMA, LONG_PROMPT, 48)
    ids, text, stats, plan = _generate_budgeted(
        GEMMA, LONG_PROMPT, 48,
        budget="8GB", max_context=2048, headroom="2GB",
    )
    assert plan.fraction < 0.25
    assert stats["misses"] > 0
    assert text == ref_text, f"text divergence:\n{text!r}\nvs\n{ref_text!r}"
