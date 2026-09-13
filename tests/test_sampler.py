# SPDX-License-Identifier: Apache-2.0
"""The explicit-key sampler: top-k cut, top-p cut, seed semantics."""
import mlx.core as mx
import pytest

from boyle.server import _seeded_sampler


def _draws(sampler, logits, n=200):
    return {int(sampler(logits).item()) for _ in range(n)}


@pytest.fixture
def logits():
    # a flat-ish distribution so unrestricted sampling reaches many tokens
    mx.random.seed(0)
    return mx.log(mx.softmax(mx.random.normal((64,)) * 2.0))


def test_top_k_one_is_greedy(logits):
    sampler = _seeded_sampler(1.0, 1.0, seed=7, top_k=1)
    assert _draws(sampler, logits) == {int(mx.argmax(logits).item())}


def test_top_k_admits_only_the_k_largest(logits):
    k = 5
    allowed = set(mx.argsort(-logits)[:k].tolist())
    sampler = _seeded_sampler(1.0, 1.0, seed=7, top_k=k)
    seen = _draws(sampler, logits)
    assert seen <= allowed
    assert len(seen) > 1  # it still samples, not a disguised greedy


def test_top_k_zero_leaves_sampling_unrestricted(logits):
    sampler = _seeded_sampler(1.0, 1.0, seed=7, top_k=0)
    assert len(_draws(sampler, logits)) > 5


def test_same_seed_same_draw(logits):
    a = _seeded_sampler(1.0, 0.95, seed=3, top_k=20)
    b = _seeded_sampler(1.0, 0.95, seed=3, top_k=20)
    assert [int(a(logits).item()) for _ in range(20)] == [int(b(logits).item()) for _ in range(20)]


def test_top_k_composes_with_top_p(logits):
    # top-p 0.95 after a top-k 20 cut can only shrink the admitted set further
    allowed = set(mx.argsort(-logits)[:20].tolist())
    sampler = _seeded_sampler(1.0, 0.95, seed=11, top_k=20)
    assert _draws(sampler, logits) <= allowed
