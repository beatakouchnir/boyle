# SPDX-License-Identifier: Apache-2.0
"""Budget arithmetic: pure math, no model, no accelerator."""

import pytest

from boyle.budget import (
    MIN_CAPACITY,
    BudgetError,
    ModelAnatomy,
    max_context_for,
    parse_size,
    plan,
)

GB = 10**9

# A 397B-class caricature: 40 GB resident, 60 layers x 512 experts x 4 GB
# tables would be absurd — keep tables at 3 GB/layer (180 GB expert mass).
BIG = ModelAnatomy(
    resident_bytes=40 * GB,
    layers=tuple((512, 3 * GB) for _ in range(60)),
    kv_bytes_per_token=200_000,
)
# A model that fits entirely: fraction must resolve to 1.0.
SMALL = ModelAnatomy(
    resident_bytes=2 * GB,
    layers=tuple((64, 100_000_000) for _ in range(20)),
    kv_bytes_per_token=50_000,
)


def test_parse_size_forms():
    assert parse_size("80GB") == 80 * GB
    assert parse_size("80 gb") == 80 * GB
    assert parse_size("80G") == 80 * GB
    assert parse_size("0.5TB") == 500 * GB
    assert parse_size("512MB") == 512 * 10**6
    assert parse_size("80GiB") == 80 * GB  # deliberate: one unit system
    assert parse_size(1234) == 1234


@pytest.mark.parametrize("bad", ["", "GB", "-5GB", "eighty", "0GB"])
def test_parse_size_rejects(bad):
    with pytest.raises(ValueError):
        parse_size(bad)


def test_small_model_goes_fully_resident():
    p = plan(SMALL, "20GB", max_context=8192)
    assert p.fraction == 1.0
    assert all(c == n for c, (n, _) in zip(p.capacities, SMALL.layers))
    assert p.planned_bytes <= p.budget_bytes


def test_big_model_partial_fraction_fits_budget():
    p = plan(BIG, "90GB", max_context=8192)
    assert 0 < p.fraction < 1
    assert p.planned_bytes <= p.budget_bytes
    # slots absorb most of what remains: within one layer-row of the budget
    slack = p.budget_bytes - p.planned_bytes
    per_step = max(b // n for n, b in BIG.layers)
    assert slack <= per_step * len(BIG.layers)


def test_bigger_budget_never_means_fewer_slots():
    caps = [
        plan(BIG, f"{g}GB", max_context=8192).capacities
        for g in (70, 90, 110)
    ]
    for lo, hi in zip(caps, caps[1:]):
        assert all(a <= b for a, b in zip(lo, hi))


def test_context_is_honored_before_slots():
    short = plan(BIG, "90GB", max_context=1024)
    long = plan(BIG, "90GB", max_context=65536)
    assert long.kv_bytes > short.kv_bytes
    assert long.slots_bytes <= short.slots_bytes


def test_refusal_names_the_numbers():
    with pytest.raises(BudgetError) as exc:
        plan(BIG, "45GB", max_context=8192)
    msg = str(exc.value)
    assert "45.00 GB" in msg
    assert "minimum" in msg
    assert "--max-context" in msg


def test_capacity_floor_is_respected():
    p = plan(BIG, "58GB", max_context=1024)
    assert all(c >= MIN_CAPACITY for c in p.capacities)


def test_max_context_for_monotone_in_fraction():
    lo = max_context_for(BIG, "90GB", 0.05)
    hi = max_context_for(BIG, "90GB", 0.15)
    assert lo > hi > 0


def test_max_context_for_zero_when_slots_eat_the_budget():
    # 0.3 x 180 GB expert mass + 40 GB resident + headroom > 90 GB: no room
    # for any KV. predict renders 0 as a refusal, never a silent clamp.
    assert max_context_for(BIG, "90GB", 0.3) == 0
