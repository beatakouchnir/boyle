# SPDX-License-Identifier: Apache-2.0
"""Memory-budget arithmetic: what fits, what refuses, and by how much.

The budget covers everything boyle wires: resident (non-expert) weights,
expert slots, the KV cache at the declared context, and a fixed activation
headroom. Slots absorb whatever remains — they are the speed dial. Uniform
allocation across layers is deliberate: routing is flat and the hit curve
concave in every family measured, so per-layer cleverness cannot beat
uniform (Jensen). A request that cannot fit raises :class:`BudgetError`
naming the shortfall instead of degrading silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Mirrors the runtime's per-layer floor: below 8 slots the cache thrashes
# within a single token's top-k and the fraction knob stops meaning anything.
MIN_CAPACITY = 8

_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)I?B?\s*$", re.I)


def parse_size(size: int | float | str) -> int:
    """``"80GB"`` / ``"80G"`` / ``"0.5TB"`` / ``85899345920`` -> bytes.

    Decimal units (GB = 1e9), matching how machines and model cards are
    advertised. Accepts GiB spellings but treats them identically — a 7%
    fudge is not worth two unit systems in error messages.
    """
    if isinstance(size, (int, float)):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size!r}")
        return int(size)
    m = _SIZE_RE.match(size)
    if not m:
        raise ValueError(f"cannot parse size {size!r} (try e.g. '80GB')")
    value = float(m.group(1))
    scale = {"": 1, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}[
        m.group(2).upper()
    ]
    n = int(value * scale)
    if n <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    return n


def fmt_size(n: int) -> str:
    for unit, scale in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


class BudgetError(RuntimeError):
    """The request cannot fit; the message names the numbers."""


@dataclass(frozen=True)
class ModelAnatomy:
    """The memory-relevant shape of a model, independent of any machine."""

    resident_bytes: int  # non-expert weights: always wired
    layers: tuple[tuple[int, int], ...]  # per MoE layer: (n_experts, table_bytes)
    kv_bytes_per_token: int = 0

    @property
    def expert_bytes(self) -> int:
        return sum(b for _, b in self.layers)

    def slots_bytes(self, capacities: tuple[int, ...]) -> int:
        return sum(
            b * c // n for (n, b), c in zip(self.layers, capacities, strict=True)
        )

    def capacities_for(self, fraction: float) -> tuple[int, ...]:
        return tuple(
            max(MIN_CAPACITY, min(n, round(n * fraction))) for n, _ in self.layers
        )


@dataclass(frozen=True)
class BudgetPlan:
    """A resolved budget: what runs, at what fraction, with what context."""

    budget_bytes: int
    max_context: int
    headroom_bytes: int
    kv_bytes: int
    fraction: float
    capacities: tuple[int, ...]
    slots_bytes: int
    resident_bytes: int

    @property
    def planned_bytes(self) -> int:
        return (
            self.resident_bytes + self.slots_bytes + self.kv_bytes + self.headroom_bytes
        )


def plan(
    anatomy: ModelAnatomy,
    budget: int | float | str,
    max_context: int = 8192,
    headroom: int | float | str = "4GB",
) -> BudgetPlan:
    """Resolve a budget into per-layer slot capacities, or refuse.

    Context is honored before slots: serving promises the declared context,
    so KV is reserved first and the remainder becomes slot capacity (speed).
    Fully-resident layers are free wins and fall out of fraction=1.0.
    """
    budget_bytes = parse_size(budget)
    headroom_bytes = parse_size(headroom)
    kv_bytes = anatomy.kv_bytes_per_token * max_context
    for_experts = budget_bytes - anatomy.resident_bytes - kv_bytes - headroom_bytes

    floor_caps = anatomy.capacities_for(0.0)
    floor_bytes = anatomy.slots_bytes(floor_caps)
    if for_experts < floor_bytes:
        minimum = (
            anatomy.resident_bytes + kv_bytes + headroom_bytes + floor_bytes
        )
        raise BudgetError(
            f"budget {fmt_size(budget_bytes)} cannot fit this model at context "
            f"{max_context}: non-expert weights {fmt_size(anatomy.resident_bytes)} "
            f"+ KV cache {fmt_size(kv_bytes)} + headroom "
            f"{fmt_size(headroom_bytes)} + minimum expert slots "
            f"{fmt_size(floor_bytes)} = {fmt_size(minimum)} minimum. "
            f"Raise the budget to at least {fmt_size(minimum)}, lower "
            f"--max-context, or pick a smaller model/quant."
        )

    # slots_bytes(capacities_for(f)) is a monotone step function of f;
    # bisect for the largest affordable fraction, then recompute exactly.
    lo, hi = 0.0, 1.0
    if anatomy.slots_bytes(anatomy.capacities_for(1.0)) <= for_experts:
        lo = 1.0
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            if anatomy.slots_bytes(anatomy.capacities_for(mid)) <= for_experts:
                lo = mid
            else:
                hi = mid
    capacities = anatomy.capacities_for(lo)
    return BudgetPlan(
        budget_bytes=budget_bytes,
        max_context=max_context,
        headroom_bytes=headroom_bytes,
        kv_bytes=kv_bytes,
        fraction=lo,
        capacities=capacities,
        slots_bytes=anatomy.slots_bytes(capacities),
        resident_bytes=anatomy.resident_bytes,
    )


def max_context_for(
    anatomy: ModelAnatomy,
    budget: int | float | str,
    fraction: float,
    headroom: int | float | str = "4GB",
) -> int:
    """The largest context the budget supports at a given slot fraction.

    ``predict`` reports this next to the speed forecast: on very large
    models the KV cost of agent-sized contexts is the surprise, and saying
    the number before download is the feature.
    """
    if anatomy.kv_bytes_per_token == 0:
        return 10**9
    slots = anatomy.slots_bytes(anatomy.capacities_for(fraction))
    left = (
        parse_size(budget) - anatomy.resident_bytes - slots - parse_size(headroom)
    )
    return max(0, left // anatomy.kv_bytes_per_token)
