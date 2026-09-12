# SPDX-License-Identifier: Apache-2.0
"""boyle trace — capture a decode routing trace and turn it into a predict curve.

The runtime records, per offloaded layer, the sequence of routed expert ids
(``RECORD_TRACE``). We replay that sequence through an LRU cache of every
budget fraction and read off the decode hit rate, exactly the family curve
``predict`` consumes. Per-access LRU has the inclusion property, so the hit
rate is monotone in capacity by construction — the invariant the curve tests
enforce. The routing itself is budget-independent, so one streaming run yields
the whole curve; we only need the cache non-warm so it keeps recording.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import boyle._runtime as rt

# The fraction grid the shipped curves use; predict interpolates within it.
FRACTIONS = [0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.36, 0.45, 0.55, 0.64, 0.75, 0.9, 1.0]

_DEFAULT_PROMPT = "Write a detailed, factual overview of how solid-state drives work."


def _lru_hit_rate(seq: list[int], capacity: int) -> float:
    """Decode hit rate of an LRU cache of ``capacity`` slots over ``seq``."""
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    for e in seq:
        if e in cache:
            cache.move_to_end(e)
            hits += 1
        else:
            cache[e] = None
            if len(cache) > capacity:
                cache.popitem(last=False)  # evict LRU
    return hits / len(seq) if seq else 0.0


def curve_from_traces(traces, n_experts, k, model, anatomy, config, fractions=FRACTIONS):
    """Assemble a curves.json family entry from per-layer routing traces."""
    from boyle.predict import _quant_bits

    # Decode steps route exactly k experts; the prefill entry is prompt_len*k
    # long. Keep the k-length entries and flatten each layer to one sequence.
    seqs = []
    for layer in traces:
        steps = [e for e in layer if len(e) == k]
        seq = [x for step in steps for x in step]
        if seq:
            seqs.append(seq)
    if not seqs:
        raise SystemExit("trace captured no decode steps — generate more tokens")

    n_layers = len(seqs)
    hit = []
    for f in fractions:
        cap = max(1, round(f * n_experts))
        rates = [_lru_hit_rate(s, cap) for s in seqs]
        hit.append(sum(rates) / len(rates))
    for i in range(1, len(hit)):  # numerical-noise guard; LRU is already monotone
        hit[i] = max(hit[i], hit[i - 1])
    misses = [k * n_layers * (1.0 - h) for h in hit]

    expert_bytes = anatomy.expert_bytes // (n_layers * n_experts)
    return {
        "model": model,
        "n_layers": n_layers,
        "n_experts": int(n_experts),
        "k": int(k),
        "expert_bytes": int(expert_bytes),
        "trace_quant_bits": int(_quant_bits(config, anatomy)),
        "fractions": list(fractions),
        "decode_hit_rate": [round(h, 4) for h in hit],
        "decode_misses_per_token": [round(m, 2) for m in misses],
    }


def _family_of(config: dict | None, model: str) -> str:
    cfg = (config or {}).get("text_config", config or {})
    mt = (config or {}).get("model_type") or cfg.get("model_type") or ""
    mt = mt.lower().removesuffix("_text")
    return mt or Path(model).name.lower()


def capture_curve(model, budget, *, max_tokens=512, max_context=8192,
                  headroom="4GB", colo_dir=None, prompt=None):
    """Load at a streaming budget, decode, and return (family, curve entry)."""
    from boyle.loader import load
    from boyle.predict import _anatomy

    k = None
    anatomy, config, _ = _anatomy(model)
    cfg = (config or {}).get("text_config", config or {})
    k = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token")
    if not k:
        raise SystemExit("could not read num_experts_per_tok from the config")

    rt.RECORD_TRACE = True
    try:
        m = load(model, budget=budget, max_context=max_context,
                 headroom=headroom, colo_dir=colo_dir)
        if m.plan.fraction >= 1.0:
            raise SystemExit(
                f"trace needs a streaming budget so the cache keeps recording, "
                f"but this budget is fully resident (fraction {m.plan.fraction:.2f}). "
                f"Pass a smaller --budget."
            )
        for _ in m.generate(prompt or _DEFAULT_PROMPT, max_tokens=max_tokens):
            pass
        traces, n_experts = rt.offload_traces(m.model)
    finally:
        rt.RECORD_TRACE = False

    if not traces:
        raise SystemExit("no routing captured (cache stayed warm?) — use a smaller --budget")
    curve = curve_from_traces(traces, n_experts, int(k), model, anatomy, config)
    return _family_of(config, model), curve


def write_curve(family: str, curve: dict, path: Path | None = None) -> Path:
    """Merge a family curve into curves.json (create/replace the family key)."""
    path = path or (Path(__file__).parent / "data" / "curves.json")
    data = json.loads(path.read_text())
    data[family] = curve
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path
