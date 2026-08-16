# SPDX-License-Identifier: Apache-2.0
"""boyle build: colocated expert stores.

One contiguous record per (layer, expert) holding all projection fields
back-to-back, so a cache miss becomes ONE contiguous ~7 MB pread instead
of nine scattered reads, six of them ~131 KB latency-bound. Measured
+13% on the diverse-serving ceiling; the runtime reads this format via
``--colo``. Experts are written in id order in v1 — trace-driven
co-activation ordering lands with `boyle trace`.

Stacked checkpoints only (3-D ``[E, ...]`` projection tensors — Qwen,
gemma lineage). Per-expert-scheme checkpoints are already read directly
by the runtime; a colo store for them is not supported in v1.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

from boyle.loader import _resolve_model_dir

_PROJ_RE = re.compile(
    r"^(.*)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
_FIELD_ORDER = tuple(
    f"{p}.{f}"
    for p in ("gate_proj", "up_proj", "down_proj")
    for f in ("weight", "scales", "biases")
)
_CHUNK = 64 << 20


def default_store_dir(model: str) -> Path:
    safe = str(model).replace("/", "--")
    return Path.home() / ".cache" / "boyle" / "stores" / safe


def build_store(
    model: str | Path,
    out_dir: str | Path | None = None,
    progress=print,
) -> Path:
    """Write colo.bin + colo.json for a stacked-layout checkpoint."""
    model_dir = _resolve_model_dir(model)
    out = Path(out_dir) if out_dir else default_store_dir(model)
    out.mkdir(parents=True, exist_ok=True)

    # pass 1: stacked projection tensors from shard headers
    specs: dict[str, tuple[Path, str, tuple, int, int]] = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        for name, spec in hdr.items():
            if name == "__metadata__" or len(spec.get("shape", ())) != 3:
                continue
            if _PROJ_RE.match(name):
                b0, b1 = spec["data_offsets"]
                specs[name] = (
                    shard, spec["dtype"], tuple(spec["shape"]), base + b0, b1 - b0
                )
    if not specs:
        raise SystemExit(
            "no stacked expert tensors found — this checkpoint either has no "
            "MoE layers or uses the per-expert naming scheme, which the "
            "runtime reads directly (a colo store adds nothing there in v1)"
        )

    layers: dict[str, dict[str, str]] = {}
    for name in specs:
        m = _PROJ_RE.match(name)
        layers.setdefault(m.group(1), {})[f"{m.group(2)}.{m.group(3)}"] = name
    prefixes = sorted(
        layers,
        key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p)],
    )

    # pass 2: records, expert-id order
    index = {}
    pos = 0
    with open(out / "colo.bin", "wb") as sink:
        for li, prefix in enumerate(prefixes):
            fields = layers[prefix]
            n_experts = specs[next(iter(fields.values()))][2][0]
            recs = {}
            for e in range(n_experts):
                rec_off, rel, cursor = pos, {}, 0
                for key in _FIELD_ORDER:
                    name = fields.get(key)
                    if name is None:
                        continue
                    shard, dtype, shape, off, nb = specs[name]
                    slab = nb // shape[0]
                    with open(shard, "rb") as f:
                        f.seek(off + e * slab)
                        left = slab
                        while left:
                            buf = f.read(min(_CHUNK, left))
                            sink.write(buf)
                            left -= len(buf)
                    rel[key] = [cursor, dtype, list(shape[1:])]
                    cursor += slab
                pos += cursor
                recs[str(e)] = {"offset": rec_off, "fields": rel}
            index[prefix] = {"order": list(range(n_experts)), "records": recs}
            if li % 10 == 0 or li == len(prefixes) - 1:
                progress(f"[build] layer {li + 1}/{len(prefixes)}: "
                         f"{pos / 1e9:.1f} GB written")
    with open(out / "colo.json", "w") as f:
        json.dump(index, f)
    progress(f"[build] colo store: {pos / 1e9:.2f} GB -> {out}")
    progress(f"[build] use it: boyle serve {model} --budget <B> --colo {out}")
    return out
