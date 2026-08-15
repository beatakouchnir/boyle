# SPDX-License-Identifier: Apache-2.0
"""predict: speed, context, and measured accuracy — before any download.

The forecast rests on three measured legs. (1) Anatomy: exact tensor
shapes/dtypes, read locally from checkpoint headers or remotely via ranged
HTTP reads of each shard's safetensors header — a few hundred KB, never the
weights. (2) The family hit-curve: decode hit rate and misses-per-token vs
budget fraction, distilled from routing traces (quant-independent —
measured identical at 4-bit and 8-bit). (3) This machine's fetch bandwidth,
from a ~10 s first-run probe, cached.

Speed model: step_ms = base_ms + misses_per_token(f) x expert_bytes / bw.
base_ms (compute + per-layer sync) is derived from a measured anchor of the
same family; cross-quant or cross-machine extrapolation widens the band.
Accuracy is *never* forecast — the accuracy section is lookup into measured
rows, or silence.
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.request
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from boyle.budget import (
    BudgetError,
    BudgetPlan,
    ModelAnatomy,
    fmt_size,
    max_context_for,
    parse_size,
    plan as resolve_budget,
)
from boyle.loader import _resolve_model_dir, classify_specs, read_anatomy

_TTFT_OVERHEAD = 2.0  # measured fill-vs-raw-bytes ratio at the 397B anchor
_CAL_PATH = Path.home() / ".cache" / "boyle" / "calibration.json"
_FALLBACK_BW = 5e9


def _data(name: str) -> dict:
    with resources.files("boyle.data").joinpath(name).open() as f:
        return json.load(f)


def family_key(config: dict | None) -> str | None:
    if not config:
        return None
    cfg = config.get("text_config", config)
    mt = (config.get("model_type") or cfg.get("model_type") or "").lower()
    for suffix in ("_text",):
        mt = mt.removesuffix(suffix)
    curves = _data("curves.json")
    if mt in curves:
        return mt
    # gemma4_text -> gemma4 handled by suffix strip; try prefix matches last
    for key in curves:
        if mt.startswith(key):
            return key
    return None


def interp(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


# --- machine calibration -------------------------------------------------


_IMPLAUSIBLE_BW = 25e9  # no consumer NVMe does this; above it = cache, not disk


def _cold_probe_files(min_total: int = 30 << 30) -> list[tuple[str, int]]:
    """Big local safetensors shards — the only reliably cold bytes around.

    A written-then-read probe file measures the page cache, not the disk:
    F_NOCACHE stops pages being *added*, not already-cached pages being
    *served* (a fresh 1 GB probe read back at 90 GB/s). Model shards dwarf
    RAM, so random offsets across them are cold with high probability.
    """
    root = Path.home() / ".cache" / "huggingface" / "hub"
    files = []
    if root.exists():
        for f in root.rglob("*.safetensors"):
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size >= (1 << 30):
                files.append((str(f), size))
    files.sort(key=lambda x: -x[1])
    files = files[:64]
    return files if sum(s for _, s in files) >= min_total else []


def calibrate(force: bool = False) -> dict:
    """Parallel-random pread bandwidth, measured once and cached.

    Mirrors the runtime's real access pattern: F_NOCACHE fds, 7 MB reads at
    random offsets across 8 threads (sequential-read numbers overstate what
    expert fetches see; parallel-random is what the drive actually does).
    """
    if not force and _CAL_PATH.exists():
        return json.loads(_CAL_PATH.read_text())
    _CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
        import random
        import threading

        sources = _cold_probe_files()
        if not sources:
            raise RuntimeError(
                "no large local model files to probe against — download a "
                "model first, then re-run"
            )
        read_bytes = [0] * 8
        stop = time.monotonic() + 3.0

        def worker(i):
            rng = random.Random(i)
            fds = {}
            size = 7 << 20
            try:
                while time.monotonic() < stop:
                    path, fsize = sources[rng.randrange(len(sources))]
                    fd = fds.get(path)
                    if fd is None:
                        fd = fds[path] = os.open(path, os.O_RDONLY)
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    off = rng.randrange(0, max(1, fsize - size))
                    read_bytes[i] += len(os.pread(fd, size, off))
            finally:
                for fd in fds.values():
                    os.close(fd)

        t0 = time.monotonic()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bw = sum(read_bytes) / (time.monotonic() - t0)
        if bw > _IMPLAUSIBLE_BW:
            raise RuntimeError(
                f"measured {bw / 1e9:.0f} GB/s — that is the page cache, "
                "not the disk; refusing to calibrate on it"
            )
        cal = {"bandwidth_bytes_s": bw, "measured_at": time.strftime("%Y-%m-%d")}
    except Exception as e:  # non-macOS, no big files, cache-contaminated
        cal = {
            "bandwidth_bytes_s": _FALLBACK_BW,
            "note": f"bandwidth probe unavailable ({e}); using a conservative "
            f"{_FALLBACK_BW / 1e9:.0f} GB/s default",
        }
    _CAL_PATH.write_text(json.dumps(cal))
    return cal


# --- remote anatomy ------------------------------------------------------


def _ranged(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def parse_header_bytes(prefix: bytes, header_len: int | None = None) -> dict:
    """Specs from the head of a safetensors file: {name: (shape, dtype)}."""
    if header_len is None:
        header_len = struct.unpack("<Q", prefix[:8])[0]
    header = json.loads(prefix[8 : 8 + header_len])
    return {
        name: (tuple(spec["shape"]), spec["dtype"])
        for name, spec in header.items()
        if name != "__metadata__"
    }


def anatomy_from_hub(repo: str) -> tuple[ModelAnatomy, dict]:
    """Anatomy without downloading weights: config.json + ranged header reads."""
    from huggingface_hub import HfApi, hf_hub_download

    config = json.loads(
        Path(hf_hub_download(repo, "config.json")).read_text()
    )
    files = [
        f
        for f in HfApi().list_repo_files(repo)
        if f.endswith(".safetensors")
    ]
    if not files:
        raise FileNotFoundError(f"{repo}: no safetensors files")
    specs: dict = {}
    for fname in files:
        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        head = _ranged(url, 0, 8)
        header_len = struct.unpack("<Q", head)[0]
        specs.update(
            parse_header_bytes(head + _ranged(url, 8, header_len), header_len)
        )
    return classify_specs(specs, config), config


def _anatomy(model: str) -> tuple[ModelAnatomy, dict | None, bool]:
    """(anatomy, config, is_local). Local checkpoint wins; hub needs no weights."""
    p = Path(model)
    if p.is_dir():
        cfg_p = p / "config.json"
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else None
        return read_anatomy(p), cfg, True
    try:
        d = _resolve_model_dir_local_only(model)
        cfg_p = d / "config.json"
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else None
        return read_anatomy(d), cfg, True
    except Exception:
        anatomy, cfg = anatomy_from_hub(model)
        return anatomy, cfg, False


def _resolve_model_dir_local_only(model: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model,
            local_files_only=True,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )


# --- the forecast --------------------------------------------------------


@dataclass
class Forecast:
    model: str
    fits: bool
    message: str = ""
    plan: BudgetPlan | None = None
    family: str | None = None
    curve_source: str = ""
    hit_rate: float = 0.0
    tok_s: float = 0.0
    tok_s_lo: float = 0.0
    tok_s_hi: float = 0.0
    ttft_cold_s: float = 0.0
    max_context_headroom: int = 0
    store_bytes: int = 0
    bandwidth_bytes_s: float = 0.0
    accuracy_rows: list = field(default_factory=list)
    accuracy_notes: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def render(self) -> str:
        if not self.fits:
            return f"boyle predict — {self.model}\n  DOES NOT FIT: {self.message}"
        p = self.plan
        lines = [
            f"boyle predict — {self.model}",
            f"  budget {fmt_size(p.budget_bytes)}: FITS "
            f"(fraction {p.fraction:.2f}, slots {fmt_size(p.slots_bytes)}, "
            f"resident {fmt_size(p.resident_bytes)})",
            f"  decode ~{self.tok_s:.1f} tok/s "
            f"(band {self.tok_s_lo:.1f}–{self.tok_s_hi:.1f}) — "
            f"expert hit rate ~{100 * self.hit_rate:.0f}% "
            f"[{self.curve_source}]",
            f"  cold fill-heavy TTFT ~{self.ttft_cold_s:.0f} s (worst case); "
            f"warm turns are prefix-cached",
            f"  context: {p.max_context} guaranteed at this budget "
            f"(headroom to ~{self.max_context_headroom})",
            f"  disk: {fmt_size(self.store_bytes)} checkpoint",
        ]
        for row in self.accuracy_rows:
            ref = f" ({row['reference']})" if row.get("reference") else ""
            lines.append(
                f"  accuracy [measured]: {row['task']} = {row['score']}"
                f" (n={row['n']}){ref} — {row['source']}"
            )
        if not self.accuracy_rows:
            lines.append(
                "  accuracy: no measured rows for this model/quant — "
                "boyle never forecasts accuracy"
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def _quant_bits(config: dict | None, anatomy: ModelAnatomy) -> float:
    if config:
        q = config.get("quantization") or config.get("quantization_config") or {}
        if isinstance(q, dict) and q.get("bits"):
            return float(q["bits"])
    # fall back to bytes-per-parameter arithmetic on the expert mass
    return 4.0


def predict(
    model: str,
    budget: int | float | str,
    max_context: int = 8192,
    headroom: int | float | str = "4GB",
) -> Forecast:
    anatomy, config, is_local = _anatomy(model)
    curves = _data("curves.json")
    anchors_data = _data("anchors.json")
    accuracy = _data("accuracy.json")
    cal = calibrate()
    bw = float(cal["bandwidth_bytes_s"])

    try:
        p = resolve_budget(anatomy, budget, max_context=max_context, headroom=headroom)
    except BudgetError as e:
        return Forecast(model=model, fits=False, message=str(e))

    fam = family_key(config)
    notes = []
    if "note" in cal:
        notes.append(cal["note"])

    n_layers = len(anatomy.layers)
    total_expert_slots = sum(n for n, _ in anatomy.layers)
    avg_expert_bytes = anatomy.expert_bytes / total_expert_slots if total_expert_slots else 0

    if fam and fam in curves:
        c = curves[fam]
        curve_source = f"{fam} curve, measured"
        if c["n_experts"] != anatomy.layers[0][0] or c["n_layers"] != n_layers:
            curve_source = f"{fam} curve transferred across sizes"
            notes.append(
                f"curve measured on {c['n_layers']}x{c['n_experts']} experts; "
                f"this model is {n_layers}x{anatomy.layers[0][0]} — same family, "
                f"wider band"
            )
        hit = interp(c["fractions"], c["decode_hit_rate"], p.fraction)
        # misses scale with this model's layer/expert structure
        mpt_curve = interp(c["fractions"], c["decode_misses_per_token"], p.fraction)
        mpt = mpt_curve * (n_layers * anatomy.layers[0][0]) / (
            c["n_layers"] * c["n_experts"]
        )
        band = 1.25
    else:
        curve_source = "flat-routing prior (family unmeasured)"
        hits, mpts = [], []
        for c in curves.values():
            hits.append(interp(c["fractions"], c["decode_hit_rate"], p.fraction))
            m = interp(c["fractions"], c["decode_misses_per_token"], p.fraction)
            mpts.append(m * (n_layers * anatomy.layers[0][0]) / (c["n_layers"] * c["n_experts"]))
        hit = sum(hits) / len(hits)
        mpt = sum(mpts) / len(mpts)
        band = 1.6
        notes.append(
            "routing curve is a prior from 4 measured families; "
            "`boyle trace` then `bench` tightens this"
        )

    io_ms = 1000 * mpt * avg_expert_bytes / bw

    fam_anchors = [
        a
        for a in anchors_data["anchors"]
        if a["family"] == fam and not a.get("exclude_from_base")
    ]
    if fam_anchors:
        anchor_bw = anchors_data["bandwidth_bytes_s"]
        c = curves[fam]
        bases = []
        for a in fam_anchors:
            a_mpt = interp(c["fractions"], c["decode_misses_per_token"], a["fraction"])
            a_bytes = c["expert_bytes"] * a["quant_bits"] / c["trace_quant_bits"]
            base = 1000 / a["tok_s"] - 1000 * a_mpt * a_bytes / anchor_bw
            bases.append(max(1.0, base))
        base_ms = sum(bases) / len(bases)
        if max(bases) / min(bases) > 1.5:
            band = max(band, 1.5)
            notes.append("family anchors disagree on the compute floor; band widened")
        step_ms = base_ms + io_ms
        tok_s = 1000 / step_ms
    else:
        tok_s = 1000 / io_ms if io_ms > 0 else 0.0
        band = 2.0
        notes.append(
            "no compute anchor for this family — the estimate is I/O-only "
            "and therefore an upper bound; `boyle bench` measures the truth"
        )
        if avg_expert_bytes < 2e6:
            notes.append(
                f"expert records are small ({fmt_size(int(avg_expert_bytes))}): "
                "per-read latency dominates over bandwidth and the upper bound "
                "is loose — definitely bench this one"
            )

    ttft = _TTFT_OVERHEAD * anatomy.expert_bytes * (1 - p.fraction) / bw

    rows = [
        r
        for r in accuracy["rows"]
        if r.get("model") == model
        or (fam and r.get("family") == fam and r.get("quant_bits") == _quant_bits(config, anatomy))
    ]

    return Forecast(
        model=model,
        fits=True,
        plan=p,
        family=fam,
        curve_source=curve_source,
        hit_rate=hit,
        tok_s=tok_s,
        tok_s_lo=tok_s / band,
        tok_s_hi=tok_s * band,
        ttft_cold_s=ttft,
        max_context_headroom=max_context_for(anatomy, budget, p.fraction, headroom=headroom),
        store_bytes=anatomy.resident_bytes + anatomy.expert_bytes,
        bandwidth_bytes_s=bw,
        accuracy_rows=rows,
        accuracy_notes=accuracy["notes"],
        notes=notes,
    )
