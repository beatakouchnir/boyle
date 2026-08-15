# SPDX-License-Identifier: Apache-2.0
"""predict internals: interpolation, header parsing, data contracts.

No network, no model, no accelerator — these run in CI. The live
out-of-sample check (predict vs bench on a fraction never measured) is a
manual acceptance step recorded in the project log.
"""

import json
import struct

import pytest

from boyle.predict import (
    _FALLBACK_BW,
    _data,
    calibrate,
    family_key,
    interp,
    parse_header_bytes,
)


def test_interp_edges_and_midpoint():
    xs, ys = [0.1, 0.5, 1.0], [10.0, 50.0, 100.0]
    assert interp(xs, ys, 0.05) == 10.0
    assert interp(xs, ys, 2.0) == 100.0
    assert interp(xs, ys, 0.3) == pytest.approx(30.0)
    assert interp(xs, ys, 0.75) == pytest.approx(75.0)


def test_family_key_normalization():
    assert family_key({"model_type": "qwen3_moe"}) == "qwen3_moe"
    assert family_key({"model_type": "qwen3_5_moe_text"}) == "qwen3_5_moe"
    assert family_key({"text_config": {"model_type": "gemma4_text"}}) == "gemma4"
    assert family_key({"model_type": "olmoe"}) is None
    assert family_key(None) is None


def test_parse_header_bytes_roundtrip():
    header = {
        "a.weight": {"dtype": "F32", "shape": [4, 8], "data_offsets": [0, 128]},
        "__metadata__": {"format": "pt"},
    }
    raw = json.dumps(header).encode()
    blob = struct.pack("<Q", len(raw)) + raw
    specs = parse_header_bytes(blob)
    assert specs == {"a.weight": ((4, 8), "F32")}


def test_curves_are_monotone_and_complete():
    curves = _data("curves.json")
    assert set(curves) >= {"qwen3_moe", "qwen3_5_moe", "glm_moe_dsa", "gemma4"}
    for name, c in curves.items():
        fr, hit, mpt = c["fractions"], c["decode_hit_rate"], c["decode_misses_per_token"]
        assert len(fr) == len(hit) == len(mpt), name
        assert fr == sorted(fr), name
        assert all(b >= a - 1e-9 for a, b in zip(hit, hit[1:])), f"{name}: hit not monotone"
        assert all(b <= a + 1e-9 for a, b in zip(mpt, mpt[1:])), f"{name}: misses not monotone"
        for key in ("n_layers", "n_experts", "k", "expert_bytes", "trace_quant_bits"):
            assert c[key] > 0, (name, key)


def test_anchor_data_contract():
    curves = _data("curves.json")
    anchors = _data("anchors.json")
    assert anchors["bandwidth_bytes_s"] > 1e9
    for a in anchors["anchors"]:
        assert a["family"] in curves, a
        assert 0 < a["fraction"] <= 1 and a["tok_s"] > 0, a
        if a.get("exclude_from_base"):
            assert a.get("exclude_reason"), f"excluded anchor needs a reason: {a}"
        else:
            # base_ms derived from this anchor must be positive: the anchor's
            # I/O share may not exceed its whole measured step.
            c = curves[a["family"]]
            mpt = interp(c["fractions"], c["decode_misses_per_token"], a["fraction"])
            a_bytes = c["expert_bytes"] * a["quant_bits"] / c["trace_quant_bits"]
            io_ms = 1000 * mpt * a_bytes / anchors["bandwidth_bytes_s"]
            assert io_ms < 1000 / a["tok_s"], (
                f"{a.get('model', a['family'])}: io {io_ms:.0f}ms exceeds "
                f"step {1000 / a['tok_s']:.0f}ms — anchor and curve disagree"
            )


def test_accuracy_rows_well_formed():
    acc = _data("accuracy.json")
    assert acc["notes"]
    for r in acc["rows"]:
        for key in ("model", "family", "task", "score", "n", "source"):
            assert r.get(key) is not None, (r, key)


def test_calibrate_falls_back_without_cold_files(monkeypatch, tmp_path):
    import boyle.predict as p

    monkeypatch.setattr(p, "_CAL_PATH", tmp_path / "calibration.json")
    monkeypatch.setattr(p, "_cold_probe_files", lambda **kw: [])
    cal = calibrate(force=True)
    assert cal["bandwidth_bytes_s"] == _FALLBACK_BW
    assert "note" in cal
