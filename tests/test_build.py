# SPDX-License-Identifier: Apache-2.0
"""build round-trip: the writer must produce exactly what the runtime reads.

Synthetic stacked checkpoint -> build_store -> read every field back
through the runtime's own ColoStore/ColoGLUView and compare byte-for-byte
against the source slabs. No model, CI-safe."""

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from test_store import write_shard  # noqa: E402

from boyle.build import build_store  # noqa: E402


def _stacked_checkpoint(tmp_path, n_layers=2, e=4):
    rng = np.random.default_rng(11)
    tensors = {}
    for li in range(n_layers):
        prefix = f"model.layers.{li}.mlp.switch_mlp"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"{prefix}.{proj}.weight"] = rng.integers(
                0, 2**32, (e, 8, 4), dtype=np.uint32
            ).astype(np.uint32)
            tensors[f"{prefix}.{proj}.scales"] = rng.standard_normal(
                (e, 8, 2), dtype=np.float32
            )
    tensors["lm_head.weight"] = rng.standard_normal((16, 8), dtype=np.float32)
    write_shard(tmp_path / "model.safetensors", tensors)
    return tensors


def test_build_roundtrip_through_runtime_reader(tmp_path):
    from boyle._runtime import ColoGLUView, ColoStore

    tensors = _stacked_checkpoint(tmp_path)
    out = build_store(tmp_path, tmp_path / "store", progress=lambda *a: None)
    store = ColoStore(str(out), direct=True)
    for li in range(2):
        prefix = f"model.layers.{li}.mlp.switch_mlp"
        view = ColoGLUView(store, prefix)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for field in ("weight", "scales"):
                src = tensors[f"{prefix}.{proj}.{field}"]
                for e in range(4):
                    got = np.array(view.fetch(proj, field, e))
                    np.testing.assert_array_equal(got, src[e])


def test_build_rejects_per_expert_scheme(tmp_path):
    tensors = {
        f"layers.0.mlp.experts.{e}.gate_proj.weight": np.zeros((8, 4), np.float32)
        for e in range(4)
    }
    write_shard(tmp_path / "model.safetensors", tensors)
    with pytest.raises(SystemExit, match="per-expert naming"):
        build_store(tmp_path, tmp_path / "store", progress=lambda *a: None)
