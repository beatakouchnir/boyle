# SPDX-License-Identifier: Apache-2.0
"""CheckpointExpertStore against a hand-crafted safetensors shard.

The shard is built with struct + numpy (no safetensors/mlx dependency in
the writer), so this exercises the real header parser and both read paths
— memmap and F_NOCACHE pread — byte for byte. mlx itself is required to
import the runtime; the test skips where mlx is unavailable.
"""

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from boyle._runtime import CheckpointExpertStore  # noqa: E402


def write_shard(path, tensors):
    """Minimal safetensors writer: {name: np.ndarray} -> one shard."""
    header, blobs, offset = {}, [], 0
    dtype_tag = {np.dtype(np.float32): "F32", np.dtype(np.uint32): "U32"}
    for name, arr in tensors.items():
        blob = arr.tobytes()
        header[name] = {
            "dtype": dtype_tag[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for blob in blobs:
            f.write(blob)


@pytest.fixture()
def shard_dir(tmp_path):
    rng = np.random.default_rng(7)
    tensors = {
        "layers.0.mlp.gate_proj.weight": rng.standard_normal(
            (4, 8, 16), dtype=np.float32
        ),
        "layers.0.mlp.gate_proj.scales": (
            rng.integers(0, 2**32, (4, 8), dtype=np.uint32).astype(np.uint32)
        ),
    }
    write_shard(tmp_path / "model.safetensors", tensors)
    return tmp_path, tensors


@pytest.mark.parametrize("direct", [False, True])
def test_expert_slab_reads_match_source(shard_dir, direct):
    path, tensors = shard_dir
    store = CheckpointExpertStore(path, direct=direct)
    name = "layers.0.mlp.gate_proj.weight"
    shape, dtype = store.spec(name)
    assert shape == (4, 8, 16) and dtype == "F32"
    for e in range(4):
        raw, tag = store.read_raw(*store.raw_expert_args(name, e))
        assert tag == "F32"
        np.testing.assert_array_equal(raw, tensors[name][e])


@pytest.mark.parametrize("direct", [False, True])
def test_whole_tensor_read_and_uint_dtype(shard_dir, direct):
    path, tensors = shard_dir
    store = CheckpointExpertStore(path, direct=direct)
    name = "layers.0.mlp.gate_proj.scales"
    raw, tag = store.read_raw(*store.raw_tensor_args(name))
    assert tag == "U32"
    np.testing.assert_array_equal(raw, tensors[name])


def test_fetch_expert_roundtrip_via_mx(shard_dir):
    path, tensors = shard_dir
    store = CheckpointExpertStore(path, direct=True)
    name = "layers.0.mlp.gate_proj.weight"
    out = np.array(store.fetch_expert(name, 2))
    np.testing.assert_array_equal(out, tensors[name][2])


def test_missing_dir_is_empty_store(tmp_path):
    assert not CheckpointExpertStore(tmp_path / "nope_does_not_exist")


def test_resolve_model_dir_rejects_shardless_cache(tmp_path, monkeypatch):
    """A config.json-only snapshot (left by headers-only predict) must not
    satisfy local resolution — the download branch has to run."""
    import boyle.loader as loader

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    full = tmp_path / "full"
    full.mkdir()
    (full / "model.safetensors").write_bytes(b"x")
    calls = []

    def fake_snapshot_download(repo, local_files_only=False, **kw):
        calls.append(local_files_only)
        return str(partial if local_files_only else full)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    resolved = loader._resolve_model_dir("org/some-model")
    assert resolved == full
    assert calls == [True, False]  # local tried, rejected, download ran


# --- prefixed checkpoint names ------------------------------------------


def _quantized_glu():
    """A stock SwitchGLU quantized the way a 4-bit checkpoint loads it."""
    from mlx import nn
    from mlx_lm.models.switch_layers import SwitchGLU

    glu = SwitchGLU(32, 32, 4)
    nn.quantize(glu, group_size=32, bits=4)
    return glu


def _glu_tensors(glu, prefix):
    """{checkpoint name: numpy array} for one GLU under ``prefix``."""
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(glu, proj)
        for field in ("weight", "scales", "biases"):
            out[f"{prefix}.{proj}.{field}"] = np.array(lin[field])
    return out


def test_store_view_resolves_prefixed_checkpoint_name(tmp_path):
    """mlx-vlm conversions key the text model as ``language_model.model.``
    while some mlx-lm ports build a ``model.`` tree: the store reads by
    checkpoint name, so the module path must resolve to the prefixed one."""
    from boyle._runtime import _resolve_store_view

    glu = _quantized_glu()
    write_shard(
        tmp_path / "model.safetensors",
        _glu_tensors(glu, "language_model.model.layers.0.mlp.switch_mlp"),
    )
    store = CheckpointExpertStore(tmp_path, direct=False)
    view, reason = _resolve_store_view(glu, store, "model.layers.0.mlp.switch_mlp")
    assert reason is None
    assert view._prefix == "language_model.model.layers.0.mlp.switch_mlp"
    np.testing.assert_array_equal(
        np.array(view.fetch("gate_proj", "weight", 2)),
        np.array(glu.gate_proj["weight"][2]),
    )


def test_store_view_leaves_ambiguous_prefix_unresolved(tmp_path):
    """Two prefixes matching the same suffix: refuse rather than guess."""
    from boyle._runtime import _resolve_store_view

    glu = _quantized_glu()
    write_shard(
        tmp_path / "model.safetensors",
        {
            **_glu_tensors(glu, "language_model.model.layers.0.mlp.switch_mlp"),
            **_glu_tensors(glu, "mtp.model.layers.0.mlp.switch_mlp"),
        },
    )
    store = CheckpointExpertStore(tmp_path, direct=False)
    view, reason = _resolve_store_view(glu, store, "model.layers.0.mlp.switch_mlp")
    assert view is None and "no tensor" in reason

