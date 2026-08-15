# SPDX-License-Identifier: Apache-2.0
"""Budgeted MoE expert offloading: stream non-resident experts from disk.

Provenance: descends from the omlx expert-offload patch (PR jundot/omlx#2595,
origin c9924dbb, Apache-2.0 — see NOTICE) by way of a measurement workbench.
This port keeps exactly the levers that survived measurement: direct
(F_NOCACHE) pread fetches with a parallel install pool, a colocated
per-expert record store, and pooled record-based expert-major prefill as the
only over-capacity prefill path. Measured-dead alternatives (learned
prefetch, eviction-policy variants, sentinel-polled sync, a userspace L2
behind the slots, read-order sorting) are deliberately absent; the research
record explains each verdict.

For Mixture-of-Experts models whose expert tables do not fit in the budget,
keep ``capacity`` of each layer's experts in a contiguous slot tensor with
LRU eviction and fetch the rest on demand. Routing is computed exactly as
shipped; a cache miss changes *when* an expert's weights are read, never
*which* expert runs.

Numerical contract, measured against pinned mlx-lm: decode and
within-capacity prefill are bit-identical to the resident model at any
budget. Over-capacity prefill runs expert-major — each needed expert is
installed exactly once per call, groups are evaluated as they complete —
which is rounding-equivalent to the resident computation (same math,
different batching), not bit-identical.

Applied once post-load, before lazy weights materialize: each stock
``SwitchGLU`` whose projections are quantized and fully covered by the
checkpoint is replaced with an ``OffloadSwitchGLU``. The original module —
and with it the lazy references to the full expert tensors — is dropped, so
non-resident experts are never materialized. Unsupported instances
(non-quantized, fused ``gate_up_proj``, per-expert ``bias``, or tensor names
the checkpoint does not contain) are left untouched and reported.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

logger = logging.getLogger(__name__)

# Wall-clock buckets for `boyle bench` (attribution as-executed under lazy
# eval; valid for io_workers <= 1 except the pure-I/O buckets).
INSTRUMENT = False
# Routing capture for `boyle trace` (host-side, off in normal serving).
RECORD_ROUTING = False
RECORD_TRACE = False

_TIMERS: dict[str, list] = {}


@contextmanager
def timed(name: str):
    if not INSTRUMENT:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rec = _TIMERS.setdefault(name, [0.0, 0])
        rec[0] += time.perf_counter() - t0
        rec[1] += 1


def reset_timers() -> None:
    _TIMERS.clear()


def get_timers() -> dict:
    return {k: {"s": round(v[0], 3), "n": v[1]} for k, v in _TIMERS.items()}


_io_pools: dict[int, ThreadPoolExecutor] = {}


def _get_io_pool(n: int) -> ThreadPoolExecutor:
    pool = _io_pools.get(n)
    if pool is None:
        pool = _io_pools[n] = ThreadPoolExecutor(
            max_workers=n, thread_name_prefix="boyle-io"
        )
    return pool


def _sync_and_clear_cache() -> None:
    mx.synchronize()
    mx.clear_cache()


_PROJS = ("gate_proj", "up_proj", "down_proj")

# safetensors dtype tag -> (numpy transport dtype, mlx dtype to view as).
# bf16 has no numpy equivalent, so it travels as raw uint16 and is
# reinterpreted on the mlx side; everything else converts directly.
_DTYPES = {
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, None),
    "F32": (np.float32, None),
    "U32": (np.uint32, None),
    "I32": (np.int32, None),
    "U8": (np.uint8, None),
}


class CheckpointExpertStore:
    """Per-expert slab reads from a model directory's safetensors shards.

    Expert tables are stored stacked with the expert axis leading
    (``[num_experts, ...]``), so one expert is a contiguous byte range in the
    shard. ``direct=True`` (the default) reads via pread on F_NOCACHE fds,
    bypassing the page cache: the memmap path pays GIL-held copies and
    page-cache double-buffering, measured 4x slower end to end. No MLX/Metal
    calls happen here except the final host-side ``mx.array`` construction,
    so raw fetches are safe on any thread.
    """

    def __init__(self, model_path: str | Path, direct: bool = True):
        self._direct = direct
        self._specs: dict[str, tuple[Path, str, tuple[int, ...], int]] = {}
        self._mm: dict[Path, np.memmap] = {}
        self._fd: dict[Path, int] = {}
        model_path = Path(model_path)
        for shard in sorted(model_path.glob("*.safetensors")):
            with open(shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            data_base = 8 + header_len
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                self._specs[name] = (
                    shard,
                    spec["dtype"],
                    tuple(spec["shape"]),
                    data_base + spec["data_offsets"][0],
                )

    def __bool__(self) -> bool:
        return bool(self._specs)

    def has(self, name: str) -> bool:
        return name in self._specs

    def spec(self, name: str) -> tuple[tuple[int, ...], str]:
        _, dtype, shape, _ = self._specs[name]
        return shape, dtype

    def read_raw(self, name: str, start_elem: int, n_elems: int,
                 out_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
        """Pure I/O + numpy: safe on any thread (no MLX calls). Returns the
        reshaped numpy array plus the safetensors dtype tag."""
        shard, dtype, _, offset = self._specs[name]
        np_dtype, _ = _DTYPES[dtype]
        itemsize = np.dtype(np_dtype).itemsize
        start = offset + start_elem * itemsize
        n_bytes = n_elems * itemsize
        if self._direct:
            fd = self._fd.get(shard)
            if fd is None:
                fd = self._fd[shard] = os.open(shard, os.O_RDONLY)
                import fcntl

                fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
            raw = np.frombuffer(os.pread(fd, n_bytes, start), dtype=np.uint8)
        else:
            mm = self._mm.get(shard)
            if mm is None:
                mm = self._mm[shard] = np.memmap(shard, dtype=np.uint8, mode="r")
            raw = np.array(mm[start : start + n_bytes])  # one copy
        return raw.view(np_dtype).reshape(out_shape), dtype

    @staticmethod
    def to_mx(raw: np.ndarray, dtype_tag: str) -> mx.array:
        """numpy -> mx, calling-thread only (Metal/stream affinity)."""
        mx_view = _DTYPES[dtype_tag][1]
        out = mx.array(raw)
        return out.view(mx_view) if mx_view is not None else out

    def raw_expert_args(self, name: str, expert: int):
        _, _, shape, _ = self._specs[name]
        slab = int(np.prod(shape[1:]))
        return (name, expert * slab, slab, shape[1:])

    def raw_tensor_args(self, name: str):
        _, _, shape, _ = self._specs[name]
        return (name, 0, int(np.prod(shape)), shape)

    def fetch_expert(self, name: str, expert: int) -> mx.array:
        """One expert's slab of a stacked ``[num_experts, ...]`` tensor."""
        return self.to_mx(*self.read_raw(*self.raw_expert_args(name, expert)))

    def fetch_tensor(self, name: str) -> mx.array:
        """A whole tensor (per-expert checkpoint layouts)."""
        return self.to_mx(*self.read_raw(*self.raw_tensor_args(name)))


class ColoStore:
    """Colocated per-expert-record store (``boyle build`` output).

    One pread per miss: the record holds all of an expert's fields
    back-to-back, ordered by co-activation; slicing happens in-memory on the
    calling thread. Measured +13% on the diverse-serving ceiling over
    scattered field reads."""

    def __init__(self, colo_dir, direct: bool = True):
        self.index = json.load(open(f"{colo_dir}/colo.json"))
        self.path = f"{colo_dir}/colo.bin"
        self._direct = direct
        self._fd = None
        self._mm = None

    def _get_fd(self):
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
            if self._direct:
                import fcntl

                fcntl.fcntl(self._fd, fcntl.F_NOCACHE, 1)
        return self._fd

    to_mx = staticmethod(CheckpointExpertStore.to_mx)

    def read_raw(self, dtype_tag: str, byte_off: int, shape):
        """Expert-major path shim: consumes ColoGLUView.raw_args tuples."""
        np_dtype, _ = _DTYPES[dtype_tag]
        nb = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        return (self.read_record(byte_off, nb).view(np_dtype).reshape(shape),
                dtype_tag)

    def read_record(self, offset: int, nbytes: int) -> np.ndarray:
        """Thread-safe raw record read (pure I/O + numpy)."""
        if self._direct:
            return np.frombuffer(os.pread(self._get_fd(), nbytes, offset),
                                 dtype=np.uint8)
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return np.array(self._mm[offset:offset + nbytes])


class ColoGLUView:
    """Duck-types the parts of _GLUStoreView the cache uses, plus the
    record interface the colo-aware installer prefers."""

    def __init__(self, store: ColoStore, prefix: str):
        self.store = store
        self._recs = store.index[prefix]["records"]
        self._nbytes = {e: sum(
            _colo_field_bytes(m) for m in r["fields"].values())
            for e, r in self._recs.items()}

    def has(self, proj: str, field: str) -> bool:
        return f"{proj}.{field}" in self._recs["0"]["fields"]

    def record_args(self, e: int):
        rec = self._recs[str(e)]
        return rec["offset"], self._nbytes[str(e)], rec["fields"]

    def raw_args(self, proj: str, field: str, e: int):
        """Expert-major path: pairs with ColoStore.read_raw."""
        rec = self._recs[str(e)]
        rel, dtype, shape = rec["fields"][f"{proj}.{field}"]
        return (dtype, rec["offset"] + rel, shape)

    def slice_record(self, raw: np.ndarray, fields: dict):
        out = {}
        for key, (rel, dtype, shape) in fields.items():
            np_dtype, _ = _DTYPES[dtype]
            nb = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
            out[key] = (raw[rel:rel + nb].view(np_dtype).reshape(shape), dtype)
        return out

    def fetch(self, proj: str, field: str, e: int) -> mx.array:
        off, nb, fields = self.record_args(e)
        sliced = self.slice_record(self.store.read_record(off, nb), fields)
        arr, dtype = sliced[f"{proj}.{field}"]
        return CheckpointExpertStore.to_mx(arr, dtype)


def _colo_field_bytes(meta):
    rel, dtype, shape = meta
    np_dtype, _ = _DTYPES[dtype]
    return int(np.prod(shape)) * np.dtype(np_dtype).itemsize


class _GLUStoreView:
    """Adapt the flat store to one SwitchGLU's checkpoint naming scheme.

    Two layouts exist in the wild. Newer conversions store experts stacked
    under the module-tree name (``<glu>.gate_proj.weight`` with shape
    ``[E, ...]``). Older ones store one tensor per expert under the GLU's
    parent (``<parent>.experts.<e>.gate_proj.weight``), which mlx-lm's
    ``sanitize()`` stacks at load — so the stacked names never exist in the
    file. The view hides the difference from :class:`ExpertCache`.
    """

    def __init__(self, store: CheckpointExpertStore, prefix: str,
                 per_expert: bool = False):
        self._store = store
        self._prefix = prefix  # stacked: the GLU path; per-expert: its parent
        self._per_expert = per_expert

    def _name(self, proj: str, field: str, expert: int) -> str:
        if self._per_expert:
            return f"{self._prefix}.experts.{expert}.{proj}.{field}"
        return f"{self._prefix}.{proj}.{field}"

    def has(self, proj: str, field: str) -> bool:
        return self._store.has(self._name(proj, field, 0))

    def fetch(self, proj: str, field: str, expert: int) -> mx.array:
        if self._per_expert:
            return self._store.fetch_tensor(self._name(proj, field, expert))
        return self._store.fetch_expert(self._name(proj, field, 0), expert)

    def raw_args(self, proj: str, field: str, expert: int):
        """Arguments for ``store.read_raw`` — lets callers batch raw reads
        across a thread pool without any MLX involvement."""
        if self._per_expert:
            return self._store.raw_tensor_args(self._name(proj, field, expert))
        return self._store.raw_expert_args(self._name(proj, field, 0), expert)

    @property
    def store(self) -> CheckpointExpertStore:
        return self._store


class ExpertCache:
    """Contiguous resident slots over one layer's experts, LRU eviction.

    LRU because the capacity law says so: routing is flat, LFU loses badly,
    and clairvoyant OPT beats LRU by +0.07 hit rate at real budgets — there
    is no policy headroom worth code. Holds no reference to the wrapped
    module's expert tensors — only the resident slots and the store view.
    That is the difference between saving memory and adding it: keeping the
    source tensors referenced alongside the slots costs the full expert set
    *plus* the cache.
    """

    def __init__(self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView,
                 io_workers: int = 8):
        self.io_workers = io_workers
        self.n_experts = glu.gate_proj["weight"].shape[0]
        self.capacity = min(capacity, self.n_experts)
        self.projs = _PROJS
        self.disk = disk
        self.resident: dict[str, list] = {}
        for name in self.projs:
            lin = getattr(glu, name)
            has_b = lin.get("biases") is not None
            w, s = lin["weight"], lin["scales"]
            b = lin["biases"] if has_b else None
            self.resident[name] = [
                mx.zeros((self.capacity,) + w.shape[1:], dtype=w.dtype),
                mx.zeros((self.capacity,) + s.shape[1:], dtype=s.dtype),
                (
                    None
                    if b is None
                    else mx.zeros((self.capacity,) + b.shape[1:], dtype=b.dtype)
                ),
            ]
        self.group = glu.gate_proj.group_size
        self.bits = glu.gate_proj.bits
        self.mode = glu.gate_proj.mode
        self.slot_of: dict[int, int] = {}  # expert id -> slot, LRU ordered
        self.free = list(range(self.capacity))
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.hits = self.misses = 0
        self.warm = False
        self.expert_counts = (
            np.zeros(self.n_experts, dtype=np.int64) if RECORD_ROUTING else None
        )
        self.trace: list | None = [] if RECORD_TRACE else None

    def _evict_slot(self) -> int:
        old_e = next(iter(self.slot_of))  # LRU victim
        slot = self.slot_of.pop(old_e)
        self.map[old_e] = -1
        return slot

    def _install(self, e: int) -> int:
        slot = self.free.pop() if self.free else self._evict_slot()
        st = self.disk.store
        if hasattr(self.disk, "record_args"):
            # Colo path: one contiguous record read instead of nine
            # scattered field reads.
            off, nb, fields = self.disk.record_args(e)
            with timed("read_io"):
                buf = st.read_record(off, nb)
            sliced = self.disk.slice_record(buf, fields)
            with timed("install_mx"):
                self._fill_slot(slot, sliced, st.to_mx)
        else:
            for name in self.projs:
                rw, rs, rb = self.resident[name]
                with timed("read_io"):
                    w_raw = st.read_raw(*self.disk.raw_args(name, "weight", e))
                    s_raw = st.read_raw(*self.disk.raw_args(name, "scales", e))
                    b_raw = (
                        st.read_raw(*self.disk.raw_args(name, "biases", e))
                        if rb is not None and self.disk.has(name, "biases")
                        else None
                    )
                with timed("install_mx"):
                    rw[slot] = st.to_mx(*w_raw)
                    rs[slot] = st.to_mx(*s_raw)
                    if b_raw is not None:
                        rb[slot] = st.to_mx(*b_raw)
        self.slot_of[e] = slot
        self.map[e] = slot
        # once every expert has a slot no eviction can occur, so residency is
        # permanently satisfied and the per-token check is pure overhead
        self.warm = len(self.slot_of) == self.n_experts
        return slot

    def _fill_slot(self, slot: int, sliced: dict, to_mx) -> None:
        for name in self.projs:
            rw, rs, rb = self.resident[name]
            arr, dt = sliced[f"{name}.weight"]
            rw[slot] = to_mx(arr, dt)
            arr, dt = sliced[f"{name}.scales"]
            rs[slot] = to_mx(arr, dt)
            if f"{name}.biases" in sliced and rb is not None:
                arr, dt = sliced[f"{name}.biases"]
                rb[slot] = to_mx(arr, dt)

    def ensure(self, idx: mx.array) -> None:
        """Make every expert in ``idx`` resident.

        The ``.tolist()`` is a device->host readback and therefore a sync per
        MoE layer per step. That sync is architectural — the router's output
        genuinely gates which weights must be present — and the measured
        alternatives (sentinel polling, event gating) lose or break even, so
        this stays the simple honest version.
        """
        if self.warm:  # nothing can miss; skip it
            return
        with timed("ensure_sync"):
            flat = idx.reshape(-1).tolist()
        if self.expert_counts is not None:
            np.add.at(self.expert_counts, flat, 1)
        if self.trace is not None:
            self.trace.append([int(e) for e in flat])
        needed = set(int(e) for e in flat)
        missing = []
        for e in needed:
            if e in self.slot_of:
                slot = self.slot_of.pop(e)  # re-insert: LRU order
                self.slot_of[e] = slot
                self.hits += 1
            else:
                self.misses += 1
                missing.append(e)
        if not missing:
            return
        if self.io_workers >= 2 and len(missing) * len(self.projs) >= 2:
            self._install_parallel(sorted(missing))
        else:
            for e in sorted(missing):
                self._install(e)
        # No mx.eval here: installs are already-materialized host arrays, and
        # evaluating every resident tensor on every miss measured 22% slower
        # at identical peak memory. Prefill's transient is bounded by the
        # per-group eval in the expert-major path, a different mechanism.

    def _install_parallel_colo(self, missing: list[int]) -> None:
        """One contiguous read per missing expert across the I/O pool; field
        slicing and mx construction on the calling thread."""
        pool = _get_io_pool(max(self.io_workers, 2))
        read = self.disk.store.read_record
        args = {e: self.disk.record_args(e) for e in missing}
        with timed("read_raw"):
            futures = {e: pool.submit(read, args[e][0], args[e][1])
                       for e in missing}
            raw = {e: fut.result() for e, fut in futures.items()}
        to_mx = CheckpointExpertStore.to_mx
        with timed("install_graph"):
            for e in missing:
                sliced = self.disk.slice_record(raw[e], args[e][2])
                slot = self.free.pop() if self.free else self._evict_slot()
                self._fill_slot(slot, sliced, to_mx)
                self.slot_of[e] = slot
                self.map[e] = slot
        self.warm = len(self.slot_of) == self.n_experts

    def _install_parallel(self, missing: list[int]) -> None:
        """Batch every raw read for this step's misses across a thread pool
        (pure I/O + numpy — GIL-released), then do slot bookkeeping and all
        mx array construction on the calling thread. Measured 4x over serial
        memmap fetches at 397B scale."""
        if hasattr(self.disk, "record_args"):
            self._install_parallel_colo(missing)
            return
        tasks = {}
        for e in missing:
            for name in self.projs:
                rb = self.resident[name][2]
                fields = ["weight", "scales"] + (
                    ["biases"]
                    if rb is not None and self.disk.has(name, "biases")
                    else []
                )
                for field in fields:
                    tasks[(e, name, field)] = self.disk.raw_args(name, field, e)
        pool = _get_io_pool(self.io_workers)
        read = self.disk.store.read_raw
        with timed("read_raw"):
            futures = {key: pool.submit(read, *args) for key, args in tasks.items()}
            raw = {key: fut.result() for key, fut in futures.items()}
        to_mx = self.disk.store.to_mx
        with timed("install_graph"):
            for e in missing:
                slot = self.free.pop() if self.free else self._evict_slot()
                for name in self.projs:
                    rw, rs, rb = self.resident[name]
                    rw[slot] = to_mx(*raw[(e, name, "weight")])
                    rs[slot] = to_mx(*raw[(e, name, "scales")])
                    if (e, name, "biases") in raw:
                        rb[slot] = to_mx(*raw[(e, name, "biases")])
                self.slot_of[e] = slot
                self.map[e] = slot
        self.warm = len(self.slot_of) == self.n_experts

    def qmm(
        self, name: str, x: mx.array, slots: mx.array, sorted_indices: bool = False
    ) -> mx.array:
        # sorted_indices selects a different kernel; the wrapper mirrors the
        # stock SwitchGLU's sort decision so the kernel choice — and with it
        # the numerics — matches the path the resident model would take.
        rw, rs, rb = self.resident[name]
        return mx.gather_qmm(
            x,
            rw,
            rs,
            rb,
            rhs_indices=slots,
            transpose=True,
            group_size=self.group,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


class OffloadSwitchGLU(nn.Module):
    """SwitchGLU whose experts live in an :class:`ExpertCache`."""

    def __init__(self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView,
                 io_workers: int = 8):
        super().__init__()
        self.cache = ExpertCache(glu, capacity, disk, io_workers=io_workers)
        self.activation = glu.activation

    def _forward(self, x: mx.array, indices: mx.array) -> mx.array:
        c = self.cache
        c.ensure(indices)
        slots = mx.take(c.map, indices)
        x = mx.expand_dims(x, (-2, -3))
        # Mirror the stock SwitchGLU's sort rule exactly (threshold and all):
        # decode calls are far below it, and forcing the sort there measured
        # slower than it saved.
        do_sort = indices.size >= 64
        inv = None
        if do_sort:
            x, slots, inv = _gather_sort(x, slots)
        up = c.qmm("up_proj", x, slots, do_sort)
        gate = c.qmm("gate_proj", x, slots, do_sort)
        out = c.qmm("down_proj", self.activation(up, gate), slots, do_sort)
        if do_sort:
            out = _scatter_unsort(out, inv, indices.shape)
        return out.squeeze(-2)

    def _forward_expert_major(self, flat_x: mx.array, rows: list) -> mx.array:
        """Over-capacity prefill: iterate expert groups, not token chunks.

        (token, k-slot) pairs are independent inside SwitchGLU — the
        cross-expert weighted sum happens in the caller — so the output is
        filled group by group: install <= capacity experts once (pooled
        record reads exercise NVMe parallelism, measured -47% TTFT on
        fill-heavy prompts), gather every pair routed to them, run the
        projections with per-pair single-expert indices, scatter into the
        output. Each needed expert is installed exactly once per call; group
        evals bound the transient. Uses the unsorted kernel
        (any-permutation bit-exact); output is rounding-equivalent to the
        resident path, not bit-identical — same math, different batching.
        """
        c = self.cache
        n_tok, k = len(rows), len(rows[0])
        d_model = flat_x.shape[-1]

        if c.expert_counts is not None:
            np.add.at(c.expert_counts, [e for row in rows for e in row], 1)
        by_expert: dict[int, list[int]] = {}
        for t, row in enumerate(rows):
            for j, e in enumerate(row):
                by_expert.setdefault(e, []).append(t * k + j)
        needed = sorted(by_expert)
        groups = [
            needed[i : i + c.capacity] for i in range(0, len(needed), c.capacity)
        ]

        out = mx.zeros((n_tok * k, d_model), dtype=flat_x.dtype)
        for grp in groups:
            grp_missing = []
            for e in grp:
                if e in c.slot_of:
                    slot = c.slot_of.pop(e)  # refresh LRU
                    c.slot_of[e] = slot
                    c.hits += 1
                else:
                    c.misses += 1
                    grp_missing.append(e)
            if grp_missing:
                if hasattr(c.disk, "record_args") and c.io_workers >= 2:
                    c._install_parallel_colo(grp_missing)
                else:
                    for e in grp_missing:
                        c._install(e)
            pos, slots, t_idx = [], [], []
            for e in grp:
                s = c.slot_of[e]
                for p in by_expert[e]:
                    pos.append(p)
                    slots.append(s)
                    t_idx.append(p // k)
            xp = flat_x[mx.array(t_idx)]  # (m, D), duplicates fine
            xe = mx.expand_dims(xp, (-2, -3))  # (m, 1, 1, D)
            sl = mx.array(slots, dtype=mx.int32).reshape(-1, 1)  # (m, 1)
            up = c.qmm("up_proj", xe, sl, False)
            gate = c.qmm("gate_proj", xe, sl, False)
            o = c.qmm("down_proj", self.activation(up, gate), sl, False)
            with timed("em_scatter"):
                out[mx.array(pos)] = o.squeeze(-2)[:, 0, :]
            with timed("chunk_eval"):
                mx.eval(out)  # bound the per-group transient
        return out

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        # A single _forward must have every expert it routes to resident AT
        # ONCE: a long prefill can route to more distinct experts than the
        # cache holds, in which case earlier installs would be evicted before
        # the gather runs and their slots would read garbage. Decode (working
        # set = batch x top_k) takes the no-extra-sync fast path; larger
        # calls pay one readback to decide, and go expert-major only when
        # the distinct working set genuinely exceeds capacity.
        c = self.cache
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        if n_tok * k <= c.capacity or n_tok == 1:
            return self._forward(x, indices)

        with timed("chunk_search"):
            rows = flat_i.tolist()
            expert_major = len({e for row in rows for e in row}) > c.capacity
        if not expert_major:
            return self._forward(x, indices)
        flat_x = x.reshape(-1, x.shape[-1])
        out = self._forward_expert_major(flat_x, rows)
        return out.reshape(indices.shape + (x.shape[-1],))


def _resolve_model_dir(model_path: str | Path) -> Path | None:
    """Resolve a model name to its local checkpoint directory.

    Local directories pass through; hub repo ids resolve against the local
    HF cache only (the model was just loaded from it, so it is present) —
    this never triggers a download.
    """
    p = Path(model_path)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download

        # Restrict to the shards (all the store reads) so an mlx-lm-style
        # partial cache — model files only, no README etc. — resolves. A
        # patternless local_files_only lookup would demand the repo's full
        # file list and fail on exactly such caches.
        return Path(
            snapshot_download(
                str(model_path),
                allow_patterns=["*.safetensors"],
                local_files_only=True,
            )
        )
    except Exception:
        logger.warning(
            "boyle: cannot resolve %r to a local checkpoint directory",
            str(model_path),
        )
        return None


def _iter_switch_glus(model):
    """Yield ``(parent, key, module, tree_path)`` for every stock SwitchGLU.

    mlx ``nn.Module`` subclasses ``dict`` — children are dict items, not
    attributes — so this walks ``.items()`` and list entries, building the
    same dotted paths ``tree_flatten`` produces (which is what checkpoint
    tensor names are matched against at load time).
    """
    seen = set()

    def walk(parent, key, obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if type(obj) is SwitchGLU:
            yield (parent, key, obj, path)
            return
        if isinstance(obj, dict):  # includes nn.Module
            for k, v in obj.items():
                yield from walk(obj, k, v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from walk(obj, i, v, f"{path}.{i}")

    yield from walk(None, None, model, "")


def _resolve_store_view(
    glu: SwitchGLU, store: CheckpointExpertStore, path: str
) -> tuple[_GLUStoreView | None, str | None]:
    """Validate coverage and return a view in whichever naming scheme the
    checkpoint uses, or ``(None, reason)``.

    Stacked scheme: tensors live under the GLU's own tree path with shape
    ``[E, ...]``. Per-expert scheme: one tensor per expert under the GLU's
    parent (``<parent>.experts.<e>.<proj>.<field>`` — the layout mlx-lm's
    ``sanitize()`` stacks at load, e.g. OLMoE / Qwen2-MoE conversions);
    every expert's tensor is verified. Anything else — including layouts
    that also rename the projections, like Mixtral's ``w1/w2/w3`` — is
    reported for a graceful skip. Unknown storage dtypes are rejected here
    so the failure mode stays "runs resident" instead of a fetch-time
    KeyError mid-generation.
    """
    n = None
    fields_of: dict[str, list[str]] = {}
    for proj in _PROJS:
        lin = getattr(glu, proj, None)
        if not isinstance(lin, QuantizedSwitchLinear):
            return None, f"{proj} is not QuantizedSwitchLinear"
        if "bias" in lin:
            return None, f"{proj} has per-expert bias (unsupported)"
        n = lin["weight"].shape[0] if n is None else n
        fields_of[proj] = ["weight", "scales"] + (
            ["biases"] if lin.get("biases") is not None else []
        )

    stacked = _GLUStoreView(store, path)
    parent = path.rsplit(".", 1)[0] if "." in path else ""
    view = (
        stacked
        if stacked.has("gate_proj", "weight")
        else _GLUStoreView(store, parent, per_expert=True)
    )

    for proj in _PROJS:
        lin = getattr(glu, proj)
        for field in fields_of[proj]:
            module_shape = tuple(lin[field].shape)
            if view is stacked:
                checks = [(view._name(proj, field, 0), module_shape)]
            else:
                checks = [
                    (view._name(proj, field, e), module_shape[1:]) for e in range(n)
                ]
            for name, want_shape in checks:
                if not store.has(name):
                    return None, f"checkpoint has no tensor {name!r}"
                shape, dtype = store.spec(name)
                if shape != want_shape:
                    return None, f"{name!r} shape {shape} != expected {want_shape}"
                if dtype not in _DTYPES:
                    return None, f"{name!r} has unsupported dtype {dtype!r}"
    return view, None


def apply_expert_offload(
    model,
    model_path: str | Path,
    resident_fraction: float = 0.25,
    io_workers: int = 8,
    direct: bool = True,
    capacity_plan: list | None = None,
    layer_filter=None,
    colo_dir: str | None = None,
) -> int:
    """Replace covered SwitchGLU instances with offloaded ones.

    Returns the number of layers wrapped (0 when the model has no stock
    SwitchGLU or the checkpoint does not cover them). Must run before lazy
    weights are materialized for the memory saving to exist.
    ``capacity_plan`` takes :class:`boyle.budget.BudgetPlan.capacities`,
    consumed in discovery order (the tree walk is deterministic).
    """
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    store = CheckpointExpertStore(model_dir, direct=direct)
    if not store:
        logger.warning("boyle: no safetensors under %s", model_dir)
        return 0

    wrapped = 0
    total_bytes = resident_bytes = 0
    colo_store = ColoStore(colo_dir, direct=direct) if colo_dir else None
    for parent, key, glu, path in list(_iter_switch_glus(model)):
        if layer_filter is not None and not layer_filter(path):
            continue
        view, reason = _resolve_store_view(glu, store, path)
        if view is not None and colo_store is not None:
            if path in colo_store.index:
                view = ColoGLUView(colo_store, path)
            else:
                logger.warning("boyle: colo store missing %s — plain view", path)
        if view is None:
            logger.info("boyle: skipping %s (%s)", path, reason)
            continue
        n_experts = glu.gate_proj["weight"].shape[0]
        if capacity_plan is not None:
            capacity = max(8, min(n_experts, int(capacity_plan[wrapped])))
        else:
            capacity = max(8, min(n_experts, round(n_experts * resident_fraction)))
        layer_bytes = sum(
            int(np.prod(lin[f].shape)) * lin[f].dtype.size
            for p in _PROJS
            for lin in (getattr(glu, p),)
            for f in (
                ["weight", "scales"]
                + (["biases"] if lin.get("biases") is not None else [])
            )
        )
        total_bytes += layer_bytes
        resident_bytes += layer_bytes * capacity // n_experts
        new = OffloadSwitchGLU(glu, capacity, view, io_workers=io_workers)
        if isinstance(parent, nn.Module):
            setattr(parent, key, new)  # registers via Module.__setattr__
        else:
            parent[key] = new  # plain list / plain dict
        wrapped += 1
        # Dropped source buffers land in the MLX pool; drain per layer to
        # bound the load transient.
        _sync_and_clear_cache()

    if wrapped:
        logger.info(
            "boyle: wrapped %d layers (expert tables: %.2f GB total, "
            "%.2f GB resident)",
            wrapped,
            total_bytes / 1e9,
            resident_bytes / 1e9,
        )
    return wrapped


def offload_stats(model) -> dict:
    """Aggregate hit/miss counters over all offloaded layers."""
    hits = misses = layers = 0
    stack = [model]
    seen = set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, OffloadSwitchGLU):
            hits += obj.cache.hits
            misses += obj.cache.misses
            layers += 1
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
    total = hits + misses
    return {
        "layers": layers,
        "hits": hits,
        "misses": misses,
        "hit_rate": (hits / total) if total else None,
    }


__all__ = [
    "CheckpointExpertStore",
    "ColoStore",
    "OffloadSwitchGLU",
    "apply_expert_offload",
    "offload_stats",
    "get_timers",
    "reset_timers",
]
