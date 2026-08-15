# SPDX-License-Identifier: Apache-2.0
"""load(): from a model id and a budget to a generating model.

Order matters and is load-bearing: anatomy is read from the checkpoint
headers (no weights touched), the budget resolves against it — refusal
happens *before* any download-sized allocation — the model loads lazy,
offload wrapping replaces expert tables before they materialize, and only
then do the surviving weights hit memory. The wired limit is set last, to
the plan's budget capped at the device recommendation: custom decode loops
without it churn Metal residency per step (measured 88x pathology).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from boyle._runtime import (
    CheckpointExpertStore,
    apply_expert_offload,
    offload_stats,
)
from boyle.budget import BudgetPlan, ModelAnatomy, plan as resolve_budget

logger = logging.getLogger(__name__)

_MODEL_FILES = ["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"]


def _resolve_model_dir(model: str | Path) -> Path:
    """Local dir, else HF cache (pattern-restricted: an mlx-style partial
    cache — model files, no README — must resolve), else download."""
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                str(model), local_files_only=True, allow_patterns=_MODEL_FILES
            )
        )
    except Exception:
        return Path(snapshot_download(str(model), allow_patterns=_MODEL_FILES))


_PER_EXPERT = re.compile(r"\.layers\.(\d+)\..*\.experts\.(\d+)\.")
_STACKED = re.compile(
    r"\.layers\.(\d+)\..*\.(?:gate_proj|up_proj|down_proj)\.(?:weight|scales|biases)$"
)


def read_anatomy(model_dir: str | Path) -> ModelAnatomy:
    """Model memory anatomy from checkpoint headers + config.json only.

    Expert tensors are recognized structurally, not by family knowledge:
    either the per-expert naming scheme (``...experts.<e>...``) or stacked
    switch projections (3-D ``[E, out, in]`` gate/up/down tensors). The
    expert count per layer comes from the tensors themselves, so a new MoE
    family needs no config-key archaeology to plan a budget.
    """
    model_dir = Path(model_dir)
    store = CheckpointExpertStore(model_dir, direct=False)
    if not store:
        raise FileNotFoundError(f"no safetensors under {model_dir}")

    layer_bytes: dict[int, int] = {}
    layer_experts: dict[int, set] = {}
    expert_total = 0
    total = 0
    itemsize = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1}
    for name in list(store._specs):
        shape, dtype = store.spec(name)
        nbytes = int(np.prod(shape)) * itemsize.get(dtype, 2) if shape else 0
        total += nbytes
        m = _PER_EXPERT.search(name)
        if m:
            layer = int(m.group(1))
            layer_bytes[layer] = layer_bytes.get(layer, 0) + nbytes
            layer_experts.setdefault(layer, set()).add(int(m.group(2)))
            expert_total += nbytes
            continue
        m = _STACKED.search(name)
        if m and len(shape) == 3:
            layer = int(m.group(1))
            layer_bytes[layer] = layer_bytes.get(layer, 0) + nbytes
            layer_experts.setdefault(layer, set()).add(shape[0])
            expert_total += nbytes

    layers = []
    for layer in sorted(layer_bytes):
        marks = layer_experts[layer]
        # per-expert scheme: distinct expert ids; stacked scheme: E itself
        n = len(marks) if len(marks) > 1 else next(iter(marks))
        layers.append((int(n), layer_bytes[layer]))

    kv_per_token = 0
    config_path = model_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        cfg = cfg.get("text_config", cfg)
        n_layers = cfg.get("num_hidden_layers")
        kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
        head_dim = cfg.get("head_dim") or (
            cfg["hidden_size"] // cfg["num_attention_heads"]
            if cfg.get("hidden_size") and cfg.get("num_attention_heads")
            else None
        )
        if n_layers and kv_heads and head_dim:
            kv_per_token = int(n_layers) * 2 * int(kv_heads) * int(head_dim) * 2
        else:
            logger.warning("boyle: config.json missing KV keys — planning with 0")

    return ModelAnatomy(
        resident_bytes=total - expert_total,
        layers=tuple(layers),
        kv_bytes_per_token=kv_per_token,
    )


@dataclass
class BoyleModel:
    """A loaded, budgeted model. ``generate`` streams text."""

    model: object
    tokenizer: object
    plan: BudgetPlan
    model_dir: Path
    wrapped_layers: int

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs):
        from mlx_lm import stream_generate

        yield from stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens, **kwargs
        )

    def stats(self) -> dict:
        return offload_stats(self.model)


def load(
    model: str | Path,
    budget: int | float | str,
    max_context: int = 8192,
    io_workers: int = 8,
    colo_dir: str | None = None,
    headroom: int | float | str = "4GB",
) -> BoyleModel:
    from mlx_lm.utils import load_model, load_tokenizer

    model_dir = _resolve_model_dir(model)

    anatomy = read_anatomy(model_dir)
    budget_plan = resolve_budget(
        anatomy, budget, max_context=max_context, headroom=headroom
    )

    mdl, _cfg = load_model(model_dir, lazy=True)
    wrapped = apply_expert_offload(
        mdl,
        model_dir,
        capacity_plan=list(budget_plan.capacities),
        io_workers=io_workers,
        direct=True,
        colo_dir=colo_dir,
    )
    if wrapped != len(anatomy.layers):
        logger.warning(
            "boyle: planned %d MoE layers but wrapped %d — budget accounting "
            "is off for this model; please report it",
            len(anatomy.layers),
            wrapped,
        )
    mx.eval(mdl.parameters())
    tokenizer = load_tokenizer(model_dir)

    device_cap = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(min(budget_plan.budget_bytes, int(device_cap)))
    logger.info(
        "boyle: loaded %s at fraction %.3f (%d/%d layers wrapped)",
        model_dir.name,
        budget_plan.fraction,
        wrapped,
        len(anatomy.layers),
    )
    return BoyleModel(
        model=mdl,
        tokenizer=tokenizer,
        plan=budget_plan,
        model_dir=model_dir,
        wrapped_layers=wrapped,
    )
