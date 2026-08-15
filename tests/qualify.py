# SPDX-License-Identifier: Apache-2.0
"""Qualify a model for COMPATIBILITY.md — the standard battery, one command.

    BOYLE_LOCAL_MODELS=1 uv run python tests/qualify.py <hf-repo> --budget 8GB

Downloads the model if absent. Prints PASS/FLAG/NA per step and ends with a
markdown row to paste into COMPATIBILITY.md (plus a JSON record for
boyle/data/tested_models.json).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--budget", required=True)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--headroom", default="2GB")
    ap.add_argument("--skip-identity", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx

    from boyle.budget import BudgetError, fmt_size
    from boyle.loader import _resolve_model_dir, load, read_anatomy
    from boyle.server import GenerationCore, classify_prefix_behavior

    checks: list[str] = []
    flags: list[str] = []

    # 1 — anatomy + plan
    model_dir = _resolve_model_dir(args.model)
    anatomy = read_anatomy(model_dir)
    dense = len(anatomy.layers) == 0
    print(f"[1] anatomy: {len(anatomy.layers)} MoE layers, "
          f"experts {fmt_size(anatomy.expert_bytes)}, "
          f"resident {fmt_size(anatomy.resident_bytes)}"
          + (" — DENSE model (runs resident, budget still enforced)" if dense else ""))
    try:
        m = load(args.model, budget=args.budget, max_context=args.max_context,
                 headroom=args.headroom)
    except BudgetError as e:
        print(f"    budget refusal (working as designed): {e}")
        return 1
    print(f"    plan: fraction {m.plan.fraction:.2f}, wrapped {m.wrapped_layers}")
    checks.append("anatomy/plan")
    if not dense and m.wrapped_layers != len(anatomy.layers):
        flags.append(f"wrapped {m.wrapped_layers}/{len(anatomy.layers)} MoE layers "
                     "— unsupported layout for the rest? check logs")

    # 2 — template roundtrip + cache class
    cls, detail = classify_prefix_behavior(m)
    print(f"[2] prefix behavior [{cls}]: {detail}")
    checks.append(f"prefix cache: {cls}")
    if cls == "unknown":
        flags.append("template probe failed — chat use unverified")

    core = GenerationCore(m, args.model, tools_supported=True)

    def turn(msgs, tools=None, n=48):
        tokens = core._tokenize_chat(msgs, tools)
        final = None
        for kind, payload in core.generate(tokens, n, 0.0, 1.0, bool(tools)):
            if kind == "final":
                final = payload
        return final

    # 3 — bit-identity vs resident (size-gated)
    if args.skip_identity or dense:
        print("[3] bit-identity: skipped" + (" (dense)" if dense else " (flag)"))
    else:
        total = anatomy.resident_bytes + anatomy.expert_bytes
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if total > 0.55 * ram:
            print(f"[3] bit-identity: NA — model ({fmt_size(total)}) exceeds "
                  "resident headroom; contract carried by smaller family members")
            checks.append("bit-identity: NA (size)")
        else:
            from mlx_lm import load as mload, stream_generate

            prompt = "The capital of France is"
            ids_b = [r.token for r in stream_generate(
                m.model, m.tokenizer, prompt, max_tokens=32)]
            rm, rt = mload(str(model_dir))
            ids_r = [r.token for r in stream_generate(rm, rt, prompt, max_tokens=32)]
            del rm
            gc.collect(); mx.clear_cache()
            if ids_b == ids_r:
                note = ""
                if not dense and m.plan.fraction >= 1.0:
                    note = (" — NOTE: fully resident at this budget; re-run "
                            "with a tighter --budget to exercise offload misses")
                print(f"[3] bit-identity: PASS (32 tokens exact){note}")
                checks.append("bit-identity vs resident"
                              + (" (resident-fraction)" if note else ""))
            else:
                print("[3] bit-identity: FLAG — token divergence")
                flags.append("bit-identity divergence — investigate before listing")

    # 4 — tool dialect
    TOOLS = [{"type": "function", "function": {
        "name": "get_time", "description": "Get the current time",
        "parameters": {"type": "object", "properties": {
            "tz": {"type": "string"}}, "required": ["tz"]}}}]
    f = turn([{"role": "system", "content":
               "You must use the provided tool to answer. Always call it."},
              {"role": "user", "content": "What time is it in Paris?"}],
             TOOLS, n=120)
    raw = f.text
    if f.tool_calls:
        print(f"[4] tool dialect: PASS — parsed {f.tool_calls[0]['function']}")
        checks.append("tool calls parsed")
    elif "<tool_call" in raw or "<function" in raw:
        print(f"[4] tool dialect: FLAG — unparsed tool syntax: {raw[:120]!r}")
        flags.append("unparsed tool dialect — parser gap, file with the sample")
    else:
        print(f"[4] tool dialect: model declined ({raw[:80]!r}) — parser untested")
        checks.append("tool calls: model declined (untested)")

    # 5 — prefix warm ratio
    msgs = [{"role": "system", "content": "Be terse. " * 60},
            {"role": "user", "content": "Say A."}]
    f1 = turn(msgs, n=8)
    msgs += [{"role": "assistant", "content": f1.text},
             {"role": "user", "content": "Say B."}]
    f2 = turn(msgs, n=8)
    ratio = f2.cached_tokens / max(1, f2.prompt_tokens)
    print(f"[5] prefix warm: {f2.cached_tokens}/{f2.prompt_tokens} cached "
          f"({100 * ratio:.0f}%)")
    if cls in ("full", "aligned") and ratio < 0.7:
        flags.append(f"warm ratio {100 * ratio:.0f}% despite '{cls}' class — investigate")
    else:
        checks.append(f"prefix warm {100 * ratio:.0f}%")

    # 6 — bench vs predict pointer
    print("[6] run `boyle bench` for the forecast-band check "
          "(separate process; out-of-band = bug report)")

    date = time.strftime("%Y-%m-%d")
    verdict = "FLAGS: " + "; ".join(flags) if flags else "clean"
    print("\n=== qualification " + ("FLAGGED" if flags else "PASSED") + f" — {verdict}")
    print("\nCOMPATIBILITY.md row:")
    print(f"| {args.model} | {date} | {'; '.join(checks)} |")
    print("\ntested_models.json record:")
    print(json.dumps({"model": args.model, "verified": date, "checks": checks}))
    return 2 if flags else 0


if __name__ == "__main__":
    sys.exit(main())
