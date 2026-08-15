# SPDX-License-Identifier: Apache-2.0
"""boyle command-line interface.

Verbs: predict · build · run · serve · bench · trace. Wiring lands verb by
verb; anything not yet implemented says so and exits non-zero rather than
pretending.
"""

from __future__ import annotations

import argparse
import sys

from boyle import __version__

_VERBS = {
    "predict": "forecast speed, max context, and measured accuracy before downloading",
    "build": "download/convert a model and build its colocated expert store",
    "run": "generate under a memory budget (one-shot or REPL)",
    "serve": "OpenAI-compatible server for local harnesses (OpenCode, aider, ...)",
    "bench": "measure this machine against the prediction",
    "trace": "capture a routing trace to add a model family to predict",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="boyle",
        description="Run the model you want at the memory pressure you specify.",
    )
    parser.add_argument("--version", action="version", version=f"boyle {__version__}")
    sub = parser.add_subparsers(dest="verb")
    for verb, help_text in _VERBS.items():
        p = sub.add_parser(verb, help=help_text)
        p.add_argument("model", nargs="?", help="HF repo id or local path")
        p.add_argument("--budget", help="memory budget, e.g. 80GB")
        if verb == "run":
            p.add_argument("-p", "--prompt", required=True)
            p.add_argument("--max-tokens", type=int, default=512)
            p.add_argument("--max-context", type=int, default=8192)
            p.add_argument("--colo", help="colocated store dir (boyle build output)")
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return 0
    if args.verb == "run":
        return _run(args)
    print(f"boyle {args.verb}: not implemented yet (pre-release scaffold)",
          file=sys.stderr)
    return 2


def _run(args) -> int:
    import time

    from boyle.budget import BudgetError, fmt_size
    from boyle.loader import load

    if not args.model or not args.budget:
        print("boyle run: model and --budget are required", file=sys.stderr)
        return 2
    try:
        m = load(
            args.model,
            budget=args.budget,
            max_context=args.max_context,
            colo_dir=args.colo,
        )
    except BudgetError as e:
        print(f"boyle: {e}", file=sys.stderr)
        return 1
    print(
        f"[boyle] fraction={m.plan.fraction:.3f} "
        f"slots={fmt_size(m.plan.slots_bytes)} "
        f"max_context={args.max_context}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    n = 0
    for r in m.generate(args.prompt, max_tokens=args.max_tokens):
        print(r.text, end="", flush=True)
        n += 1
    dt = time.perf_counter() - t0
    s = m.stats()
    hit = f"{100 * s['hit_rate']:.1f}%" if s["hit_rate"] is not None else "n/a"
    print(
        f"\n[boyle] {n} tokens in {dt:.1f}s ({n / dt:.1f} tok/s) — "
        f"expert cache hit rate {hit}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
