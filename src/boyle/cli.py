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
        if verb in ("predict", "bench"):
            p.add_argument("--max-context", type=int, default=8192)
            p.add_argument("--headroom", default="4GB")
        if verb == "bench":
            p.add_argument("--max-tokens", type=int, default=96)
            p.add_argument("--colo", help="colocated store dir")
        if verb == "serve":
            p.add_argument("--port", type=int, help="default: 11434 if free, else 11435")
            p.add_argument("--host", default="127.0.0.1")
            # agent harnesses carry 5-8k-token system prompts; 8192 starves
            # them (measured: OpenCode overflowed on its second task)
            p.add_argument("--max-context", type=int, default=32768)
            p.add_argument("--headroom", default="4GB")
            p.add_argument("--colo", help="colocated store dir")
            p.add_argument("--temperature", type=float, default=0.7,
                           help="server default when the client sends none")
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return 0
    if args.verb == "run":
        return _run(args)
    if args.verb == "predict":
        return _predict(args)
    if args.verb == "bench":
        return _bench(args)
    if args.verb == "serve":
        return _serve(args)
    print(f"boyle {args.verb}: not implemented yet (pre-release scaffold)",
          file=sys.stderr)
    return 2


def _predict(args) -> int:
    from boyle.predict import predict

    if not args.model or not args.budget:
        print("boyle predict: model and --budget are required", file=sys.stderr)
        return 2
    f = predict(
        args.model, args.budget,
        max_context=args.max_context, headroom=args.headroom,
    )
    print(f.render())
    return 0 if f.fits else 1


def _bench(args) -> int:
    """Measure this machine against the forecast — the trust loop."""
    import time

    from boyle.loader import load
    from boyle.predict import predict

    if not args.model or not args.budget:
        print("boyle bench: model and --budget are required", file=sys.stderr)
        return 2
    fc = predict(
        args.model, args.budget,
        max_context=args.max_context, headroom=args.headroom,
    )
    if not fc.fits:
        print(fc.render())
        return 1
    print(f"[bench] predicted {fc.tok_s:.1f} tok/s "
          f"(band {fc.tok_s_lo:.1f}–{fc.tok_s_hi:.1f}); loading...",
          file=sys.stderr)
    m = load(
        args.model, budget=args.budget,
        max_context=args.max_context, headroom=args.headroom,
        colo_dir=args.colo,
    )
    prompt = "Write a detailed, factual overview of how solid-state drives work."
    t0 = time.perf_counter()
    times = []
    for r in m.generate(prompt, max_tokens=args.max_tokens):
        times.append(time.perf_counter())
    ttft = times[0] - t0
    steady = times[len(times) // 3 :]
    tok_s = (len(steady) - 1) / (steady[-1] - steady[0])
    s = m.stats()
    within = fc.tok_s_lo <= tok_s <= fc.tok_s_hi
    print(
        f"[bench] measured {tok_s:.1f} tok/s steady "
        f"(TTFT {ttft:.1f}s, hit rate {100 * (s['hit_rate'] or 0):.1f}%) — "
        f"{'WITHIN' if within else 'OUTSIDE'} the predicted band "
        f"{fc.tok_s_lo:.1f}–{fc.tok_s_hi:.1f}"
    )
    return 0 if within else 3


def _serve(args) -> int:
    from boyle.budget import BudgetError
    from boyle.loader import load
    from boyle.server import run_server

    if not args.model or not args.budget:
        print("boyle serve: model and --budget are required", file=sys.stderr)
        return 2
    try:
        m = load(
            args.model, budget=args.budget,
            max_context=args.max_context, headroom=args.headroom,
            colo_dir=args.colo,
        )
    except BudgetError as e:
        print(f"boyle: {e}", file=sys.stderr)
        return 1
    tools_supported = "qwen" in args.model.lower()
    run_server(m, args.model, tools_supported, host=args.host, port=args.port,
               default_temperature=args.temperature)
    return 0


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
