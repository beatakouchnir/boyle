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
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return 0
    print(f"boyle {args.verb}: not implemented yet (pre-release scaffold)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
