# SPDX-License-Identifier: Apache-2.0
"""Tool-call extraction from model output.

v1 parses the hermes-style ``<tool_call>{...}</tool_call>`` format used by
the Qwen family — the family with measured curves and the strongest local
tool-calling. Families without a parser pass their text through untouched
and the support matrix says so; a wrong parse is worse than no parse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

OPENER = "<tool_call>"
_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class ParsedOutput:
    content: str
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self.tool_calls else "stop"


def parse_tool_calls(text: str) -> ParsedOutput:
    """Split generated text into user-visible content + structured calls.

    Malformed JSON inside a block is left in the content untouched — the
    harness sees exactly what the model said instead of a silent drop.
    """
    calls = []
    kept = []
    last = 0
    for m in _BLOCK.finditer(text):
        try:
            obj = json.loads(m.group(1))
            name = obj["name"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # leave the malformed block in content
        kept.append(text[last : m.start()])
        last = m.end()
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(obj.get("arguments", {})),
                },
            }
        )
    kept.append(text[last:])
    return ParsedOutput(content="".join(kept).strip(), tool_calls=calls)


def safe_emit_split(pending: str, done: bool) -> tuple[str, str]:
    """(emit_now, hold_back) for streaming.

    Never emit text that could be the start of a ``<tool_call>`` block:
    once an opener is present, everything from it onward is held; otherwise
    hold a small tail in case an opener spans a chunk boundary.
    """
    idx = pending.find(OPENER)
    if idx != -1:
        return pending[:idx], pending[idx:]
    if done:
        return pending, ""
    # hold the longest suffix that is a prefix of the opener
    for k in range(min(len(OPENER) - 1, len(pending)), 0, -1):
        if pending.endswith(OPENER[:k]):
            return pending[:-k], pending[-k:]
    return pending, ""
