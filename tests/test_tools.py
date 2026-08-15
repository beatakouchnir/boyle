# SPDX-License-Identifier: Apache-2.0
"""Tool-call parsing and streaming hold-back: pure logic, CI-safe."""

import json

from boyle.tools import OPENER, parse_tool_calls, safe_emit_split


def test_single_tool_call():
    text = ('I will check the weather.\n<tool_call>\n'
            '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            '</tool_call>')
    p = parse_tool_calls(text)
    assert p.finish_reason == "tool_calls"
    assert p.content == "I will check the weather."
    assert len(p.tool_calls) == 1
    tc = p.tool_calls[0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}


def test_multiple_and_nested_arguments():
    text = (
        '<tool_call>{"name": "a", "arguments": {"x": {"y": [1, 2]}}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    p = parse_tool_calls(text)
    assert [c["function"]["name"] for c in p.tool_calls] == ["a", "b"]
    assert p.tool_calls[0]["id"] != p.tool_calls[1]["id"]


def test_malformed_json_left_in_content():
    text = "before <tool_call>{not json}</tool_call> after"
    p = parse_tool_calls(text)
    assert p.tool_calls == []
    assert p.finish_reason == "stop"
    assert "{not json}" in p.content


def test_plain_text_untouched():
    p = parse_tool_calls("just an answer")
    assert p.content == "just an answer" and p.tool_calls == []


def test_safe_emit_holds_opener():
    emit, held = safe_emit_split("hello " + OPENER + '{"name"', False)
    assert emit == "hello "
    assert held.startswith(OPENER)


def test_safe_emit_holds_partial_opener_tail():
    emit, held = safe_emit_split("text <tool_c", False)
    assert emit == "text "
    assert held == "<tool_c"
    # and flushes when generation is done and it was a false alarm
    emit, held = safe_emit_split("text <tool_c", True)
    assert emit == "text <tool_c" and held == ""


def test_safe_emit_passthrough():
    emit, held = safe_emit_split("no tags here. ", False)
    assert emit == "no tags here. " and held == ""
