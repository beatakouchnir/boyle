# SPDX-License-Identifier: Apache-2.0
"""Server integration on OLMoE: both API surfaces, prefix cache, refusal.

    BOYLE_LOCAL_MODELS=1 uv run pytest tests/test_server_local.py -v
"""

import json
import os
import threading
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BOYLE_LOCAL_MODELS") != "1",
    reason="local-model integration (set BOYLE_LOCAL_MODELS=1)",
)

OLMOE = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
LONG_SYSTEM = (
    "You are a meticulous assistant. " +
    "Context document: " + "the quick brown fox jumps over the lazy dog. " * 120
)


@pytest.fixture(scope="module")
def server():
    from boyle.loader import load
    from boyle.server import serve

    m = load(OLMOE, budget="6GB", max_context=2048, headroom="1GB")
    httpd, port = serve(m, OLMOE, tools_supported=True, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return r.status, r.read()


def test_discovery_endpoints(server):
    status, body = _get(server, "/")
    assert status == 200 and b"Ollama is running" in body
    status, body = _get(server, "/api/tags")
    assert status == 200
    models = json.loads(body)["models"]
    assert models and models[0]["name"].endswith(":latest")
    status, body = _get(server, "/v1/models")
    assert json.loads(body)["data"][0]["id"] == OLMOE
    status, body = _post(server, "/api/show", {"model": "x"})
    assert "completion" in json.loads(body)["capabilities"]


def test_openai_chat_roundtrip_and_prefix_cache(server):
    convo = [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "Say the word 'ready' and nothing else."},
    ]
    status, body = _post(server, "/v1/chat/completions", {
        "messages": convo, "max_tokens": 16, "temperature": 0.0,
    })
    assert status == 200
    r1 = json.loads(body)
    cold_prompt = r1["usage"]["prompt_tokens"]
    cold_cached = r1["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cold_prompt > 600  # the long system prompt is real
    text1 = r1["choices"][0]["message"]["content"]

    convo += [
        {"role": "assistant", "content": text1},
        {"role": "user", "content": "Now say 'again'."},
    ]
    status, body = _post(server, "/v1/chat/completions", {
        "messages": convo, "max_tokens": 16, "temperature": 0.0,
    })
    r2 = json.loads(body)
    warm_cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert warm_cached > cold_cached
    assert warm_cached > 0.8 * cold_prompt, (
        f"prefix cache ineffective: cached {warm_cached} of "
        f"{r2['usage']['prompt_tokens']}"
    )


def test_openai_streaming_sse(server):
    status, body = _post(server, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Count to three."}],
        "max_tokens": 24, "stream": True,
    })
    assert status == 200
    lines = [ln for ln in body.decode().split("\n") if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert any(c["choices"][0]["delta"].get("content") for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_ollama_chat_ndjson_with_real_timings(server):
    status, body = _post(server, "/api/chat", {
        "model": "whatever:latest",
        "messages": [{"role": "user", "content": "Say hi."}],
        "options": {"num_predict": 16},
    })
    assert status == 200
    objs = [json.loads(ln) for ln in body.decode().strip().split("\n")]
    assert objs[-1]["done"] is True
    assert objs[-1]["eval_count"] > 0
    assert objs[-1]["eval_duration"] > 0  # real timings -> UIs show true tok/s
    assert any(o["message"]["content"] for o in objs[:-1])


def test_ollama_cli_protocol_regressions(server):
    """The official ollama CLI exposed two wire bugs: it health-checks with
    HEAD / (auto-501'd), and POST /api/show's unread body desynced the
    keep-alive connection so the next request line began mid-JSON."""
    import http.client

    host, port = server.replace("http://", "").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=60)
    # HEAD / must be 200, empty body
    conn.request("HEAD", "/")
    r = conn.getresponse()
    assert r.status == 200 and r.read() == b""
    # POST /api/show then ANOTHER request on the same socket must parse
    conn.request("POST", "/api/show",
                 json.dumps({"model": "x", "system": "", "template": "",
                             "verbose": False}),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    assert r.status == 200 and "capabilities" in json.loads(r.read())
    conn.request("GET", "/api/tags")
    r = conn.getresponse()
    assert r.status == 200 and json.loads(r.read())["models"]
    # empty-prompt generate = Ollama's load probe: ack, don't generate
    conn.request("POST", "/api/generate",
                 json.dumps({"model": "x", "prompt": ""}),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    body = json.loads(r.read())
    assert body["done"] is True and body["done_reason"] == "load"
    conn.close()


def test_ollama_generate_applies_chat_template(server):
    """Old ollama CLIs drive `run` through /api/generate; per Ollama
    semantics the prompt goes through the chat template unless raw:true.
    Bare text fed to an instruct model free-associates (M1, year-old CLI)."""
    status, body = _post(server, "/api/generate", {
        "model": "x", "prompt": "Reply with the single word: ready",
        "stream": False, "options": {"num_predict": 12, "temperature": 0.0}})
    assert status == 200
    r = json.loads(body)
    assert r["done"] and r["done_reason"] in ("stop", "length")
    assert "ready" in r["response"].lower()  # instruct behavior => template applied


def test_finish_reason_reports_length_on_cap(server):
    """A generation that hits max_tokens must say so: finish_reason length /
    done_reason length — szilard's first smoke caught truncated thinking
    scored as clean because this defaulted to stop."""
    status, body = _post(server, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Count upward forever: 1, 2, 3,"}],
        "max_tokens": 8, "temperature": 0.0})
    r = json.loads(body)
    assert r["choices"][0]["finish_reason"] == "length"
    status, body = _post(server, "/api/chat", {
        "model": "x", "stream": False,
        "messages": [{"role": "user", "content": "Count upward forever: 1, 2, 3,"}],
        "options": {"num_predict": 8}})
    assert json.loads(body)["done_reason"] == "length"


def test_logprobs_and_entropy_extension(server):
    """S3b/S4 dependency: OpenAI logprobs shape plus the token_entropies
    extension (full-distribution entropy, which top-k cannot reconstruct)."""
    status, body = _post(server, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 12, "temperature": 0.0,
        "logprobs": True, "top_logprobs": 3})
    assert status == 200
    r = json.loads(body)
    lp = r["choices"][0]["logprobs"]
    n = r["usage"]["completion_tokens"]
    assert len(lp["content"]) == n and len(lp["token_entropies"]) == n
    for entry, ent in zip(lp["content"], lp["token_entropies"]):
        assert entry["logprob"] <= 0.0
        assert ent >= 0.0
        tops = entry["top_logprobs"]
        assert len(tops) == 3
        assert tops == sorted(tops, key=lambda e: -e["logprob"])
    # greedy: chosen token should be the argmax => matches top-1
    assert lp["content"][0]["logprob"] == lp["content"][0]["top_logprobs"][0]["logprob"]


def test_context_overflow_refusal(server):
    huge = "word " * 4000  # ~4k tokens >> 2048 context
    status, body = _post(server, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": huge}], "max_tokens": 16,
    })
    assert status == 400
    err = json.loads(body)["error"]
    assert err["type"] == "context_overflow"
    assert "2048" in err["message"]


def test_qwen35_template_alignment_tokenizer_only():
    """Regression for the 397B demo's 813-token re-prefill: the Qwen3.5
    template seeds an empty <think> block in every generation prompt but
    re-renders plain-content assistant history WITHOUT it, so an unaligned
    cache (prompt + raw generation) diverges at each assistant reply. The
    alignment rule — trim to the prefix the canonical re-render agrees
    with — must leave the next turn a small suffix, not the conversation."""
    from mlx_lm.utils import load_tokenizer

    from boyle.loader import _resolve_model_dir
    from boyle.server import _common_prefix_len

    tok = load_tokenizer(_resolve_model_dir("mlx-community/Qwen3.5-397B-A17B-4bit"))

    def render(msgs, gen):
        return list(tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=gen, enable_thinking=False))

    base = [{"role": "system", "content": "Agent."},
            {"role": "user", "content": "Run the tests."}]
    reply = "The tests passed. All good."

    p1 = render(base, True)
    candidate = p1 + tok.encode(reply, add_special_tokens=False) + [tok.eos_token_id]
    # sentinel user message: the reply must render as HISTORY (the template
    # keeps the think block on a final assistant message, strips it later)
    canonical = render(base + [{"role": "assistant", "content": reply},
                               {"role": "user", "content": ""}], False)
    aligned = candidate[: _common_prefix_len(candidate, canonical)]

    p2 = render(base + [{"role": "assistant", "content": reply},
                        {"role": "user", "content": "Thanks."}], True)
    # unaligned cache diverges before the reply; aligned cache is a clean prefix
    assert _common_prefix_len(candidate, p2) < len(p1)  # the bug, demonstrated
    assert p2[: len(aligned)] == aligned  # the fix: full reuse of aligned cache
    assert len(p2) - len(aligned) < 60  # next turn pays a small suffix only


def test_thinking_flag_changes_generation_prompt():
    """Per-request thinking control (szilard S1 dependency): thinking=False
    pre-closes the think block in the generation prompt; True leaves it to
    the model. Tokenizer-only, real Qwen3.5 template."""
    from mlx_lm.utils import load_tokenizer

    from boyle.loader import _resolve_model_dir

    tok = load_tokenizer(_resolve_model_dir("mlx-community/Qwen3.5-397B-A17B-4bit"))
    msgs = [{"role": "user", "content": "Hi."}]

    def render(thinking):
        ids = tok.apply_chat_template(msgs, tokenize=True,
                                      add_generation_prompt=True,
                                      enable_thinking=thinking)
        return tok.decode(list(ids)[-12:])

    off, on = render(False), render(True)
    assert off != on
    assert "<think>" in off and "</think>" in off  # pre-closed = skip
    assert "</think>" not in on  # model gets to think


def test_streaming_overflow_is_clean_400_and_socket_survives(server):
    """Regression: overflow on a STREAMING request once fired after the 200 +
    chunked headers were sent (lazy generator), writing a 400 status line
    into the live stream — clients reported InvalidHTTPResponse and the
    keep-alive socket was poisoned for the next request (seen in a real
    OpenCode session at 8.2k tokens against the 8192 default)."""
    import http.client

    host, port = server.replace("http://", "").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=120)
    huge = "word " * 4000
    conn.request("POST", "/v1/chat/completions",
                 json.dumps({"messages": [{"role": "user", "content": huge}],
                             "max_tokens": 16, "stream": True}),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 400  # clean pre-stream refusal, not mid-stream junk
    assert json.loads(resp.read())["error"]["type"] == "context_overflow"
    # same socket must still serve the next request
    conn.request("POST", "/v1/chat/completions",
                 json.dumps({"messages": [{"role": "user", "content": "Say ok."}],
                             "max_tokens": 8, "stream": True}),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    assert b"data: [DONE]" in resp.read()
    conn.close()
