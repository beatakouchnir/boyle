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


def test_context_overflow_refusal(server):
    huge = "word " * 4000  # ~4k tokens >> 2048 context
    status, body = _post(server, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": huge}], "max_tokens": 16,
    })
    assert status == 400
    err = json.loads(body)["error"]
    assert err["type"] == "context_overflow"
    assert "2048" in err["message"]


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
