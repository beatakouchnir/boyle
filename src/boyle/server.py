# SPDX-License-Identifier: Apache-2.0
"""boyle serve: one generation core, two API surfaces.

OpenAI-compatible (/v1/*, SSE) covers OpenCode, aider, Cline, Continue,
Zed, SillyTavern, LibreChat and anything built on the openai client.
Ollama-compatible (/api/*, NDJSON) covers apps that only know Ollama;
timing fields are real measurements, so UIs display true tok/s.

Execution is single-stream by design: requests queue on a lock. Diverse
batching is drive-bound (measured ~9.5 tok/s aggregate at any batch size),
so concurrency would move latency around, not create throughput.

The prompt-prefix cache is the harness-latency lever: agents resend the
whole conversation every turn, so warm turns re-prefill only the new
suffix. Single slot — the single-user agent pattern — with trim-based
recovery when histories diverge.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from boyle.budget import fmt_size
from boyle.tools import parse_tool_calls, safe_emit_split

logger = logging.getLogger(__name__)

OLLAMA_PORT = 11434
FALLBACK_PORT = 11435


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class ContextOverflow(Exception):
    pass


@dataclass
class Reply:
    """Filled in as generation streams; final state after the generator ends."""

    text: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0


class GenerationCore:
    """Serialized generation over one loaded BoyleModel, with prefix cache.

    All Metal work happens on one dedicated worker thread: mlx-lm's
    generation stream lives on the thread that first uses it, and
    ThreadingHTTPServer handles each request on a fresh thread — generating
    there dies with "no Stream(gpu, N) in current thread". The worker also
    makes request ordering genuinely FIFO, which a bare Lock is not.
    """

    def __init__(self, bmodel, model_id: str, tools_supported: bool):
        import queue

        self.m = bmodel
        self.model_id = model_id
        self.tools_supported = tools_supported
        self._cache = None
        self._cache_ids: list[int] = []
        # Warm up ON THE LOADING THREAD before any worker generation:
        # measured (not just theorized) — if the first-ever generation with
        # an offloaded model happens on a secondary thread, its prefill dies
        # with "no Stream(gpu, 1) in current thread"; after one main-thread
        # token, generation works from any thread indefinitely.
        from mlx_lm import stream_generate

        for _ in stream_generate(bmodel.model, bmodel.tokenizer, "warmup",
                                 max_tokens=1):
            pass
        self._jobs: "queue.Queue" = queue.Queue()
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

    def _work_loop(self):
        while True:
            args, out = self._jobs.get()
            try:
                for event in self._generate_on_worker(*args):
                    out.put(event)
            except Exception as e:  # surfaces on the requesting thread
                out.put(("error", e))
            out.put(None)

    # -- prompt assembly ---------------------------------------------------

    def _normalize(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    c = json.loads(json.dumps(c))  # deep copy
                    fn = c.get("function", {})
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except json.JSONDecodeError:
                            pass
                    calls.append(c)
                m["tool_calls"] = calls
            if m.get("content") is None:
                m["content"] = ""
            out.append(m)
        return out

    def _tokenize_chat(self, messages, tools) -> list[int]:
        kwargs = {"add_generation_prompt": True, "tokenize": True}
        if tools:
            kwargs["tools"] = tools
        ids = self.m.tokenizer.apply_chat_template(self._normalize(messages), **kwargs)
        return list(ids)

    # -- prefix cache ------------------------------------------------------

    def _prepare_cache(self, tokens: list[int]) -> tuple[list[int], int]:
        """Returns (suffix_to_prefill, cached_common_len); mutates cache state."""
        from mlx_lm.models.cache import (
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )

        common = _common_prefix_len(self._cache_ids, tokens)
        # always leave at least one token to feed generation
        common = min(common, len(tokens) - 1)
        if self._cache is None or common <= 0:
            self._cache = make_prompt_cache(self.m.model)
            self._cache_ids = []
            return tokens, 0
        if common < len(self._cache_ids):
            if can_trim_prompt_cache(self._cache):
                trim_prompt_cache(self._cache, len(self._cache_ids) - common)
                self._cache_ids = self._cache_ids[:common]
            else:
                self._cache = make_prompt_cache(self.m.model)
                self._cache_ids = []
                return tokens, 0
        return tokens[common:], common

    # -- generation --------------------------------------------------------

    def generate(
        self,
        tokens: list[int],
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        parse_tools: bool,
    ):
        """Returns an iterator of ("delta", text) events ending with
        ("final", Reply). Validation is EAGER — deliberately not a generator
        function: ContextOverflow must fire on this call, while the handler
        can still send a clean 400. (It once fired lazily, after the 200 +
        chunked headers were already out, and the error body wrote a fresh
        status line into the live stream — clients saw InvalidHTTPResponse.)
        """
        import queue

        limit = self.m.plan.max_context
        want = max_tokens or 2048
        if len(tokens) + want > limit:
            if len(tokens) >= limit:
                raise ContextOverflow(
                    f"prompt is {len(tokens)} tokens; the budget guarantees "
                    f"{limit}. Raise --budget or --max-context at serve time, "
                    f"or shorten the conversation."
                )
            want = limit - len(tokens)
        out: "queue.Queue" = queue.Queue()
        self._jobs.put(((tokens, want, temperature, top_p, parse_tools), out))

        def _events():
            while True:
                event = out.get()
                if event is None:
                    return
                if event[0] == "error":
                    raise event[1]
                yield event

        return _events()

    def _generate_on_worker(self, tokens, want, temperature, top_p, parse_tools):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        suffix, cached = self._prepare_cache(tokens)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        r = Reply(prompt_tokens=len(tokens), cached_tokens=cached)
        t0 = time.perf_counter()
        t_first = None
        gen_ids = []
        held = ""
        for out in stream_generate(
            self.m.model,
            self.m.tokenizer,
            suffix,
            max_tokens=want,
            sampler=sampler,
            prompt_cache=self._cache,
        ):
            if t_first is None:
                t_first = time.perf_counter()
            gen_ids.append(out.token)
            if parse_tools:
                emit, held = safe_emit_split(held + out.text, False)
                if emit:
                    r.text += emit
                    yield ("delta", emit)
            else:
                r.text += out.text
                yield ("delta", out.text)
        t_end = time.perf_counter()
        if parse_tools:
            parsed = parse_tool_calls(r.text + held)
            r.tool_calls = parsed.tool_calls
            r.finish_reason = parsed.finish_reason
            if parsed.tool_calls:
                r.text = parsed.content
            elif held:  # held tail was a false alarm, flush it
                r.text += held
                yield ("delta", held)
        r.completion_tokens = len(gen_ids)
        r.prompt_eval_s = (t_first or t_end) - t0
        r.eval_s = t_end - (t_first or t_end)
        self._cache_ids = tokens + gen_ids
        yield ("final", r)


# --- HTTP layer -----------------------------------------------------------


def make_handler(core: GenerationCore):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            logger.debug("http: " + fmt, *args)

        # -- plumbing ------------------------------------------------------

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                return {}

        def _json(self, obj, status=200):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _start_stream(self, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._streaming = content_type

        def _chunk(self, data: bytes):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")

        def _end_chunks(self):
            self.wfile.write(b"0\r\n\r\n")

        def _sse(self, obj):
            self._chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")

        def _ndjson(self, obj):
            self._chunk(json.dumps(obj).encode() + b"\n")

        # -- routing -------------------------------------------------------

        def do_GET(self):
            try:
                self._do_get()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as e:
                logger.exception("GET failed")
                try:
                    self._json({"error": str(e)}, 500)
                except Exception:
                    self.close_connection = True

        def _do_get(self):
            if self.path == "/":
                data = b"Ollama is running"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/version":
                self._json({"version": "0.11.0"})
            elif self.path == "/api/tags":
                self._json({"models": [self._ollama_model_card()]})
            elif self.path == "/api/ps":
                card = self._ollama_model_card()
                card["expires_at"] = "never"
                self._json({"models": [card]})
            elif self.path == "/v1/models":
                self._json(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": core.model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "boyle",
                                "max_context_length": core.m.plan.max_context,
                            }
                        ],
                    }
                )
            else:
                self._json({"error": f"unknown path {self.path}"}, 404)

        def do_POST(self):
            self._streaming = None
            try:
                if self.path == "/v1/chat/completions":
                    self._oai_chat()
                elif self.path == "/v1/completions":
                    self._oai_completions()
                elif self.path == "/api/chat":
                    self._ollama_chat()
                elif self.path == "/api/generate":
                    self._ollama_generate()
                elif self.path == "/api/show":
                    self._ollama_show()
                elif self.path.startswith(("/api/pull", "/api/push", "/api/create",
                                           "/api/delete", "/api/copy")):
                    self._json(
                        {"error": "boyle does not manage models over the API — "
                                  "models come from `boyle run/serve <hf-repo>` "
                                  "(and `boyle build` for colocated stores)"},
                        501,
                    )
                else:
                    self._json({"error": f"unknown path {self.path}"}, 404)
            except ContextOverflow as e:
                self._error_out({"message": str(e), "type": "context_overflow"}, 400)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as e:
                logger.exception("request failed")
                self._error_out({"message": str(e), "type": "server_error"}, 500)

        def _error_out(self, err: dict, status: int):
            """Report an error without ever corrupting the wire protocol.

            Before streaming: a normal JSON error response. After the 200 +
            chunked headers are out, a status line would land inside the
            chunk framing (clients report InvalidHTTPResponse and the
            keep-alive socket is poisoned) — so instead terminate the stream
            with an in-band error event and drop the connection.
            """
            try:
                if not getattr(self, "_streaming", None):
                    self._json({"error": err}, status)
                    return
                if self._streaming == "text/event-stream":
                    self._sse({"error": err})
                    self._chunk(b"data: [DONE]\n\n")
                else:
                    self._ndjson({"error": err["message"], "done": True})
                self._end_chunks()
            except Exception:
                pass
            self.close_connection = True

        # -- shared bits ---------------------------------------------------

        def _ollama_model_card(self):
            short = core.model_id.split("/")[-1].lower()
            return {
                "name": f"{short}:latest",
                "model": f"{short}:latest",
                "modified_at": _now_iso(),
                "size": core.m.plan.resident_bytes + core.m.plan.slots_bytes,
                "digest": f"{abs(hash(core.model_id)):064x}"[:64],
                "details": {
                    "format": "safetensors (mlx)",
                    "family": core.model_id.split("/")[-1].split("-")[0].lower(),
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }

        def _gen_params(self, body):
            opts = body.get("options") or {}
            return (
                body.get("max_tokens") or body.get("max_completion_tokens")
                or opts.get("num_predict"),
                float(body.get("temperature", opts.get("temperature", 0.7))),
                float(body.get("top_p", opts.get("top_p", 1.0))),
            )

        # -- OpenAI surface ------------------------------------------------

        def _oai_chat(self):
            body = self._body()
            messages = body.get("messages") or []
            tools = body.get("tools") or None
            max_tokens, temp, top_p = self._gen_params(body)
            tokens = core._tokenize_chat(messages, tools)
            parse = bool(tools) and core.tools_supported
            events = core.generate(tokens, max_tokens, temp, top_p, parse)
            rid = f"chatcmpl-{int(time.time() * 1000)}"

            if body.get("stream"):
                self._start_stream("text/event-stream")
                base = {
                    "id": rid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": core.model_id,
                }
                self._sse({**base, "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]})
                final = None
                for kind, payload in events:
                    if kind == "delta" and payload:
                        self._sse({**base, "choices": [
                            {"index": 0, "delta": {"content": payload},
                             "finish_reason": None}
                        ]})
                    elif kind == "final":
                        final = payload
                if final.tool_calls:
                    self._sse({**base, "choices": [
                        {"index": 0,
                         "delta": {"tool_calls": [
                             {"index": i, **tc} for i, tc in enumerate(final.tool_calls)
                         ]},
                         "finish_reason": None}
                    ]})
                self._sse({**base,
                           "choices": [{"index": 0, "delta": {},
                                        "finish_reason": final.finish_reason}],
                           "usage": self._usage(final)})
                self._chunk(b"data: [DONE]\n\n")
                self._end_chunks()
            else:
                final = None
                for kind, payload in events:
                    if kind == "final":
                        final = payload
                msg = {"role": "assistant", "content": final.text or None}
                if final.tool_calls:
                    msg["tool_calls"] = final.tool_calls
                self._json({
                    "id": rid, "object": "chat.completion",
                    "created": int(time.time()), "model": core.model_id,
                    "choices": [{"index": 0, "message": msg,
                                 "finish_reason": final.finish_reason}],
                    "usage": self._usage(final),
                })

        def _usage(self, r: Reply):
            return {
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.prompt_tokens + r.completion_tokens,
                "prompt_tokens_details": {"cached_tokens": r.cached_tokens},
            }

        def _oai_completions(self):
            body = self._body()
            prompt = body.get("prompt") or ""
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            max_tokens, temp, top_p = self._gen_params(body)
            tokens = list(core.m.tokenizer.encode(prompt))
            events = core.generate(tokens, max_tokens, temp, top_p, False)
            rid = f"cmpl-{int(time.time() * 1000)}"
            if body.get("stream"):
                self._start_stream("text/event-stream")
                base = {"id": rid, "object": "text_completion",
                        "created": int(time.time()), "model": core.model_id}
                final = None
                for kind, payload in events:
                    if kind == "delta" and payload:
                        self._sse({**base, "choices": [
                            {"index": 0, "text": payload, "finish_reason": None}]})
                    elif kind == "final":
                        final = payload
                self._sse({**base, "choices": [
                    {"index": 0, "text": "", "finish_reason": final.finish_reason}],
                    "usage": self._usage(final)})
                self._chunk(b"data: [DONE]\n\n")
                self._end_chunks()
            else:
                final = None
                for kind, payload in events:
                    if kind == "final":
                        final = payload
                self._json({
                    "id": rid, "object": "text_completion",
                    "created": int(time.time()), "model": core.model_id,
                    "choices": [{"index": 0, "text": final.text,
                                 "finish_reason": final.finish_reason}],
                    "usage": self._usage(final),
                })

        # -- Ollama surface ------------------------------------------------

        def _ollama_timings(self, r: Reply):
            total = r.prompt_eval_s + r.eval_s
            return {
                "total_duration": int(total * 1e9),
                "load_duration": 0,
                "prompt_eval_count": r.prompt_tokens - r.cached_tokens,
                "prompt_eval_duration": int(r.prompt_eval_s * 1e9),
                "eval_count": r.completion_tokens,
                "eval_duration": int(r.eval_s * 1e9),
            }

        def _ollama_chat(self):
            body = self._body()
            messages = body.get("messages") or []
            tools = body.get("tools") or None
            max_tokens, temp, top_p = self._gen_params(body)
            tokens = core._tokenize_chat(messages, tools)
            parse = bool(tools) and core.tools_supported
            events = core.generate(tokens, max_tokens, temp, top_p, parse)
            name = body.get("model") or core.model_id
            stream = body.get("stream", True)

            def tool_calls_ollama(r):
                return [
                    {"function": {
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"]["arguments"]),
                    }}
                    for tc in r.tool_calls
                ]

            if stream:
                self._start_stream("application/x-ndjson")
                final = None
                for kind, payload in events:
                    if kind == "delta" and payload:
                        self._ndjson({
                            "model": name, "created_at": _now_iso(),
                            "message": {"role": "assistant", "content": payload},
                            "done": False,
                        })
                    elif kind == "final":
                        final = payload
                if final.tool_calls:
                    self._ndjson({
                        "model": name, "created_at": _now_iso(),
                        "message": {"role": "assistant", "content": "",
                                    "tool_calls": tool_calls_ollama(final)},
                        "done": False,
                    })
                self._ndjson({
                    "model": name, "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True, "done_reason": final.finish_reason,
                    **self._ollama_timings(final),
                })
                self._end_chunks()
            else:
                final = None
                for kind, payload in events:
                    if kind == "final":
                        final = payload
                msg = {"role": "assistant", "content": final.text}
                if final.tool_calls:
                    msg["tool_calls"] = tool_calls_ollama(final)
                self._json({
                    "model": name, "created_at": _now_iso(), "message": msg,
                    "done": True, "done_reason": final.finish_reason,
                    **self._ollama_timings(final),
                })

        def _ollama_generate(self):
            body = self._body()
            prompt = body.get("prompt") or ""
            max_tokens, temp, top_p = self._gen_params(body)
            tokens = list(core.m.tokenizer.encode(prompt))
            events = core.generate(tokens, max_tokens, temp, top_p, False)
            name = body.get("model") or core.model_id
            if body.get("stream", True):
                self._start_stream("application/x-ndjson")
                final = None
                for kind, payload in events:
                    if kind == "delta" and payload:
                        self._ndjson({"model": name, "created_at": _now_iso(),
                                      "response": payload, "done": False})
                    elif kind == "final":
                        final = payload
                self._ndjson({"model": name, "created_at": _now_iso(),
                              "response": "", "done": True,
                              "done_reason": final.finish_reason,
                              **self._ollama_timings(final)})
                self._end_chunks()
            else:
                final = None
                for kind, payload in events:
                    if kind == "final":
                        final = payload
                self._json({"model": name, "created_at": _now_iso(),
                            "response": final.text, "done": True,
                            "done_reason": final.finish_reason,
                            **self._ollama_timings(final)})

        def _ollama_show(self):
            caps = ["completion"]
            if core.tools_supported:
                caps.append("tools")
            fam = core.model_id.split("/")[-1].split("-")[0].lower()
            self._json({
                "modelfile": "# served by boyle",
                "parameters": "",
                "template": "",
                "details": self._ollama_model_card()["details"],
                "model_info": {
                    "general.architecture": fam,
                    f"{fam}.context_length": core.m.plan.max_context,
                },
                "capabilities": caps,
            })

    return Handler


def serve(
    bmodel,
    model_id: str,
    tools_supported: bool,
    host: str = "127.0.0.1",
    port: int | None = None,
) -> tuple[ThreadingHTTPServer, int]:
    """Bind (with the polite port policy) and return the server, unstarted."""
    core = GenerationCore(bmodel, model_id, tools_supported)
    handler = make_handler(core)
    candidates = [port] if port else [OLLAMA_PORT, FALLBACK_PORT, 0]
    last_err = None
    for cand in candidates:
        try:
            httpd = ThreadingHTTPServer((host, cand), handler)
            httpd.daemon_threads = True
            actual = httpd.server_address[1]
            if not port and cand != OLLAMA_PORT:
                print(f"[boyle] port {OLLAMA_PORT} is taken (a real Ollama?) — "
                      f"serving on {actual}; Ollama-first apps need the URL "
                      f"http://{host}:{actual} configured explicitly")
            return httpd, actual
        except OSError as e:
            last_err = e
    raise last_err


def run_server(bmodel, model_id, tools_supported, host="127.0.0.1", port=None):
    httpd, actual = serve(bmodel, model_id, tools_supported, host, port)
    plan = bmodel.plan
    print(f"[boyle] serving {model_id}")
    print(f"[boyle]   budget: fraction {plan.fraction:.2f}, "
          f"slots {fmt_size(plan.slots_bytes)}, context {plan.max_context}")
    print(f"[boyle]   OpenAI-compatible: http://{host}:{actual}/v1")
    print(f"[boyle]   Ollama-compatible: http://{host}:{actual}")
    print(f"[boyle]   tool calls: {'parsed (hermes)' if tools_supported else 'passthrough only'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[boyle] shutting down")
        httpd.shutdown()
