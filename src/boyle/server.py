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


def _seeded_sampler(temperature: float, top_p: float, seed: int | None):
    """Categorical/top-p sampler over an explicit key chain."""
    import os as _os

    import mlx.core as mx

    if seed is None:
        seed = int.from_bytes(_os.urandom(4), "little")
    state = {"key": mx.random.key(seed)}

    def sample(logprobs: mx.array) -> mx.array:
        state["key"], sub = mx.random.split(state["key"])
        logits = logprobs * (1 / temperature)
        if 0 < top_p < 1:
            probs = mx.softmax(logits, axis=-1)
            order = mx.argsort(-logits, axis=-1)
            cum = mx.cumsum(mx.take_along_axis(probs, order, axis=-1), axis=-1)
            keep_sorted = cum - mx.take_along_axis(probs, order, axis=-1) < top_p
            keep = mx.zeros_like(keep_sorted)
            keep = mx.put_along_axis(keep, order, keep_sorted, axis=-1)
            logits = mx.where(keep, logits, mx.array(-float("inf")))
        return mx.random.categorical(logits, key=sub)

    return sample


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
    logprobs: list | None = None
    token_entropies: list | None = None


class _EventStream:
    """Iterator of generation events that also exposes .cancel (an Event)."""

    __slots__ = ("_it", "cancel")

    def __init__(self, it, cancel):
        self._it = it
        self.cancel = cancel

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class GenerationCore:
    """Serialized generation over one loaded BoyleModel, with prefix cache.

    All Metal work happens on one dedicated worker thread: mlx-lm's
    generation stream lives on the thread that first uses it, and
    ThreadingHTTPServer handles each request on a fresh thread — generating
    there dies with "no Stream(gpu, N) in current thread". The worker also
    makes request ordering genuinely FIFO, which a bare Lock is not.
    """

    def __init__(self, bmodel, model_id: str, tools_supported: bool,
                 default_temperature: float = 0.7):
        import queue

        self.m = bmodel
        self.model_id = model_id
        self.tools_supported = tools_supported
        self.default_temperature = default_temperature
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
        self._active_cancel: threading.Event | None = None
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

    def _work_loop(self):
        while True:
            args, out = self._jobs.get()
            if args is None:             # liveness ping — proves the WORKER
                out.put(("pong", None))  # services the queue, not merely that
                out.put(None)            # an HTTP thread answered
                continue
            self._active_cancel = args[-1]   # this job's cancel Event
            try:
                for event in self._generate_on_worker(*args):
                    out.put(event)
            except Exception as e:  # surfaces on the requesting thread
                out.put(("error", e))
            finally:
                self._active_cancel = None
            out.put(None)

    def ping(self, timeout: float = 2.0) -> bool:
        """True iff the worker services a trivial job within `timeout`.

        Behind a long generation the ping queues, times out, and returns
        False — the honest "busy/backed-up" signal that `/v1/models` cannot
        give, since that route never touches the worker at all.
        """
        import queue

        out: queue.Queue = queue.Queue()
        self._jobs.put((None, out))
        try:
            return out.get(timeout=timeout) == ("pong", None)
        except queue.Empty:
            return False

    def cancel_active(self) -> None:
        """Ask the running generation to stop at its next token — used when
        the requesting client vanished, and on shutdown."""
        c = getattr(self, "_active_cancel", None)
        if c is not None:
            c.set()

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

    def _tokenize_chat(self, messages, tools, generation: bool = True,
                       thinking: bool = False) -> list[int]:
        # thinking defaults OFF: serving targets agents and chat UIs, where
        # interleaved reasoning in message.content breaks clients. Callers
        # opt in per request (OpenAI extra field "enable_thinking"; Ollama
        # "think"). Templates without the variable simply ignore it
        # (Qwen3/3.5 honor it).
        kwargs = {"add_generation_prompt": generation, "tokenize": True,
                  "enable_thinking": thinking}
        if tools:
            kwargs["tools"] = tools
        ids = self.m.tokenizer.apply_chat_template(self._normalize(messages), **kwargs)
        return list(ids)

    @staticmethod
    def reply_message(r: Reply) -> dict:
        """The assistant message exactly as handlers return it to clients —
        and therefore exactly as clients will echo it back next turn."""
        msg = {"role": "assistant", "content": r.text or ""}
        if r.tool_calls:
            msg["tool_calls"] = r.tool_calls
        return msg

    def _align_cache(self, candidate: list[int], messages, tools, r: Reply) -> None:
        """Trim the cache to the prefix the NEXT turn's render will agree with.

        What we cache is prompt + raw generated tokens; what the next prompt
        contains is the template's RE-RENDER of that exchange, and templates
        are not roundtrip-faithful — measured on Qwen3.5: the generation
        prompt seeds an empty <think> block that is kept while the assistant
        message is FINAL but stripped once it becomes HISTORY, so an
        unaligned cache misses from that point on (an 813-token re-prefill
        in the 397B demo). The sentinel user message below forces the new
        reply to render as history — exactly as every future prompt will
        render it; the sentinel itself sits beyond any matchable prefix, so
        its content is irrelevant. Aligning after every reply caps the
        damage at the current reply's length instead of the rest of the
        conversation.
        """
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        try:
            canonical = self._tokenize_chat(
                list(messages)
                + [self.reply_message(r), {"role": "user", "content": ""}],
                tools,
                generation=False,
            )
        except Exception as e:  # template quirk: keep the unaligned cache
            logger.warning("cache alignment render failed (%s) — cache kept "
                           "unaligned; warm turns may re-prefill", e)
            self._cache_ids = candidate
            return
        common = _common_prefix_len(candidate, canonical)
        if common < len(candidate):
            if can_trim_prompt_cache(self._cache):
                trim_prompt_cache(self._cache, len(candidate) - common)
                self._cache_ids = candidate[:common]
                return
            # Hybrid-attention caches (e.g. Qwen3.5) cannot rewind, so the
            # aligned point is unreachable: keep the full cache. Tool loops
            # within one user turn still reuse it 100% (the template keeps
            # think blocks until the next user message); each NEW user turn
            # then re-prefills once. Measured on 397B: warm tool turn 5.5s,
            # per-user-turn rebuild ~35s. v1.1 design (fork-and-advance)
            # is in the scope doc.
            logger.info(
                "cache alignment wants to trim %d tokens but this model's "
                "cache is not trimmable — new user turns will re-prefill",
                len(candidate) - common,
            )
        self._cache_ids = candidate

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
        if self._cache_ids:
            logger.debug(
                "prefix cache: %d cached, %d prompt, %d common (%.0f%%)",
                len(self._cache_ids), len(tokens), common,
                100 * common / max(1, min(len(self._cache_ids), len(tokens))),
            )
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
        chat_ctx: tuple | None = None,
        logprobs_k: int | None = None,
        seed: int | None = None,
    ):
        """Returns an iterator of ("delta", text) events ending with
        ("final", Reply). Validation is EAGER — deliberately not a generator
        function: ContextOverflow must fire on this call, while the handler
        can still send a clean 400. (It once fired lazily, after the 200 +
        chunked headers were already out, and the error body wrote a fresh
        status line into the live stream — clients saw InvalidHTTPResponse.)
        ``chat_ctx=(messages, tools)`` enables post-generation cache
        alignment for chat endpoints.
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
        cancel = threading.Event()
        self._jobs.put(((tokens, want, temperature, top_p, parse_tools,
                         chat_ctx, logprobs_k, seed, cancel), out))

        def _events():
            while True:
                event = out.get()
                if event is None:
                    return
                if event[0] == "error":
                    raise event[1]
                yield event

        # The handler cancels through .cancel when its client disconnects:
        # an abandoned generation otherwise runs to full length on the single
        # worker, blocking every queued request behind it (including the
        # client's own retry) while /v1/models keeps answering 200.
        # (A plain generator cannot carry the attribute — no __dict__.)
        return _EventStream(_events(), cancel)

    def _generate_on_worker(self, tokens, want, temperature, top_p, parse_tools,
                            chat_ctx, logprobs_k=None, seed=None, cancel=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        suffix, cached = self._prepare_cache(tokens)
        # Explicit-KEY sampling, not global-state sampling. Measured: in the
        # serving path, sampled generations were identical across requests
        # at any temperature — global-state seeding on the worker did not
        # reach the draws (an mlx state/thread/stream interaction; minimal
        # two-thread repros behave, the full stream_generate path does not).
        # Explicit keys are immune to threads, streams, and compile capture,
        # and give OpenAI seed semantics for free: same seed, same draw.
        if temperature > 0:
            sampler = _seeded_sampler(temperature, top_p, seed)
        else:
            sampler = make_sampler(temp=0.0)
        r = Reply(prompt_tokens=len(tokens), cached_tokens=cached)
        t0 = time.perf_counter()
        t_first = None
        gen_ids = []
        held = ""
        gen_finish = "stop"
        if logprobs_k is not None:
            import mlx.core as mx

            r.logprobs, r.token_entropies = [], []
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
            if cancel is not None and cancel.is_set():
                gen_finish = "cancelled"
                break
            gen_ids.append(out.token)
            if getattr(out, "finish_reason", None):
                gen_finish = out.finish_reason
            if logprobs_k is not None and out.logprobs is not None:
                # per-token signal: chosen logprob, full-distribution entropy,
                # top-k — the uncertainty machinery szilard/route consume.
                # Each .item() syncs; the cost exists only when requested.
                lp = out.logprobs
                chosen = float(lp[out.token].item())
                ent = float((-(mx.exp(lp) * lp)).sum().item())
                entry = {"token": self.m.tokenizer.decode([out.token]),
                         "logprob": round(chosen, 6)}
                if logprobs_k > 0:
                    idx = mx.argpartition(-lp, kth=logprobs_k - 1)[:logprobs_k]
                    pairs = sorted(
                        ((int(i), float(lp[int(i)].item())) for i in idx.tolist()),
                        key=lambda x: -x[1])
                    entry["top_logprobs"] = [
                        {"token": self.m.tokenizer.decode([i]),
                         "logprob": round(v, 6)} for i, v in pairs]
                r.logprobs.append(entry)
                r.token_entropies.append(round(ent, 6))
            if parse_tools:
                emit, held = safe_emit_split(held + out.text, False)
                if emit:
                    r.text += emit
                    yield ("delta", emit)
            else:
                r.text += out.text
                yield ("delta", out.text)
        t_end = time.perf_counter()
        # honest termination cause: a hit max_tokens must read "length", not
        # "stop" — szilard's smoke caught truncated thinking scored as clean
        # because this defaulted to stop (the truncation-masquerade lesson)
        r.finish_reason = gen_finish
        if parse_tools:
            parsed = parse_tool_calls(r.text + held)
            r.tool_calls = parsed.tool_calls
            if parsed.tool_calls:
                r.finish_reason = parsed.finish_reason
            if parsed.tool_calls:
                r.text = parsed.content
            elif held:  # held tail was a false alarm, flush it
                r.text += held
                yield ("delta", held)
        r.completion_tokens = len(gen_ids)
        r.prompt_eval_s = (t_first or t_end) - t0
        r.eval_s = t_end - (t_first or t_end)
        if chat_ctx is not None:
            self._align_cache(tokens + gen_ids, chat_ctx[0], chat_ctx[1], r)
        else:
            self._cache_ids = tokens + gen_ids
        yield ("final", r)


def tested_models() -> set:
    from importlib import resources

    with resources.files("boyle.data").joinpath("tested_models.json").open() as f:
        return {r["model"] for r in json.load(f)["models"]}


def classify_prefix_behavior(bmodel) -> tuple[str, str]:
    """Probe, tokenizer-only, how this model's cache will behave across turns.

    New families arrive with new templates and new cache types; the two
    failure classes measured so far (Qwen3.5: history render differs from
    generation render; hybrid caches cannot rewind) are both detectable
    up front without loading a weight. Returns (class, human detail) where
    class is "full" | "aligned" | "per-user-turn" | "unknown".
    """
    from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

    tok = bmodel.tokenizer
    try:
        msgs = [{"role": "user", "content": "Hi."}]
        reply = "Hello there."
        p1 = list(tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            enable_thinking=False))
        candidate = p1 + list(tok.encode(reply, add_special_tokens=False))
        hist = list(tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": reply},
                    {"role": "user", "content": ""}],
            tokenize=True, add_generation_prompt=False, enable_thinking=False))
        faithful = _common_prefix_len(candidate, hist) >= len(p1)
    except Exception as e:
        return "unknown", f"template probe failed ({e}) — expect re-prefills"
    if faithful:
        return "full", "template roundtrip faithful — warm turns reuse everything"
    if can_trim_prompt_cache(make_prompt_cache(bmodel.model)):
        return "aligned", ("template re-renders history differently — handled "
                           "by cache alignment; warm turns reuse everything")
    return "per-user-turn", (
        "template re-renders history differently AND this cache type cannot "
        "rewind — tool loops stay warm within a user turn; each new user "
        "turn re-prefills the conversation once"
    )


# --- HTTP layer -----------------------------------------------------------


def make_handler(core: GenerationCore):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            logger.debug("http: " + fmt, *args)

        # -- plumbing ------------------------------------------------------

        def _body(self) -> dict:
            return getattr(self, "_payload", {})

        def _drain_body(self) -> None:
            """Read the request body exactly once, before routing. A handler
            that skips its body leaves the bytes in the socket and corrupts
            the next request line on a keep-alive connection (the official
            ollama CLI hit this via POST /api/show)."""
            n = int(self.headers.get("Content-Length") or 0)
            self._payload = {}
            if n:
                raw = self.rfile.read(n)
                try:
                    self._payload = json.loads(raw)
                except json.JSONDecodeError:
                    pass

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

        def do_HEAD(self):
            # the official ollama CLI health-checks with HEAD / before every
            # command; an auto-501 here reads as "server broken"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "0")
            self.end_headers()

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
            elif self.path == "/health":
                # Deliberately NOT a "did an HTTP thread answer" check:
                # generation is serialized on one worker, so the question
                # that matters is whether the WORKER is servicing its queue.
                # Behind a long generation this reports busy, with 503 so
                # scripts and load balancers see it without parsing.
                alive = core.ping()
                self._json({"status": "ok" if alive else "busy",
                            "model": core.model_id,
                            "worker": "idle" if alive else "generating"},
                           200 if alive else 503)
            elif self.path == "/api/version":
                self._json({"version": "0.11.0"})
            elif self.path == "/api/tags":
                self._json({"models": [self._ollama_model_card()]})
            elif self.path == "/api/ps":
                card = self._ollama_model_card()
                # Go clients unmarshal this as time.Time — "never" breaks them
                card["expires_at"] = "2099-01-01T00:00:00Z"
                card["size_vram"] = card["size"]
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
            self._events = None   # set by handlers that start a generation
            self._drain_body()
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
                # The client vanished mid-stream. Stop the generation it
                # owns — without this the worker runs it to full length and
                # every queued request (including this client's retry) waits
                # behind a reply nobody will ever read.
                self._cancel_events()
                self.close_connection = True
            except Exception as e:
                logger.exception("request failed")
                self._cancel_events()
                self._error_out({"message": str(e), "type": "server_error"}, 500)

        def _cancel_events(self):
            ev = getattr(self, "_events", None)
            if ev is not None and getattr(ev, "cancel", None) is not None:
                ev.cancel.set()

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
            seed = body.get("seed", opts.get("seed"))
            return (
                body.get("max_tokens") or body.get("max_completion_tokens")
                or opts.get("num_predict"),
                float(body.get("temperature",
                               opts.get("temperature", core.default_temperature))),
                float(body.get("top_p", opts.get("top_p", 1.0))),
                int(seed) if seed is not None else None,
            )

        # -- OpenAI surface ------------------------------------------------

        def _oai_chat(self):
            body = self._body()
            messages = body.get("messages") or []
            tools = body.get("tools") or None
            max_tokens, temp, top_p, seed = self._gen_params(body)
            thinking = bool(body.get("enable_thinking") or body.get("think"))
            logprobs_k = (int(body.get("top_logprobs") or 0)
                          if body.get("logprobs") else None)
            tokens = core._tokenize_chat(messages, tools, thinking=thinking)
            parse = bool(tools) and core.tools_supported
            events = core.generate(tokens, max_tokens, temp, top_p, parse,
                                   chat_ctx=(messages, tools),
                                   logprobs_k=logprobs_k, seed=seed)
            self._events = events   # so a dead client cancels it
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
                choice = {"index": 0, "message": msg,
                          "finish_reason": final.finish_reason}
                if final.logprobs is not None:
                    # OpenAI logprobs shape + token_entropies extension (the
                    # full-distribution signal top-k cannot reconstruct)
                    choice["logprobs"] = {
                        "content": final.logprobs,
                        "token_entropies": final.token_entropies,
                    }
                self._json({
                    "id": rid, "object": "chat.completion",
                    "created": int(time.time()), "model": core.model_id,
                    "choices": [choice],
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
            max_tokens, temp, top_p, seed = self._gen_params(body)
            tokens = list(core.m.tokenizer.encode(prompt))
            events = core.generate(tokens, max_tokens, temp, top_p, False,
                                   seed=seed)
            self._events = events   # so a dead client cancels it
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
            if not messages:
                # Ollama convention: empty request = load/keep-alive management
                self._json({"model": body.get("model") or core.model_id,
                            "created_at": _now_iso(),
                            "message": {"role": "assistant", "content": ""},
                            "done": True, "done_reason": "load"})
                return
            tools = body.get("tools") or None
            max_tokens, temp, top_p, seed = self._gen_params(body)
            thinking = bool(body.get("think") or body.get("enable_thinking"))
            tokens = core._tokenize_chat(messages, tools, thinking=thinking)
            parse = bool(tools) and core.tools_supported
            events = core.generate(tokens, max_tokens, temp, top_p, parse,
                                   chat_ctx=(messages, tools), seed=seed)
            self._events = events   # so a dead client cancels it
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
            if not prompt:
                self._json({"model": body.get("model") or core.model_id,
                            "created_at": _now_iso(), "response": "",
                            "done": True, "done_reason": "load"})
                return
            max_tokens, temp, top_p, seed = self._gen_params(body)
            if body.get("raw"):
                tokens = list(core.m.tokenizer.encode(prompt))
            else:
                # Ollama semantics: /api/generate applies the chat template
                # unless raw:true. Older ollama CLIs drive `run` through this
                # endpoint; feeding an instruct model bare text produced
                # free-associating repetition loops (found by a year-old CLI
                # on the M1).
                msgs = []
                if body.get("system"):
                    msgs.append({"role": "system", "content": body["system"]})
                msgs.append({"role": "user", "content": prompt})
                tokens = core._tokenize_chat(msgs, None)
            events = core.generate(tokens, max_tokens, temp, top_p, False,
                                   seed=seed)
            self._events = events   # so a dead client cancels it
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
    default_temperature: float = 0.7,
) -> tuple[ThreadingHTTPServer, int]:
    """Bind (with the polite port policy) and return the server, unstarted."""
    core = GenerationCore(bmodel, model_id, tools_supported,
                          default_temperature=default_temperature)
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
                      f"http://{host}:{actual} configured explicitly, e.g.\n"
                      f"[boyle]   OLLAMA_HOST={host}:{actual} ollama run <model> ...")
            return httpd, actual
        except OSError as e:
            last_err = e
    raise last_err


def run_server(bmodel, model_id, tools_supported, host="127.0.0.1", port=None,
               default_temperature=0.7):
    httpd, actual = serve(bmodel, model_id, tools_supported, host, port,
                          default_temperature=default_temperature)
    plan = bmodel.plan
    print(f"[boyle] serving {model_id}")
    print(f"[boyle]   budget: fraction {plan.fraction:.2f}, "
          f"slots {fmt_size(plan.slots_bytes)}, context {plan.max_context}")
    print(f"[boyle]   OpenAI-compatible: http://{host}:{actual}/v1")
    print(f"[boyle]   Ollama-compatible: http://{host}:{actual}")
    print(f"[boyle]   tool calls: "
          f"{'parsed (hermes + XML dialects)' if tools_supported else 'passthrough only — untested family'}")
    cache_class, detail = classify_prefix_behavior(bmodel)
    print(f"[boyle]   prefix cache [{cache_class}]: {detail}")
    if model_id not in tested_models():
        print("[boyle]   note: this exact model is not on the tested list "
              "(see COMPATIBILITY.md) — the probes above are automatic, but "
              "behavior beyond them is unverified")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[boyle] shutting down")
        httpd.shutdown()
