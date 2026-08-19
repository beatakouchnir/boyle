# SPDX-License-Identifier: Apache-2.0
"""Worker cancellation, liveness, and drain — no model, no Metal.

These cover the failure that cost a machine reboot on 2026-08-18: a client
disconnected mid-generation, nothing cancelled the abandoned job, and every
later request queued behind it on the single generation worker while
/v1/models kept answering 200.
"""

import threading
import time

from boyle.server import GenerationCore


class _FakePlan:
    max_context = 32768


class _FakeModel:
    plan = _FakePlan()


class _FakeCore(GenerationCore):
    """GenerationCore with the model/warmup replaced — the queue, worker
    loop, cancellation and ping logic under test are model-independent."""

    def __init__(self, tokens_per_job=200, tick=0.005):
        import queue

        self.m = _FakeModel()          # generate() checks m.plan.max_context
        self.model_id = "fake/model"
        self.tools_supported = False
        self.default_temperature = 0.0
        self._cache = None
        self._cache_ids = []
        self.tokens_per_job = tokens_per_job
        self.tick = tick
        self.emitted = 0
        self._jobs = queue.Queue()
        self._active_cancel = None
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

    def _generate_on_worker(self, tokens, want, temperature, top_p,
                            parse_tools, chat_ctx, logprobs_k=None,
                            seed=None, cancel=None):
        for _ in range(self.tokens_per_job):
            if cancel is not None and cancel.is_set():
                break
            self.emitted += 1
            time.sleep(self.tick)
            yield ("delta", "x")
        yield ("final", object())


def _drain_one(core):
    """Start a job, read one event, then abandon it like a dead client."""
    events = core.generate([1, 2, 3], 999, 0.0, 1.0, False)
    next(iter(events))
    return events


def test_abandoned_generation_is_cancelled_and_frees_the_worker():
    core = _FakeCore(tokens_per_job=2000, tick=0.002)   # ~4s if uncancelled
    events = _drain_one(core)
    events.cancel.set()                                  # what do_POST does

    t0 = time.perf_counter()
    assert core.ping(timeout=3.0), "worker never became available again"
    waited = time.perf_counter() - t0
    assert waited < 2.0, f"queue stayed blocked for {waited:.1f}s"
    assert core.emitted < 2000, "generation ran to completion despite cancel"


def test_ping_reports_busy_while_a_generation_holds_the_worker():
    core = _FakeCore(tokens_per_job=400, tick=0.01)      # ~4s of work
    _drain_one(core)                                     # no cancel: still running
    assert core.ping(timeout=0.3) is False, "ping must report busy, not lie"


def test_ping_is_true_on_an_idle_worker():
    assert _FakeCore().ping(timeout=2.0) is True


def test_cancel_active_targets_the_running_job():
    core = _FakeCore(tokens_per_job=2000, tick=0.002)
    _drain_one(core)
    core.cancel_active()                                 # what SIGTERM does
    assert core.ping(timeout=3.0), "drain did not release the worker"


def test_events_stream_exposes_cancel_attribute():
    # a bare generator cannot carry attributes — regression guard
    core = _FakeCore(tokens_per_job=1, tick=0.0)
    events = core.generate([1], 8, 0.0, 1.0, False)
    assert isinstance(events.cancel, threading.Event)
    for _ in events:
        pass
