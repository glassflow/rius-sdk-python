"""Debounced partial spans: delay emission, cancel on fast finish.

Timing is injected everywhere (fake monotonic clocks, bounded Event waits) —
no test sleeps.
"""

from __future__ import annotations

import threading

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.pending import PendingScheduler
from rius.semconv import GLASSFLOW_SPAN_PENDING

# --- config resolution -------------------------------------------------------


def test_delay_defaults_to_zero() -> None:
    from rius.config import resolve_config

    assert resolve_config().partial_spans_delay == 0.0


def test_delay_env_var_and_clamp(monkeypatch) -> None:
    from rius.config import resolve_config

    monkeypatch.setenv("GLASSFLOW_PARTIAL_SPANS_DELAY", "2.5")
    assert resolve_config().partial_spans_delay == 2.5
    # explicit argument wins; out-of-range clamps instead of crashing
    assert resolve_config(partial_spans_delay=9999).partial_spans_delay == 60.0
    assert resolve_config(partial_spans_delay=-1).partial_spans_delay == 0.0


# --- scheduler core (pure logic, no thread) -----------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _scheduler(emitted: list, clock: _Clock, delay: float = 5.0) -> PendingScheduler:
    return PendingScheduler(emit=emitted.append, delay=delay, clock=clock, start_thread=False)


def test_snapshot_not_due_before_delay() -> None:
    emitted: list = []
    clock = _Clock()
    s = _scheduler(emitted, clock)
    s.schedule(("t", 1), "snapshot-1")
    s.pop_due()
    assert emitted == []


def test_snapshot_emitted_once_after_delay() -> None:
    emitted: list = []
    clock = _Clock()
    s = _scheduler(emitted, clock)
    s.schedule(("t", 1), "snapshot-1")
    clock.now += 5.0
    s.pop_due()
    s.pop_due()  # idempotent: never emitted twice
    assert emitted == ["snapshot-1"]


def test_cancel_before_due_means_no_emission() -> None:
    emitted: list = []
    clock = _Clock()
    s = _scheduler(emitted, clock)
    s.schedule(("t", 1), "snapshot-1")
    s.cancel(("t", 1))
    clock.now += 60.0
    s.pop_due()
    assert emitted == []


def test_shutdown_drops_scheduled_pendings() -> None:
    emitted: list = []
    clock = _Clock()
    s = _scheduler(emitted, clock)
    s.schedule(("t", 1), "snapshot-1")
    s.shutdown()
    clock.now += 60.0
    s.pop_due()
    assert emitted == []
    # post-shutdown schedules are ignored, not errors
    s.schedule(("t", 2), "snapshot-2")
    clock.now += 60.0
    s.pop_due()
    assert emitted == []


def test_thread_emits_when_due() -> None:
    """One real-thread smoke test: emission signals an Event (bounded wait)."""
    done = threading.Event()
    s = PendingScheduler(emit=lambda _snap: done.set(), delay=0.01, start_thread=True)
    s.schedule(("t", 1), "snapshot-1")
    assert done.wait(timeout=5.0), "scheduler thread never emitted the due snapshot"
    s.shutdown()


# --- end-to-end through init() ------------------------------------------------


def _memory_client(**kwargs: object):
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        **kwargs,  # type: ignore[arg-type]
    )
    return client, exporter


def test_fast_span_produces_no_pending_on_the_wire() -> None:
    """The whole point: a span finishing within the delay costs zero network."""
    client, exporter = _memory_client(partial_spans=True, partial_spans_delay=30.0)
    with client.get_tracer().start_as_current_span("quick"):
        pass
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert GLASSFLOW_SPAN_PENDING not in spans[0].attributes


def test_delay_zero_keeps_immediate_emission() -> None:
    client, exporter = _memory_client(partial_spans=True, partial_spans_delay=0.0)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    markers = [
        bool(s.attributes.get(GLASSFLOW_SPAN_PENDING)) for s in exporter.get_finished_spans()
    ]
    assert sorted(markers) == [False, True]
