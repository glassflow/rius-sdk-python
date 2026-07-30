"""Partial (pending) spans: a content-free snapshot exported at span start.

Spans normally leave the process only when they END, so an in-flight agent
run is invisible and a crashed one exports nothing. With ``partial_spans``
enabled, every sampled span additionally exports a snapshot at START; the
backend stores it as an unfinished row that the real span replaces at end
(same trace/span id and start timestamp — the identity the storage layer
keys replacement on), and a snapshot that is never replaced is the durable
record of what a crashed agent was doing.

Wire contract (GLA2-195):

- Same trace_id, span_id, parent, name, and start timestamp as the final
  span; ``end_time == start_time`` (OTLP cannot represent an unfinished
  span, so the snapshot is an ended zero-duration span with a marker).
- The ``glassflow.span.pending`` marker attribute (see ``semconv.py`` for
  why a vendor-namespaced key is unavoidable here).
- Identity/taxonomy attributes only (``PENDING_IDENTITY_ATTRIBUTES`` /
  ``_PREFIXES``); NEVER content, whatever instrumentation set it.

Debounce (GLA2-244): with ``partial_spans_delay > 0`` the snapshot is held
for N seconds and only emitted if the span is STILL OPEN then — a span that
finishes first costs zero network. Most agent spans live milliseconds, so a
small delay cuts pending volume drastically while keeping the live view
useful (anything worth watching live is open longer than the delay). The
snapshot is still built at ``on_start`` and held, never rebuilt at emit
time: content set during the delay (``set_input`` etc.) can never leak onto
a pending. A delayed pending is byte-identical to an immediate one — zero
wire/backend/UI impact.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import os
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import Status, StatusCode

from .semconv import (
    GLASSFLOW_SPAN_PENDING,
    PENDING_IDENTITY_ATTRIBUTES,
    PENDING_IDENTITY_PREFIXES,
)

logger = logging.getLogger(__name__)

_SpanKey = tuple[int, int]  # (trace_id, span_id)

# ``os.register_at_fork`` callbacks can never be unregistered, so the hook is
# installed once at module level over a weak set of live schedulers — the same
# pattern as the heartbeat sender. A forked child re-arms its scheduler thread
# with an EMPTY registry: the parent's open spans are not the child's.
_active_schedulers: weakref.WeakSet[PendingScheduler] = weakref.WeakSet()
_fork_hook_installed = False
_fork_lock = threading.Lock()


def _reset_schedulers_in_child() -> None:  # pragma: no cover - exercised via fork
    for scheduler in list(_active_schedulers):
        scheduler._at_fork_reinit()


def _install_fork_hook() -> None:
    global _fork_hook_installed
    with _fork_lock:
        if _fork_hook_installed or not hasattr(os, "register_at_fork"):
            return
        os.register_at_fork(after_in_child=_reset_schedulers_in_child)
        _fork_hook_installed = True


class PendingScheduler:
    """Delays snapshot emission; a span ending first cancels its snapshot.

    ONE daemon thread regardless of span volume: deadlines live in a heap,
    snapshots in a key->snapshot registry. ``cancel`` just drops the registry
    entry (heap entries for cancelled keys are discarded lazily), so both
    ``schedule`` and ``cancel`` are O(log n) / O(1) — safe on the span hot
    path. ``clock`` and ``start_thread`` are injectable for tests.
    """

    def __init__(
        self,
        *,
        emit: Callable[[ReadableSpan], None],
        delay: float,
        clock: Callable[[], float] = time.monotonic,
        start_thread: bool = True,
    ) -> None:
        self._emit_fn = emit
        self._delay = delay
        self._clock = clock
        self._cond = threading.Condition()
        self._heap: list[tuple[float, int, _SpanKey]] = []
        self._snapshots: dict[_SpanKey, ReadableSpan] = {}
        self._counter = itertools.count()  # heap tiebreaker
        self._stopped = False
        self._thread: threading.Thread | None = None
        if start_thread:
            _active_schedulers.add(self)
            _install_fork_hook()
            self._start_thread()

    def _start_thread(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="glassflow-pending-scheduler", daemon=True
        )
        self._thread.start()

    def schedule(self, key: _SpanKey, snapshot: ReadableSpan) -> None:
        with self._cond:
            if self._stopped:
                return
            self._snapshots[key] = snapshot
            heapq.heappush(self._heap, (self._clock() + self._delay, next(self._counter), key))
            self._cond.notify_all()

    def cancel(self, key: _SpanKey) -> None:
        """Span ended before its deadline: the pending never hits the wire."""
        with self._cond:
            self._snapshots.pop(key, None)

    def pop_due(self) -> None:
        """Emit every snapshot whose deadline has passed (thread and tests)."""
        due: list[ReadableSpan] = []
        with self._cond:
            now = self._clock()
            while self._heap and self._heap[0][0] <= now:
                _, _, key = heapq.heappop(self._heap)
                snapshot = self._snapshots.pop(key, None)
                if snapshot is not None:  # None = cancelled, discard lazily
                    due.append(snapshot)
        for snapshot in due:
            try:
                self._emit_fn(snapshot)
            except Exception:  # noqa: BLE001 - never propagate into the SDK
                logger.debug("pending snapshot emission failed", exc_info=True)

    def shutdown(self) -> None:
        """Drop everything not yet due: the final spans are being flushed at
        this moment, so any pending emitted now would be instantly superseded."""
        with self._cond:
            self._stopped = True
            self._snapshots.clear()
            self._heap.clear()
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _at_fork_reinit(self) -> None:  # pragma: no cover - exercised via fork
        # Fresh lock (the parent's may be held mid-fork), empty registry, new
        # thread: parent spans do not exist in the child.
        self._cond = threading.Condition()
        self._heap = []
        self._snapshots = {}
        if not self._stopped:
            self._start_thread()

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._stopped:
                    return
                # discard cancelled heads so the timeout tracks a LIVE deadline
                while self._heap and self._heap[0][2] not in self._snapshots:
                    heapq.heappop(self._heap)
                timeout = None
                if self._heap:
                    timeout = max(0.0, self._heap[0][0] - self._clock())
                self._cond.wait(timeout)
                if self._stopped:
                    return
            self.pop_due()


def _identity_attributes(attributes: Any) -> dict[str, Any]:
    """Filter a span's start-time attributes down to the pending allowlist."""
    if not attributes:
        return {}
    return {
        key: value
        for key, value in attributes.items()
        if key in PENDING_IDENTITY_ATTRIBUTES or key.startswith(PENDING_IDENTITY_PREFIXES)
    }


def _span_key(context: Any) -> _SpanKey:
    return (context.trace_id, context.span_id)


class PendingSpanProcessor(SpanProcessor):
    """Exports a pending snapshot of every sampled span at ``on_start``.

    Delegates the snapshot to the provider's existing batch processor
    (``delegate.on_end``), so pendings share the exporter, batching, retry,
    and masking pipeline with final spans — nothing bespoke on the wire path.
    ``on_start`` stays an in-memory enqueue: the never-block guarantee holds.

    With ``delay > 0`` (GLA2-244) emission is debounced through a
    :class:`PendingScheduler`; ``delay == 0`` keeps the emit-immediately
    behavior with no scheduler thread at all.
    """

    def __init__(self, delegate: SpanProcessor, *, delay: float = 0.0) -> None:
        self._delegate = delegate
        self._scheduler: PendingScheduler | None = None
        if delay > 0:
            self._scheduler = PendingScheduler(emit=delegate.on_end, delay=delay)

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        if not span.is_recording():
            return
        snapshot = self._snapshot(span)
        if self._scheduler is not None:
            self._scheduler.schedule(_span_key(span.get_span_context()), snapshot)
        else:
            self._emit(snapshot)

    @staticmethod
    def _snapshot(span: Span) -> ReadableSpan:
        attributes = _identity_attributes(span.attributes)
        attributes[GLASSFLOW_SPAN_PENDING] = True
        return ReadableSpan(
            name=span.name,
            context=span.get_span_context(),
            parent=span.parent,
            resource=span.resource,
            attributes=attributes,
            events=(),
            links=(),
            kind=span.kind,
            instrumentation_scope=span.instrumentation_scope,
            status=Status(StatusCode.UNSET),
            start_time=span.start_time,
            end_time=span.start_time,  # zero duration: unfinished, marked
        )

    def _emit(self, snapshot: ReadableSpan) -> None:
        self._delegate.on_end(snapshot)

    def on_end(self, span: ReadableSpan) -> None:
        # The debounce cancellation hook: a span that ends within the delay
        # never sends its pending at all.
        if self._scheduler is not None and span.context is not None:
            self._scheduler.cancel(_span_key(span.context))

    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # Deliberately NOT a drop (deviation from the ticket's prose, kept to
        # its ACs): flush() happens mid-operation — killing scheduled pendings
        # here would silently disable liveness for spans that stay open. The
        # batch delegate flushes its own queue; not-yet-due pendings simply
        # emit later if their spans are still open.
        return True
