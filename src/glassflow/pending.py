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

v1 emits immediately on start (Logfire-style). The emission seam
(:meth:`PendingSpanProcessor._emit`) exists so a debounce ("only emit if
still open after N seconds" — the volume escape valve) can be added later
without changing the wire contract.
"""

from __future__ import annotations

from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import Status, StatusCode

from .semconv import (
    GLASSFLOW_SPAN_PENDING,
    PENDING_IDENTITY_ATTRIBUTES,
    PENDING_IDENTITY_PREFIXES,
)


def _identity_attributes(attributes: Any) -> dict[str, Any]:
    """Filter a span's start-time attributes down to the pending allowlist."""
    if not attributes:
        return {}
    return {
        key: value
        for key, value in attributes.items()
        if key in PENDING_IDENTITY_ATTRIBUTES or key.startswith(PENDING_IDENTITY_PREFIXES)
    }


class PendingSpanProcessor(SpanProcessor):
    """Exports a pending snapshot of every sampled span at ``on_start``.

    Delegates the snapshot to the provider's existing batch processor
    (``delegate.on_end``), so pendings share the exporter, batching, retry,
    and masking pipeline with final spans — nothing bespoke on the wire path.
    ``on_start`` stays an in-memory enqueue: the never-block guarantee holds.
    """

    def __init__(self, delegate: SpanProcessor) -> None:
        self._delegate = delegate

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        if not span.is_recording():
            return
        self._emit(self._snapshot(span))

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
        # The debounce seam: a future timer wraps THIS call (delay + cancel on
        # early end), leaving the snapshot construction and wire shape alone.
        self._delegate.on_end(snapshot)

    def on_end(self, span: ReadableSpan) -> None:  # pragma: no cover - no-op
        pass

    def shutdown(self) -> None:  # pragma: no cover - delegate owns the exporter
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
