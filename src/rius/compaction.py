"""Context compaction: marking where an agent's history was shrunk.

A provider that compacts the context says so in its response, and the SDK
reads that back at export (see ``_provider_compaction``). An application that
shrinks its own history, by summarizing it or dropping old tool results, is
the only one who knows, so it says so with :func:`mark_compaction`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, Literal

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from ._provider_response import is_compact_reply, response_json
from ._serde import text_value
from ._tracer import sdk_tracer
from .semconv import (
    GEN_AI_CONVERSATION_COMPACTED,
    GEN_AI_OPERATION_NAME,
    OPENINFERENCE_SPAN_KIND,
    RIUS_COMPACTION_EXECUTOR,
    RIUS_COMPACTION_EXECUTOR_CLIENT,
    RIUS_COMPACTION_KIND,
    RIUS_COMPACTION_KIND_CLEAR_THINKING,
    RIUS_COMPACTION_KIND_CLEAR_TOOL_RESULTS,
    RIUS_COMPACTION_KIND_SUMMARIZE,
    RIUS_COMPACTION_KIND_TRUNCATE,
    RIUS_COMPACTION_MESSAGES_COMPACTED,
    RIUS_COMPACTION_MESSAGES_KEPT,
    RIUS_COMPACTION_SUMMARY_BYTES,
    RIUS_COMPACTION_SUMMARY_TEXT,
    RIUS_COMPACTION_TOKENS_AFTER,
    RIUS_COMPACTION_TOKENS_BEFORE,
    RIUS_COMPACTION_TOKENS_DROPPED,
    RIUS_COMPACTION_TRIGGER,
    RIUS_COMPACTION_TRIGGER_AUTO,
    RIUS_COMPACTION_TRIGGER_MANUAL,
    RIUS_SPAN_PENDING,
    SpanKind,
    kind_for_operation,
    otel_span_kind,
)

CompactionKind = Literal["summarize", "clear_tool_results", "clear_thinking", "truncate"]
CompactionTrigger = Literal["auto", "manual"]

_KINDS = (
    RIUS_COMPACTION_KIND_SUMMARIZE,
    RIUS_COMPACTION_KIND_CLEAR_TOOL_RESULTS,
    RIUS_COMPACTION_KIND_CLEAR_THINKING,
    RIUS_COMPACTION_KIND_TRUNCATE,
)
_TRIGGERS = (RIUS_COMPACTION_TRIGGER_AUTO, RIUS_COMPACTION_TRIGGER_MANUAL)

#: ``gen_ai.operation.name`` of the span: the registry's taxonomy table knows
#: it (it maps to CHAIN) but has no constant for it, and semconv.py defines none.
OPERATION_COMPACTION = "compaction"


def dropped_tokens(before: int | None, after: int | None) -> int | None:
    """What a compaction removed: ``before`` minus ``after``, when both are known.

    A context that did not shrink has dropped nothing to report, so a larger
    ``after`` yields None rather than a negative count.
    """
    if before is None or after is None or after > before:
        return None
    return before - after


def mark_compaction(
    kind: CompactionKind = "summarize",
    *,
    trigger: CompactionTrigger | None = None,
    tokens_before: int | None = None,
    tokens_after: int | None = None,
    messages_compacted: int | None = None,
    messages_kept: int | None = None,
    summary: str | None = None,
) -> None:
    """Record that the application just shrank the context it sends to the model.

    Call it once the history has been replaced. It records a finished
    ``compaction`` span, a child of the current span (a root when there is
    none), and flags the next model call made in the same context with
    ``gen_ai.conversation.compacted``, so that call can be told apart from one
    that ran on the full history.

    ``kind`` is what the compaction did: ``"summarize"`` (the default) replaced
    old turns with a summary, ``"clear_tool_results"`` and ``"clear_thinking"``
    dropped those blocks, ``"truncate"`` cut the oldest turns. ``trigger`` is
    ``"auto"`` when the application decided and ``"manual"`` when a person
    asked. An unknown ``kind`` or ``trigger`` raises ``ValueError``.

    The counts are whatever the application knows; one it does not know is
    left out. What was dropped is derived, ``tokens_before - tokens_after``,
    when both are given and the context did not grow.

    ``summary`` is the text that replaced the history. Its size in bytes is
    always sent; the text itself is content, so it follows ``capture_content``
    and ``mask`` like a prompt does. The flag is spent by the first model call
    that sees it, including in a copied context (a thread or task started
    after this call), and never spent twice. It does not expire: it flags the
    next model call in this context whenever that comes. A mark made inside a
    child task or thread (``create_task``, ``to_thread``, an executor) is not
    seen by the code that started it; it propagates into copies of the context,
    never out of them.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown compaction kind {kind!r}; expected one of {_KINDS}")
    if trigger is not None and trigger not in _TRIGGERS:
        raise ValueError(f"unknown compaction trigger {trigger!r}; expected one of {_TRIGGERS}")
    attributes: dict[str, str | int] = {
        OPENINFERENCE_SPAN_KIND: SpanKind.CHAIN.value,
        GEN_AI_OPERATION_NAME: OPERATION_COMPACTION,
        RIUS_COMPACTION_EXECUTOR: RIUS_COMPACTION_EXECUTOR_CLIENT,
        RIUS_COMPACTION_KIND: kind,
    }
    optional: dict[str, str | int | None] = {
        RIUS_COMPACTION_TRIGGER: trigger,
        RIUS_COMPACTION_TOKENS_BEFORE: tokens_before,
        RIUS_COMPACTION_TOKENS_AFTER: tokens_after,
        RIUS_COMPACTION_TOKENS_DROPPED: dropped_tokens(tokens_before, tokens_after),
        RIUS_COMPACTION_MESSAGES_COMPACTED: messages_compacted,
        RIUS_COMPACTION_MESSAGES_KEPT: messages_kept,
        RIUS_COMPACTION_SUMMARY_BYTES: (
            None if summary is None else len(summary.encode("utf-8", "replace"))
        ),
    }
    attributes.update({key: value for key, value in optional.items() if value is not None})
    span = sdk_tracer().start_span(
        OPERATION_COMPACTION, kind=otel_span_kind(SpanKind.CHAIN), attributes=attributes
    )
    # Armed before the end, so a model call that ending the span itself sets
    # off is the next one.
    _pending.set(_Pending())
    # After the start: the pending snapshot a span sends when it begins is
    # built from its creation attributes, and the text is content.
    if summary is not None:
        span.set_attribute(RIUS_COMPACTION_SUMMARY_TEXT, text_value(summary))
    span.end()


class _Pending:
    """A compaction mark waiting for the model call it applies to.

    Shared by reference between a context and every copy of it, so the first
    call in any of them takes it and the others find it spent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._armed = True

    @property
    def armed(self) -> bool:
        return self._armed

    def take(self) -> bool:
        with self._lock:
            armed, self._armed = self._armed, False
            return armed


_pending: ContextVar[_Pending | None] = ContextVar("rius_compaction_pending", default=None)

#: Upper bound on the spans remembered at once; the oldest goes first. A span
#: normally leaves the books when it ends or is exported, so the bound only
#: protects a process whose spans are never exported.
_MAX_TRACKED = 10_000


def _model_call(attributes: Mapping[str, Any]) -> bool | None:
    """Whether the attributes make a model call: None while nothing says what the span is."""
    kind = attributes.get(OPENINFERENCE_SPAN_KIND)
    if kind is None:
        operation = attributes.get(GEN_AI_OPERATION_NAME)
        kind = kind_for_operation(operation) if isinstance(operation, str) else None
    return None if kind is None else kind == SpanKind.LLM.value


class CompactedContextProcessor(SpanProcessor):
    """Flag the first model call after :func:`mark_compaction` or the compact endpoint.

    Add-only. A call known to be a model call when it starts takes the mark at
    start, so the flag rides the pending snapshot too. LangChain, LiteLLM and
    LlamaIndex learn that a span is a model call only before it ends: such a
    span remembers the mark at start without taking it, and takes it at end if
    it became a model call by then. The exporter then adds the flag
    (:meth:`take`), as it does for a wrapper's usage. A value the span already
    carries is the producer's own and is kept, but it still spends the mark:
    that call is the one the mark was for.

    The reply of ``POST /responses/compact`` is the compaction, not a call on a
    compacted context; ending it marks the call that follows.
    """

    def __init__(self, *, max_tracked: int = _MAX_TRACKED) -> None:
        self._max = max_tracked
        self._lock = threading.Lock()
        self._open: OrderedDict[int, _Pending] = OrderedDict()
        self._marked: OrderedDict[int, None] = OrderedDict()

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        pending = _pending.get()
        if pending is None or not pending.armed:
            return
        model_call = _model_call(span.attributes or {})
        if model_call is None:
            with self._lock:
                self._open[span.context.span_id] = pending
                while len(self._open) > self._max:
                    self._open.popitem(last=False)
        elif model_call and pending.take():
            self._flag(span)

    def _flag(self, span: Span) -> None:
        if GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {}):
            span.set_attribute(GEN_AI_CONVERSATION_COMPACTED, True)

    def on_end(self, span: ReadableSpan) -> None:
        attributes = span.attributes or {}
        with self._lock:
            pending = self._open.pop(span.context.span_id, None)
            if pending is not None and _model_call(attributes) and pending.take():
                self._marked[span.context.span_id] = None
                while len(self._marked) > self._max:
                    self._marked.popitem(last=False)
        raw = response_json(attributes) if _model_call(attributes) else None
        if raw is not None and is_compact_reply(raw):
            _pending.set(_Pending())

    def take(self, span: ReadableSpan) -> bool:
        """Whether ``span`` became a model call that took the mark, once.

        A pending snapshot never consumes it, since the real span is exported
        after it.
        """
        if span.attributes and span.attributes.get(RIUS_SPAN_PENDING):
            return False
        with self._lock:
            return self._marked.pop(span.context.span_id, False) is None

    def shutdown(self) -> None:  # pragma: no cover - the books are bounded
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True
