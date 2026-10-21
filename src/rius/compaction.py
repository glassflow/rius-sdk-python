"""Context compaction: marking where an agent's history was shrunk.

A provider that compacts the context says so in its response, and the SDK
reads that back at export (see ``_provider_compaction``). An application that
shrinks its own history, by summarizing it or dropping old tool results, is
the only one who knows, so it says so with :func:`mark_compaction`.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Literal

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

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
    after this call), and never spent twice.
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

    def take(self) -> bool:
        with self._lock:
            armed, self._armed = self._armed, False
            return armed


_pending: ContextVar[_Pending | None] = ContextVar("rius_compaction_pending", default=None)


def _is_model_call(span: Span) -> bool:
    attributes = span.attributes or {}
    kind = attributes.get(OPENINFERENCE_SPAN_KIND)
    if kind is None:
        operation = attributes.get(GEN_AI_OPERATION_NAME)
        kind = kind_for_operation(operation) if isinstance(operation, str) else None
    return kind == SpanKind.LLM.value


class CompactedContextProcessor(SpanProcessor):
    """Flag the first model call after :func:`mark_compaction`.

    Add-only and at span start, like the other stampers, so the flag rides the
    pending snapshot too. A value the span already carries is the producer's
    own and is kept, but it still spends the mark: that call is the one the
    mark was for.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        pending = _pending.get()
        if pending is None or not _is_model_call(span):
            return
        if not pending.take():
            return
        if GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {}):
            span.set_attribute(GEN_AI_CONVERSATION_COMPACTED, True)

    def on_end(self, span: ReadableSpan) -> None:  # pragma: no cover - nothing to do
        pass

    def shutdown(self) -> None:  # pragma: no cover - nothing to hold
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True
