"""Count a model call's usage once when a wrapper span repeats it.

LiteLLM calls the OpenAI client for OpenAI models. With both instrumentors
enabled (the rius SDK enables every one it finds), each such call is recorded
twice: on LiteLLM's ``completion`` span and on the openai instrumentor's
``ChatCompletion`` span, its child, both carrying the call's usage. Every
backend sum over ``gen_ai.usage.*`` then counts the call twice.

The child is the call itself, so it keeps the usage. The wrapper's copy moves
to ``rius.aggregated_usage.*``, the namespace the TypeScript SDK already uses
for an aggregating producer's invoke_agent span: still on the span for display,
no longer summed.

Which wrapper repeats its child's usage is decided by end order, never by
guessing: a child ends before its parent, so by the time a LiteLLM span ends,
:class:`WrappedCallTracker` already knows whether an LLM child with usage ended
under it. The exporter then takes that mark when it exports the wrapper. A
LiteLLM span with no such child (Anthropic goes through LiteLLM's own HTTP
handler, so it has none) keeps its usage: it is the only record of the call.

Streamed LiteLLM calls depend on :mod:`rius.litellm_stream_context`:
openinference-instrumentation-litellm starts the stream's span without making
it current, so on its own the OpenAI call is not its child (it is a root span
in a trace of its own) and no mark is set. That module makes the span current
for the instrumentor versions it was checked against; on any other version a
streamed call is still counted twice.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from .semconv import GEN_AI_USAGE_PREFIX, RIUS_AGGREGATED_USAGE_PREFIX, RIUS_SPAN_PENDING

#: The tracer scope openinference-instrumentation-litellm starts its spans under.
LITELLM_SCOPE = "openinference.instrumentation.litellm"

#: OpenInference's usage family: the child's usage before normalization has
#: renamed it. A child carrying either spelling repeats the call's usage.
_LLM_TOKEN_COUNT_PREFIX = "llm.token_count."

#: Upper bound on the spans remembered at once. A wrapper normally leaves the
#: books when it is exported, milliseconds after it ends; the bound only
#: protects a process whose wrappers are never exported (sampled out, or a
#: provider shut down mid-flight). The oldest entry goes first.
_MAX_TRACKED = 10_000


def _has_usage(attributes: Mapping[str, Any] | None) -> bool:
    if not attributes:
        return False
    return any(
        key.startswith(GEN_AI_USAGE_PREFIX) or key.startswith(_LLM_TOKEN_COUNT_PREFIX)
        for key in attributes
    )


class WrappedCallTracker(SpanProcessor):
    """Remember which LiteLLM spans have an LLM child that recorded usage.

    ``on_start`` notes every open LiteLLM span; a child that ends with usage
    under one marks it; :meth:`take` hands the mark to the exporter once. Reads
    only a span's scope, ids and attributes, so a replay can drive it with
    ReadableSpans.
    """

    def __init__(self, *, max_tracked: int = _MAX_TRACKED) -> None:
        self._max = max_tracked
        self._lock = threading.Lock()
        self._open: OrderedDict[int, None] = OrderedDict()
        self._marked: OrderedDict[int, None] = OrderedDict()

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        scope = span.instrumentation_scope
        if scope is None or scope.name != LITELLM_SCOPE:
            return
        with self._lock:
            self._remember(self._open, span.context.span_id)

    def on_end(self, span: ReadableSpan) -> None:
        span_id = span.context.span_id
        parent = span.parent
        with self._lock:
            self._open.pop(span_id, None)
            if parent is not None and parent.span_id in self._open and _has_usage(span.attributes):
                self._remember(self._marked, parent.span_id)

    def take(self, span: ReadableSpan) -> bool:
        """Whether ``span`` is a LiteLLM span whose child recorded the usage.

        True once per span: the mark is consumed. A pending snapshot never
        consumes it, since the real span is exported after it.
        """
        if span.attributes and span.attributes.get(RIUS_SPAN_PENDING):
            return False
        with self._lock:
            return self._marked.pop(span.context.span_id, False) is None

    def _remember(self, books: OrderedDict[int, None], span_id: int) -> None:
        books[span_id] = None
        books.move_to_end(span_id)
        while len(books) > self._max:
            books.popitem(last=False)

    def shutdown(self) -> None:
        with self._lock:
            self._open.clear()
            self._marked.clear()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def move_aggregated_usage(attributes: Mapping[str, Any]) -> dict[str, Any] | None:
    """``attributes`` with every ``gen_ai.usage.*`` key under ``rius.aggregated_usage.*``.

    Returns None when there is no usage to move. Runs on normalized attributes,
    so a legacy spelling has already become its canonical key and moves under
    the name the backend reads.
    """
    if not any(key.startswith(GEN_AI_USAGE_PREFIX) for key in attributes):
        return None
    moved: dict[str, Any] = {}
    for key, value in attributes.items():
        if key.startswith(GEN_AI_USAGE_PREFIX):
            moved[RIUS_AGGREGATED_USAGE_PREFIX + key[len(GEN_AI_USAGE_PREFIX) :]] = value
        else:
            moved[key] = value
    return moved
