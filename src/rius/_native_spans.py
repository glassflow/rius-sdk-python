"""The anthropic SDK's own chat span, demoted when OpenInference covers the call.

From ``anthropic`` 1.14.0 the SDK starts a CLIENT span of its own for every
API call, under the scope ``com.anthropic.sdk.python``, on the OpenTelemetry
global provider. Its chat span carries ``gen_ai.operation.name=chat``,
``gen_ai.request.model`` and ``gen_ai.usage.*``, so next to the span of the
OpenInference Anthropic instrumentor it was normalized as a second LLM span:
the backend priced the call twice, a failure carried a second ``error.type``
in the API's own spelling (``not_found_error``), and the call after
``mark_compaction()`` had an unflagged twin.

When rius registered that instrumentor, :class:`NativeChatSpanDemoter` strips
the chat span's GenAI keys, so it is exported as a plain client span: its
HTTP, server, ``anthropic.*`` and ``error.type`` attributes stay, as do its
children. It is never dropped. The match is on the scope, not on parentage:
the span of the ``messages.stream()`` helper is not under OpenInference's
span but the root of a trace of its own. The SDK's tool and agent spans are
left alone, and without the instrumentor the SDK's span stays the LLM span.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from ._attributes import replacement_attributes
from .semconv import GEN_AI_OPERATION_NAME, GEN_AI_PROVIDER_NAME, OPENINFERENCE_SPAN_KIND

logger = logging.getLogger(__name__)

#: The scope prefix of the spans the anthropic SDKs start themselves
#: (``com.anthropic.sdk.python`` in anthropic 1.14.0 through 2.1.0).
ANTHROPIC_SDK_SCOPE_PREFIX = "com.anthropic.sdk."

#: Not in the rius registry: written by the anthropic SDK, kept as it names
#: the conversation and not the model call.
_GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
_GEN_AI_PREFIX = "gen_ai."
_KEPT_GEN_AI = frozenset({GEN_AI_PROVIDER_NAME, _GEN_AI_CONVERSATION_ID})
_CHAT = "chat"


def _removed(key: str) -> bool:
    if key == OPENINFERENCE_SPAN_KIND:
        return True
    return key.startswith(_GEN_AI_PREFIX) and key not in _KEPT_GEN_AI


def demoted_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """``attributes`` without what makes a span a model call.

    Every ``gen_ai.*`` key goes, the message content included, except the
    provider and the conversation id; so does ``openinference.span.kind``.
    """
    return {key: value for key, value in attributes.items() if not _removed(key)}


def _is_sdk_chat_span(span: Span) -> bool:
    scope = span.instrumentation_scope
    if scope is None or not scope.name.startswith(ANTHROPIC_SDK_SCOPE_PREFIX):
        return False
    return (span.attributes or {}).get(GEN_AI_OPERATION_NAME) == _CHAT


class NativeChatSpanDemoter(SpanProcessor):
    """Strip the GenAI keys off the anthropic SDK's chat span while ``active()``.

    Registered ahead of every processor that reads a span at start
    (normalization, the compaction flag, the pending snapshot), which then see
    a plain client span. Keys the SDK sets later (usage, the response model)
    are stripped again at ``on_end``, ahead of the export processor. A bridged
    span reaches the pipeline as rius's private twin and copy, so the other
    provider's own span is never touched. Never raises.
    """

    def __init__(self, active: Callable[[], bool]) -> None:
        self._active = active
        self._demoted: set[tuple[int, int]] = set()
        self._lock = threading.Lock()

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        try:
            if not _is_sdk_chat_span(span) or not self._active():
                return
            context = span.get_span_context()
            with self._lock:
                self._demoted.add((context.trace_id, context.span_id))
            live = span._attributes  # noqa: SLF001 - a live Span has no public delete
            if not isinstance(live, MutableMapping):
                return
            for key in [key for key in live if _removed(key)]:
                del live[key]
        except Exception:
            logger.debug("could not demote the anthropic SDK's chat span", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        try:
            context = span.get_span_context()
            if context is None:
                return
            key = (context.trace_id, context.span_id)
            with self._lock:
                if key not in self._demoted:
                    return
                self._demoted.discard(key)
            attributes = span.attributes or {}
            if any(_removed(k) for k in attributes):
                span._attributes = replacement_attributes(  # noqa: SLF001
                    span, demoted_attributes(attributes)
                )
        except Exception:
            logger.debug("could not demote the anthropic SDK's chat span", exc_info=True)

    def shutdown(self) -> None:
        with self._lock:
            self._demoted.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True
