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

Whether or not it is demoted, a failed span of the SDK's is exported with
``error.type`` naming the exception class the SDK raised, as every other
span names a failure, instead of the API's error code: see
:func:`native_error_type`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from ._attributes import replacement_attributes
from .semconv import (
    ERROR_TYPE,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    OPENINFERENCE_SPAN_KIND,
)

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


#: Not in the rius registry: the OpenTelemetry HTTP convention's key, written
#: by the anthropic SDK on its span for the call's final response.
_HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"

#: The class every Anthropic client, in Python and TypeScript alike, raises
#: for a status. Other statuses (413, 503, 504, 529) raise a class that
#: differs by client or language, so they keep the SDK's own error.type.
_EXCEPTION_CLASS_BY_STATUS = {
    400: "anthropic.BadRequestError",
    401: "anthropic.AuthenticationError",
    403: "anthropic.PermissionDeniedError",
    404: "anthropic.NotFoundError",
    409: "anthropic.ConflictError",
    422: "anthropic.UnprocessableEntityError",
    429: "anthropic.RateLimitError",
    500: "anthropic.InternalServerError",
}


def _is_sdk_span(span: ReadableSpan) -> bool:
    scope = span.instrumentation_scope
    return scope is not None and scope.name.startswith(ANTHROPIC_SDK_SCOPE_PREFIX)


def native_error_type(span: ReadableSpan) -> dict[str, Any]:
    """``error.type`` naming the exception class, for a failed span of the anthropic SDK's.

    The SDK writes the code of the API's error envelope (``not_found_error``)
    or, without one, the HTTP status. Every other span, OpenInference's
    included, names the exception class (``anthropic.NotFoundError``), so the
    same failure grouped under two names depending on which instrumentation
    was on. The class is the one the SDK raises for the span's
    ``http.response.status_code``, which is how the SDK itself picks it.

    Empty, so ``error.type`` stays as written, for a span from another scope,
    one without ``error.type``, one with no status (a connection error or a
    timeout) and one whose status maps to no single class. The SDK records
    no exception event on this span, so there is no ``exception.type`` to
    keep in step.
    """
    if not _is_sdk_span(span):
        return {}
    attributes = span.attributes or {}
    if ERROR_TYPE not in attributes:
        return {}
    status = attributes.get(_HTTP_RESPONSE_STATUS_CODE)
    if not isinstance(status, int) or isinstance(status, bool):
        return {}
    name = _EXCEPTION_CLASS_BY_STATUS.get(status)
    return {} if name is None else {ERROR_TYPE: name}


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


def _is_sdk_chat_span(span: ReadableSpan) -> bool:
    if not _is_sdk_span(span):
        return False
    return (span.attributes or {}).get(GEN_AI_OPERATION_NAME) == _CHAT


class NativeChatSpanDemoter(SpanProcessor):
    """Strip the GenAI keys off the anthropic SDK's chat span while ``active()``.

    Registered ahead of every processor that reads a span at start
    (normalization, the compaction flag, the pending snapshot), which then see
    a plain client span. Keys the SDK sets later (usage, the response model)
    are stripped again at ``on_end``, ahead of the export processor.

    ``on_end`` decides on its own: a span demoted at start (which no longer
    carries its operation) is known by its span context, and any other span
    is judged again by its own scope and operation. A bridged span reaches
    the pipeline as rius's private twin at start and a separate copy at end;
    only those are ever written, so the other provider's span is never
    touched. Never raises.
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
                started_demoted = key in self._demoted
                self._demoted.discard(key)
            if not started_demoted and not (_is_sdk_chat_span(span) and self._active()):
                return
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
