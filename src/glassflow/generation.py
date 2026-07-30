"""LLM generation capture helpers.

``start_as_current_generation`` (context manager) and ``start_generation`` (manual,
requires ``.end()``) open an LLM-kind span and return a ``Generation`` handle for
recording gen_ai-native attributes (messages, model, usage, finish reason). LLM
spans are therefore readable by any gen_ai-compatible consumer.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span

from . import __version__
from ._serde import serialize
from .semconv import (
    GEN_AI_FIRST_TOKEN_EVENT,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PREFIX,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    TRACER_NAME,
    SpanKind,
    kind_attributes,
    set_span_kind,
)

Messages = str | list[Any]


def _normalize_part(item: Any) -> dict[str, Any]:
    """Convert one content-list entry into a spec message part."""
    if isinstance(item, dict):
        # OpenAI multimodal text part {"type": "text", "text": ...}
        if item.get("type") == "text" and "text" in item:
            return {"type": "text", "content": item["text"]}
        if "type" in item:
            return item  # already-typed part: pass through
    return {"type": "text", "content": item if isinstance(item, str) else serialize(item)}


def _normalize_message(message: Any, default_role: str) -> dict[str, Any]:
    """Convert one message into the spec ``{"role", "parts": [...]}`` shape.

    Accepts spec-conformant messages (passed through), OpenAI-style
    ``{"role", "content"}`` / ``tool_calls`` / tool-response messages, and
    falls back to a serialized text part for anything else.
    """
    if isinstance(message, str):
        return {"role": default_role, "parts": [{"type": "text", "content": message}]}
    if not isinstance(message, dict):
        return {"role": default_role, "parts": [{"type": "text", "content": serialize(message)}]}
    if "parts" in message:
        normalized = dict(message)
        normalized.setdefault("role", default_role)
        return normalized

    role = message.get("role", default_role)
    if role == "tool" and "tool_call_id" in message:
        return {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": message["tool_call_id"],
                    "response": message.get("content"),
                }
            ],
        }

    parts: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append({"type": "text", "content": content})
    elif isinstance(content, list):
        parts.extend(_normalize_part(item) for item in content)
    elif content is not None:
        parts.append({"type": "text", "content": serialize(content)})
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            function = call.get("function") or {}
            parts.append(
                {
                    "type": "tool_call",
                    "id": call.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                }
            )
    return {"role": role, "parts": parts}


def _serialize_messages(messages: Messages, default_role: str) -> str:
    """Serialize to the spec message-array shape (JSON string attribute)."""
    if isinstance(messages, str):
        normalized = [_normalize_message(messages, default_role)]
    else:
        normalized = [_normalize_message(message, default_role) for message in messages]
    return serialize(normalized)


class Generation:
    """Handle for recording gen_ai attributes on an LLM span.

    Returned by ``start_generation`` and ``start_as_current_generation``.
    Messages passed to :meth:`set_input` / :meth:`set_output` are normalized
    to the GenAI message shape (``{"role", "parts": [...]}``): bare strings,
    OpenAI-style dicts (including ``tool_calls`` and tool responses), and
    multimodal content lists are all accepted.
    """

    def __init__(self, span: Span) -> None:
        self._span = span
        self._first_token_recorded = False

    def set_input(self, messages: Messages) -> None:
        """Record the request messages (``gen_ai.input.messages``).

        Args:
            messages: A string or list of messages in any supported format;
                bare strings default to the ``user`` role.
        """
        self._span.set_attribute(GEN_AI_INPUT_MESSAGES, _serialize_messages(messages, "user"))

    def set_output(self, messages: Messages) -> None:
        """Record the response messages (``gen_ai.output.messages``).

        Args:
            messages: A string or list of messages in any supported format;
                bare strings default to the ``assistant`` role.
        """
        self._span.set_attribute(GEN_AI_OUTPUT_MESSAGES, _serialize_messages(messages, "assistant"))

    def set_response_model(self, response_model: str) -> None:
        """Record the model that produced the response (``gen_ai.response.model``).

        Args:
            response_model: Model identifier as reported by the provider,
                which may differ from the requested model.
        """
        self._span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)

    def set_usage(
        self, *, input_tokens: int | None = None, output_tokens: int | None = None
    ) -> None:
        """Record token usage (``gen_ai.usage.input_tokens`` / ``output_tokens``).

        Send token counts, never cost: cost is computed server-side from
        model pricing.

        Args:
            input_tokens: Prompt tokens consumed, when known.
            output_tokens: Completion tokens produced, when known.
        """
        if input_tokens is not None:
            self._span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        if output_tokens is not None:
            self._span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

    def record_first_token(self) -> None:
        """Mark the arrival of the first streamed token (``gen_ai.first_token`` event).

        Call from a streaming loop when the first content chunk arrives; the
        backend derives time-to-first-token as the event time minus the span
        start. Idempotent: only the first call records; safe to call
        unconditionally per chunk. A no-op after ``end()``.
        """
        if self._first_token_recorded or not self._span.is_recording():
            return
        self._span.add_event(GEN_AI_FIRST_TOKEN_EVENT)
        self._first_token_recorded = True

    def set_finish_reasons(self, reasons: str | list[str]) -> None:
        """Record why generation stopped (``gen_ai.response.finish_reasons``).

        Args:
            reasons: A single reason (wrapped into a list) or a list of
                reasons, e.g. ``"stop"``, ``"length"``, ``"tool_calls"``.
        """
        if isinstance(reasons, str):
            reasons = [reasons]
        self._span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, reasons)

    def update(self, *, input: Messages | None = None, output: Messages | None = None) -> None:
        """Record input and/or output messages in one call.

        Args:
            input: When not ``None``, forwarded to :meth:`set_input`.
            output: When not ``None``, forwarded to :meth:`set_output`.
        """
        if input is not None:
            self.set_input(input)
        if output is not None:
            self.set_output(output)

    def end(self) -> None:
        """End the underlying span.

        Required for generations created with ``start_generation``; spans from
        ``start_as_current_generation`` end automatically when the block exits.
        """
        self._span.end()


def _configure(
    generation: Generation,
    *,
    model: str | None,
    provider: str | None,
    input: Messages | None,
    model_parameters: dict[str, Any] | None,
    operation: str,
) -> None:
    span = generation._span
    set_span_kind(span, SpanKind.LLM)
    if operation != "chat":  # set_span_kind already stamped the default
        span.set_attribute(GEN_AI_OPERATION_NAME, operation)
    if model is not None:
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
    if provider is not None:
        span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
    for key, value in (model_parameters or {}).items():
        span.set_attribute(f"{GEN_AI_REQUEST_PREFIX}{key}", value)
    if input is not None:
        generation.set_input(input)


def _creation_attributes(model: str | None, provider: str | None, operation: str) -> dict[str, str]:
    """Identity attributes for an LLM span at CREATION (pending snapshots
    are built at on_start; anything set later is invisible to them)."""
    attributes = kind_attributes(SpanKind.LLM)
    attributes[GEN_AI_OPERATION_NAME] = operation
    if model is not None:
        attributes[GEN_AI_REQUEST_MODEL] = model
    if provider is not None:
        attributes[GEN_AI_PROVIDER_NAME] = provider
    return attributes


def start_generation(
    name: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    input: Messages | None = None,
    model_parameters: dict[str, Any] | None = None,
    operation: str = "chat",
) -> Generation:
    """Create an LLM-kind span and return a ``Generation``. You MUST call ``.end()``.

    Manual counterpart to ``start_as_current_generation``: not set as current, no
    auto-recording of exceptions.

    Args:
        name: Span name.
        model: Requested model (``gen_ai.request.model``).
        provider: Provider name (``gen_ai.provider.name``), e.g. ``"openai"``.
        input: Request messages, recorded immediately via ``set_input``.
        model_parameters: Request parameters, each recorded as
            ``gen_ai.request.<key>``.
        operation: Operation name (``gen_ai.operation.name``); default ``"chat"``.

    Returns:
        A ``Generation`` handle; call ``.end()`` when the call completes.
    """
    span = trace.get_tracer(TRACER_NAME, __version__).start_span(
        name, attributes=_creation_attributes(model, provider, operation)
    )
    generation = Generation(span)
    _configure(
        generation,
        model=model,
        provider=provider,
        input=input,
        model_parameters=model_parameters,
        operation=operation,
    )
    return generation


@contextmanager
def start_as_current_generation(
    name: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    input: Messages | None = None,
    model_parameters: dict[str, Any] | None = None,
    operation: str = "chat",
) -> Iterator[Generation]:
    """Open an LLM-kind span as the current span and yield a ``Generation``; auto-ends.

    Children created inside the block nest under this span, and exceptions
    raised in the block are recorded with ERROR status, then re-raised.
    Accepts the same arguments as ``start_generation``.

    Yields:
        A ``Generation`` handle for recording messages, usage, and response
        metadata; the span ends when the block exits.
    """
    tracer = trace.get_tracer(TRACER_NAME, __version__)
    with tracer.start_as_current_span(
        name, attributes=_creation_attributes(model, provider, operation)
    ) as span:
        generation = Generation(span)
        _configure(
            generation,
            model=model,
            provider=provider,
            input=input,
            model_parameters=model_parameters,
            operation=operation,
        )
        yield generation
