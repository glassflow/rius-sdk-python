"""LLM generation capture helpers.

``start_as_current_generation`` (context manager) and ``start_generation`` (manual,
requires ``.end()``) open an LLM-kind span and return a ``Generation`` handle for
recording gen_ai-native attributes (messages, model, usage, finish reason). LLM
spans are therefore readable by any gen_ai-compatible consumer.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from opentelemetry.trace import Span

from ._errors import error_type
from ._serde import serialize
from ._tracer import sdk_tracer
from .semconv import (
    ERROR_TYPE,
    GEN_AI_FIRST_TOKEN_EVENT,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PREFIX,
    GEN_AI_REQUEST_REASONING_LEVEL,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    USER_ID,
    SpanKind,
    kind_attributes,
    otel_span_kind,
)
from .user import user

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
        # Set by _configure; drives the Anthropic input-token summing in
        # set_usage. A bare Generation(span) has no provider and never sums.
        self._provider: str | None = None

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

    def set_tool_definitions(self, tools: list[Any]) -> None:
        """Record the request's tool/function definitions (``gen_ai.tool.definitions``).

        Definitions are serialized verbatim, in whatever shape the provider
        request used (OpenAI nests each tool under ``function``, Anthropic uses
        top-level ``name``/``input_schema``) — no normalization, so what is
        recorded is exactly what the model was shown. Definitions are content,
        not identity: they are masked/stripped under ``capture_content=False``
        like messages are.

        Args:
            tools: The tools/functions list passed to the provider, verbatim.
        """
        self._span.set_attribute(GEN_AI_TOOL_DEFINITIONS, serialize(tools))

    def set_response_model(self, response_model: str) -> None:
        """Record the model that produced the response (``gen_ai.response.model``).

        Args:
            response_model: Model identifier as reported by the provider,
                which may differ from the requested model.
        """
        self._span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)

    def set_usage(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_input_tokens: int | None = None,
        cache_write_input_tokens: int | None = None,
        reasoning_output_tokens: int | None = None,
    ) -> None:
        """Record token usage (``gen_ai.usage.*`` attributes).

        Send token counts, never cost: cost is computed server-side from
        model pricing.

        Pass provider-reported values as-is; never pre-add anything. Per the
        GenAI conventions, ``gen_ai.usage.input_tokens`` is the total
        including cached tokens (the cache counts are subsets of it).
        Anthropic's API reports ``input_tokens`` excluding the cache counts,
        and the conventions require the instrumentation to do the summing,
        so when the generation's provider is ``"anthropic"`` the emitted
        total is ``input_tokens`` plus both cache counts. For every other
        provider the values are recorded verbatim.

        Args:
            input_tokens: Prompt tokens consumed, when known.
            output_tokens: Completion tokens produced, when known.
            cache_read_input_tokens: Input tokens served from a
                provider-managed prompt cache
                (``gen_ai.usage.cache_read.input_tokens``).
            cache_write_input_tokens: Input tokens written to a
                provider-managed prompt cache
                (``gen_ai.usage.cache_write.input_tokens``, called
                "cache creation" by Anthropic).
            reasoning_output_tokens: Output tokens spent on reasoning /
                extended thinking (``gen_ai.usage.reasoning.output_tokens``).
                A subset of ``output_tokens``, never in addition to it:
                providers already include reasoning tokens in the output
                total, so pass both as reported and do no arithmetic.
        """
        if input_tokens is not None:
            total = input_tokens
            if self._provider is not None and self._provider.lower() == "anthropic":
                total += (cache_read_input_tokens or 0) + (cache_write_input_tokens or 0)
            self._span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, total)
        if output_tokens is not None:
            self._span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        if cache_read_input_tokens is not None:
            self._span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_input_tokens)
        if cache_write_input_tokens is not None:
            self._span.set_attribute(
                GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, cache_write_input_tokens
            )
        if reasoning_output_tokens is not None:
            self._span.set_attribute(GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning_output_tokens)

    def record_first_token(self) -> None:
        """Mark the arrival of the first streamed token.

        Call from a streaming loop when the first content chunk arrives. Records
        the ``gen_ai.first_token`` event (the timestamp the backend derives
        time-to-first-token from) and, per the GenAI conventions, the derived
        ``gen_ai.response.time_to_first_chunk`` (seconds since span start) plus
        ``gen_ai.request.stream = True``: a first chunk arriving is what tells
        the SDK the request streamed. Idempotent: only the first call records;
        safe to call unconditionally per chunk. A no-op after ``end()``.
        """
        if self._first_token_recorded or not self._span.is_recording():
            return
        now_ns = time.time_ns()
        self._span.add_event(GEN_AI_FIRST_TOKEN_EVENT, timestamp=now_ns)
        self._first_token_recorded = True
        self._span.set_attribute(GEN_AI_REQUEST_STREAM, True)
        # Only the SDK span knows its start; a non-SDK Span (or one whose
        # start_time is unset) keeps the event and skips the derived value.
        start_ns = getattr(self._span, "start_time", None)
        if isinstance(start_ns, int):
            self._span.set_attribute(
                GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK, max(now_ns - start_ns, 0) / 1e9
            )

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


_PRIMITIVES = (str, bool, int, float)


def _attribute_value(value: Any) -> Any:
    """A value OTel will store: primitives and primitive sequences as-is, else JSON."""
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, _PRIMITIVES) for v in value):
        return list(value)
    return serialize(value)


def _configure(
    generation: Generation,
    *,
    model: str | None,
    provider: str | None,
    input: Messages | None,
    model_parameters: dict[str, Any] | None,
    operation: str,
    reasoning_level: str | None,
    tools: list[Any] | None,
) -> None:
    span = generation._span
    # Kind, operation, model and provider are already on the span from
    # _creation_attributes; each set_attribute here was a second locked write
    # of the same value (the review counted up to five per generation).
    if provider is not None:
        generation._provider = provider
    for key, value in (model_parameters or {}).items():
        # OTel accepts primitives and homogeneous primitive sequences; anything
        # else (response_format dicts, nested tool choices) was dropped with a
        # warning from the OTel logger and never reached the span. None is
        # "not set", not a value.
        if value is None:
            continue
        span.set_attribute(f"{GEN_AI_REQUEST_PREFIX}{key}", _attribute_value(value))
    if reasoning_level is not None:
        span.set_attribute(GEN_AI_REQUEST_REASONING_LEVEL, reasoning_level)
    if tools is not None:
        generation.set_tool_definitions(tools)
    if input is not None:
        generation.set_input(input)


def _creation_attributes(
    model: str | None, provider: str | None, operation: str, user_id: str | None = None
) -> dict[str, str]:
    """Identity attributes for an LLM span at CREATION (pending snapshots
    are built at on_start; anything set later is invisible to them)."""
    attributes = kind_attributes(SpanKind.LLM)
    attributes[GEN_AI_OPERATION_NAME] = operation
    if model is not None:
        attributes[GEN_AI_REQUEST_MODEL] = model
    if provider is not None:
        attributes[GEN_AI_PROVIDER_NAME] = provider
    if user_id is not None:
        attributes[USER_ID] = user_id
    return attributes


def start_generation(
    name: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    input: Messages | None = None,
    model_parameters: dict[str, Any] | None = None,
    operation: str = "chat",
    reasoning_level: str | None = None,
    tools: list[Any] | None = None,
    user_id: str | None = None,
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
        reasoning_level: Requested reasoning/thinking effort level
            (``gen_ai.request.reasoning.level``), e.g. OpenAI's
            ``reasoning.effort`` values. Provider-defined string, recorded
            verbatim.
        tools: The request's tool/function definitions, recorded immediately
            via ``set_tool_definitions`` (verbatim, any provider shape).
        user_id: End-user identity (``user.id``) stamped on this span. Sugar
            for a single call; to attribute a whole request use ``user()``.

    Returns:
        A ``Generation`` handle; call ``.end()`` when the call completes.
    """
    span = sdk_tracer().start_span(
        name,
        kind=otel_span_kind(SpanKind.LLM),
        attributes=_creation_attributes(model, provider, operation, user_id),
    )
    generation = Generation(span)
    _configure(
        generation,
        model=model,
        provider=provider,
        input=input,
        model_parameters=model_parameters,
        operation=operation,
        reasoning_level=reasoning_level,
        tools=tools,
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
    reasoning_level: str | None = None,
    tools: list[Any] | None = None,
    user_id: str | None = None,
) -> Iterator[Generation]:
    """Open an LLM-kind span as the current span and yield a ``Generation``; auto-ends.

    Children created inside the block nest under this span, and exceptions
    raised in the block are recorded with ERROR status, then re-raised.
    Accepts the same arguments as ``start_generation``.

    Yields:
        A ``Generation`` handle for recording messages, usage, and response
        metadata; the span ends when the block exits.
    """
    tracer = sdk_tracer()
    with (
        # user_id is sugar for user(user_id) around the block: children opened
        # inside inherit it through UserSpanProcessor, this span at creation.
        user(user_id) if user_id is not None else nullcontext(),
        tracer.start_as_current_span(
            name,
            kind=otel_span_kind(SpanKind.LLM),
            attributes=_creation_attributes(model, provider, operation, user_id),
        ) as span,
    ):
        generation = Generation(span)
        _configure(
            generation,
            model=model,
            provider=provider,
            input=input,
            model_parameters=model_parameters,
            operation=operation,
            reasoning_level=reasoning_level,
            tools=tools,
        )
        try:
            yield generation
        except BaseException as exc:
            # The OTel context manager above records the exception event and
            # ERROR status as the block unwinds; error.type is the one thing
            # it does not set, and the GenAI conventions require it on an
            # inference span that ends in an error. Set it before re-raising
            # so the event and this attribute land on the same span.
            span.set_attribute(ERROR_TYPE, error_type(exc))
            raise
