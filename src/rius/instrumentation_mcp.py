"""First-class spans for MCP tool calls.

Wraps ``mcp.ClientSession.call_tool`` so every tool invocation an agent makes
over MCP becomes a TOOL-kind span: tool name, arguments, result, latency, and
error status. Generic instrumentation SDKs cover MCP unevenly (the OpenInference
MCP package only propagates context; it creates no spans), so we instrument it
ourselves. Registered in :mod:`rius.instrumentation` under ``"mcp"``; the
top-level import of ``mcp`` below makes an environment without the package look
"not installed" to the registry, exactly like a missing third-party instrumentor.

Tool arguments/results are recorded as ``input.value`` / ``output.value``, so
they are covered by the same ``mask`` / ``capture_content`` controls as all
other content.
"""

from __future__ import annotations

import functools
from typing import Any

from mcp import ClientSession  # ImportError => registry treats "mcp" as not installed
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from . import __version__
from ._serde import serialize
from .semconv import (
    GEN_AI_TOOL_NAME,
    INPUT_VALUE,
    MCP_RESULT_TYPE,
    OUTPUT_VALUE,
    TRACER_NAME,
    SpanKind,
    set_span_kind,
)

# mcp 2.x (spec 2026-07-28) renamed CallToolResult's fields to snake_case and
# dropped the camelCase attributes entirely; reads must try both spellings to
# stay correct on whichever major the host application has installed.


def _serialize_result(result: Any) -> str:
    """Best-effort serialization of a CallToolResult."""
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)  # mcp 1.x
    if structured is not None:
        return serialize(structured)
    content = getattr(result, "content", None)
    if content is not None:
        texts = [block.text for block in content if getattr(block, "text", None) is not None]
        if texts:
            return texts[0] if len(texts) == 1 else serialize(texts)
    return serialize(result)


def _result_is_error(result: Any) -> bool:
    """The MCP error-result flag, on either major (never raises)."""
    return bool(getattr(result, "is_error", getattr(result, "isError", False)))


def _record_result(span: Any, result: Any) -> None:
    """Record a tools/call result on the span, on either mcp major.

    An interim ``InputRequiredResult`` (MRTR, mcp 2.x) is NOT the tool's
    output: its ``input_requests`` carry elicitation/sampling content, so
    recording them as ``output.value`` would leak conversation content into
    a tool span. Interim rounds get only the ``mcp.result_type`` marker; the
    final round of the retry loop records output as usual.
    """
    if getattr(result, "result_type", None) == "input_required":
        span.set_attribute(MCP_RESULT_TYPE, "input_required")
        return
    span.set_attribute(OUTPUT_VALUE, _serialize_result(result))
    if _result_is_error(result):
        span.set_status(Status(StatusCode.ERROR, "tool returned an error result"))


class MCPInstrumentor:
    """Duck-types the OTel instrumentor interface (instrument/uninstrument)."""

    _instrumented = False
    _original_call_tool: Any = None
    _tracer: trace.Tracer | None = None

    @property
    def is_instrumented_by_opentelemetry(self) -> bool:
        return type(self)._instrumented

    def instrument(self, *, tracer_provider: Any = None, **kwargs: Any) -> None:
        cls = type(self)
        if cls._instrumented:
            return
        provider = tracer_provider if tracer_provider is not None else trace.get_tracer_provider()
        cls._tracer = provider.get_tracer(TRACER_NAME, __version__)
        original = ClientSession.call_tool
        cls._original_call_tool = original

        @functools.wraps(original)
        async def instrumented_call_tool(
            session: ClientSession,
            name: str,
            arguments: dict[str, Any] | None = None,
            *args: Any,
            **kw: Any,
        ) -> Any:
            tracer = cls._tracer
            if tracer is None:  # uninstrumented mid-flight; fall through
                return await original(session, name, arguments, *args, **kw)
            with tracer.start_as_current_span(
                f"execute_tool {name}",
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                set_span_kind(span, SpanKind.TOOL)
                span.set_attribute(GEN_AI_TOOL_NAME, name)
                if arguments is not None:
                    span.set_attribute(INPUT_VALUE, serialize(arguments))
                try:
                    result = await original(session, name, arguments, *args, **kw)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                _record_result(span, result)
                return result

        setattr(ClientSession, "call_tool", instrumented_call_tool)  # noqa: B010
        cls._instrumented = True

    def uninstrument(self) -> None:
        cls = type(self)
        if not cls._instrumented:
            return
        ClientSession.call_tool = cls._original_call_tool  # type: ignore[method-assign]
        cls._original_call_tool = None
        cls._tracer = None
        cls._instrumented = False
