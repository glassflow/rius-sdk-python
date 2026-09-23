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

The span also carries the OTel MCP semantic conventions (``mcp.method.name``,
``mcp.protocol.version``), so it is identifiable AS an MCP call — a local
``@observe(kind=TOOL)`` tool produces an otherwise identical span. How the
span composes the two conventions it sits under:

- The OTel ``SpanKind`` field is ``CLIENT``, the MCP client-span value. The
  two conventions disagree here: the GenAI execute-tool span says INTERNAL,
  the MCP client span says CLIENT. CLIENT wins because the call crosses a
  process boundary and the MCP convention is the more specific one; a local
  ``@observe(kind=TOOL)`` tool stays INTERNAL. That field is orthogonal to
  ``openinference.span.kind=TOOL``, our product taxonomy attribute, which
  stays.
- The name is the GenAI execute-tool one, ``execute_tool {tool}``, not the
  MCP ``tools/call {tool}``. The MCP convention resolves that collision by
  consolidation: when a tool-execution span already exists, MCP
  instrumentation SHOULD NOT open a second span and SHOULD add its
  attributes to the existing one — which is exactly this span.

``ClientSession`` is built from a bare stream pair and keeps nothing about its
connection — not the transport and not the server address, so server identity
is not obtainable from here. The negotiated protocol version is: mcp 2.x
sessions expose it as ``protocol_version`` (set by initialize, discover or
adopt); mcp 1.x validates it in ``initialize()`` and drops it, so on that
major ``initialize`` is wrapped too, only to remember the value per session.
That 1.x path is order-dependent by nature: a session initialized BEFORE
``rius.init()`` installed the wrap never gets the attribute (its tool calls
are still traced). 2.x reads the session's own state and is order-independent.
"""

from __future__ import annotations

import functools
import weakref
from typing import Any

from mcp import ClientSession  # ImportError => registry treats "mcp" as not installed
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from . import __version__
from ._errors import error_type
from ._serde import serialize, truncate
from .semconv import (
    ERROR_TYPE,
    ERROR_TYPE_TOOL_ERROR,
    GEN_AI_TOOL_NAME,
    INPUT_VALUE,
    MCP_METHOD_NAME,
    MCP_METHOD_TOOLS_CALL,
    MCP_PROTOCOL_VERSION,
    MCP_RESULT_TYPE,
    OUTPUT_VALUE,
    TRACER_NAME,
    SpanKind,
    kind_attributes,
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
            # A lone text block is recorded raw (it usually IS the answer),
            # but bounded like every other attribute: tool results are the
            # payloads most likely to be huge.
            return truncate(texts[0]) if len(texts) == 1 else serialize(texts)
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
        span.set_attribute(ERROR_TYPE, ERROR_TYPE_TOOL_ERROR)


def _call_attributes(name: str, *, protocol_version: str | None) -> dict[str, str | int]:
    """Every identity attribute of a tools/call span, for setting at CREATION.

    Pending snapshots are built at ``on_start``, so anything set afterwards
    never reaches them — that includes the MCP marker, which an interim MRTR
    round must carry as much as a final one.
    """
    attributes = {
        **kind_attributes(SpanKind.TOOL),
        GEN_AI_TOOL_NAME: name,
        MCP_METHOD_NAME: MCP_METHOD_TOOLS_CALL,
    }
    if protocol_version is not None:
        attributes[MCP_PROTOCOL_VERSION] = protocol_version
    return attributes


def _negotiated_version(init_result: Any) -> str | None:
    """The protocol version an InitializeResult carries, on either mcp major."""
    version = getattr(init_result, "protocolVersion", None)
    if version is None:
        version = getattr(init_result, "protocol_version", None)  # mcp 2.x spelling
    return version if isinstance(version, str) else None


def _session_protocol_version(session: Any) -> str | None:
    """The protocol version this session negotiated, on either mcp major.

    mcp 2.x sessions expose ``protocol_version``, set by whichever of
    ``initialize()``, ``discover()`` or ``adopt()`` ran — and ``Client``
    prefers ``discover``, so the ``initialize`` wrap below never fires on that
    path. mcp 1.x has no such property and drops the value after validating
    it, so there the wrap is the only source.
    """
    version = getattr(session, "protocol_version", None)
    if isinstance(version, str):
        return version
    return MCPInstrumentor._protocol_versions.get(session)


class MCPInstrumentor:
    """Duck-types the OTel instrumentor interface (instrument/uninstrument)."""

    _instrumented = False
    _original_call_tool: Any = None
    _original_initialize: Any = None
    _tracer: trace.Tracer | None = None
    # Negotiated protocol version per live session. Weak keys: the session
    # owns its lifetime, and this must never keep one alive.
    _protocol_versions: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()

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
        original_initialize = ClientSession.initialize
        cls._original_initialize = original_initialize

        @functools.wraps(original_initialize)
        async def instrumented_initialize(session: ClientSession, *args: Any, **kw: Any) -> Any:
            result = await original_initialize(session, *args, **kw)
            version = _negotiated_version(result)
            if version is not None:
                cls._protocol_versions[session] = version
            return result

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
                # The OTel SpanKind FIELD (orthogonal to our openinference
                # taxonomy attribute): a remote tool call is CLIENT under both
                # the MCP and the GenAI execute-tool conventions.
                kind=trace.SpanKind.CLIENT,
                attributes=_call_attributes(
                    name, protocol_version=_session_protocol_version(session)
                ),
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                if arguments is not None:
                    span.set_attribute(INPUT_VALUE, serialize(arguments))
                try:
                    result = await original(session, name, arguments, *args, **kw)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.set_attribute(ERROR_TYPE, error_type(exc))
                    raise
                _record_result(span, result)
                return result

        setattr(ClientSession, "call_tool", instrumented_call_tool)  # noqa: B010
        setattr(ClientSession, "initialize", instrumented_initialize)  # noqa: B010
        cls._instrumented = True

    def uninstrument(self) -> None:
        cls = type(self)
        if not cls._instrumented:
            return
        ClientSession.call_tool = cls._original_call_tool  # type: ignore[method-assign]
        ClientSession.initialize = cls._original_initialize  # type: ignore[method-assign]
        cls._original_call_tool = None
        cls._original_initialize = None
        cls._protocol_versions.clear()
        cls._tracer = None
        cls._instrumented = False
