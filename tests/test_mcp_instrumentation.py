"""First-class MCP tool-call spans (client side: ClientSession.call_tool).

Runs against BOTH mcp majors: the default suite exercises the locked 1.x,
and the ci `mcp-v2` job re-runs this module against ``mcp>=2`` (spec
2026-07-28), whose ``CallToolResult`` renamed its fields to snake_case and
whose tool calls can return interim ``InputRequiredResult``s.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest.importorskip("mcp")

try:  # mcp >= 2
    from mcp.server import MCPServer

    MCP_V2 = True
except ImportError:  # mcp 1.x: FastMCP is the same decorator surface
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[assignment]

    MCP_V2 = False

from rius import init  # noqa: E402
from rius.instrumentation import REGISTRY  # noqa: E402
from rius.instrumentation_mcp import MCPInstrumentor  # noqa: E402


def _make_server() -> Any:
    server = MCPServer("test-server")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool()
    def boom() -> str:
        """Always fails."""
        raise ValueError("tool failed")

    return server


@asynccontextmanager
async def _connected_session(server: Any) -> Any:
    """Yield a live ClientSession against an in-memory server, on either major."""
    if MCP_V2:
        from mcp import Client

        async with Client(server) as client:
            yield client.session
    else:
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(server._mcp_server) as session:
            yield session


def _result_error_flag(result: Any) -> Any:
    """Version-agnostic read of the result error flag, for assertions."""
    return getattr(result, "is_error", getattr(result, "isError", None))


@pytest.fixture(autouse=True)
def _fresh_mcp_instrumentor() -> Any:
    instrumentor = MCPInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    yield
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


def _run_tool_call(
    tool: str,
    arguments: dict[str, Any] | None,
    **init_kwargs: Any,
) -> tuple[list[ReadableSpan], Any]:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=["mcp"], **init_kwargs)

    async def scenario() -> Any:
        server = _make_server()
        async with _connected_session(server) as session:
            return await session.call_tool(tool, arguments)

    result = asyncio.run(scenario())
    client.flush()
    return list(inner.get_finished_spans()), result


def test_mcp_is_a_registry_instrument() -> None:
    assert "mcp" in {spec.name for spec in REGISTRY}


def test_call_tool_creates_tool_span() -> None:
    spans, _result = _run_tool_call("add", {"a": 2, "b": 3})
    tool_spans = [s for s in spans if s.name == "execute_tool add"]
    assert tool_spans, f"no tool span; got {[s.name for s in spans]}"
    attrs = tool_spans[0].attributes
    assert attrs is not None
    assert attrs["openinference.span.kind"] == "TOOL"
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "add"
    assert json.loads(attrs["input.value"]) == {"a": 2, "b": 3}
    assert "5" in attrs["output.value"]


def test_tool_error_result_marks_span_error() -> None:
    spans, result = _run_tool_call("boom", None)
    # the server converts tool exceptions into error results on both majors
    assert _result_error_flag(result)
    (tool_span,) = [s for s in spans if s.name == "execute_tool boom"]
    assert not tool_span.status.is_ok


def test_tool_span_nests_under_current_span() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=["mcp"])

    async def scenario() -> None:
        server = _make_server()
        with client.get_tracer().start_as_current_span("agent-step"):
            async with _connected_session(server) as session:
                await session.call_tool("add", {"a": 1, "b": 1})

    asyncio.run(scenario())
    client.flush()
    spans = {s.name: s for s in inner.get_finished_spans()}
    tool_span = spans["execute_tool add"]
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == spans["agent-step"].context.span_id


def test_capture_content_false_strips_tool_io_but_keeps_tool_name() -> None:
    spans, _result = _run_tool_call("add", {"a": 2, "b": 3}, capture_content=False)
    (tool_span,) = [s for s in spans if s.name == "execute_tool add"]
    attrs = tool_span.attributes
    assert attrs is not None
    assert "input.value" not in attrs
    assert "output.value" not in attrs
    assert attrs["gen_ai.tool.name"] == "add"


def test_uninstrument_restores_call_tool() -> None:
    spans, _result = _run_tool_call("add", {"a": 1, "b": 2})
    assert any(s.name == "execute_tool add" for s in spans)

    MCPInstrumentor().uninstrument()

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)  # scoped, no instruments

    async def scenario() -> None:
        server = _make_server()
        async with _connected_session(server) as session:
            await session.call_tool("add", {"a": 1, "b": 2})

    asyncio.run(scenario())
    client.flush()
    assert not any(s.name.startswith("execute_tool") for s in inner.get_finished_spans())


# --- Result-shape compatibility -------------------------------
# mcp 2.x renamed CallToolResult's fields to snake_case (isError -> is_error,
# structuredContent -> structured_content) and added result_type; tool calls
# can return an interim InputRequiredResult (result_type "input_required")
# whose input_requests must never be recorded as tool output. The fakes below
# mimic each major's exact attribute surface, so these tests pin the compat
# behavior regardless of which mcp is installed.


class _V1Result:
    """Attribute surface of mcp 1.x CallToolResult (camelCase)."""

    def __init__(
        self,
        *,
        isError: bool = False,
        structuredContent: Any = None,
        content: Any = None,
    ) -> None:
        self.isError = isError
        self.structuredContent = structuredContent
        self.content = content


class _V2Result:
    """Attribute surface of mcp 2.x CallToolResult (snake_case + result_type)."""

    def __init__(
        self,
        *,
        is_error: bool = False,
        structured_content: Any = None,
        content: Any = None,
        result_type: str = "complete",
    ) -> None:
        self.is_error = is_error
        self.structured_content = structured_content
        self.content = content
        self.result_type = result_type


class _V2InputRequired:
    """Attribute surface of mcp 2.x InputRequiredResult (no error/content fields)."""

    def __init__(self) -> None:
        self.result_type = "input_required"
        self.input_requests = [{"type": "elicitation", "message": "which account?"}]
        self.request_state = "opaque-token"


def _record_on_fresh_span(result: Any) -> ReadableSpan:
    from rius.instrumentation_mcp import _record_result

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    span = client.get_tracer().start_span("execute_tool fake")
    _record_result(span, result)
    span.end()
    client.flush()
    (finished,) = inner.get_finished_spans()
    return finished


def test_v2_error_flag_marks_span_error() -> None:
    span = _record_on_fresh_span(_V2Result(is_error=True))
    assert not span.status.is_ok


def test_v1_error_flag_still_marks_span_error() -> None:
    span = _record_on_fresh_span(_V1Result(isError=True))
    assert not span.status.is_ok


def test_v2_structured_content_is_recorded_as_output() -> None:
    span = _record_on_fresh_span(_V2Result(structured_content={"result": 5}))
    assert span.attributes is not None
    assert json.loads(span.attributes["output.value"]) == {"result": 5}


def test_v1_structured_content_is_still_recorded_as_output() -> None:
    span = _record_on_fresh_span(_V1Result(structuredContent={"result": 5}))
    assert span.attributes is not None
    assert json.loads(span.attributes["output.value"]) == {"result": 5}


def test_input_required_round_is_marked_and_records_no_output() -> None:
    span = _record_on_fresh_span(_V2InputRequired())
    assert span.attributes is not None
    # the interim payload (input_requests) is a content surface that is NOT
    # the tool's output; it must never land in output.value
    assert "output.value" not in span.attributes
    assert span.attributes["mcp.result_type"] == "input_required"
    assert span.status.is_ok


def test_complete_result_carries_no_result_type_attribute() -> None:
    span = _record_on_fresh_span(_V2Result(structured_content={"ok": True}))
    assert span.attributes is not None
    assert "mcp.result_type" not in span.attributes


def test_tool_span_carries_kind_and_name_at_start() -> None:
    """Pending snapshots are built at on_start; the tool span used to set its
    kind and gen_ai.tool.name afterwards, so its snapshots were unclassifiable."""
    from opentelemetry.sdk.trace import SpanProcessor

    seen: dict[str, dict[str, Any]] = {}

    class Recorder(SpanProcessor):
        def on_start(self, span, parent_context=None) -> None:  # noqa: ANN001
            seen[span.name] = dict(span.attributes or {})

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=["mcp"])
    client._provider.add_span_processor(Recorder())

    async def scenario() -> Any:
        server = _make_server()
        async with _connected_session(server) as session:
            return await session.call_tool("add", {"a": 1, "b": 2})

    asyncio.run(scenario())
    attrs = seen["execute_tool add"]
    assert attrs["openinference.span.kind"] == "TOOL"
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "add"
    assert attrs["mcp.method.name"] == "tools/call"


# --- OTel MCP semantic conventions --------------------------------------------
# The client-side tools/call span must be identifiable AS an MCP call: a local
# @observe(kind=TOOL) tool produces the same kind, name and I/O shape, so only
# the `mcp.*` attributes separate the two downstream.


def test_mcp_call_carries_the_mcp_method_marker() -> None:
    spans, _result = _run_tool_call("add", {"a": 2, "b": 3})
    (tool_span,) = [s for s in spans if s.name == "execute_tool add"]
    attrs = tool_span.attributes
    assert attrs is not None
    assert attrs["mcp.method.name"] == "tools/call"
    assert attrs["gen_ai.operation.name"] == "execute_tool"


def test_error_result_span_still_carries_the_mcp_method_marker() -> None:
    spans, _result = _run_tool_call("boom", None)
    (tool_span,) = [s for s in spans if s.name == "execute_tool boom"]
    assert not tool_span.status.is_ok
    assert tool_span.attributes is not None
    assert tool_span.attributes["mcp.method.name"] == "tools/call"


def test_negotiated_protocol_version_is_recorded() -> None:
    from mcp.types import LATEST_PROTOCOL_VERSION

    spans, _result = _run_tool_call("add", {"a": 1, "b": 1})
    (tool_span,) = [s for s in spans if s.name == "execute_tool add"]
    assert tool_span.attributes is not None
    # the in-memory server and the client share one library, so the version
    # the handshake negotiates is that library's latest
    assert tool_span.attributes["mcp.protocol.version"] == LATEST_PROTOCOL_VERSION


def test_local_tool_span_carries_no_mcp_marker(exported_spans: InMemorySpanExporter) -> None:
    from rius import SpanKind, observe

    @observe(kind=SpanKind.TOOL)
    def local_tool() -> int:
        return 1

    local_tool()
    (span,) = exported_spans.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["openinference.span.kind"] == "TOOL"
    assert "mcp.method.name" not in span.attributes


def test_interim_input_required_round_keeps_the_mcp_marker() -> None:
    """MRTR interim rounds used to be marked only by mcp.result_type; the
    method marker must be there too, or the round is invisible to MCP rollups."""
    from rius.instrumentation_mcp import _call_attributes

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    span = client.get_tracer().start_span(
        "execute_tool fake", attributes=_call_attributes("fake", protocol_version=None)
    )
    from rius.instrumentation_mcp import _record_result

    _record_result(span, _V2InputRequired())
    span.end()
    client.flush()
    (finished,) = inner.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["mcp.result_type"] == "input_required"
    assert finished.attributes["mcp.method.name"] == "tools/call"
    assert "mcp.protocol.version" not in finished.attributes


def test_single_text_result_is_bounded() -> None:
    from types import SimpleNamespace

    from rius._serde import MAX_ATTR_CHARS, TRUNCATION_MARKER
    from rius.instrumentation_mcp import _serialize_result

    result = SimpleNamespace(content=[SimpleNamespace(text="x" * 200_000)])
    text = _serialize_result(result)
    assert text.endswith(TRUNCATION_MARKER)
    assert len(text) == MAX_ATTR_CHARS + len(TRUNCATION_MARKER)
