"""First-class MCP tool-call spans (client side: ClientSession.call_tool).

Runs against BOTH mcp majors: the default suite exercises the locked 1.x,
and the ci `mcp-v2` job re-runs this module against ``mcp>=2`` (spec
2026-07-28), whose ``CallToolResult`` renamed its fields to snake_case and
whose tool calls can return interim ``InputRequiredResult``s (GLA2-300).
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


# --- Result-shape compatibility (GLA2-300) -------------------------------
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
