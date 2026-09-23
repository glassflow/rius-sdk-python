"""``gen_ai.agent.name`` on execute-tool spans: the agent DOING the call.

The GenAI conventions make the key Conditionally Required on an execute-tool
span, where it means "the human-readable name of the agent executing the
tool". That is NOT what the same key means on an invoke-agent span, where it
names the agent BEING INVOKED. These tests pin both readings so a refactor
cannot quietly merge them.
"""

from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init, observe, start_as_current_span, start_span
from rius.semconv import SpanKind


def _client(exporter: InMemorySpanExporter, **kwargs: object):
    return init(span_exporter=exporter, instruments=[], **kwargs)  # type: ignore[arg-type]


def _attrs(exporter: InMemorySpanExporter, name: str) -> dict:
    (span,) = [s for s in exporter.get_finished_spans() if s.name == name]
    assert span.attributes is not None
    return dict(span.attributes)


# --- the enclosing AGENT scope ---


def test_tool_span_inside_an_agent_scope_carries_the_enclosing_agent(
    exported_spans: InMemorySpanExporter,
) -> None:
    with (
        start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="researcher"),
        start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"),
    ):
        pass
    assert _attrs(exported_spans, "weather")["gen_ai.agent.name"] == "researcher"


def test_tool_span_inside_an_observed_agent_carries_the_enclosing_agent(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe(kind=SpanKind.AGENT, agent_name="researcher")
    def run() -> None:
        with start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"):
            pass

    run()
    assert _attrs(exported_spans, "weather")["gen_ai.agent.name"] == "researcher"


def test_an_observed_tool_inside_an_observed_agent_carries_the_enclosing_agent(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe(kind=SpanKind.TOOL, tool_name="weather")
    def weather() -> None:
        pass

    @observe(kind=SpanKind.AGENT, agent_name="researcher")
    def run() -> None:
        weather()

    run()
    assert _attrs(exported_spans, "execute_tool weather")["gen_ai.agent.name"] == "researcher"


def test_the_innermost_agent_scope_wins(exported_spans: InMemorySpanExporter) -> None:
    """A supervisor delegating to a sub-agent: the tool was executed by the
    sub-agent, not by the one further up the stack."""
    with (
        start_as_current_span("outer", kind=SpanKind.AGENT, agent_name="supervisor"),
        start_as_current_span("inner", kind=SpanKind.AGENT, agent_name="researcher"),
        start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"),
    ):
        pass
    assert _attrs(exported_spans, "weather")["gen_ai.agent.name"] == "researcher"


def test_the_agent_scope_unwinds_with_its_block(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="researcher"):
        pass
    with start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"):
        pass
    assert "gen_ai.agent.name" not in _attrs(exported_spans, "weather")


def test_start_span_does_not_open_an_agent_scope(
    exported_spans: InMemorySpanExporter,
) -> None:
    """``start_span`` does not activate context at all, so it cannot carry the
    scope — the same documented limitation ``user_id=`` has there."""
    agent = start_span("plan", kind=SpanKind.AGENT, agent_name="researcher")
    with start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"):
        pass
    agent.end()
    assert "gen_ai.agent.name" not in _attrs(exported_spans, "weather")


# --- the two meanings of one key ---


def test_the_agent_span_names_the_invoked_agent_and_the_tool_span_the_executor(
    exported_spans: InMemorySpanExporter,
) -> None:
    """One key, two meanings, disambiguated by ``gen_ai.operation.name``. On
    ``invoke_agent`` it is the agent BEING INVOKED; on ``execute_tool`` it is
    the agent DOING the call."""
    with (
        start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="researcher"),
        start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"),
    ):
        pass
    agent = _attrs(exported_spans, "plan")
    tool = _attrs(exported_spans, "weather")
    assert agent["gen_ai.operation.name"] == "invoke_agent"
    assert agent["gen_ai.agent.name"] == "researcher"
    assert tool["gen_ai.operation.name"] == "execute_tool"
    assert tool["gen_ai.agent.name"] == "researcher"


def test_no_other_kind_picks_the_key_up_inside_an_agent_scope(
    exported_spans: InMemorySpanExporter,
) -> None:
    """The key has no defined meaning on a chat, retrieval or chain span, so
    the scope must not leak onto them."""
    with start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="researcher"):
        with start_as_current_span("step", kind=SpanKind.CHAIN):
            pass
        with start_as_current_span("search", kind=SpanKind.RETRIEVER):
            pass
    assert "gen_ai.agent.name" not in _attrs(exported_spans, "step")
    assert "gen_ai.agent.name" not in _attrs(exported_spans, "search")


# --- the span name is unaffected ---


def test_a_tool_span_in_an_agent_scope_is_still_named_after_its_tool(
    exported_spans: InMemorySpanExporter,
) -> None:
    """``compose_span_name`` maps TOOL to ``gen_ai.tool.name``; adding a
    second, differently-sourced key to the same map must not move the name."""
    with (
        start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="researcher"),
        start_as_current_span(kind=SpanKind.TOOL, tool_name="weather"),
    ):
        pass
    assert {s.name for s in exported_spans.get_finished_spans()} == {
        "plan",
        "execute_tool weather",
    }


# --- fallback to the configured agent name ---


def test_a_tool_span_with_no_enclosing_agent_falls_back_to_the_configured_name() -> None:
    """A single-agent process knows who ran the tool even without a scope."""
    exporter = InMemorySpanExporter()
    client = _client(exporter, service_name="svc", agent_name="configured")
    try:
        with start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"):
            pass
    finally:
        client.shutdown()
    assert _attrs(exporter, "weather")["gen_ai.agent.name"] == "configured"


def test_an_enclosing_agent_scope_wins_over_the_configured_name() -> None:
    exporter = InMemorySpanExporter()
    client = _client(exporter, service_name="svc", agent_name="configured")
    try:
        with (
            start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="researcher"),
            start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"),
        ):
            pass
    finally:
        client.shutdown()
    assert _attrs(exporter, "weather")["gen_ai.agent.name"] == "researcher"


def test_a_process_that_named_nothing_emits_no_agent_name_on_a_tool_span() -> None:
    """Same placeholder suppression the invoked-agent name applies: claiming
    ``unknown_service`` executed the tool is worse than saying nothing."""
    exporter = InMemorySpanExporter()
    client = _client(exporter)
    try:
        with start_as_current_span("weather", kind=SpanKind.TOOL, tool_name="weather"):
            pass
    finally:
        client.shutdown()
    assert "gen_ai.agent.name" not in _attrs(exporter, "weather")
