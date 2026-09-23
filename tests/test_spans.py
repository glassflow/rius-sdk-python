import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind as OtelSpanKind
from opentelemetry.trace import StatusCode

from rius import start_as_current_span, start_span
from rius.semconv import SpanKind

# --- context manager: start_as_current_span ---


def test_cm_default_kind_is_chain(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("step"):
        pass
    span = exported_spans.get_finished_spans()[0]
    assert span.name == "step"
    assert span.attributes["openinference.span.kind"] == "CHAIN"


def test_cm_custom_kind(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("t", kind=SpanKind.TOOL):
        pass
    assert exported_spans.get_finished_spans()[0].attributes["openinference.span.kind"] == "TOOL"


def test_cm_input_and_output(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("s", input={"q": "hi"}) as obs:
        obs.set_output(["a", "b"])
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "hi" in attrs["input.value"]
    assert "a" in attrs["output.value"]


def test_cm_set_attribute(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("s") as obs:
        obs.set_attribute("custom.tag", "x")
    assert exported_spans.get_finished_spans()[0].attributes["custom.tag"] == "x"


def test_cm_nesting(exported_spans: InMemorySpanExporter) -> None:
    # entered left-to-right, so "inner" nests under "outer"
    with start_as_current_span("outer"), start_as_current_span("inner"):
        pass
    spans = {s.name: s for s in exported_spans.get_finished_spans()}
    assert spans["inner"].parent.span_id == spans["outer"].context.span_id


def test_cm_exception_sets_error_status(exported_spans: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError, match="boom"), start_as_current_span("op"):
        raise ValueError("boom")
    span = exported_spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)


# --- manual lifecycle: start_span / update / end ---


def test_manual_requires_end(exported_spans: InMemorySpanExporter) -> None:
    obs = start_span("job", kind=SpanKind.TOOL)
    obs.update(output="done")
    assert exported_spans.get_finished_spans() == ()  # not ended -> not exported
    obs.end()
    spans = exported_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "job"
    assert spans[0].attributes["openinference.span.kind"] == "TOOL"
    assert "done" in spans[0].attributes["output.value"]


def test_manual_span_is_not_current(exported_spans: InMemorySpanExporter) -> None:
    obs = start_span("manual")
    with start_as_current_span("inner"):
        pass
    obs.end()
    spans = {s.name: s for s in exported_spans.get_finished_spans()}
    assert spans["inner"].parent is None  # manual span is not activated as current


# --- error.type ---


def test_cm_exception_sets_error_type(exported_spans: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError, match="boom"), start_as_current_span("op"):
        raise ValueError("boom")
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["error.type"] == "ValueError"


def test_cm_success_sets_no_error_type(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("op"):
        pass
    assert "error.type" not in exported_spans.get_finished_spans()[0].attributes


# --- gen_ai.tool.name: Required on execute_tool spans by the GenAI convention ---


def test_cm_tool_kind_sets_gen_ai_tool_name(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("weather", kind=SpanKind.TOOL):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.tool.name"] == "weather"


def test_manual_tool_kind_sets_gen_ai_tool_name(exported_spans: InMemorySpanExporter) -> None:
    start_span("weather", kind=SpanKind.TOOL).end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.tool.name"] == "weather"


def test_cm_explicit_tool_name_wins_over_the_span_name(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span("execute_tool weather", kind=SpanKind.TOOL, tool_name="weather"):
        pass
    span = exported_spans.get_finished_spans()[0]
    assert span.name == "execute_tool weather"
    assert span.attributes["gen_ai.tool.name"] == "weather"


def test_manual_explicit_tool_name_wins_over_the_span_name(
    exported_spans: InMemorySpanExporter,
) -> None:
    start_span("execute_tool weather", kind=SpanKind.TOOL, tool_name="weather").end()
    span = exported_spans.get_finished_spans()[0]
    assert span.attributes["gen_ai.tool.name"] == "weather"


def test_cm_chain_kind_has_no_gen_ai_tool_name(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("step"):
        pass
    assert "gen_ai.tool.name" not in exported_spans.get_finished_spans()[0].attributes


# --- RETRIEVER: gen_ai.operation.name and gen_ai.data_source.id ---


def test_cm_retriever_carries_both_taxonomy_keys(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("search", kind=SpanKind.RETRIEVER, data_source_id="docs-index"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["openinference.span.kind"] == "RETRIEVER"
    assert attrs["gen_ai.operation.name"] == "retrieval"
    assert attrs["gen_ai.data_source.id"] == "docs-index"


def test_manual_retriever_carries_the_data_source(exported_spans: InMemorySpanExporter) -> None:
    start_span("search", kind=SpanKind.RETRIEVER, data_source_id="docs-index").end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.data_source.id"] == "docs-index"


def test_retriever_without_a_data_source_omits_the_key(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span("search", kind=SpanKind.RETRIEVER):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.operation.name"] == "retrieval"
    assert "gen_ai.data_source.id" not in attrs


def test_chain_has_no_operation_name(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("step"):
        pass
    assert "gen_ai.operation.name" not in exported_spans.get_finished_spans()[0].attributes


# --- RETRIEVER: top_k describes the request, documents describe the result ---


def test_retriever_carries_top_k_at_creation(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("search", kind=SpanKind.RETRIEVER, top_k=5):
        pass
    assert exported_spans.get_finished_spans()[0].attributes["gen_ai.retrieval.top_k"] == 5


def test_top_k_is_ignored_on_non_retriever_kinds(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("step", kind=SpanKind.CHAIN, top_k=5):
        pass
    assert "gen_ai.retrieval.top_k" not in exported_spans.get_finished_spans()[0].attributes


def test_retrieved_documents_are_recorded_as_json(exported_spans: InMemorySpanExporter) -> None:
    obs = start_span("search", kind=SpanKind.RETRIEVER, data_source_id="docs", top_k=2)
    obs.set_retrieved_documents([{"id": "doc-1", "score": 0.91}, {"id": "doc-2", "score": 0.4}])
    obs.end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert json.loads(attrs["gen_ai.retrieval.documents"]) == [
        {"id": "doc-1", "score": 0.91},
        {"id": "doc-2", "score": 0.4},
    ]
    # The request half stays alongside the result half.
    assert attrs["gen_ai.data_source.id"] == "docs"
    assert attrs["gen_ai.retrieval.top_k"] == 2


# --- the OTel SpanKind FIELD, derived from the taxonomy ---


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (SpanKind.LLM, OtelSpanKind.CLIENT),
        (SpanKind.EMBEDDING, OtelSpanKind.CLIENT),
        (SpanKind.RETRIEVER, OtelSpanKind.CLIENT),
        (SpanKind.TOOL, OtelSpanKind.INTERNAL),
        (SpanKind.AGENT, OtelSpanKind.INTERNAL),
        (SpanKind.CHAIN, OtelSpanKind.INTERNAL),
    ],
)
def test_otel_kind_field_follows_taxonomy(
    exported_spans: InMemorySpanExporter, kind: SpanKind, expected: OtelSpanKind
) -> None:
    start_span("manual", kind=kind).end()
    with start_as_current_span("scoped", kind=kind):
        pass
    kinds = {s.name: s.kind for s in exported_spans.get_finished_spans()}
    assert kinds == {"manual": expected, "scoped": expected}


def test_default_chain_span_is_internal(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("step"):
        pass
    assert exported_spans.get_finished_spans()[0].kind is OtelSpanKind.INTERNAL


# --- gen_ai.agent.name: Conditionally Required on invoke_agent spans ---


def test_cm_agent_kind_takes_an_explicit_agent_name(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="planner"):
        pass
    assert exported_spans.get_finished_spans()[0].attributes["gen_ai.agent.name"] == "planner"


def test_manual_agent_kind_takes_an_explicit_agent_name(
    exported_spans: InMemorySpanExporter,
) -> None:
    start_span("plan", kind=SpanKind.AGENT, agent_name="planner").end()
    assert exported_spans.get_finished_spans()[0].attributes["gen_ai.agent.name"] == "planner"


def test_agent_name_is_never_derived_from_the_span_name(
    exported_spans: InMemorySpanExporter,
) -> None:
    """Unlike the tool name, which falls back to the span name for
    compatibility, an agent name is never invented: a function name is not an
    agent's identity, and a wrong name here mislabels a whole waterfall."""
    with start_as_current_span("plan_the_trip", kind=SpanKind.AGENT):
        pass
    assert "gen_ai.agent.name" not in exported_spans.get_finished_spans()[0].attributes


def test_only_agent_spans_carry_the_agent_name(exported_spans: InMemorySpanExporter) -> None:
    """The key means "the agent this span invokes". On any other kind it would
    read as "the agent that produced this span", which is the resource's job."""
    with start_as_current_span("step", kind=SpanKind.CHAIN, agent_name="planner"):
        pass
    assert "gen_ai.agent.name" not in exported_spans.get_finished_spans()[0].attributes


def test_agent_kind_takes_an_optional_agent_id(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="p", agent_id="ag_1"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.agent.id"] == "ag_1"


def test_agent_id_is_omitted_when_not_given(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="planner"):
        pass
    assert "gen_ai.agent.id" not in exported_spans.get_finished_spans()[0].attributes


def test_agent_name_falls_back_to_the_configured_agent_name() -> None:
    """The conventions make the key Conditionally Required on invoke_agent
    spans, and a single-agent process does know the answer, so the span says
    it rather than leaving the reader to find the resource. Resource
    attributes do not survive every collector pipeline; the span is
    self-describing either way."""
    from rius import init

    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter, service_name="svc", agent_name="configured", instruments=[]
    )
    try:
        with start_as_current_span("plan", kind=SpanKind.AGENT):
            pass
    finally:
        client.shutdown()
    assert exporter.get_finished_spans()[0].attributes["gen_ai.agent.name"] == "configured"


def test_explicit_agent_name_wins_over_the_configured_one() -> None:
    """The case the multi-agent product is built around: one process, several
    agents, each invocation naming the agent it invokes."""
    from rius import init

    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter, service_name="svc", agent_name="configured", instruments=[]
    )
    try:
        with start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="researcher"):
            pass
    finally:
        client.shutdown()
    assert exporter.get_finished_spans()[0].attributes["gen_ai.agent.name"] == "researcher"


# --- default span names: "{operation} {target}" when no name is given ---


def test_cm_names_a_tool_span_after_the_operation_and_the_tool(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span(kind=SpanKind.TOOL, tool_name="get_weather"):
        pass
    span = exported_spans.get_finished_spans()[0]
    assert span.name == "execute_tool get_weather"
    # The rendered name never feeds the attribute: gen_ai.tool.name is a
    # Required metric dimension and a prefix there splits the series.
    assert span.attributes["gen_ai.tool.name"] == "get_weather"


def test_manual_names_a_tool_span_after_the_operation_and_the_tool(
    exported_spans: InMemorySpanExporter,
) -> None:
    start_span(kind=SpanKind.TOOL, tool_name="get_weather").end()
    assert exported_spans.get_finished_spans()[0].name == "execute_tool get_weather"


def test_tool_span_without_a_tool_name_is_the_bare_operation(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span(kind=SpanKind.TOOL):
        pass
    span = exported_spans.get_finished_spans()[0]
    assert span.name == "execute_tool"
    assert "gen_ai.tool.name" not in span.attributes


def test_cm_names_an_agent_span_after_the_agent_it_invokes(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span(kind=SpanKind.AGENT, agent_name="planner"):
        pass
    assert exported_spans.get_finished_spans()[0].name == "invoke_agent planner"


def test_manual_names_an_agent_span_after_the_agent_it_invokes(
    exported_spans: InMemorySpanExporter,
) -> None:
    start_span(kind=SpanKind.AGENT, agent_name="planner").end()
    assert exported_spans.get_finished_spans()[0].name == "invoke_agent planner"


def test_agent_span_name_uses_the_configured_agent_name() -> None:
    """The name composes from the SAME resolved agent name the attribute
    carries, so a single-agent process gets a named span for free."""
    from rius import init

    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter, service_name="svc", agent_name="configured", instruments=[]
    )
    try:
        with start_as_current_span(kind=SpanKind.AGENT):
            pass
    finally:
        client.shutdown()
    assert exporter.get_finished_spans()[0].name == "invoke_agent configured"


def test_agent_span_without_any_agent_name_is_the_bare_operation(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span(kind=SpanKind.AGENT):
        pass
    assert exported_spans.get_finished_spans()[0].name == "invoke_agent"


def test_cm_names_a_retriever_span_after_the_data_source(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span(kind=SpanKind.RETRIEVER, data_source_id="product-kb"):
        pass
    assert exported_spans.get_finished_spans()[0].name == "retrieval product-kb"


def test_manual_names_a_retriever_span_after_the_data_source(
    exported_spans: InMemorySpanExporter,
) -> None:
    start_span(kind=SpanKind.RETRIEVER, data_source_id="product-kb").end()
    assert exported_spans.get_finished_spans()[0].name == "retrieval product-kb"


def test_retriever_span_without_a_data_source_is_the_bare_operation(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_span(kind=SpanKind.RETRIEVER):
        pass
    assert exported_spans.get_finished_spans()[0].name == "retrieval"


def test_unnamed_chain_span_is_the_degenerate_literal(
    exported_spans: InMemorySpanExporter,
) -> None:
    """The one case with nothing to compose from: CHAIN has no operation, and
    a manual helper has no function whose qualname it could borrow."""
    with start_as_current_span(kind=SpanKind.CHAIN):
        pass
    start_span().end()
    assert [s.name for s in exported_spans.get_finished_spans()] == ["chain", "chain"]


@pytest.mark.parametrize(
    "kind",
    [
        SpanKind.CHAIN,
        SpanKind.TOOL,
        SpanKind.AGENT,
        SpanKind.RETRIEVER,
        SpanKind.LLM,
        SpanKind.EMBEDDING,
    ],
)
def test_an_explicit_name_always_wins(kind: SpanKind, exported_spans: InMemorySpanExporter) -> None:
    """Composition changes the DEFAULT, never the ability to choose."""
    with start_as_current_span("my-step", kind=kind, tool_name="t", agent_name="a"):
        pass
    start_span("my-step", kind=kind, tool_name="t", agent_name="a").end()
    assert [s.name for s in exported_spans.get_finished_spans()] == ["my-step", "my-step"]


def test_model_kinds_on_the_generic_helper_get_the_bare_operation(
    exported_spans: InMemorySpanExporter,
) -> None:
    """This surface takes no model, so it cannot compose one in; a name with
    a model in it comes from the generation helpers."""
    with start_as_current_span(kind=SpanKind.LLM):
        pass
    with start_as_current_span(kind=SpanKind.EMBEDDING):
        pass
    assert [s.name for s in exported_spans.get_finished_spans()] == ["chat", "embeddings"]
