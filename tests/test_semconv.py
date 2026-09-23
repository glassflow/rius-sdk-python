import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind as OtelSpanKind

from rius.semconv import (
    GEN_AI_OPERATION_NAME,
    OPENINFERENCE_SPAN_KIND,
    SpanKind,
    otel_span_kind,
    set_span_kind,
)


def test_span_kind_values_are_openinference_strings() -> None:
    assert SpanKind.LLM.value == "LLM"
    assert SpanKind.TOOL.value == "TOOL"
    assert SpanKind.RETRIEVER.value == "RETRIEVER"
    assert SpanKind.EMBEDDING.value == "EMBEDDING"
    assert SpanKind.AGENT.value == "AGENT"
    assert SpanKind.CHAIN.value == "CHAIN"


def test_set_span_kind_stamps_kind_and_operation(exported_spans: InMemorySpanExporter) -> None:
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("s") as span:
        set_span_kind(span, SpanKind.LLM)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs[OPENINFERENCE_SPAN_KIND] == "LLM"
    assert attrs[GEN_AI_OPERATION_NAME] == "chat"


def test_set_span_kind_without_operation(exported_spans: InMemorySpanExporter) -> None:
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("s") as span:
        set_span_kind(span, SpanKind.CHAIN)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs[OPENINFERENCE_SPAN_KIND] == "CHAIN"
    assert GEN_AI_OPERATION_NAME not in attrs


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
def test_otel_span_kind_follows_the_genai_conventions(
    kind: SpanKind, expected: OtelSpanKind
) -> None:
    """Inference, embeddings and retrieval cross a process boundary (CLIENT);
    an in-process tool, agent or step does not (INTERNAL)."""
    assert otel_span_kind(kind) is expected


def test_every_taxonomy_kind_has_an_otel_kind() -> None:
    for kind in SpanKind:
        assert isinstance(otel_span_kind(kind), OtelSpanKind)


# --- RETRIEVER conformance: the retrieval operation and gen_ai.data_source.id ---


def test_retriever_maps_to_the_retrieval_operation() -> None:
    from rius.semconv import kind_attributes

    assert kind_attributes(SpanKind.RETRIEVER)[GEN_AI_OPERATION_NAME] == "retrieval"


def test_chain_stays_unmapped() -> None:
    """The conventions define no operation for a generic step, so we emit none."""
    from rius.semconv import kind_attributes

    assert GEN_AI_OPERATION_NAME not in kind_attributes(SpanKind.CHAIN)


def test_data_source_id_is_set_only_on_retriever_spans() -> None:
    from rius.semconv import GEN_AI_DATA_SOURCE_ID, kind_attributes

    assert GEN_AI_DATA_SOURCE_ID == "gen_ai.data_source.id"
    attributes = kind_attributes(SpanKind.RETRIEVER, data_source_id="docs-index")
    assert attributes[GEN_AI_DATA_SOURCE_ID] == "docs-index"
    assert GEN_AI_DATA_SOURCE_ID not in kind_attributes(SpanKind.RETRIEVER)
    assert GEN_AI_DATA_SOURCE_ID not in kind_attributes(SpanKind.TOOL, data_source_id="docs-index")


def test_data_source_id_is_pending_identity_not_content() -> None:
    from rius.semconv import (
        CONTENT_ATTRIBUTES,
        GEN_AI_DATA_SOURCE_ID,
        PENDING_IDENTITY_ATTRIBUTES,
    )

    assert GEN_AI_DATA_SOURCE_ID in PENDING_IDENTITY_ATTRIBUTES
    assert GEN_AI_DATA_SOURCE_ID not in CONTENT_ATTRIBUTES


def test_top_k_is_request_identity_and_documents_are_result_metadata() -> None:
    from rius.semconv import (
        CONTENT_ATTRIBUTES,
        GEN_AI_RETRIEVAL_DOCUMENTS,
        GEN_AI_RETRIEVAL_TOP_K,
        PENDING_IDENTITY_ATTRIBUTES,
    )

    # top_k describes the request, so it is known before the search runs.
    assert GEN_AI_RETRIEVAL_TOP_K in PENDING_IDENTITY_ATTRIBUTES
    assert GEN_AI_RETRIEVAL_TOP_K not in CONTENT_ATTRIBUTES

    # The documents describe the result: unknown at span start, so never on a
    # snapshot. Not content either, because the conventions define the entries
    # as ids and scores rather than document text, and do not mark the
    # attribute sensitive. It must therefore survive capture_content=False.
    assert GEN_AI_RETRIEVAL_DOCUMENTS not in PENDING_IDENTITY_ATTRIBUTES
    assert GEN_AI_RETRIEVAL_DOCUMENTS not in CONTENT_ATTRIBUTES


def test_kind_attributes_takes_a_tool_name_not_a_span_name() -> None:
    from rius.semconv import GEN_AI_TOOL_NAME, kind_attributes

    assert kind_attributes(SpanKind.TOOL, "get_weather")[GEN_AI_TOOL_NAME] == "get_weather"
    # Only TOOL spans carry it, and only when a tool name is given.
    assert GEN_AI_TOOL_NAME not in kind_attributes(SpanKind.TOOL)
    assert GEN_AI_TOOL_NAME not in kind_attributes(SpanKind.CHAIN, "get_weather")


def test_context_sizes_is_not_content_and_not_pending_identity() -> None:
    from rius.semconv import CONTENT_ATTRIBUTES, PENDING_IDENTITY_ATTRIBUTES, RIUS_CONTEXT_SIZES

    assert RIUS_CONTEXT_SIZES == "rius.context.sizes"
    assert RIUS_CONTEXT_SIZES not in CONTENT_ATTRIBUTES
    assert RIUS_CONTEXT_SIZES not in PENDING_IDENTITY_ATTRIBUTES


def test_kind_attributes_sets_the_agent_name_only_on_agent_spans() -> None:
    from rius.semconv import GEN_AI_AGENT_ID, GEN_AI_AGENT_NAME, kind_attributes

    attributes = kind_attributes(SpanKind.AGENT, agent_name="planner", agent_id="ag_1")
    assert attributes[GEN_AI_AGENT_NAME] == "planner"
    assert attributes[GEN_AI_AGENT_ID] == "ag_1"

    # Never invented, and meaningless on every other kind.
    assert GEN_AI_AGENT_NAME not in kind_attributes(SpanKind.AGENT)
    assert GEN_AI_AGENT_NAME not in kind_attributes(SpanKind.CHAIN, agent_name="planner")
    assert GEN_AI_AGENT_ID not in kind_attributes(SpanKind.LLM, agent_id="ag_1")


def test_agent_identity_is_pending_allowlisted_and_not_content() -> None:
    """Which agent a span invokes is known at start and is identity, so a
    still-running agent must be attributable in the live view."""
    from rius.semconv import (
        CONTENT_ATTRIBUTES,
        GEN_AI_AGENT_ID,
        GEN_AI_AGENT_NAME,
        PENDING_IDENTITY_ATTRIBUTES,
    )

    for key in (GEN_AI_AGENT_NAME, GEN_AI_AGENT_ID):
        assert key in PENDING_IDENTITY_ATTRIBUTES
        assert key not in CONTENT_ATTRIBUTES


# --- default span names: "{operation} {target}", per the GenAI conventions ---


def test_compose_span_name_composes_the_operation_and_the_target() -> None:
    """The conventions' SHOULD for every operation they define: the operation,
    then the one identifier that says which model/tool/agent/index it hit."""
    from rius.semconv import compose_span_name

    cases = [
        (SpanKind.LLM, {"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4o"}),
        (
            SpanKind.EMBEDDING,
            {
                "gen_ai.operation.name": "embeddings",
                "gen_ai.request.model": "text-embedding-3-small",
            },
        ),
        (
            SpanKind.TOOL,
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_weather"},
        ),
        (
            SpanKind.AGENT,
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "planner"},
        ),
        (
            SpanKind.RETRIEVER,
            {"gen_ai.operation.name": "retrieval", "gen_ai.data_source.id": "kb"},
        ),
    ]
    expected = [
        "chat gpt-4o",
        "embeddings text-embedding-3-small",
        "execute_tool get_weather",
        "invoke_agent planner",
        "retrieval kb",
    ]
    assert [compose_span_name(kind, attrs) for kind, attrs in cases] == expected


def test_compose_span_name_composes_from_what_the_span_actually_carries() -> None:
    """Reading the creation attributes rather than the caller's arguments is
    what makes the name and the attributes agree by construction. It is also
    why an operation overridden per generation is picked up for free: the
    override is already in the map."""
    from rius.semconv import compose_span_name, kind_attributes

    # Exactly the map kind_attributes builds, so this is the real input.
    attributes = kind_attributes(SpanKind.TOOL, "get_weather")
    assert compose_span_name(SpanKind.TOOL, attributes) == "execute_tool get_weather"

    overridden = {"gen_ai.operation.name": "embeddings", "gen_ai.request.model": "m"}
    assert compose_span_name(SpanKind.LLM, overridden) == "embeddings m"


def test_compose_span_name_falls_back_to_the_bare_operation() -> None:
    """The identifier is optional on every kind, so the name degrades to the
    operation alone rather than to a name with a hole in it."""
    from rius.semconv import compose_span_name

    for kind, operation in (
        (SpanKind.LLM, "chat"),
        (SpanKind.EMBEDDING, "embeddings"),
        (SpanKind.TOOL, "execute_tool"),
        (SpanKind.AGENT, "invoke_agent"),
        (SpanKind.RETRIEVER, "retrieval"),
    ):
        assert compose_span_name(kind, {"gen_ai.operation.name": operation}) == operation


def test_compose_span_name_of_a_chain_is_the_degenerate_literal() -> None:
    """CHAIN has no operation in the conventions and the manual helpers have
    no function to borrow a qualname from, so the name is the bare literal."""
    from rius.semconv import compose_span_name, kind_attributes

    assert compose_span_name(SpanKind.CHAIN, kind_attributes(SpanKind.CHAIN)) == "chain"


def test_compose_span_name_ignores_targets_from_another_kind() -> None:
    """Each kind reads exactly one identifier; a tool name never leaks into an
    agent's name, which is what makes the composed name a reliable filter."""
    from rius.semconv import compose_span_name

    assert (
        compose_span_name(
            SpanKind.AGENT,
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.tool.name": "get_weather"},
        )
        == "invoke_agent"
    )
    assert (
        compose_span_name(
            SpanKind.TOOL,
            {"gen_ai.operation.name": "execute_tool", "gen_ai.agent.name": "planner"},
        )
        == "execute_tool"
    )
