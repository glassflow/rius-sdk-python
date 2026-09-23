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


# --- the generation conformance keys: gen_ai.output.type / gen_ai.response.id ---


def test_output_type_is_creation_identity_and_response_id_is_metadata() -> None:
    """The requested output type is part of the request, so it is known before
    the call and rides pending snapshots. The provider's completion id is not
    knowable until the call returns, so it never reaches one. Neither is
    content: both survive capture_content=False."""
    from rius.semconv import (
        CONTENT_ATTRIBUTES,
        GEN_AI_OUTPUT_TYPE,
        GEN_AI_RESPONSE_ID,
        PENDING_IDENTITY_ATTRIBUTES,
        PENDING_IDENTITY_PREFIXES,
    )

    assert GEN_AI_OUTPUT_TYPE == "gen_ai.output.type"
    assert GEN_AI_RESPONSE_ID == "gen_ai.response.id"

    # Neither key sits under gen_ai.request., so the prefix rule does not
    # cover the output type; it needs its own allowlist entry.
    assert not GEN_AI_OUTPUT_TYPE.startswith(PENDING_IDENTITY_PREFIXES)
    assert GEN_AI_OUTPUT_TYPE in PENDING_IDENTITY_ATTRIBUTES
    assert GEN_AI_RESPONSE_ID not in PENDING_IDENTITY_ATTRIBUTES

    assert GEN_AI_OUTPUT_TYPE not in CONTENT_ATTRIBUTES
    assert GEN_AI_RESPONSE_ID not in CONTENT_ATTRIBUTES


# --- the tool conformance keys: gen_ai.tool.call.id / gen_ai.tool.type ---


def test_tool_call_id_and_tool_type_are_set_only_on_tool_spans() -> None:
    from rius.semconv import GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_TYPE, kind_attributes

    assert GEN_AI_TOOL_CALL_ID == "gen_ai.tool.call.id"
    assert GEN_AI_TOOL_TYPE == "gen_ai.tool.type"

    attributes = kind_attributes(
        SpanKind.TOOL, "get_weather", tool_call_id="call_1", tool_type="function"
    )
    assert attributes[GEN_AI_TOOL_CALL_ID] == "call_1"
    assert attributes[GEN_AI_TOOL_TYPE] == "function"

    # Never invented, and meaningless on every other kind — the same scoping
    # gen_ai.tool.name and the agent keys already use.
    assert GEN_AI_TOOL_CALL_ID not in kind_attributes(SpanKind.TOOL, "get_weather")
    assert GEN_AI_TOOL_TYPE not in kind_attributes(SpanKind.TOOL, "get_weather")
    for kind in (SpanKind.CHAIN, SpanKind.LLM, SpanKind.AGENT, SpanKind.RETRIEVER):
        assert GEN_AI_TOOL_CALL_ID not in kind_attributes(kind, tool_call_id="call_1")
        assert GEN_AI_TOOL_TYPE not in kind_attributes(kind, tool_type="function")


def test_tool_call_id_and_tool_type_are_pending_identity_not_content() -> None:
    """Both are caller-supplied at span creation, so a still-running tool call
    is already attributable to the model's tool-call message in the live view."""
    from rius.semconv import (
        CONTENT_ATTRIBUTES,
        GEN_AI_TOOL_CALL_ID,
        GEN_AI_TOOL_TYPE,
        PENDING_IDENTITY_ATTRIBUTES,
    )

    for key in (GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_TYPE):
        assert key in PENDING_IDENTITY_ATTRIBUTES
        assert key not in CONTENT_ATTRIBUTES


def test_the_tool_identifiers_never_reach_the_span_name() -> None:
    """A TOOL span is named after its tool, never after the call id or the
    tool type: _NAME_TARGET_BY_KIND maps TOOL to gen_ai.tool.name alone."""
    from rius.semconv import compose_span_name, kind_attributes

    attributes = kind_attributes(
        SpanKind.TOOL, "get_weather", tool_call_id="call_1", tool_type="function"
    )
    assert compose_span_name(SpanKind.TOOL, attributes) == "execute_tool get_weather"

    # And a tool span with no tool name degrades to the bare operation rather
    # than borrowing either identifier.
    nameless = kind_attributes(SpanKind.TOOL, tool_call_id="call_1", tool_type="function")
    assert compose_span_name(SpanKind.TOOL, nameless) == "execute_tool"


def test_kind_attributes_sets_the_agent_version_only_on_agent_spans() -> None:
    """gen_ai.agent.version describes the agent this span INVOKED, the same
    callee gen_ai.agent.name and .id describe. No other kind has a callee."""
    from rius.semconv import GEN_AI_AGENT_VERSION, kind_attributes

    assert (
        kind_attributes(SpanKind.AGENT, agent_version="2025-05-01")[GEN_AI_AGENT_VERSION]
        == "2025-05-01"
    )

    assert GEN_AI_AGENT_VERSION not in kind_attributes(SpanKind.AGENT)
    assert GEN_AI_AGENT_VERSION not in kind_attributes(SpanKind.CHAIN, agent_version="1.0.0")
    assert GEN_AI_AGENT_VERSION not in kind_attributes(SpanKind.TOOL, agent_version="1.0.0")
    assert GEN_AI_AGENT_VERSION not in kind_attributes(SpanKind.LLM, agent_version="1.0.0")


def test_the_agent_version_never_reaches_the_span_name() -> None:
    """_NAME_TARGET_BY_KIND maps AGENT to the agent NAME alone."""
    from rius.semconv import compose_span_name, kind_attributes

    attributes = kind_attributes(SpanKind.AGENT, agent_name="planner", agent_version="7")
    assert compose_span_name(SpanKind.AGENT, attributes) == "invoke_agent planner"


def test_the_agent_version_is_pending_identity_not_content() -> None:
    """Set at creation, so a still-running agent span must carry it into the
    pending snapshot; the allowlist is what lets it through."""
    from rius.semconv import (
        CONTENT_ATTRIBUTES,
        GEN_AI_AGENT_VERSION,
        PENDING_IDENTITY_ATTRIBUTES,
    )

    assert GEN_AI_AGENT_VERSION in PENDING_IDENTITY_ATTRIBUTES
    assert GEN_AI_AGENT_VERSION not in CONTENT_ATTRIBUTES


def test_the_main_agent_resource_keys_are_neither_span_identity_nor_content() -> None:
    """They ride the Resource once per batch and are never set on a span, so
    they belong to neither span-level set."""
    from rius.semconv import (
        CONTENT_ATTRIBUTES,
        PENDING_IDENTITY_ATTRIBUTES,
        RIUS_MAIN_AGENT_DESCRIPTION,
        RIUS_MAIN_AGENT_ID,
        RIUS_MAIN_AGENT_NAME,
        RIUS_MAIN_AGENT_VERSION,
    )

    for key in (
        RIUS_MAIN_AGENT_NAME,
        RIUS_MAIN_AGENT_ID,
        RIUS_MAIN_AGENT_DESCRIPTION,
        RIUS_MAIN_AGENT_VERSION,
    ):
        assert key.startswith("rius.main_agent.")
        assert key not in PENDING_IDENTITY_ATTRIBUTES
        assert key not in CONTENT_ATTRIBUTES


# --- request-parameter namespaces (RIUS-926) ---


def test_rius_request_prefix_rides_pending_snapshots() -> None:
    """Caller parameters are identity, set at creation, exactly like the
    spec-defined ones; the live view must show them the same way."""
    from rius.semconv import PENDING_IDENTITY_PREFIXES, RIUS_REQUEST_PREFIX

    assert RIUS_REQUEST_PREFIX in PENDING_IDENTITY_PREFIXES


def test_request_namespaces_are_tied_for_masking() -> None:
    """RIUS-917: rius.request.* follows gen_ai.request.* exactly. Neither is
    content today. If one is ever added to the allowlist, this fails and
    makes the other's absence the explicit decision it has to be."""
    from rius.semconv import (
        CONTENT_ATTRIBUTE_PREFIXES,
        CONTENT_ATTRIBUTES,
        GEN_AI_REQUEST_PREFIX,
        RIUS_REQUEST_PREFIX,
    )

    gen_ai_is_content = (
        GEN_AI_REQUEST_PREFIX in CONTENT_ATTRIBUTE_PREFIXES
        or GEN_AI_REQUEST_PREFIX.rstrip(".") in CONTENT_ATTRIBUTES
    )
    rius_is_content = (
        RIUS_REQUEST_PREFIX in CONTENT_ATTRIBUTE_PREFIXES
        or RIUS_REQUEST_PREFIX.rstrip(".") in CONTENT_ATTRIBUTES
    )
    assert gen_ai_is_content is rius_is_content, (
        "gen_ai.request.* and rius.request.* must move together; "
        "neither may join the content allowlist alone"
    )
    assert not gen_ai_is_content, "today neither namespace is content"


def test_recognised_parameters_map_only_onto_spec_defined_keys() -> None:
    """Every normalisation target must be an attribute the GenAI registry
    actually defines. Pinned to open-telemetry/semantic-conventions-genai
    at commit 8ffdf568e1b4391a99adb081db16e8102e36918e (2026-09-22),
    model/gen-ai/registry.yaml; the repo cuts no releases."""
    from rius.semconv import GEN_AI_REQUEST_PARAMETERS

    spec_defined = {
        "gen_ai.request.model",
        "gen_ai.request.max_tokens",
        "gen_ai.request.choice.count",
        "gen_ai.request.temperature",
        "gen_ai.request.top_p",
        "gen_ai.request.top_k",
        "gen_ai.request.stop_sequences",
        "gen_ai.request.frequency_penalty",
        "gen_ai.request.presence_penalty",
        "gen_ai.request.encoding_formats",
        "gen_ai.request.seed",
        "gen_ai.request.stream",
        "gen_ai.request.reasoning.level",
        "gen_ai.request.previous_response.id",
        "gen_ai.request.stream_cursor",
    }
    assert set(GEN_AI_REQUEST_PARAMETERS.values()) <= spec_defined
    # Every canonical key is reachable under its own spelling.
    for canonical in set(GEN_AI_REQUEST_PARAMETERS.values()):
        tail = canonical[len("gen_ai.request.") :]
        assert GEN_AI_REQUEST_PARAMETERS.get(tail) == canonical


def test_tool_definition_members_are_content_in_both_request_namespaces() -> None:
    """The named exception to the RIUS-917 blanket rule. The member list is
    shared with llm.invocation_parameters redaction so the routes to the same
    tool definitions cannot drift apart, and it applies symmetrically: a
    caller can pass `tools` and reach either namespace."""
    from rius.semconv import (
        CONTENT_ATTRIBUTES,
        GEN_AI_REQUEST_PREFIX,
        INVOCATION_PARAMETERS_CONTENT_MEMBERS,
        RIUS_REQUEST_PREFIX,
    )

    for member in INVOCATION_PARAMETERS_CONTENT_MEMBERS:
        assert f"{GEN_AI_REQUEST_PREFIX}{member}" in CONTENT_ATTRIBUTES
        assert f"{RIUS_REQUEST_PREFIX}{member}" in CONTENT_ATTRIBUTES
