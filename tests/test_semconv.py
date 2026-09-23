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
    attributes = kind_attributes(SpanKind.RETRIEVER, None, "docs-index")
    assert attributes[GEN_AI_DATA_SOURCE_ID] == "docs-index"
    assert GEN_AI_DATA_SOURCE_ID not in kind_attributes(SpanKind.RETRIEVER)
    assert GEN_AI_DATA_SOURCE_ID not in kind_attributes(SpanKind.TOOL, None, "docs-index")


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


def test_context_sizes_is_not_content_and_not_pending_identity() -> None:
    from rius.semconv import CONTENT_ATTRIBUTES, PENDING_IDENTITY_ATTRIBUTES, RIUS_CONTEXT_SIZES

    assert RIUS_CONTEXT_SIZES == "rius.context.sizes"
    assert RIUS_CONTEXT_SIZES not in CONTENT_ATTRIBUTES
    assert RIUS_CONTEXT_SIZES not in PENDING_IDENTITY_ATTRIBUTES
