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
