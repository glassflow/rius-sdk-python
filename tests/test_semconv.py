from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius.semconv import (
    GEN_AI_OPERATION_NAME,
    OPENINFERENCE_SPAN_KIND,
    SpanKind,
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
