import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
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
