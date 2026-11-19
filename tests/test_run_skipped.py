"""``mark_run_skipped``: a run that had nothing to do says so on its agent span.

The empty-run detector (CF-05) reads ``rius.run.skipped_reason`` on the run's
root agent span; a marked run is an info finding, never a warning.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rius
from rius import _tracer
from rius.masking import MaskingSpanExporter
from rius.run import MAX_SKIPPED_REASON_CHARS, RUN_SKIPPED_REASON
from rius.semconv import SpanKind


class _Pipeline:
    def __init__(self, *, capture_content: bool = True) -> None:
        self.exporter = InMemorySpanExporter()
        sink: Any = self.exporter
        if not capture_content:
            sink = MaskingSpanExporter(sink, capture_content=False)
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(sink))
        _tracer.publish(self.provider)

    def named(self, name: str) -> ReadableSpan:
        (span,) = [s for s in self.exporter.get_finished_spans() if s.name == name]
        return span


@pytest.fixture
def pipeline() -> Iterator[_Pipeline]:
    yield _Pipeline()


def _reason(span: ReadableSpan) -> Any:
    return dict(span.attributes or {}).get(RUN_SKIPPED_REASON)


def test_marks_the_current_agent_span(pipeline: _Pipeline) -> None:
    with rius.start_as_current_span("triage", kind=SpanKind.AGENT):
        assert rius.mark_run_skipped("no new tickets") is True
    assert _reason(pipeline.named("triage")) == "no new tickets"


def test_marks_an_observed_agent(pipeline: _Pipeline) -> None:
    @rius.observe(name="nightly", kind=SpanKind.AGENT)
    def nightly() -> None:
        rius.mark_run_skipped("  input   already\nprocessed ")

    nightly()
    assert _reason(pipeline.named("nightly")) == "input already processed"


def test_observation_marks_its_own_span(pipeline: _Pipeline) -> None:
    with (
        rius.start_as_current_span("run", kind=SpanKind.AGENT) as run,
        rius.start_as_current_span("step"),
    ):
        assert run.mark_skipped("guard said no") is True
    assert _reason(pipeline.named("run")) == "guard said no"
    assert _reason(pipeline.named("step")) is None


def test_a_non_agent_span_is_not_marked(pipeline: _Pipeline) -> None:
    with rius.start_as_current_span("step"):
        assert rius.mark_run_skipped("nothing to do") is False
    assert _reason(pipeline.named("step")) is None


def test_no_span_is_not_an_error(pipeline: _Pipeline) -> None:
    assert rius.mark_run_skipped("nothing to do") is False


@pytest.mark.parametrize("reason", ["", "   ", "\n\t"])
def test_an_empty_reason_is_refused(pipeline: _Pipeline, reason: str) -> None:
    with (
        rius.start_as_current_span("run", kind=SpanKind.AGENT),
        pytest.raises(ValueError, match="must not be empty"),
    ):
        rius.mark_run_skipped(reason)
    assert _reason(pipeline.named("run")) is None


def test_a_non_string_reason_is_refused(pipeline: _Pipeline) -> None:
    with rius.start_as_current_span("run", kind=SpanKind.AGENT), pytest.raises(TypeError):
        rius.mark_run_skipped(None)  # type: ignore[arg-type]


def test_an_oversized_reason_is_cut(pipeline: _Pipeline) -> None:
    with rius.start_as_current_span("run", kind=SpanKind.AGENT):
        rius.mark_run_skipped("x" * 5000)
    assert _reason(pipeline.named("run")) == "x" * MAX_SKIPPED_REASON_CHARS


def test_the_marker_survives_content_masking() -> None:
    pipeline = _Pipeline(capture_content=False)
    with rius.start_as_current_span("run", kind=SpanKind.AGENT):
        rius.mark_run_skipped("no new tickets")
    assert _reason(pipeline.named("run")) == "no new tickets"
