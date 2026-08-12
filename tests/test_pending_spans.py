"""Partial (pending) spans: a content-free snapshot exported at span start.

Wire contract under test: same trace/span/parent ids, same name and start
timestamp as the final span; zero duration; the pending marker attribute;
identity/taxonomy attributes only, never content.
"""

from __future__ import annotations

from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.semconv import GLASSFLOW_SPAN_PENDING


def _memory_client(**kwargs: object):
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        **kwargs,  # type: ignore[arg-type]
    )
    return client, exporter


def _split(spans):
    pending = [s for s in spans if s.attributes.get(GLASSFLOW_SPAN_PENDING)]
    final = [s for s in spans if not s.attributes.get(GLASSFLOW_SPAN_PENDING)]
    return pending, final


def test_flag_off_by_default_behavior_unchanged() -> None:
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert GLASSFLOW_SPAN_PENDING not in spans[0].attributes


def test_pending_snapshot_mirrors_identity_of_final_span() -> None:
    client, exporter = _memory_client(partial_spans=True)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    (pending,), (final,) = _split(exporter.get_finished_spans())

    assert pending.attributes[GLASSFLOW_SPAN_PENDING] is True
    # identical identity: the ClickHouse sort key must match for replacement
    assert pending.context.trace_id == final.context.trace_id
    assert pending.context.span_id == final.context.span_id
    assert pending.name == final.name
    assert pending.start_time == final.start_time
    # zero duration: OTLP cannot represent an unfinished span
    assert pending.end_time == pending.start_time
    assert final.end_time > final.start_time


def test_pending_preserves_parent_linkage() -> None:
    client, exporter = _memory_client(partial_spans=True)
    tracer = client.get_tracer()
    with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
        pass
    client.flush()
    pending, final = _split(exporter.get_finished_spans())
    pending_child = next(s for s in pending if s.name == "child")
    final_child = next(s for s in final if s.name == "child")
    assert pending_child.parent is not None
    assert pending_child.parent.span_id == final_child.parent.span_id


def test_pending_carries_identity_attributes_but_never_content() -> None:
    client, exporter = _memory_client(partial_spans=True)
    with client.get_tracer().start_as_current_span(
        "chat",
        attributes={
            "openinference.span.kind": "LLM",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.request.temperature": 0.2,
            "input.value": "SECRET-CONTENT",
        },
    ):
        pass
    client.flush()
    (pending,), (final,) = _split(exporter.get_finished_spans())
    assert pending.attributes["openinference.span.kind"] == "LLM"
    assert pending.attributes["gen_ai.request.model"] == "gpt-4o"
    assert pending.attributes["gen_ai.request.temperature"] == 0.2
    assert "input.value" not in pending.attributes, "content must NEVER ride a pending span"
    assert final.attributes["input.value"] == "SECRET-CONTENT"


def test_sampled_out_spans_produce_no_pending() -> None:
    client, exporter = _memory_client(partial_spans=True, sample_rate=0.0)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans() == ()


def test_disabled_kills_pendings_too() -> None:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter, set_global=False, disabled=True, partial_spans=True, instruments=[]
    )
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans() == ()


def test_env_var_enables_partial_spans(monkeypatch) -> None:
    from rius.config import resolve_config

    monkeypatch.setenv("GLASSFLOW_PARTIAL_SPANS", "true")
    assert resolve_config().partial_spans is True
    # explicit argument wins over the environment
    assert resolve_config(partial_spans=False).partial_spans is False


class _StartAttributeRecorder(_SpanProcessor):
    """Span processor recording each span's attributes as seen at on_start."""

    def __init__(self) -> None:
        self.seen: dict[str, dict] = {}

    def on_start(self, span, parent_context=None) -> None:  # noqa: ANN001
        self.seen[span.name] = dict(span.attributes or {})

    def on_end(self, span) -> None:  # noqa: ANN001
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def test_sdk_helpers_expose_identity_attributes_at_span_start() -> None:
    """Pending snapshots are built at on_start, so the SDK's own APIs must
    attach kind/model/provider at CREATION — set_attribute after the fact is
    invisible to the snapshot."""
    from opentelemetry import trace as otel_trace

    import rius

    recorder = _StartAttributeRecorder()
    otel_trace.get_tracer_provider().add_span_processor(recorder)  # type: ignore[attr-defined]

    with rius.start_as_current_span("kindly", kind=rius.SpanKind.RETRIEVER):
        pass
    with rius.start_as_current_generation("genny", model="gpt-4o", provider="openai"):
        pass

    assert recorder.seen["kindly"]["openinference.span.kind"] == "RETRIEVER"
    genny = recorder.seen["genny"]
    assert genny["openinference.span.kind"] == "LLM"
    assert genny["gen_ai.operation.name"] == "chat"
    assert genny["gen_ai.request.model"] == "gpt-4o"
    assert genny["gen_ai.provider.name"] == "openai"
