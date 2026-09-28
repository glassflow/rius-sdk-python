"""Another SDK's tracer provider holding the OpenTelemetry global (RIUS-1070).

The global provider is write-once per process and the suite's conftest has
already claimed it, so each test installs its own stand-in: a write-once global
the client reads and sets through ``opentelemetry.trace``, the same seam the
lifecycle tests patch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import rius
from rius.config import resolve_config
from rius.foreign import ForeignParentDetector
from rius.semconv import RIUS_PARENT_FOREIGN, RIUS_SDK_GLOBAL_PROVIDER, SERVICE_INSTANCE_ID

InstallGlobal = Callable[[trace.TracerProvider], None]


class _RecordingProvider(TracerProvider):
    """A foreign SDK provider that records what is done to it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.added: list[SpanProcessor] = []

    def add_span_processor(self, span_processor: SpanProcessor) -> None:
        self.added.append(span_processor)
        super().add_span_processor(span_processor)


class _LifecycleSpy(SpanProcessor):
    """The foreign SDK's own processor; rius must never shut it down or flush it."""

    def __init__(self) -> None:
        self.shutdowns = 0
        self.flushes = 0

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        self.shutdowns += 1

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self.flushes += 1
        return True


@pytest.fixture
def install_global(monkeypatch: pytest.MonkeyPatch) -> Iterator[InstallGlobal]:
    holder: list[trace.TracerProvider] = [trace.ProxyTracerProvider()]

    def set_once(provider: trace.TracerProvider) -> None:
        if isinstance(holder[0], trace.ProxyTracerProvider):
            holder[0] = provider

    def install(provider: trace.TracerProvider) -> None:
        holder[0] = provider

    monkeypatch.setattr("rius.client.trace.get_tracer_provider", lambda: holder[0])
    monkeypatch.setattr("rius.client.trace.set_tracer_provider", set_once)
    yield install


@pytest.fixture
def foreign(install_global: InstallGlobal) -> tuple[_RecordingProvider, InMemorySpanExporter]:
    provider = _RecordingProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    install_global(provider)
    return provider, exporter


def _init(exporter: InMemorySpanExporter, **kwargs: Any) -> rius.GlassflowClient:
    return rius.init(
        span_exporter=exporter,
        service_name="rius-svc",
        instruments=[],
        **kwargs,
    )


def _job_with_llm_child(parent: trace.TracerProvider, client: rius.GlassflowClient) -> None:
    """The RIUS-1070 shape: an LLM span on rius's provider inside a job span of another."""
    with parent.get_tracer("customer.jobs").start_as_current_span("job"):
        client.get_tracer().start_span("llm").end()


def _by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {span.name: span for span in exporter.get_finished_spans()}


def _flagged(exporter: InMemorySpanExporter) -> list[str]:
    return [
        span.name
        for span in exporter.get_finished_spans()
        if (span.attributes or {}).get(RIUS_PARENT_FOREIGN) is True
    ]


# --- the bridge ---------------------------------------------------------------


def test_bridge_exports_the_foreign_parent_so_the_tree_is_whole(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    client = _init(exporter)
    _job_with_llm_child(provider, client)
    client.flush()

    spans = _by_name(exporter)
    assert set(spans) == {"job", "llm"}
    assert spans["llm"].parent is not None
    assert spans["llm"].parent.span_id == spans["job"].context.span_id
    assert spans["job"].parent is None
    assert _flagged(exporter) == []


def test_bridge_leaves_the_foreign_providers_own_export_alone(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, foreign_exporter = foreign
    client = _init(InMemorySpanExporter())
    _job_with_llm_child(provider, client)
    assert [span.name for span in foreign_exporter.get_finished_spans()] == ["job"]


def test_bridged_spans_are_exported_under_the_rius_resource_identity(
    install_global: InstallGlobal,
) -> None:
    provider = TracerProvider(
        resource=Resource.create({"service.name": "langfuse-app", "deployment.environment": "x"})
    )
    install_global(provider)
    exporter = InMemorySpanExporter()
    client = _init(exporter, agent_name="checkout-agent")
    _job_with_llm_child(provider, client)
    client.flush()

    spans = _by_name(exporter)
    job, llm = spans["job"].resource.attributes, spans["llm"].resource.attributes
    for key in ("service.name", SERVICE_INSTANCE_ID, "gen_ai.agent.name", RIUS_SDK_GLOBAL_PROVIDER):
        assert job[key] == llm[key]
    assert job["service.name"] == "rius-svc"
    assert job["deployment.environment"] == "x"


def test_rius_spans_keep_their_own_resource_object(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    exporter = InMemorySpanExporter()
    client = _init(exporter)
    client.get_tracer().start_span("own").end()
    client.flush()
    assert exporter.get_finished_spans()[0].resource is client._provider.resource


def test_opt_out_keeps_rius_to_its_own_provider_and_flags_the_orphan(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    with caplog.at_level(logging.WARNING, logger="rius.client"):
        client = _init(exporter, bridge_foreign_provider=False)
    _job_with_llm_child(provider, client)
    client.flush()

    assert set(_by_name(exporter)) == {"llm"}
    assert _flagged(exporter) == ["llm"]
    assert any("bridge_foreign_provider is off" in r.message for r in caplog.records)


def test_opt_out_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_BRIDGE_FOREIGN_PROVIDER", "false")
    assert resolve_config().bridge_foreign_provider is False
    assert resolve_config(bridge_foreign_provider=True).bridge_foreign_provider is True


def test_bridge_is_on_by_default() -> None:
    assert resolve_config().bridge_foreign_provider is True


def test_shutdown_makes_the_bridge_inert_without_touching_foreign_processors(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, foreign_exporter = foreign
    spy = _LifecycleSpy()
    provider.add_span_processor(spy)
    client = _init(InMemorySpanExporter(), session_id="session-1")
    client.shutdown()

    provider.get_tracer("customer.jobs").start_span("after-shutdown").end()

    # A live pipeline would have stamped the session on the span it saw start.
    [span] = foreign_exporter.get_finished_spans()
    assert span.name == "after-shutdown"
    assert "session.id" not in (span.attributes or {})
    assert (spy.shutdowns, spy.flushes) == (0, 0)


def test_foreign_shutdown_does_not_shut_down_the_rius_pipeline(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    client = _init(exporter)
    provider.shutdown()
    client.get_tracer().start_span("still-alive").end()
    client.flush()
    assert [s.name for s in exporter.get_finished_spans()] == ["still-alive"]


def test_reinit_reuses_the_one_bridge_instead_of_attaching_another(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    attached_by_fixture = len(provider.added)
    _init(InMemorySpanExporter()).shutdown()
    second_exporter = InMemorySpanExporter()
    second = _init(second_exporter)
    _job_with_llm_child(provider, second)
    second.flush()

    assert len(provider.added) == attached_by_fixture + 1
    assert sorted(s.name for s in second_exporter.get_finished_spans()) == ["job", "llm"]


def test_a_non_sdk_global_is_not_bridged_and_does_not_crash(
    install_global: InstallGlobal, caplog: pytest.LogCaptureFixture
) -> None:
    install_global(trace.NoOpTracerProvider())
    exporter = InMemorySpanExporter()
    with caplog.at_level(logging.WARNING, logger="rius.client"):
        client = _init(exporter)
    client.get_tracer().start_span("own").end()
    client.flush()

    assert [s.name for s in exporter.get_finished_spans()] == ["own"]
    assert (
        exporter.get_finished_spans()[0].resource.attributes[RIUS_SDK_GLOBAL_PROVIDER]
        == "foreign:opentelemetry.trace.NoOpTracerProvider"
    )
    assert any("not an OpenTelemetry SDK TracerProvider" in r.message for r in caplog.records)


# --- the resource records the conflict ------------------------------------------


def test_resource_names_the_foreign_global_provider(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    exporter = InMemorySpanExporter()
    client = _init(exporter)
    client.get_tracer().start_span("own").end()
    client.flush()
    assert exporter.get_finished_spans()[0].resource.attributes[RIUS_SDK_GLOBAL_PROVIDER] == (
        f"foreign:{__name__}._RecordingProvider"
    )


def test_no_conflict_is_recorded_when_rius_claims_the_global(
    install_global: InstallGlobal,
) -> None:
    exporter = InMemorySpanExporter()
    client = _init(exporter)
    assert trace.get_tracer_provider() is client._provider
    assert RIUS_SDK_GLOBAL_PROVIDER not in client._provider.resource.attributes


def test_an_earlier_rius_provider_holding_the_global_is_not_foreign(
    install_global: InstallGlobal,
) -> None:
    _init(InMemorySpanExporter()).shutdown()
    second = _init(InMemorySpanExporter())
    assert RIUS_SDK_GLOBAL_PROVIDER not in second._provider.resource.attributes


def test_a_scoped_client_records_no_conflict(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    scoped = _init(InMemorySpanExporter(), set_global=False)
    assert RIUS_SDK_GLOBAL_PROVIDER not in scoped._provider.resource.attributes


# --- detection ------------------------------------------------------------------


def test_a_remote_parent_is_not_flagged(install_global: InstallGlobal) -> None:
    exporter = InMemorySpanExporter()
    client = _init(exporter)
    carrier = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
    remote = TraceContextTextMapPropagator().extract(carrier)
    client.get_tracer().start_span("server", context=remote).end()
    client.flush()

    span = exporter.get_finished_spans()[0]
    assert span.parent is not None and span.parent.is_remote
    assert _flagged(exporter) == []


def test_an_own_parent_is_not_flagged_even_after_it_ended(install_global: InstallGlobal) -> None:
    exporter = InMemorySpanExporter()
    client = _init(exporter)
    tracer = client.get_tracer()
    parent = tracer.start_span("parent")
    parent.end()
    tracer.start_span("late-child", context=trace.set_span_in_context(parent)).end()
    client.flush()
    assert _flagged(exporter) == []


def test_detector_flags_only_a_local_parent_it_never_saw() -> None:
    detector = ForeignParentDetector()
    own = TracerProvider()
    own.add_span_processor(detector)
    other = TracerProvider()
    exporter = InMemorySpanExporter()
    own.add_span_processor(SimpleSpanProcessor(exporter))

    with own.get_tracer("t").start_as_current_span("own-parent"):
        own.get_tracer("t").start_span("own-child").end()
    with other.get_tracer("t").start_as_current_span("foreign-parent"):
        own.get_tracer("t").start_span("foreign-child").end()

    assert _flagged(exporter) == ["foreign-child"]
    assert detector.count == 1


def _heartbeat_client(
    exporter: InMemorySpanExporter, pings: list[dict[str, Any]], **kwargs: Any
) -> rius.GlassflowClient:
    return _init(exporter, heartbeat=True, heartbeat_transport=pings.append, **kwargs)


def test_heartbeat_counts_foreign_parents_when_the_bridge_is_off(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    pings: list[dict[str, Any]] = []
    client = _heartbeat_client(InMemorySpanExporter(), pings, bridge_foreign_provider=False)
    _job_with_llm_child(provider, client)
    _job_with_llm_child(provider, client)
    client.shutdown()
    assert pings[-1]["stopped"] is True
    assert pings[-1]["foreign_parent_spans"] == 2


def test_heartbeat_counts_nothing_when_the_bridge_is_on(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    pings: list[dict[str, Any]] = []
    client = _heartbeat_client(InMemorySpanExporter(), pings)
    _job_with_llm_child(provider, client)
    client.shutdown()
    assert pings[-1]["foreign_parent_spans"] == 0


def test_bridged_spans_get_the_same_processing_as_rius_spans(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    client = _init(exporter, session_id="session-1")
    _job_with_llm_child(provider, client)
    client.flush()
    assert {(s.attributes or {}).get("session.id") for s in exporter.get_finished_spans()} == {
        "session-1"
    }


def test_after_reinit_the_earlier_rius_global_feeds_the_new_client(
    install_global: InstallGlobal,
) -> None:
    first = _init(InMemorySpanExporter())
    first.shutdown()
    exporter = InMemorySpanExporter()
    second = _init(exporter)
    trace.get_tracer_provider().get_tracer("third.party").start_span("global").end()
    second.flush()
    assert [s.name for s in exporter.get_finished_spans()] == ["global"]
