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
from rius.config import GlassflowConfig, resolve_config
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


def _init_bridged(exporter: InMemorySpanExporter, **kwargs: Any) -> rius.GlassflowClient:
    return _init(exporter, bridge_foreign_provider=True, **kwargs)


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
    client = _init_bridged(exporter)
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
    client = _init_bridged(InMemorySpanExporter())
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
    client = _init_bridged(exporter, agent_name="checkout-agent")
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


def test_by_default_rius_keeps_to_its_own_provider_and_flags_the_orphan(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    with caplog.at_level(logging.WARNING, logger="rius.client"):
        client = _init(exporter)
    _job_with_llm_child(provider, client)
    client.flush()

    assert set(_by_name(exporter)) == {"llm"}
    assert _flagged(exporter) == ["llm"]
    assert any("bridge_foreign_provider is off" in r.message for r in caplog.records)


def test_opt_in_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_BRIDGE_FOREIGN_PROVIDER", "true")
    assert resolve_config().bridge_foreign_provider is True
    assert resolve_config(bridge_foreign_provider=False).bridge_foreign_provider is False


def test_bridge_is_off_by_default() -> None:
    assert resolve_config().bridge_foreign_provider is False
    assert GlassflowConfig("https://x", None, "svc").bridge_foreign_provider is False


def test_shutdown_makes_the_bridge_inert_without_touching_foreign_processors(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, foreign_exporter = foreign
    spy = _LifecycleSpy()
    provider.add_span_processor(spy)
    client = _init_bridged(InMemorySpanExporter(), session_id="session-1")
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
    client = _init_bridged(exporter)
    provider.shutdown()
    client.get_tracer().start_span("still-alive").end()
    client.flush()
    assert [s.name for s in exporter.get_finished_spans()] == ["still-alive"]


def test_reinit_reuses_the_one_bridge_instead_of_attaching_another(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    attached_by_fixture = len(provider.added)
    _init_bridged(InMemorySpanExporter()).shutdown()
    second_exporter = InMemorySpanExporter()
    second = _init_bridged(second_exporter)
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
        client = _init_bridged(exporter)
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


def test_by_default_rius_detects_the_other_sdk_end_to_end(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, _ = foreign
    owner = f"{__name__}._RecordingProvider"
    exporter = InMemorySpanExporter()
    pings: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING, logger="rius.foreign"):
        client = _init(exporter, heartbeat=True, heartbeat_transport=pings.append)
        _job_with_llm_child(provider, client)
        _job_with_llm_child(provider, client)
    client.shutdown()

    spans = exporter.get_finished_spans()
    assert _flagged(exporter) == ["llm", "llm"] == [s.name for s in spans]
    assert pings[-1]["foreign_parent_spans"] == 2
    assert spans[0].resource.attributes[RIUS_SDK_GLOBAL_PROVIDER] == f"foreign:{owner}"
    [warning] = [r.getMessage() for r in caplog.records if r.name == "rius.foreign"]
    assert f"another OpenTelemetry SDK ({owner}) owns the global tracer provider" in warning
    assert "RIUS_BRIDGE_FOREIGN_PROVIDER=true" in warning


def _heartbeat_client(
    exporter: InMemorySpanExporter, pings: list[dict[str, Any]], **kwargs: Any
) -> rius.GlassflowClient:
    return _init(exporter, heartbeat=True, heartbeat_transport=pings.append, **kwargs)


def test_heartbeat_counts_foreign_parents_when_the_bridge_is_off(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    pings: list[dict[str, Any]] = []
    client = _heartbeat_client(InMemorySpanExporter(), pings)
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
    client = _heartbeat_client(InMemorySpanExporter(), pings, bridge_foreign_provider=True)
    _job_with_llm_child(provider, client)
    client.shutdown()
    assert pings[-1]["foreign_parent_spans"] == 0


def test_bridged_spans_get_the_same_processing_as_rius_spans(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    client = _init_bridged(exporter, session_id="session-1")
    _job_with_llm_child(provider, client)
    client.flush()
    assert {(s.attributes or {}).get("session.id") for s in exporter.get_finished_spans()} == {
        "session-1"
    }


def test_after_reinit_the_earlier_rius_global_feeds_the_new_client(
    install_global: InstallGlobal,
) -> None:
    first = _init_bridged(InMemorySpanExporter())
    first.shutdown()
    exporter = InMemorySpanExporter()
    second = _init_bridged(exporter)
    trace.get_tracer_provider().get_tracer("third.party").start_span("global").end()
    second.flush()
    assert [s.name for s in exporter.get_finished_spans()] == ["global"]


# --- content capture on bridged spans -------------------------------------------

# What Langfuse 3.15 wrote on the bridged job and tool spans of the RIUS-1070
# e2e that ran with capture_content=False.
_LANGFUSE_JOB = {
    "langfuse.observation.type": "span",
    "langfuse.observation.input": '{"label": "job-0"}',
    "langfuse.observation.output": '{"answer": "96,450,000"}',
    "langfuse.observation.metadata.note": "customer payload",
    "langfuse.trace.name": "job-0",
    "langfuse.trace.input": '{"label": "job-0"}',
    "langfuse.trace.output": '{"answer": "96,450,000"}',
    "langfuse.trace.tags": ("run-1",),
    "session.id": "run-1",
}
_LANGFUSE_CONTENT = {
    "langfuse.observation.input",
    "langfuse.observation.output",
    "langfuse.observation.metadata.note",
    "langfuse.trace.input",
    "langfuse.trace.output",
}


def _bridged_langfuse_job(
    provider: trace.TracerProvider, **init_kwargs: Any
) -> dict[str, ReadableSpan]:
    exporter = InMemorySpanExporter()
    client = _init_bridged(exporter, **init_kwargs)
    with provider.get_tracer("langfuse-sdk").start_as_current_span("job") as job:
        job.set_attributes(_LANGFUSE_JOB)
        client.get_tracer().start_span("llm").end()
    client.flush()
    return _by_name(exporter)


def test_content_off_strips_langfuse_content_from_a_bridged_span(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    attributes = dict(
        _bridged_langfuse_job(provider, capture_content=False)["job"].attributes or {}
    )
    assert _LANGFUSE_CONTENT.isdisjoint(attributes)
    assert {key: attributes.get(key) for key in _LANGFUSE_JOB.keys() - _LANGFUSE_CONTENT} == {
        key: value for key, value in _LANGFUSE_JOB.items() if key not in _LANGFUSE_CONTENT
    }


def test_content_on_keeps_langfuse_content_on_a_bridged_span(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    attributes = dict(_bridged_langfuse_job(provider)["job"].attributes or {})
    assert {key: attributes.get(key) for key in _LANGFUSE_JOB} == _LANGFUSE_JOB


def test_content_off_leaves_the_foreign_providers_own_export_untouched(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, foreign_exporter = foreign
    _bridged_langfuse_job(provider, capture_content=False)
    assert dict(_by_name(foreign_exporter)["job"].attributes or {}) == _LANGFUSE_JOB


def test_masking_treats_a_bridged_span_exactly_as_an_own_span(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    client = _init_bridged(exporter, mask=lambda value, *, key: f"masked:{key}")
    for tracer, name in (
        (provider.get_tracer("langfuse-sdk"), "bridged"),
        (client.get_tracer(), "own"),
    ):
        with tracer.start_as_current_span(name) as span:
            span.set_attributes(_LANGFUSE_JOB)
    client.flush()
    spans = _by_name(exporter)
    assert dict(spans["bridged"].attributes or {}) == dict(spans["own"].attributes or {})
    assert (spans["own"].attributes or {})["langfuse.observation.input"] == (
        "masked:langfuse.observation.input"
    )


# --- rius never writes to a span it did not create --------------------------------

# Start attributes the normalizer maps onto canonical GenAI keys at span start.
_OPENINFERENCE_LLM = {
    "openinference.span.kind": "LLM",
    "llm.provider": "openai",
    "llm.token_count.prompt": 3,
}
_PENDING = "rius.span.pending"


def _foreign_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _foreign_workload(provider: trace.TracerProvider) -> None:
    """The reviewer's probe, widened: user and workspace scopes, a GenAI child."""
    tracer = provider.get_tracer("langfuse-sdk")
    job_start = {"langfuse.trace.name": "job"}
    with (
        rius.user("u-42"),
        rius.workspace("acme"),
        tracer.start_as_current_span("job", attributes=job_start) as job,
    ):
        job.set_attribute("langfuse.observation.type", "span")
        tracer.start_span("llm-call", attributes=_OPENINFERENCE_LLM).end()


def _attributes_by_name(exporter: InMemorySpanExporter) -> dict[str, dict[str, Any]]:
    return {span.name: dict(span.attributes or {}) for span in exporter.get_finished_spans()}


def _bridge_with_every_stamp(
    install_global: InstallGlobal, provider: TracerProvider
) -> tuple[rius.GlassflowClient, InMemorySpanExporter]:
    """A client stamping session, user, route and pendings; returns acme's exporter."""
    install_global(provider)
    acme = InMemorySpanExporter()
    client = _init_bridged(
        InMemorySpanExporter(),
        session_id="sess-123",
        workspaces={"acme": "key-acme"},
        workspace_exporter_factory=lambda _key: acme,
        partial_spans=True,
        partial_spans_delay=0.0,
    )
    return client, acme


def test_the_foreign_export_is_exactly_what_it_is_without_rius(
    install_global: InstallGlobal,
) -> None:
    baseline_provider, baseline = _foreign_provider()
    _foreign_workload(baseline_provider)

    provider, foreign_exporter = _foreign_provider()
    client, _ = _bridge_with_every_stamp(install_global, provider)
    _foreign_workload(provider)
    client.flush()

    assert _attributes_by_name(foreign_exporter) == _attributes_by_name(baseline)


def test_bridged_spans_carry_the_stamps_in_the_rius_export(
    install_global: InstallGlobal,
) -> None:
    provider, _ = _foreign_provider()
    client, acme = _bridge_with_every_stamp(install_global, provider)
    _foreign_workload(provider)
    client.flush()

    spans = acme.get_finished_spans()
    final = {
        s.name: dict(s.attributes or {}) for s in spans if _PENDING not in (s.attributes or {})
    }
    assert set(final) == {"job", "llm-call"}
    for attributes in final.values():
        assert (attributes["session.id"], attributes["user.id"]) == ("sess-123", "u-42")
    assert {k: final["llm-call"][k] for k in _CANONICAL_LLM} == _CANONICAL_LLM


_CANONICAL_LLM = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "openai",
    "gen_ai.usage.input_tokens": 3,
}


def test_pending_snapshots_of_bridged_spans_carry_the_stamps(
    install_global: InstallGlobal,
) -> None:
    provider, _ = _foreign_provider()
    client, acme = _bridge_with_every_stamp(install_global, provider)
    _foreign_workload(provider)
    client.flush()

    pending = {
        s.name: dict(s.attributes or {})
        for s in acme.get_finished_spans()
        if (s.attributes or {}).get(_PENDING) is True
    }
    assert set(pending) == {"job", "llm-call"}
    for attributes in pending.values():
        assert (attributes["session.id"], attributes["user.id"]) == ("sess-123", "u-42")
    assert pending["llm-call"]["gen_ai.operation.name"] == "chat"


def test_a_bridged_span_exports_what_an_own_span_would(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    client = _init_bridged(exporter, session_id="sess-123")
    for tracer, name in ((provider.get_tracer("x"), "bridged"), (client.get_tracer(), "own")):
        with rius.user("u-42"):
            span = tracer.start_span(name, attributes={**_OPENINFERENCE_LLM, "session.id": "x"})
            # Written after start, so it wins over the start-time stamp.
            span.set_attribute("user.id", "set-later")
            span.end()
    client.flush()
    spans = _attributes_by_name(exporter)
    assert spans["bridged"] == spans["own"]
    assert (spans["own"]["session.id"], spans["own"]["user.id"]) == ("sess-123", "set-later")


def test_a_flagged_bridged_span_is_flagged_only_in_the_rius_export(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, foreign_exporter = foreign
    tracer = provider.get_tracer("customer.jobs")
    with tracer.start_as_current_span("started-before-init"):
        exporter = InMemorySpanExporter()
        client = _init_bridged(exporter)
        tracer.start_span("child").end()
    client.flush()

    assert _flagged(exporter) == ["child"]
    assert RIUS_PARENT_FOREIGN not in _attributes_by_name(foreign_exporter)["child"]


def test_the_workspace_guard_sees_the_route_of_a_bridged_parent(
    install_global: InstallGlobal, caplog: pytest.LogCaptureFixture
) -> None:
    provider, _ = _foreign_provider()
    client, _ = _bridge_with_every_stamp(install_global, provider)
    with (
        caplog.at_level(logging.WARNING, logger="rius.workspace"),
        rius.workspace("acme"),
        provider.get_tracer("x").start_as_current_span("job"),
        rius.workspace("other"),
    ):
        client.get_tracer().start_span("llm").end()
    assert any("cannot straddle two workspaces" in r.message for r in caplog.records)


# --- rius's sampling decision holds for bridged spans ------------------------------


def test_rius_sampling_drops_bridged_spans_and_their_own_children(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, foreign_exporter = foreign
    exporter = InMemorySpanExporter()
    client = _init_bridged(exporter, sample_rate=0.0)
    _job_with_llm_child(provider, client)
    client.flush()

    assert exporter.get_finished_spans() == ()
    assert [s.name for s in foreign_exporter.get_finished_spans()] == ["job"]


def test_a_trace_is_kept_or_dropped_whole_by_the_rius_ratio(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    provider, foreign_exporter = foreign
    exporter = InMemorySpanExporter()
    client = _init_bridged(exporter, sample_rate=0.5)
    tracer = provider.get_tracer("customer.jobs")
    for _ in range(64):
        with tracer.start_as_current_span("job"), tracer.start_as_current_span("step"):
            client.get_tracer().start_span("llm").end()
    client.flush()

    ratio = TraceIdRatioBased(0.5)
    expected = {
        s.context.trace_id
        for s in foreign_exporter.get_finished_spans()
        if ratio.should_sample(None, s.context.trace_id, "job").decision.is_sampled()
    }
    names_by_trace: dict[int, list[str]] = {}
    for span in exporter.get_finished_spans():
        names_by_trace.setdefault(span.context.trace_id, []).append(span.name)
    assert set(names_by_trace) == expected
    assert all(sorted(names) == ["job", "llm", "step"] for names in names_by_trace.values())
    assert 0 < len(expected) < 64


def test_a_sampled_remote_parent_keeps_a_bridged_trace_as_it_would_an_own_one(
    foreign: tuple[_RecordingProvider, InMemorySpanExporter],
) -> None:
    provider, _ = foreign
    exporter = InMemorySpanExporter()
    client = _init_bridged(exporter, sample_rate=0.0)
    carrier = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
    remote = TraceContextTextMapPropagator().extract(carrier)
    with provider.get_tracer("customer.jobs").start_as_current_span("server", context=remote):
        client.get_tracer().start_span("llm").end()
    client.flush()
    assert sorted(s.name for s in exporter.get_finished_spans()) == ["llm", "server"]
