import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.client import GlassflowClient, build_span_exporter
from rius.config import resolve_config


def _memory_client(**kwargs: object) -> tuple[GlassflowClient, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    client = init(span_exporter=exporter, set_global=False, service_name="test-svc", **kwargs)  # type: ignore[arg-type]
    return client, exporter


def test_init_returns_client() -> None:
    client, _ = _memory_client()
    assert isinstance(client, GlassflowClient)


def test_spans_are_exported() -> None:
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "op"


def test_resource_has_service_name() -> None:
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].resource.attributes["service.name"] == "test-svc"


def test_resource_uses_distro_not_sdk_identity() -> None:
    # Spec: telemetry.sdk.name MUST be "opentelemetry"; distributions identify
    # themselves via telemetry.distro.*.
    from rius import __version__

    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    resource_attrs = exporter.get_finished_spans()[0].resource.attributes
    assert resource_attrs["telemetry.sdk.name"] == "opentelemetry"
    assert resource_attrs["telemetry.distro.name"] == "glassflow-rius"
    assert resource_attrs["telemetry.distro.version"] == __version__


def test_resource_carries_the_agent_name() -> None:
    """Spans must group under the agent name their heartbeats use.

    Without this the backend falls through to service.name per span, so a
    process whose agent name differs has its agents view and its trace list
    silently disagreeing.
    """
    client, exporter = _memory_client(agent_name="checkout-agent")
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    resource_attrs = exporter.get_finished_spans()[0].resource.attributes
    assert resource_attrs["gen_ai.agent.name"] == "checkout-agent"
    assert resource_attrs["service.name"] == "test-svc"


def test_agent_name_defaults_to_the_service_name_on_the_resource() -> None:
    """A process that sets no agent name is unchanged: the two values match."""
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    resource_attrs = exporter.get_finished_spans()[0].resource.attributes
    assert resource_attrs["gen_ai.agent.name"] == resource_attrs["service.name"] == "test-svc"


def test_resource_and_heartbeat_agree_on_the_agent_name() -> None:
    """The bug this closes: heartbeats grouped under the agent name while
    every span fell through to the service name."""
    pings: list[dict] = []
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        service_name="test-svc",
        agent_name="checkout-agent",
        heartbeat=True,
        heartbeat_interval=5,
        heartbeat_transport=pings.append,
    )
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    client.shutdown()

    resource_attrs = exporter.get_finished_spans()[0].resource.attributes
    assert pings, "expected at least the initial heartbeat"
    assert {p["agent_name"] for p in pings} == {resource_attrs["gen_ai.agent.name"]}


def test_disabled_does_not_export() -> None:
    exporter = InMemorySpanExporter()
    client = init(span_exporter=exporter, set_global=False, disabled=True)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans() == ()


def test_default_exporter_is_otlp_http_targeting_traces_endpoint() -> None:
    config = resolve_config(endpoint="https://x.dev", api_key="k")
    exporter = build_span_exporter(config)
    assert isinstance(exporter, OTLPSpanExporter)
    assert exporter._endpoint == "https://x.dev/v1/traces"


def test_set_global_false_leaves_global_untouched() -> None:
    before = trace.get_tracer_provider()
    client, _ = _memory_client()
    assert trace.get_tracer_provider() is before
    assert client._provider is not before


def test_sampling_drops_all_at_zero() -> None:
    exporter = InMemorySpanExporter()
    client = init(span_exporter=exporter, set_global=False, sample_rate=0.0)
    for i in range(20):
        with client.get_tracer().start_as_current_span(f"op{i}"):
            pass
    client.flush()
    assert exporter.get_finished_spans() == ()


def test_sampling_keeps_all_at_one() -> None:
    exporter = InMemorySpanExporter()
    client = init(span_exporter=exporter, set_global=False, sample_rate=1.0)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert len(exporter.get_finished_spans()) == 1


def test_resource_carries_the_service_version() -> None:
    client, exporter = _memory_client(service_version="1.4.2")
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].resource.attributes["service.version"] == "1.4.2"


def test_an_unset_service_version_is_absent_from_the_resource() -> None:
    """Never stamped empty and never given a placeholder: a resource without
    the attribute is queryable as "unknown", a fake value is not."""
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert "service.version" not in exporter.get_finished_spans()[0].resource.attributes


def test_an_explicit_service_version_beats_the_otel_resource_attributes_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OTEL_RESOURCE_ATTRIBUTES already lands service.version on the resource,
    because Resource.create merges it. It is the weakest source of the three:
    Resource.create merges the passed attributes OVER the detected ones, so an
    explicit argument (and RIUS_SERVICE_VERSION behind it) wins."""
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.version=0.0.1-env")
    client, exporter = _memory_client(service_version="1.4.2")
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].resource.attributes["service.version"] == "1.4.2"


def test_otel_resource_attributes_still_supplies_the_version_when_nothing_else_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last rung of the resolution order, and it works without any code of
    ours; the SDK only has to stop overwriting it with a placeholder."""
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.version=0.0.1-env")
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].resource.attributes["service.version"] == "0.0.1-env"


def _resource_of(exporter: InMemorySpanExporter) -> "dict[str, object]":
    return dict(exporter.get_finished_spans()[0].resource.attributes)


def _resource_with(**kwargs: object) -> "dict[str, object]":
    client, exporter = _memory_client(**kwargs)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    return _resource_of(exporter)


def test_resource_carries_the_main_agent_identity() -> None:
    """rius.main_agent.* says which agent this PROCESS is. gen_ai.agent.name
    has no resource-level meaning in the conventions and on a span means the
    agent being INVOKED, so the process-level question gets its own namespace."""
    attrs = _resource_with(
        agent_name="checkout-agent",
        main_agent_id="ag_prod_7",
        main_agent_description="Takes a cart to a paid order",
        main_agent_version="7",
    )
    assert attrs["rius.main_agent.name"] == "checkout-agent"
    assert attrs["rius.main_agent.id"] == "ag_prod_7"
    assert attrs["rius.main_agent.description"] == "Takes a cart to a paid order"
    assert attrs["rius.main_agent.version"] == "7"


def test_the_main_agent_name_is_still_also_the_legacy_agent_name() -> None:
    """ADDITIVE, never a swap: every deployment between this SDK release and
    the sink release still has its agent identity read off gen_ai.agent.name.
    Resource attributes ride once per OTLP batch, so the duplication is free."""
    attrs = _resource_with(agent_name="checkout-agent")
    assert attrs["gen_ai.agent.name"] == "checkout-agent"
    assert attrs["rius.main_agent.name"] == attrs["gen_ai.agent.name"]


def test_an_unnamed_process_emits_no_main_agent_name() -> None:
    """The placeholder is suppressed at BOTH scopes, the same rule the span
    helpers already apply: a process that named nothing claims no identity."""
    client = init(span_exporter=(exporter := InMemorySpanExporter()), set_global=False)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    attrs = _resource_of(exporter)
    assert attrs["service.name"] == "unknown_service"
    assert "rius.main_agent.name" not in attrs


def test_unset_main_agent_attributes_are_absent_from_the_resource() -> None:
    attrs = _resource_with(agent_name="checkout-agent")
    for key in (
        "rius.main_agent.id",
        "rius.main_agent.description",
        "rius.main_agent.version",
    ):
        assert key not in attrs


def test_the_main_agent_version_and_the_service_version_are_independent() -> None:
    """A service at 2.3.1 can run an agent definition at 7. Neither key is
    ever derived from the other, in either direction."""
    versioned_service = _resource_with(service_version="2.3.1")
    assert versioned_service["service.version"] == "2.3.1"
    assert "rius.main_agent.version" not in versioned_service

    versioned_agent = _resource_with(main_agent_version="7")
    assert versioned_agent["rius.main_agent.version"] == "7"
    assert "service.version" not in versioned_agent
