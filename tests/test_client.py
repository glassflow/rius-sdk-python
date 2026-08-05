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
