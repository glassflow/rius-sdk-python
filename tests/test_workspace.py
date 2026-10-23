"""Workspaces: context-scoped routing of spans to per-customer destinations.

Wire contract under test: spans started inside ``workspace(alias)`` are
delivered by the exporter registered for that alias, spans outside any scope
go to the default exporter, and the transient routing attribute is stripped
before spans leave the process (the destination's API key already says which
workspace a span belongs to; the alias is process-local configuration).
"""

from __future__ import annotations

import logging

import pytest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init, workspace
from rius.semconv import PENDING_IDENTITY_ATTRIBUTES, RIUS_SPAN_PENDING, WORKSPACE_ROUTE


def _routed_client(**kwargs: object):
    """A scoped client with one InMemory exporter per workspace key."""
    default = InMemorySpanExporter()
    per_key: dict[str, InMemorySpanExporter] = {}

    def factory(api_key: str) -> SpanExporter:
        per_key[api_key] = InMemorySpanExporter()
        return per_key[api_key]

    client = init(
        span_exporter=default,
        workspaces={"acme": "key-acme", "globex": "key-globex"},
        workspace_exporter_factory=factory,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        **kwargs,  # type: ignore[arg-type]
    )
    return client, default, per_key


# --- the scoped API ---


def test_scope_routes_every_span_to_the_alias_exporter() -> None:
    client, default, per_key = _routed_client()
    tracer = client.get_tracer()
    with (
        workspace("acme"),
        tracer.start_as_current_span("root"),
        tracer.start_as_current_span("child"),
    ):
        pass
    client.flush()
    assert default.get_finished_spans() == ()
    spans = per_key["key-acme"].get_finished_spans()
    assert [s.name for s in spans] == ["child", "root"]


def test_no_scope_routes_to_the_default_exporter() -> None:
    client, default, per_key = _routed_client()
    with client.get_tracer().start_as_current_span("bare"):
        pass
    client.flush()
    assert len(default.get_finished_spans()) == 1
    assert "key-acme" not in per_key or per_key["key-acme"].get_finished_spans() == ()


def test_two_scopes_partition_one_batch() -> None:
    client, default, per_key = _routed_client()
    tracer = client.get_tracer()
    with workspace("acme"), tracer.start_as_current_span("for-acme"):
        pass
    with workspace("globex"), tracer.start_as_current_span("for-globex"):
        pass
    with tracer.start_as_current_span("for-default"):
        pass
    client.flush()
    assert [s.name for s in per_key["key-acme"].get_finished_spans()] == ["for-acme"]
    assert [s.name for s in per_key["key-globex"].get_finished_spans()] == ["for-globex"]
    assert [s.name for s in default.get_finished_spans()] == ["for-default"]


def test_routing_attribute_never_reaches_the_wire() -> None:
    client, default, per_key = _routed_client()
    with workspace("acme"), client.get_tracer().start_as_current_span("root"):
        pass
    with client.get_tracer().start_as_current_span("bare"):
        pass
    client.flush()
    for exporter in (default, per_key["key-acme"]):
        for span in exporter.get_finished_spans():
            assert WORKSPACE_ROUTE not in (span.attributes or {})


def test_scope_ends_at_the_block() -> None:
    client, default, per_key = _routed_client()
    tracer = client.get_tracer()
    with workspace("acme"), tracer.start_as_current_span("inside"):
        pass
    with tracer.start_as_current_span("after"):
        pass
    client.flush()
    assert [s.name for s in per_key["key-acme"].get_finished_spans()] == ["inside"]
    assert [s.name for s in default.get_finished_spans()] == ["after"]


def test_unknown_alias_falls_back_to_default_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, default, per_key = _routed_client()
    with caplog.at_level(logging.WARNING, logger="rius.workspace"):
        with workspace("no-such-customer"), client.get_tracer().start_as_current_span("lost"):
            pass
        client.flush()
    assert [s.name for s in default.get_finished_spans()] == ["lost"]
    assert any("no-such-customer" in r.message for r in caplog.records)


def test_switching_workspace_inside_a_trace_warns(caplog: pytest.LogCaptureFixture) -> None:
    client, default, per_key = _routed_client()
    tracer = client.get_tracer()
    with (
        caplog.at_level(logging.WARNING, logger="rius.workspace"),
        workspace("acme"),
        tracer.start_as_current_span("root"),
        workspace("globex"),
        tracer.start_as_current_span("child"),
    ):
        pass
    client.flush()
    assert any("workspace" in r.message and "trace" in r.message for r in caplog.records)


# --- dynamic registration ---


def test_register_workspace_after_init() -> None:
    client, default, per_key = _routed_client()
    client.register_workspace("initech", "key-initech")
    with workspace("initech"), client.get_tracer().start_as_current_span("late"):
        pass
    client.flush()
    assert [s.name for s in per_key["key-initech"].get_finished_spans()] == ["late"]


def test_register_workspace_without_routing_raises() -> None:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        service_name="test-svc",
        instruments=[],
    )
    with pytest.raises(RuntimeError, match="workspaces"):
        client.register_workspace("acme", "key-acme")


def test_empty_workspaces_dict_enables_pure_dynamic_routing() -> None:
    default = InMemorySpanExporter()
    per_key: dict[str, InMemorySpanExporter] = {}

    def factory(api_key: str) -> SpanExporter:
        per_key[api_key] = InMemorySpanExporter()
        return per_key[api_key]

    client = init(
        span_exporter=default,
        workspaces={},
        workspace_exporter_factory=factory,
        set_global=False,
        service_name="test-svc",
        instruments=[],
    )
    client.register_workspace("acme", "key-acme")
    with workspace("acme"), client.get_tracer().start_as_current_span("routed"):
        pass
    client.flush()
    assert [s.name for s in per_key["key-acme"].get_finished_spans()] == ["routed"]


# --- lifecycle fan-out ---


class _RecordingExporter(SpanExporter):
    def __init__(self) -> None:
        self.exported: list[str] = []
        self.flushed = False
        self.shut_down = False

    def export(self, spans) -> SpanExportResult:  # type: ignore[no-untyped-def]
        self.exported.extend(s.name for s in spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self.flushed = True
        return True

    def shutdown(self) -> None:
        self.shut_down = True


def test_shutdown_fans_out_and_flush_delivers_to_every_destination() -> None:
    # Note: OTel's BatchSpanProcessor.force_flush drains its queue through
    # export() and never calls exporter.force_flush(), so the delivery
    # contract under test is "flushed spans reach the destination" plus
    # "shutdown reaches every destination exporter".
    default = _RecordingExporter()
    created: dict[str, _RecordingExporter] = {}

    def factory(api_key: str) -> SpanExporter:
        created[api_key] = _RecordingExporter()
        return created[api_key]

    client = init(
        span_exporter=default,
        workspaces={"acme": "key-acme"},
        workspace_exporter_factory=factory,
        set_global=False,
        service_name="test-svc",
        instruments=[],
    )
    with workspace("acme"), client.get_tracer().start_as_current_span("s"):
        pass
    client.flush()
    assert created["key-acme"].exported == ["s"]
    client.shutdown()
    assert default.shut_down
    assert created["key-acme"].shut_down


# --- pending snapshots route too ---


def test_pending_snapshot_routes_with_the_scope() -> None:
    assert WORKSPACE_ROUTE in PENDING_IDENTITY_ATTRIBUTES
    client, default, per_key = _routed_client(partial_spans=True)
    tracer = client.get_tracer()
    with workspace("acme"), tracer.start_as_current_span("long-run"):
        client.flush()
        pending = per_key["key-acme"].get_finished_spans()
        assert len(pending) == 1
        assert pending[0].attributes is not None
        assert pending[0].attributes.get(RIUS_SPAN_PENDING) is True
        assert WORKSPACE_ROUTE not in pending[0].attributes
    client.flush()


# --- one destination's failure stays its own ---


def _stamped_span(alias: str | None, name: str):
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import SpanContext

    attributes = {} if alias is None else {WORKSPACE_ROUTE: alias}
    context = SpanContext(trace_id=1, span_id=len(name) + 1, is_remote=False)
    return ReadableSpan(name=name, context=context, attributes=attributes)


def test_a_raising_exporter_factory_fails_only_its_own_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from rius.workspace import RoutingSpanExporter

    def factory(api_key: str) -> SpanExporter:
        raise RuntimeError("bad key material")

    default = InMemorySpanExporter()
    routing = RoutingSpanExporter(default, factory, {"acme": "key-acme"})
    with caplog.at_level(logging.WARNING, logger="rius.workspace"):
        result = routing.export([_stamped_span("acme", "a"), _stamped_span(None, "default")])
    assert result is SpanExportResult.FAILURE
    assert [s.name for s in default.get_finished_spans()] == ["default"]
    assert any("acme" in r.getMessage() for r in caplog.records)


def test_a_raising_destination_exporter_fails_only_its_own_group() -> None:
    from rius.workspace import RoutingSpanExporter

    class Exploding(SpanExporter):
        def export(self, spans):  # type: ignore[no-untyped-def]
            raise ConnectionError("boom")

    default = InMemorySpanExporter()
    routing = RoutingSpanExporter(default, lambda key: Exploding(), {"acme": "key-acme"})
    result = routing.export([_stamped_span("acme", "a"), _stamped_span(None, "default")])
    assert result is SpanExportResult.FAILURE
    assert [s.name for s in default.get_finished_spans()] == ["default"]
    assert routing.last_failed == {"acme"}


def test_the_router_reports_which_destinations_failed() -> None:
    # A FAILURE result is reported the same as a raise. An unregistered
    # alias's spans go to the default destination, so its failure is the
    # default's.
    from rius.workspace import RoutingSpanExporter

    class Failing(SpanExporter):
        def export(self, spans):  # type: ignore[no-untyped-def]
            return SpanExportResult.FAILURE

    default = InMemorySpanExporter()
    routing = RoutingSpanExporter(default, lambda key: Failing(), {"acme": "key-acme"})
    routing.export([_stamped_span("acme", "a"), _stamped_span(None, "default")])
    assert routing.last_failed == {"acme"}

    routing.export([_stamped_span(None, "default")])
    assert routing.last_failed == frozenset()

    failing_default = RoutingSpanExporter(Failing(), lambda key: InMemorySpanExporter(), {})
    failing_default.export([_stamped_span("unknown", "u")])
    assert failing_default.last_failed == {None}


# --- key rotation releases the exporter it replaced ---


class _Tracked(InMemorySpanExporter):
    def __init__(self) -> None:
        super().__init__()
        self.shut = False

    def shutdown(self) -> None:
        self.shut = True
        super().shutdown()


def test_key_rotation_shuts_down_and_evicts_the_replaced_exporter() -> None:
    from rius.workspace import RoutingSpanExporter

    built: list[_Tracked] = []

    def factory(api_key: str) -> SpanExporter:
        built.append(_Tracked())
        return built[-1]

    routing = RoutingSpanExporter(InMemorySpanExporter(), factory, {"acme": "key-0"})
    for n in range(1, 6):
        routing.export([_stamped_span("acme", f"s{n}")])
        routing.register("acme", f"key-{n}")
    routing.export([_stamped_span("acme", "last")])

    assert len(built) == 6
    assert [e.shut for e in built] == [True] * 5 + [False]
    assert len(routing._exporters) == 1
    assert [s.name for s in built[-1].get_finished_spans()] == ["last"]


def test_shutdown_releases_an_exporter_rotated_out_since_the_last_export() -> None:
    # The process can exit before another export would release it.
    from rius.workspace import RoutingSpanExporter

    built: list[_Tracked] = []

    def factory(api_key: str) -> SpanExporter:
        built.append(_Tracked())
        return built[-1]

    routing = RoutingSpanExporter(InMemorySpanExporter(), factory, {"acme": "key-0"})
    routing.export([_stamped_span("acme", "s0")])
    routing.register("acme", "key-1")
    assert not built[0].shut
    routing.shutdown()
    assert built[0].shut
    assert routing._retired == []


def test_rotation_keeps_an_exporter_another_alias_still_uses() -> None:
    from rius.workspace import RoutingSpanExporter

    built: dict[str, _Tracked] = {}

    def factory(api_key: str) -> SpanExporter:
        built[api_key] = _Tracked()
        return built[api_key]

    routing = RoutingSpanExporter(
        InMemorySpanExporter(), factory, {"acme": "shared", "globex": "shared"}
    )
    routing.export([_stamped_span("acme", "a")])
    routing.register("acme", "acme-own")
    routing.export([_stamped_span("globex", "g"), _stamped_span("acme", "a2")])
    assert not built["shared"].shut
    assert [s.name for s in built["shared"].get_finished_spans()] == ["a", "g"]
    assert [s.name for s in built["acme-own"].get_finished_spans()] == ["a2"]


def test_re_registering_the_same_key_keeps_its_exporter() -> None:
    from rius.workspace import RoutingSpanExporter

    built: list[_Tracked] = []

    def factory(api_key: str) -> SpanExporter:
        built.append(_Tracked())
        return built[-1]

    routing = RoutingSpanExporter(InMemorySpanExporter(), factory, {"acme": "k"})
    routing.export([_stamped_span("acme", "a")])
    routing.register("acme", "k")
    routing.export([_stamped_span("acme", "b")])
    assert len(built) == 1 and not built[0].shut
