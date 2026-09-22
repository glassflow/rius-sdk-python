"""Users: a caller-supplied id stamped on every span created in scope.

Wire contract under test: ``user.id`` (OpenInference, Langfuse, OTel registry)
on each span, set at span START so pending snapshots carry it too. Unlike
sessions there is no process-wide default and nothing is minted when the
caller passes nothing: a user is per request, and an unscoped span is simply
anonymous.
"""

from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init, session, user
from rius.semconv import PENDING_IDENTITY_ATTRIBUTES, RIUS_SPAN_PENDING, SESSION_ID, USER_ID


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


# --- the scoped API ---


def test_scope_stamps_every_span_in_it() -> None:
    client, exporter = _memory_client()
    tracer = client.get_tracer()
    with (
        user("u-1"),
        tracer.start_as_current_span("root"),
        tracer.start_as_current_span("child"),
    ):
        pass
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert all(s.attributes[USER_ID] == "u-1" for s in spans)


def test_no_scope_no_attribute() -> None:
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("bare"):
        pass
    client.flush()
    assert USER_ID not in exporter.get_finished_spans()[0].attributes


def test_scope_ends_at_the_block() -> None:
    client, exporter = _memory_client()
    tracer = client.get_tracer()
    with user("u-1"), tracer.start_as_current_span("inside"):
        pass
    with tracer.start_as_current_span("after"):
        pass
    client.flush()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["inside"].attributes[USER_ID] == "u-1"
    assert USER_ID not in spans["after"].attributes


def test_sibling_scopes_do_not_leak() -> None:
    client, exporter = _memory_client()
    tracer = client.get_tracer()
    with user("alice"), tracer.start_as_current_span("a"):
        pass
    with user("bob"), tracer.start_as_current_span("b"):
        pass
    client.flush()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["a"].attributes[USER_ID] == "alice"
    assert spans["b"].attributes[USER_ID] == "bob"


def test_nested_scope_wins() -> None:
    client, exporter = _memory_client()
    tracer = client.get_tracer()
    with user("outer"), user("inner"), tracer.start_as_current_span("s"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].attributes[USER_ID] == "inner"


def test_user_and_session_scopes_are_independent() -> None:
    client, exporter = _memory_client()
    with user("u-1"), session("sess-1"), client.get_tracer().start_as_current_span("s"):
        pass
    client.flush()
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs[USER_ID] == "u-1"
    assert attrs[SESSION_ID] == "sess-1"


def test_scope_yields_the_id() -> None:
    with user("u-1") as uid:
        assert uid == "u-1"


# --- no process-wide default, by design ---


def test_env_var_is_not_a_default(monkeypatch) -> None:
    monkeypatch.setenv("RIUS_USER_ID", "from-env")
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("s"):
        pass
    client.flush()
    assert USER_ID not in exporter.get_finished_spans()[0].attributes


# --- pending spans ---


def test_pending_snapshot_carries_the_user_id() -> None:
    assert USER_ID in PENDING_IDENTITY_ATTRIBUTES
    client, exporter = _memory_client(partial_spans=True)
    with user("u-p"), client.get_tracer().start_as_current_span("op"):
        client.flush()  # pending snapshot exported while the span is open
    client.flush()
    pending = [s for s in exporter.get_finished_spans() if s.attributes.get(RIUS_SPAN_PENDING)]
    assert pending, "expected a pending snapshot"
    assert pending[0].attributes[USER_ID] == "u-p"


# --- user_id= sugar on the span-opening entry points ---
#
# These go through the GLOBAL tracer (conftest installs a bare provider with no
# rius processors), so they pin that the sugar stamps the attribute at creation
# itself rather than relying on UserSpanProcessor being present.


def test_start_span_user_id_stamps_the_span(exported_spans: InMemorySpanExporter) -> None:
    from rius import start_span

    start_span("manual", user_id="u-m").end()
    assert exported_spans.get_finished_spans()[0].attributes[USER_ID] == "u-m"


def test_start_as_current_span_user_id_stamps_the_span(
    exported_spans: InMemorySpanExporter,
) -> None:
    from rius import start_as_current_span

    with start_as_current_span("root", user_id="u-s"):
        pass
    assert exported_spans.get_finished_spans()[0].attributes[USER_ID] == "u-s"


def test_start_as_current_span_user_id_scopes_children(
    exported_spans: InMemorySpanExporter,
) -> None:
    # The sugar must also enter the user() scope, so children opened by any
    # tracer inherit the user through the processor, as they would in a scope.
    from opentelemetry import trace

    from rius import start_as_current_span
    from rius.user import UserSpanProcessor

    trace.get_tracer_provider().add_span_processor(UserSpanProcessor())  # type: ignore[attr-defined]
    with (
        start_as_current_span("root", user_id="u-c"),
        trace.get_tracer("t").start_as_current_span("child"),
    ):
        pass
    by_name = {s.name: s for s in exported_spans.get_finished_spans()}
    assert by_name["child"].attributes[USER_ID] == "u-c"


def test_generation_user_id_stamps_the_span(exported_spans: InMemorySpanExporter) -> None:
    from rius import start_as_current_generation, start_generation

    start_generation("gen", model="m", user_id="u-g").end()
    with start_as_current_generation("gen2", model="m", user_id="u-g2"):
        pass
    by_name = {s.name: s for s in exported_spans.get_finished_spans()}
    assert by_name["gen"].attributes[USER_ID] == "u-g"
    assert by_name["gen2"].attributes[USER_ID] == "u-g2"


def test_sugar_does_not_leak_past_the_call(exported_spans: InMemorySpanExporter) -> None:
    from opentelemetry import trace

    from rius import start_as_current_span

    with start_as_current_span("with-user", user_id="u-x"):
        pass
    with trace.get_tracer("t").start_as_current_span("after"):
        pass
    by_name = {s.name: s for s in exported_spans.get_finished_spans()}
    assert USER_ID not in by_name["after"].attributes
