"""Sessions: a caller-minted id stamped on every span created in scope.

Wire contract under test: ``session.id`` (OpenInference) on each span, set at
span START so pending snapshots carry it too. The sink groups per-span with a
TraceId fallback, so a session id must reach every span in the scope, not just
the root, or children fragment into the wrong session.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init, session
from rius.semconv import GLASSFLOW_SPAN_PENDING, PENDING_IDENTITY_ATTRIBUTES, SESSION_ID


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
        session("sess-1"),
        tracer.start_as_current_span("root"),
        tracer.start_as_current_span("child"),
    ):
        pass
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert all(s.attributes[SESSION_ID] == "sess-1" for s in spans)


def test_no_scope_no_attribute() -> None:
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("bare"):
        pass
    client.flush()
    assert SESSION_ID not in exporter.get_finished_spans()[0].attributes


def test_scope_ends_at_the_block() -> None:
    client, exporter = _memory_client()
    tracer = client.get_tracer()
    with session("sess-1"), tracer.start_as_current_span("inside"):
        pass
    with tracer.start_as_current_span("after"):
        pass
    client.flush()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["inside"].attributes[SESSION_ID] == "sess-1"
    assert SESSION_ID not in spans["after"].attributes


def test_nested_scope_wins() -> None:
    client, exporter = _memory_client()
    tracer = client.get_tracer()
    with session("outer"), session("inner"), tracer.start_as_current_span("s"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].attributes[SESSION_ID] == "inner"


def test_scope_crosses_threads_when_context_is_carried() -> None:
    # Same contract as OTel context generally: explicit propagation via the
    # executor's initializer/wrapper is the caller's job; here the closure
    # runs the span inside the scope entered by the submitting thread.
    client, exporter = _memory_client()
    tracer = client.get_tracer()

    def turn() -> None:
        with session("sess-t"), tracer.start_as_current_span("threaded"):
            pass

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(turn).result()
    client.flush()
    assert exporter.get_finished_spans()[0].attributes[SESSION_ID] == "sess-t"


# --- the init()-level default ---


def test_init_default_applies_without_scope() -> None:
    client, exporter = _memory_client(session_id="proc-wide")
    with client.get_tracer().start_as_current_span("s"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].attributes[SESSION_ID] == "proc-wide"


def test_scope_overrides_init_default() -> None:
    client, exporter = _memory_client(session_id="proc-wide")
    with session("per-turn"), client.get_tracer().start_as_current_span("s"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].attributes[SESSION_ID] == "per-turn"


def test_env_var_sets_the_default(monkeypatch) -> None:
    monkeypatch.setenv("RIUS_SESSION_ID", "from-env")
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("s"):
        pass
    client.flush()
    assert exporter.get_finished_spans()[0].attributes[SESSION_ID] == "from-env"


# --- pending spans ---


def test_pending_snapshot_carries_the_session_id() -> None:
    assert SESSION_ID in PENDING_IDENTITY_ATTRIBUTES
    client, exporter = _memory_client(partial_spans=True)
    with session("sess-p"), client.get_tracer().start_as_current_span("op"):
        client.flush()  # pending snapshot exported while the span is open
    client.flush()
    pending = [s for s in exporter.get_finished_spans() if s.attributes.get(GLASSFLOW_SPAN_PENDING)]
    assert pending, "expected a pending snapshot"
    assert pending[0].attributes[SESSION_ID] == "sess-p"
