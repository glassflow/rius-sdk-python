"""Bundled auto-instrumentation.

`init()` enables any supported third-party instrumentor whose package is
installed, passing our tracer provider so instrumentation spans nest under
ours. `instruments=[...]` restricts, `instruments=[]` disables.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rius.instrumentation as instrumentation
from rius import init
from rius.instrumentation import InstrumentorSpec


class FakeInstrumentor:
    """Duck-types BaseInstrumentor: instrument() + is_instrumented flag."""

    instrument_calls: list[Any] = []
    uninstrument_calls: list[Any] = []

    def __init__(self) -> None:
        pass

    @property
    def is_instrumented_by_opentelemetry(self) -> bool:
        return getattr(type(self), "_instrumented", False)

    def instrument(self, *, tracer_provider: Any = None, **kwargs: Any) -> None:
        type(self).instrument_calls.append(tracer_provider)
        type(self)._instrumented = True

    def uninstrument(self) -> None:
        type(self).uninstrument_calls.append(None)
        type(self)._instrumented = False


@pytest.fixture()
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> type[FakeInstrumentor]:
    """Install a fake instrumentor module + registry entry; reset per test."""

    class _Instrumentor(FakeInstrumentor):
        instrument_calls: list[Any] = []
        uninstrument_calls: list[Any] = []
        _instrumented = False

    module = types.ModuleType("fake_instrumentation_pkg")
    module.FakeProviderInstrumentor = _Instrumentor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_instrumentation_pkg", module)
    monkeypatch.setattr(
        instrumentation,
        "REGISTRY",
        (
            InstrumentorSpec("fake", "fake_instrumentation_pkg", "FakeProviderInstrumentor"),
            InstrumentorSpec("missing", "package_that_is_not_installed", "NopeInstrumentor"),
        ),
    )
    return _Instrumentor


def test_global_init_auto_instruments_available_instrumentors(
    fake_registry: type[FakeInstrumentor],
) -> None:
    client = init(span_exporter=InMemorySpanExporter(), set_global=True)
    assert fake_registry.instrument_calls == [client._provider]


def test_scoped_init_does_not_auto_instrument(fake_registry: type[FakeInstrumentor]) -> None:
    # Instrumentors are process-global singletons — a scoped client must not
    # silently reroute all LLM traffic in the process. Explicit opt-in only.
    init(span_exporter=InMemorySpanExporter(), set_global=False)
    assert fake_registry.instrument_calls == []


def test_scoped_init_with_explicit_instruments_instruments(
    fake_registry: type[FakeInstrumentor],
) -> None:
    client = init(span_exporter=InMemorySpanExporter(), set_global=False, instruments=["fake"])
    assert fake_registry.instrument_calls == [client._provider]


def test_missing_instrumentor_package_is_skipped(fake_registry: type[FakeInstrumentor]) -> None:
    # "missing" spec's package is not importable — init() must not raise.
    init(span_exporter=InMemorySpanExporter(), set_global=True)


def test_instruments_param_restricts(fake_registry: type[FakeInstrumentor]) -> None:
    init(span_exporter=InMemorySpanExporter(), set_global=False, instruments=["missing"])
    assert fake_registry.instrument_calls == []


def test_instruments_empty_disables(fake_registry: type[FakeInstrumentor]) -> None:
    init(span_exporter=InMemorySpanExporter(), set_global=False, instruments=[])
    assert fake_registry.instrument_calls == []


def test_unknown_instrument_name_warns(
    fake_registry: type[FakeInstrumentor], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        init(
            span_exporter=InMemorySpanExporter(),
            set_global=False,
            instruments=["not-a-thing"],
        )
    assert any("not-a-thing" in record.message for record in caplog.records)


def test_requested_but_uninstalled_instrument_warns(
    fake_registry: type[FakeInstrumentor], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        init(span_exporter=InMemorySpanExporter(), set_global=False, instruments=["missing"])
    assert any("missing" in record.message for record in caplog.records)


def test_no_double_instrumentation_while_client_active(
    fake_registry: type[FakeInstrumentor],
) -> None:
    init(span_exporter=InMemorySpanExporter(), set_global=True)
    init(span_exporter=InMemorySpanExporter(), set_global=True)  # warn+skip
    assert len(fake_registry.instrument_calls) == 1


def test_reinit_after_shutdown_rebinds_instrumentors(
    fake_registry: type[FakeInstrumentor],
) -> None:
    first = init(span_exporter=InMemorySpanExporter(), set_global=True)
    first.shutdown()
    second = init(span_exporter=InMemorySpanExporter(), set_global=True)
    # re-bound: uninstrumented from the dead provider, instrumented on the new one
    assert fake_registry.uninstrument_calls == [None]
    assert fake_registry.instrument_calls == [first._provider, second._provider]


def test_externally_enabled_instrumentor_is_not_stolen(
    fake_registry: type[FakeInstrumentor],
) -> None:
    external = object()
    fake_registry().instrument(tracer_provider=external)  # someone else's setup
    init(span_exporter=InMemorySpanExporter(), set_global=True)
    assert fake_registry.instrument_calls == [external]  # left alone
    assert fake_registry.uninstrument_calls == []


def test_disabled_sdk_does_not_instrument(fake_registry: type[FakeInstrumentor]) -> None:
    init(span_exporter=InMemorySpanExporter(), set_global=True, disabled=True)
    assert fake_registry.instrument_calls == []


_CHAT_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def _start_json_server(payload: dict[str, Any]) -> Any:
    import http.server
    import json
    import threading

    body = json.dumps(payload).encode()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.integration
def test_openai_instrumentation_spans_nest_under_ours() -> None:
    """Real OpenAIInstrumentor: its spans land in our exporter, nested under ours."""
    openai = pytest.importorskip("openai")
    oi = pytest.importorskip("openinference.instrumentation.openai")

    instrumentor = oi.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()  # claim the provider deterministically

    server = _start_json_server(_CHAT_COMPLETION)
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=["openai"])
    try:
        oai = openai.OpenAI(
            api_key="test-key", base_url=f"http://127.0.0.1:{server.server_port}/v1"
        )
        with client.get_tracer().start_as_current_span("agent-step") as parent:
            oai.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
            )
        client.flush()

        spans = inner.get_finished_spans()
        llm_spans = [s for s in spans if s.name != "agent-step"]
        assert llm_spans, "OpenAI instrumentor emitted no spans into our provider"
        llm = llm_spans[0]
        assert llm.parent is not None
        assert llm.parent.span_id == parent.get_span_context().span_id
        assert llm.attributes is not None
        assert llm.attributes.get("openinference.span.kind") == "LLM"
    finally:
        instrumentor.uninstrument()
        server.shutdown()
