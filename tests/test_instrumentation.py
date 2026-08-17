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


def test_missing_extra_instrument_hint_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A third-party instrumentation is installed through our extra."""
    monkeypatch.setattr(
        instrumentation,
        "REGISTRY",
        (
            InstrumentorSpec(
                "anthropic", "package_that_is_not_installed", "NopeInstrumentor", extra="anthropic"
            ),
        ),
    )
    with caplog.at_level("WARNING"):
        init(span_exporter=InMemorySpanExporter(), set_global=False, instruments=["anthropic"])
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert 'pip install "glassflow-rius[anthropic]"' in message


def test_missing_builtin_instrument_hint_names_the_library(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A built-in instrumentation has no extra, so the hint must name the
    library instead. Pointing at `glassflow-rius[mcp]` would send the user to
    an extra that does not exist."""
    monkeypatch.setattr(
        instrumentation,
        "REGISTRY",
        (
            InstrumentorSpec(
                "mcp", "package_that_is_not_installed", "NopeInstrumentor", library="mcp"
            ),
        ),
    )
    with caplog.at_level("WARNING"):
        init(span_exporter=InMemorySpanExporter(), set_global=False, instruments=["mcp"])
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "pip install mcp" in message
    assert "glassflow-rius[mcp]" not in message


def test_every_spec_declares_exactly_one_install_source() -> None:
    """Each entry is installed either through one of our extras or by the user
    installing the library a built-in instrumentation patches. Both set, or
    neither, means the requested-but-missing warning cannot name a fix."""
    for spec in instrumentation.REGISTRY:
        assert (spec.extra is None) != (spec.library is None), spec.name


def test_extras_install_instrumentation_only_never_a_runtime_library() -> None:
    """The packaging rule this SDK is built on: an extra installs the
    instrumentation FOR a library, so the SDK never pins or upgrades a version
    the user's own code runs against. `mcp` used to be the exception, and its
    extra put a runtime library into environments that never touched MCP.
    """
    from pathlib import Path

    # tomllib is stdlib from 3.11 and this package supports 3.10. The assertion
    # is about static metadata rather than runtime behaviour, so proving it on
    # the rest of the matrix is enough; a tomli dev dependency just to read one
    # file on the oldest interpreter is not worth it.
    tomllib = pytest.importorskip("tomllib", reason="stdlib tomllib requires Python 3.11+")

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]

    specs = {spec.name: spec for spec in instrumentation.REGISTRY}
    aggregate = "instruments"

    # Every extra maps to a registry entry that names it, so a stray or
    # renamed extra cannot drift away from the registry unnoticed.
    for name in extras:
        if name == aggregate:
            continue
        assert name in specs, f"extra {name!r} has no registry entry"
        assert specs[name].extra == name

    # No extra exists for a built-in instrumentation, and nothing an extra
    # installs is a library we merely instrument.
    builtin_libraries = {spec.library for spec in specs.values() if spec.library}
    for name, requirements in extras.items():
        assert name not in builtin_libraries, f"{name!r} is a built-in instrumentation"
        for requirement in requirements:
            package = requirement.split(">=")[0].split("==")[0].strip()
            assert package not in builtin_libraries, (
                f"extra {name!r} installs {package!r}, a library the SDK only instruments"
            )
