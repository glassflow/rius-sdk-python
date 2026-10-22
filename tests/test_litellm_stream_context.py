"""A streamed LiteLLM call nests its OpenAI span and counts its usage once.

openinference-instrumentation-litellm does not make a stream's span current
while it calls LiteLLM, so the openai instrumentor's span for the same call
lands in a trace of its own. ``rius.litellm_stream_context`` runs that call
with the span current. The end-to-end tests drive real LiteLLM with both
instrumentors through ``init()``, every HTTP request answered in-process.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init, litellm_stream_context
from rius.litellm_stream_context import VERIFIED_VERSIONS, _verified

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no cost-map download

litellm = pytest.importorskip("litellm")
httpx = pytest.importorskip("httpx")
litellm_instrumentation = pytest.importorskip("openinference.instrumentation.litellm")
openai_instrumentation = pytest.importorskip("openinference.instrumentation.openai")

LiteLLMInstrumentor = litellm_instrumentation.LiteLLMInstrumentor
OpenAIInstrumentor = openai_instrumentation.OpenAIInstrumentor

CHAT = "https://api.openai.com/v1/chat/completions"
USAGE = {"prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51}


def _sse_body() -> bytes:
    def chunk(delta: dict[str, Any], finish: str | None = None, usage: Any = None) -> str:
        body: dict[str, Any] = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage:
            body["usage"] = usage
        return f"data: {json.dumps(body)}\n\n"

    events = [
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "Paris."}),
        chunk({}, "stop"),
        chunk({}, usage=USAGE),
        "data: [DONE]\n\n",
    ]
    return "".join(events).encode()


def _release_instrumentors() -> None:
    """Undo both instrumentors, so init() enables and binds them afresh.

    Other tests' init() calls leave them instrumented against an older
    provider; init() would then treat them as someone else's and skip them.
    """
    litellm_stream_context.uninstall(LiteLLMInstrumentor())
    for instrumentor in (LiteLLMInstrumentor(), OpenAIInstrumentor()):
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.fixture
def openai_answers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Answer every OpenAI chat request with a streamed completion carrying usage."""

    def answer(request: Any) -> Any:
        assert str(request.url) == CHAT
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body()
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: answer(req))

    async def answer_async(self: Any, request: Any) -> Any:
        return answer(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", answer_async)
    # LiteLLM's async path defaults to its own aiohttp transport; use httpx's,
    # which the patch above answers.
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("this test must not reach the network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    _release_instrumentors()
    yield
    _release_instrumentors()


class Exported:
    """The spans init() exported, read after a flush (export is batched)."""

    def __init__(self, exporter: InMemorySpanExporter, client: Any) -> None:
        self._exporter = exporter
        self._client = client

    def by_name(self) -> dict[str, ReadableSpan]:
        self._client.flush()
        return {span.name: span for span in self._exporter.get_finished_spans()}


@pytest.fixture
def exported(openai_answers: None) -> Iterator[Exported]:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        instruments=["openai", "litellm"],
        heartbeat=False,
        partial_spans=False,
    )
    yield Exported(exporter, client)
    client.shutdown()


def _assert_counted_once(spans: dict[str, ReadableSpan], wrapper: str) -> None:
    outer, inner = spans[wrapper], spans["ChatCompletion"]
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id
    # The OpenAI span is the call, so it keeps the usage; LiteLLM's copy moves aside.
    assert (inner.attributes or {}).get("gen_ai.usage.input_tokens") == 42
    outer_attributes = dict(outer.attributes or {})
    assert not any(key.startswith("gen_ai.usage.") for key in outer_attributes)
    assert outer_attributes.get("rius.aggregated_usage.input_tokens") == 42


def test_a_streamed_completion_nests_and_counts_once(
    exported: Exported,
) -> None:
    for _ in litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
    ):
        pass

    _assert_counted_once(exported.by_name(), "completion")


def test_a_streamed_acompletion_nests_and_counts_once(
    exported: Exported,
) -> None:
    async def run() -> None:
        stream = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Capital of France?"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for _ in stream:
            pass

    asyncio.run(run())

    _assert_counted_once(exported.by_name(), "acompletion")


def test_the_stream_span_does_not_stay_current_for_the_caller(
    exported: Exported,
) -> None:
    stream = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert not trace.get_current_span().get_span_context().is_valid
    for _ in stream:
        pass


def test_a_re_init_does_not_stack_the_wrappers(openai_answers: None) -> None:
    def start() -> None:
        init(
            span_exporter=InMemorySpanExporter(),
            set_global=False,
            instruments=["openai", "litellm"],
            heartbeat=False,
            partial_spans=False,
        ).shutdown()

    try:
        start()
        start()  # the SDK's own re-bind path
        # Someone else uninstruments without us: the instrumentor puts our
        # wrapper back on litellm and records it as "original" next time.
        LiteLLMInstrumentor().uninstrument()
        start()
        entry = LiteLLMInstrumentor.original_litellm_funcs["completion"]
        inner = getattr(entry, litellm_stream_context.MARK)
        assert not hasattr(inner, litellm_stream_context.MARK)
    finally:
        _release_instrumentors()


# --- the hook itself, on a stand-in instrumentor -------------------------------


class _FakeInstrumentor:
    """The two private details the hook relies on, and a stream wrapper using them."""

    original_litellm_funcs: dict[str, Any] = {}

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def completion_stream(self, **kwargs: Any) -> Any:
        span = self._tracer.start_span("completion")
        try:
            return self.original_litellm_funcs["completion"](stream=True, **kwargs)
        finally:
            span.end()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[_FakeInstrumentor, list[Any]]]:
    seen: list[Any] = []
    tracer = TracerProvider().get_tracer("fake")
    monkeypatch.setattr(
        _FakeInstrumentor,
        "original_litellm_funcs",
        {"completion": lambda **kwargs: seen.append(trace.get_current_span())},
    )
    monkeypatch.setattr(litellm_stream_context, "version", lambda _: VERIFIED_VERSIONS[1])
    yield _FakeInstrumentor(tracer), seen


def test_the_wrapped_call_runs_under_the_span_it_was_started_for(
    fake: tuple[_FakeInstrumentor, list[Any]],
) -> None:
    instrumentor, seen = fake
    assert litellm_stream_context.install(instrumentor)

    instrumentor.completion_stream()

    assert seen[0].name == "completion"


def test_without_the_hook_the_wrapped_call_sees_no_span(
    fake: tuple[_FakeInstrumentor, list[Any]],
) -> None:
    instrumentor, seen = fake

    instrumentor.completion_stream()

    assert not seen[0].get_span_context().is_valid


def test_an_unverified_instrumentor_version_is_left_alone(
    fake: tuple[_FakeInstrumentor, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    instrumentor, seen = fake
    monkeypatch.setattr(litellm_stream_context, "version", lambda _: "0.1.51")

    assert not litellm_stream_context.install(instrumentor)
    instrumentor.completion_stream()

    assert not seen[0].get_span_context().is_valid


def test_uninstall_restores_the_instrumentor(
    fake: tuple[_FakeInstrumentor, list[Any]],
) -> None:
    instrumentor, seen = fake
    tracer = instrumentor._tracer
    original = instrumentor.original_litellm_funcs["completion"]
    litellm_stream_context.install(instrumentor)

    litellm_stream_context.uninstall(instrumentor)

    assert instrumentor._tracer is tracer
    assert instrumentor.original_litellm_funcs["completion"] is original


def test_install_never_raises() -> None:
    class Broken:
        @property
        def _tracer(self) -> Any:
            raise RuntimeError("private internals moved")

    assert not litellm_stream_context.install(Broken())


@pytest.mark.parametrize(
    ("installed", "expected"),
    [("0.1.33", False), ("0.1.34", True), ("0.1.50", True), ("0.1.51", False), ("1.0.0", False)],
)
def test_verified_versions(installed: str, expected: bool) -> None:
    assert _verified(installed) is expected
