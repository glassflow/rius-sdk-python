"""A LangChain chat model call nests its provider span and counts its usage once.

openinference-instrumentation-langchain never makes its spans current, so the
openai instrumentor's span for the same call lands in a trace of its own.
``rius.langchain_span_context`` runs the model call with the LangChain span
current. The end-to-end tests drive a real ``ChatOpenAI`` with both
instrumentors through ``init()``, every HTTP request answered in-process.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init, langchain_span_context
from rius.langchain_span_context import VERIFIED_VERSIONS, _verified

httpx = pytest.importorskip("httpx")
langchain_openai = pytest.importorskip("langchain_openai")
langchain_core = pytest.importorskip("langchain_core")
langchain_instrumentation = pytest.importorskip("openinference.instrumentation.langchain")
openai_instrumentation = pytest.importorskip("openinference.instrumentation.openai")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

ChatOpenAI = langchain_openai.ChatOpenAI
LangChainInstrumentor = langchain_instrumentation.LangChainInstrumentor
OpenAIInstrumentor = openai_instrumentation.OpenAIInstrumentor

CHAT = "https://api.openai.com/v1/chat/completions"
USAGE = {"prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51}


def _completion() -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Paris."},
                "finish_reason": "stop",
            }
        ],
        "usage": USAGE,
    }


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
        chunk({"content": "Par"}),
        chunk({"content": "is."}),
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
    langchain_span_context.uninstall(LangChainInstrumentor())
    for instrumentor in (LangChainInstrumentor(), OpenAIInstrumentor()):
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


@pytest.fixture
def openai_answers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Answer every OpenAI chat request, streamed or not, with usage."""

    def answer(request: Any) -> Any:
        assert str(request.url) == CHAT
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse_body()
            )
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: answer(req))

    async def answer_async(self: Any, request: Any) -> Any:
        return answer(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", answer_async)
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

    def spans(self) -> list[ReadableSpan]:
        self._client.flush()
        return list(self._exporter.get_finished_spans())


@pytest.fixture
def exported(openai_answers: None) -> Iterator[Exported]:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        instruments=["openai", "langchain"],
        heartbeat=False,
        partial_spans=False,
    )
    yield Exported(exporter, client)
    client.shutdown()


def _model(**kwargs: Any) -> Any:
    return ChatOpenAI(model="gpt-4o-mini", max_retries=0, **kwargs)


def _pairs(spans: list[ReadableSpan]) -> list[tuple[ReadableSpan, ReadableSpan]]:
    """Each ChatOpenAI span with the ChatCompletion span it is the parent of."""
    by_id = {span.context.span_id: span for span in spans}
    pairs = []
    for span in spans:
        if span.name != "ChatCompletion":
            continue
        assert span.parent is not None, "the provider span has no parent"
        outer = by_id[span.parent.span_id]
        assert outer.name == "ChatOpenAI"
        assert outer.context.trace_id == span.context.trace_id
        pairs.append((outer, span))
    return pairs


def _assert_counted_once(outer: ReadableSpan, inner: ReadableSpan) -> None:
    # The provider span is the call, so it keeps the usage; LangChain's copy moves aside.
    inner_attributes = dict(inner.attributes or {})
    outer_attributes = dict(outer.attributes or {})
    assert inner_attributes.get("gen_ai.usage.input_tokens") == 42
    assert inner_attributes.get("gen_ai.usage.output_tokens") == 9
    assert not any(key.startswith("gen_ai.usage.") for key in outer_attributes)
    assert outer_attributes.get("rius.aggregated_usage.input_tokens") == 42
    assert outer_attributes.get("rius.aggregated_usage.output_tokens") == 9


def test_invoke_nests_and_counts_once(exported: Exported) -> None:
    _model().invoke([HumanMessage("Capital of France?")])

    (pair,) = _pairs(exported.spans())
    _assert_counted_once(*pair)


def test_ainvoke_nests_and_counts_once(exported: Exported) -> None:
    asyncio.run(_model().ainvoke([HumanMessage("Capital of France?")]))

    (pair,) = _pairs(exported.spans())
    _assert_counted_once(*pair)


def test_stream_nests_and_counts_once(exported: Exported) -> None:
    chunks = list(_model(stream_usage=True).stream([HumanMessage("Capital of France?")]))

    assert "".join(str(chunk.content) for chunk in chunks) == "Paris."
    (pair,) = _pairs(exported.spans())
    _assert_counted_once(*pair)


def test_two_calls_in_a_chain_each_match_their_own(exported: Exported) -> None:
    prompt = ChatPromptTemplate.from_messages([("human", "Capital of {country}?")])
    model = _model()
    (prompt | model | (lambda _: {"country": "Germany"}) | prompt | model).invoke(
        {"country": "France"}
    )

    pairs = _pairs(exported.spans())
    assert len(pairs) == 2
    assert pairs[0][0].context.span_id != pairs[1][0].context.span_id
    for pair in pairs:
        _assert_counted_once(*pair)


def test_parallel_identical_calls_each_match_their_own(exported: Exported) -> None:
    # Same prompt, same model, same usage, run on several threads at once:
    # nothing but the span context can tell the calls apart.
    model = _model()
    model.batch([[HumanMessage("Capital of France?")]] * 4, config={"max_concurrency": 4})

    pairs = _pairs(exported.spans())
    assert len(pairs) == 4
    assert len({outer.context.span_id for outer, _ in pairs}) == 4
    for pair in pairs:
        _assert_counted_once(*pair)


def test_the_span_is_not_left_current(exported: Exported) -> None:
    model = _model(stream_usage=True)
    model.invoke([HumanMessage("Capital of France?")])
    assert not trace.get_current_span().get_span_context().is_valid

    # Between stream chunks the caller's own code runs: it must not run
    # under LangChain's span either.
    for _ in model.stream([HumanMessage("Capital of France?")]):
        assert not trace.get_current_span().get_span_context().is_valid
    assert not trace.get_current_span().get_span_context().is_valid


def test_an_abandoned_stream_leaves_nothing_current(exported: Exported) -> None:
    stream = _model(stream_usage=True).stream([HumanMessage("Capital of France?")])
    next(stream)
    stream.close()
    assert not trace.get_current_span().get_span_context().is_valid


def test_without_the_provider_instrumentor_langchain_keeps_its_usage(
    openai_answers: None,
) -> None:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        instruments=["langchain"],
        heartbeat=False,
        partial_spans=False,
    )
    try:
        _model().invoke([HumanMessage("Capital of France?")])
        client.flush()
        (span,) = [s for s in exporter.get_finished_spans() if s.name == "ChatOpenAI"]
        assert (span.attributes or {}).get("gen_ai.usage.input_tokens") == 42
    finally:
        client.shutdown()


def test_reinit_does_not_stack_wrappers(openai_answers: None) -> None:
    for _ in range(2):
        client = init(
            span_exporter=InMemorySpanExporter(),
            set_global=False,
            instruments=["openai", "langchain"],
            heartbeat=False,
            partial_spans=False,
        )
        client.shutdown()
        _release_instrumentors()
        client = init(
            span_exporter=InMemorySpanExporter(),
            set_global=False,
            instruments=["openai", "langchain"],
            heartbeat=False,
            partial_spans=False,
        )
        client.shutdown()
    wrapped = BaseChatModel.__dict__["_generate_with_cache"]
    inner = getattr(wrapped, langchain_span_context._MARK, wrapped)
    assert not hasattr(inner, langchain_span_context._MARK)


def test_uninstall_restores_the_chat_model(openai_answers: None) -> None:
    originals = {name: BaseChatModel.__dict__[name] for name in langchain_span_context._HOOKS}
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument()
    try:
        assert langchain_span_context.install(instrumentor)
        assert all(
            hasattr(BaseChatModel.__dict__[name], langchain_span_context._MARK)
            for name in langchain_span_context._HOOKS
        )
        langchain_span_context.uninstall(instrumentor)
        assert {name: BaseChatModel.__dict__[name] for name in originals} == originals
        assert "_start_trace" not in instrumentor._tracer.__dict__
    finally:
        instrumentor.uninstrument()


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (VERIFIED_VERSIONS[0], True),
        (VERIFIED_VERSIONS[1], True),
        ("0.1.66", False),
        ("0.1.80", False),
        ("0.2.0", False),
    ],
)
def test_only_verified_instrumentor_versions_install(installed: str, expected: bool) -> None:
    assert _verified(installed) is expected


def test_a_failing_hook_never_reaches_the_caller(
    exported: Exported, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(run_id: Any) -> Any:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(langchain_span_context._tracer, "get_span", broken)
    result = _model().invoke([HumanMessage("Capital of France?")])
    assert result.content == "Paris."
