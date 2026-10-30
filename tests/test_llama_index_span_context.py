"""A LlamaIndex LLM call nests its provider span and counts its usage once.

openinference-instrumentation-llama-index never makes its spans current, so the
openai instrumentor's span for the same call lands in a trace of its own.
``rius.llama_index_span_context`` runs the call with the LlamaIndex span
current. The end-to-end tests drive a real ``llama_index.llms.openai.OpenAI``
with both instrumentors through ``init()``, every HTTP request answered
in-process.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import socket
import threading
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init, llama_index_span_context
from rius.llama_index_span_context import VERIFIED_VERSIONS, _verified

httpx = pytest.importorskip("httpx")
llama_index_openai = pytest.importorskip("llama_index.llms.openai")
llama_index_instrumentation = pytest.importorskip("openinference.instrumentation.llama_index")
openai_instrumentation = pytest.importorskip("openinference.instrumentation.openai")

from llama_index.core.base.llms.base import BaseLLM  # noqa: E402
from llama_index.core.instrumentation import get_dispatcher  # noqa: E402
from llama_index.core.llms import ChatMessage  # noqa: E402
from llama_index.core.llms.callbacks import llm_chat_callback  # noqa: E402

OpenAI = llama_index_openai.OpenAI
LlamaIndexInstrumentor = llama_index_instrumentation.LlamaIndexInstrumentor
OpenAIInstrumentor = openai_instrumentation.OpenAIInstrumentor

CHAT = "https://api.openai.com/v1/chat/completions"
USAGE = {"prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51}
MESSAGES = [ChatMessage(role="user", content="Capital of France?")]


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
    llama_index_span_context.uninstall(LlamaIndexInstrumentor())
    for instrumentor in (LlamaIndexInstrumentor(), OpenAIInstrumentor()):
        if instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


class Answers:
    """Answers every OpenAI chat request; ``fail`` makes them fail instead."""

    def __init__(self) -> None:
        self.fail = False

    def __call__(self, request: Any) -> Any:
        assert str(request.url) == CHAT
        if self.fail:
            return httpx.Response(500, json={"error": {"message": "boom", "type": "server"}})
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse_body()
            )
        return httpx.Response(200, json=_completion())


@pytest.fixture
def openai_answers(monkeypatch: pytest.MonkeyPatch) -> Iterator[Answers]:
    """Answer every OpenAI chat request, streamed or not, with usage."""
    answer = Answers()
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: answer(req))

    async def answer_async(self: Any, request: Any) -> Any:
        return answer(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", answer_async)

    def no_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("this test must not reach the network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    _release_instrumentors()
    yield answer
    _release_instrumentors()


class Exported:
    """The spans init() exported, read after a flush (export is batched)."""

    def __init__(self, exporter: InMemorySpanExporter, client: Any, answers: Answers) -> None:
        self._exporter = exporter
        self._client = client
        self.answers = answers

    def spans(self) -> list[ReadableSpan]:
        self._client.flush()
        return list(self._exporter.get_finished_spans())


@pytest.fixture
def exported(openai_answers: Answers) -> Iterator[Exported]:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        instruments=["openai", "llama-index"],
        heartbeat=False,
        partial_spans=False,
    )
    yield Exported(exporter, client, openai_answers)
    client.shutdown()


def _llm() -> Any:
    return OpenAI(model="gpt-4o-mini", api_key="sk-test", max_retries=0)


def _provider_pairs(spans: list[ReadableSpan]) -> list[tuple[ReadableSpan, ReadableSpan]]:
    """Each ChatCompletion span with its parent, which must be in the same trace."""
    by_id = {span.context.span_id: span for span in spans}
    pairs = []
    for span in spans:
        if span.name != "ChatCompletion":
            continue
        assert span.parent is not None, "the provider span has no parent"
        outer = by_id[span.parent.span_id]
        assert outer.context.trace_id == span.context.trace_id
        pairs.append((outer, span))
    return pairs


def _pairs(spans: list[ReadableSpan], name: str) -> list[tuple[ReadableSpan, ReadableSpan]]:
    """Each LlamaIndex ``name`` span with the ChatCompletion span it is the parent of."""
    pairs = _provider_pairs(spans)
    assert [outer.name for outer, _ in pairs] == [name] * len(pairs)
    return pairs


def _assert_counted_once(outer: ReadableSpan, inner: ReadableSpan) -> None:
    # The provider span is the call, so it keeps the usage; LlamaIndex's copy moves aside.
    inner_attributes = dict(inner.attributes or {})
    outer_attributes = dict(outer.attributes or {})
    assert inner_attributes.get("gen_ai.usage.input_tokens") == 42
    assert inner_attributes.get("gen_ai.usage.output_tokens") == 9
    assert not any(key.startswith("gen_ai.usage.") for key in outer_attributes)
    assert outer_attributes.get("rius.aggregated_usage.input_tokens") == 42
    assert outer_attributes.get("rius.aggregated_usage.output_tokens") == 9


def _nothing_current() -> bool:
    return not trace.get_current_span().get_span_context().is_valid


def test_chat_nests_and_counts_once(exported: Exported) -> None:
    assert str(_llm().chat(MESSAGES).message.content) == "Paris."

    (pair,) = _pairs(exported.spans(), "OpenAI.chat")
    _assert_counted_once(*pair)
    assert _nothing_current()


def test_a_call_leaves_the_context_as_it_found_it(exported: Exported) -> None:
    # An instrumentor that makes its span current itself (4.6.1 does, through a
    # proxy of the span) must not get a second attach from us: detached out of
    # order, it would leave the instrumentor's context behind.
    before = otel_context.get_current()
    _llm().chat(MESSAGES)
    assert otel_context.get_current() is before


def test_complete_nests_and_counts_once(exported: Exported) -> None:
    assert _llm().complete("Capital of France?").text == "Paris."

    (pair,) = _pairs(exported.spans(), "OpenAI.complete")
    _assert_counted_once(*pair)


def test_achat_nests_and_counts_once(exported: Exported) -> None:
    asyncio.run(_llm().achat(MESSAGES))

    (pair,) = _pairs(exported.spans(), "OpenAI.achat")
    _assert_counted_once(*pair)


def test_acomplete_nests_and_counts_once(exported: Exported) -> None:
    asyncio.run(_llm().acomplete("Capital of France?"))

    (pair,) = _pairs(exported.spans(), "OpenAI.acomplete")
    _assert_counted_once(*pair)


def test_stream_chat_nests_and_counts_once(exported: Exported) -> None:
    chunks = list(_llm().stream_chat(MESSAGES))

    assert str(chunks[-1].message.content) == "Paris."
    (pair,) = _pairs(exported.spans(), "OpenAI.stream_chat")
    _assert_counted_once(*pair)


def test_stream_complete_nests_and_counts_once(exported: Exported) -> None:
    chunks = list(_llm().stream_complete("Capital of France?"))

    assert chunks[-1].text == "Paris."
    (pair,) = _pairs(exported.spans(), "OpenAI.stream_complete")
    _assert_counted_once(*pair)


def test_astream_chat_nests_and_counts_once(exported: Exported) -> None:
    async def run() -> list[Any]:
        return [chunk async for chunk in await _llm().astream_chat(MESSAGES)]

    chunks = asyncio.run(run())

    assert str(chunks[-1].message.content) == "Paris."
    (pair,) = _pairs(exported.spans(), "OpenAI.astream_chat")
    _assert_counted_once(*pair)


def test_astream_complete_nests_and_counts_once(exported: Exported) -> None:
    async def run() -> list[Any]:
        return [chunk async for chunk in await _llm().astream_complete("Capital of France?")]

    chunks = asyncio.run(run())

    assert chunks[-1].text == "Paris."
    (pair,) = _pairs(exported.spans(), "OpenAI.astream_complete")
    _assert_counted_once(*pair)


def test_the_span_is_not_left_current_between_stream_chunks(exported: Exported) -> None:
    # Between chunks the caller's own code runs: it must not run under
    # LlamaIndex's span.
    for _ in _llm().stream_chat(MESSAGES):
        assert _nothing_current()
    assert _nothing_current()

    async def run() -> None:
        async for _ in await _llm().astream_chat(MESSAGES):
            assert _nothing_current()
        assert _nothing_current()

    asyncio.run(run())


def test_an_abandoned_stream_leaves_nothing_current(exported: Exported) -> None:
    stream = _llm().stream_chat(MESSAGES)
    next(stream)
    stream.close()
    assert _nothing_current()

    async def run() -> None:
        stream = await _llm().astream_chat(MESSAGES)
        await stream.__anext__()
        await stream.aclose()
        assert _nothing_current()

    asyncio.run(run())


def test_closing_a_wrapped_stream_closes_the_stream_it_wraps() -> None:
    # The inner stream stays referenced here, so the garbage collector cannot
    # be what closes it: only the wrapper's own close() can.
    closed: list[str] = []

    def inner() -> Iterator[int]:
        try:
            yield 1
            yield 2
        except GeneratorExit:
            closed.append("sync")
            raise

    source = inner()
    wrapped = llama_index_span_context._steps_under(SPAN, source)
    assert next(wrapped) == 1
    wrapped.close()
    assert closed == ["sync"]
    assert _nothing_current()

    async def ainner() -> AsyncIterator[int]:
        try:
            yield 1
            yield 2
        except GeneratorExit:
            closed.append("async")
            raise

    async def run() -> None:
        asource = ainner()
        awrapped = llama_index_span_context._asteps_under(SPAN, asource)
        assert await awrapped.__anext__() == 1
        await awrapped.aclose()
        assert closed == ["sync", "async"]
        assert _nothing_current()

    asyncio.run(run())


def _usage_records(spans: list[ReadableSpan]) -> int:
    return sum(1 for s in spans if "gen_ai.usage.input_tokens" in (s.attributes or {}))


def test_an_abandoned_stream_never_counts_its_usage_twice(exported: Exported) -> None:
    # Abandoned after every chunk it yields, so the provider stream has sent
    # its usage but was never read to its end. The LlamaIndex span ends with
    # no usage; the openai instrumentor ends its span only on the provider
    # stream's end, so the call's usage is counted at most once (today: not
    # at all, since that span never ends).
    def abandon() -> None:
        stream = _llm().stream_chat(MESSAGES)
        for _ in range(5):
            next(stream)
        stream.close()

    async def aabandon() -> None:
        stream = await _llm().astream_chat(MESSAGES)
        for _ in range(5):
            await stream.__anext__()
        await stream.aclose()

    abandon()
    asyncio.run(aabandon())

    spans = exported.spans()
    outer = [s for s in spans if s.name in ("OpenAI.stream_chat", "OpenAI.astream_chat")]
    assert len(outer) == 2
    assert _usage_records(spans) <= 1
    assert _usage_records(outer) == 0


def test_a_failing_call_leaves_nothing_current(exported: Exported) -> None:
    exported.answers.fail = True
    with pytest.raises(Exception, match="boom"):
        _llm().chat(MESSAGES)
    assert _nothing_current()

    async def run() -> None:
        with pytest.raises(Exception, match="boom"):
            await _llm().achat(MESSAGES)
        assert _nothing_current()
        with pytest.raises(Exception, match="boom"):
            async for _ in await _llm().astream_chat(MESSAGES):
                pass
        assert _nothing_current()

    asyncio.run(run())
    with pytest.raises(Exception, match="boom"):
        list(_llm().stream_chat(MESSAGES))
    assert _nothing_current()


def test_a_failing_call_still_nests(exported: Exported) -> None:
    exported.answers.fail = True
    with pytest.raises(Exception, match="boom"):
        _llm().chat(MESSAGES)

    spans = exported.spans()
    (provider,) = [s for s in spans if s.name == "ChatCompletion"]
    (outer,) = [s for s in spans if s.name == "OpenAI.chat"]
    assert provider.parent is not None
    assert provider.parent.span_id == outer.context.span_id


def test_parallel_calls_in_threads_each_match_their_own(exported: Exported) -> None:
    # Same prompt, same model, same usage, on several threads at once: nothing
    # but the span context can tell the calls apart.
    llm = _llm()
    barrier = threading.Barrier(4)

    def call(stream: bool) -> None:
        barrier.wait()
        if stream:
            list(llm.stream_chat(MESSAGES))
        else:
            llm.chat(MESSAGES)
        assert _nothing_current()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(call, [False, True, False, True]))

    pairs = _provider_pairs(exported.spans())
    assert sorted(outer.name for outer, _ in pairs) == [
        "OpenAI.chat",
        "OpenAI.chat",
        "OpenAI.stream_chat",
        "OpenAI.stream_chat",
    ]
    assert len({outer.context.span_id for outer, _ in pairs}) == 4
    for pair in pairs:
        _assert_counted_once(*pair)


def test_concurrent_tasks_each_match_their_own(exported: Exported) -> None:
    llm = _llm()

    async def stream() -> None:
        async for _ in await llm.astream_chat(MESSAGES):
            await asyncio.sleep(0)

    async def run() -> None:
        await asyncio.gather(llm.achat(MESSAGES), stream(), llm.achat(MESSAGES), stream())
        assert _nothing_current()

    asyncio.run(run())

    pairs = _provider_pairs(exported.spans())
    assert len({outer.context.span_id for outer, _ in pairs}) == 4
    assert sorted(outer.name for outer, _ in pairs) == [
        "OpenAI.achat",
        "OpenAI.achat",
        "OpenAI.astream_chat",
        "OpenAI.astream_chat",
    ]
    for pair in pairs:
        _assert_counted_once(*pair)


def test_interleaved_streams_each_match_their_own(exported: Exported) -> None:
    # Two streams opened before either is read, then read alternately: each
    # step must run under its own stream's span, not the latest one's.
    llm = _llm()
    first = llm.stream_chat(MESSAGES)
    second = llm.stream_chat(MESSAGES)
    for a, b in itertools.zip_longest(first, second):
        assert a is not None and b is not None

    spans = exported.spans()
    pairs = _pairs(spans, "OpenAI.stream_chat")
    assert len(pairs) == 2
    assert len({outer.context.span_id for outer, _ in pairs}) == 2


def test_a_subclass_defined_after_init_is_covered(exported: Exported) -> None:
    # Its own stream_chat, not delegating to one wrapped at install.
    class LaterOpenAI(OpenAI):  # type: ignore[misc, valid-type]
        @llm_chat_callback()  # type: ignore[untyped-decorator, misc]
        def stream_chat(self, messages: Any, **kwargs: Any) -> Any:
            return self._stream_chat(messages, **kwargs)

    list(LaterOpenAI(model="gpt-4o-mini", api_key="sk-test", max_retries=0).stream_chat(MESSAGES))

    spans = exported.spans()
    (provider,) = [s for s in spans if s.name == "ChatCompletion"]
    assert provider.parent is not None
    names = {s.context.span_id: s.name for s in spans}
    assert names[provider.parent.span_id] == "LaterOpenAI.stream_chat"


def test_the_subclass_hook_still_runs_an_ancestors_hook(
    openai_answers: Answers, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing above BaseLLM defines this hook today; one that does (a library's
    # or a user's base, a mixin) must keep running for classes defined later.
    seen: list[Any] = []

    def recording_hook(cls: Any, /, **kwargs: Any) -> None:
        seen.append(cls)

    ancestor = BaseLLM.__mro__[1]
    monkeypatch.setattr(ancestor, "__pydantic_init_subclass__", classmethod(recording_hook))
    client = init(
        span_exporter=InMemorySpanExporter(),
        set_global=False,
        instruments=["openai", "llama-index"],
        heartbeat=False,
        partial_spans=False,
    )
    try:

        class LaterOpenAI(OpenAI):  # type: ignore[misc, valid-type]
            @llm_chat_callback()  # type: ignore[untyped-decorator, misc]
            def stream_chat(self, messages: Any, **kwargs: Any) -> Any:
                return self._stream_chat(messages, **kwargs)

        assert seen == [LaterOpenAI]
        assert hasattr(LaterOpenAI.__dict__["stream_chat"], llama_index_span_context.MARK)
    finally:
        client.shutdown()


def test_without_the_provider_instrumentor_llama_index_keeps_its_usage(
    openai_answers: Answers,
) -> None:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        instruments=["llama-index"],
        heartbeat=False,
        partial_spans=False,
    )
    try:
        _llm().chat(MESSAGES)
        client.flush()
        (span,) = [s for s in exporter.get_finished_spans() if s.name == "OpenAI.chat"]
        assert (span.attributes or {}).get("gen_ai.usage.input_tokens") == 42
    finally:
        client.shutdown()


def _stream_methods() -> dict[str, Any]:
    return {name: OpenAI.__dict__[name] for name in llama_index_span_context._STREAM_METHODS}


def test_reinit_does_not_stack_hooks(openai_answers: Answers) -> None:
    for _ in range(2):
        client = init(
            span_exporter=InMemorySpanExporter(),
            set_global=False,
            instruments=["openai", "llama-index"],
            heartbeat=False,
            partial_spans=False,
        )
        client.shutdown()
        client = init(
            span_exporter=InMemorySpanExporter(),
            set_global=False,
            instruments=["openai", "llama-index"],
            heartbeat=False,
            partial_spans=False,
        )
        client.shutdown()
    for wrapped in _stream_methods().values():
        inner = getattr(wrapped, llama_index_span_context.MARK, wrapped)
        assert not hasattr(inner, llama_index_span_context.MARK)
    ours = [
        h
        for h in get_dispatcher().span_handlers
        if isinstance(h, llama_index_span_context._handler_class())
    ]
    assert len(ours) == 1


def test_installing_twice_does_not_stack(openai_answers: Answers) -> None:
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    try:
        assert llama_index_span_context.install(instrumentor)
        assert llama_index_span_context.install(instrumentor)
        for wrapped in _stream_methods().values():
            inner = getattr(wrapped, llama_index_span_context.MARK)
            assert not hasattr(inner, llama_index_span_context.MARK)
    finally:
        llama_index_span_context.uninstall(instrumentor)
        instrumentor.uninstrument()


def test_our_handler_runs_after_the_instrumentors_after_a_reinstrument(
    openai_answers: Answers,
) -> None:
    # The instrumentor's uninstrument keeps our handler and its instrument then
    # appends its own after it; install moves ours back behind it, or the span
    # would not exist yet when ours looks it up.
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    try:
        assert llama_index_span_context.install(instrumentor)
        instrumentor.uninstrument()
        instrumentor.instrument()
        assert llama_index_span_context.install(instrumentor)
        handlers = list(get_dispatcher().span_handlers)
        ours = handlers.index(llama_index_span_context._handler)
        assert ours > handlers.index(instrumentor._span_handler)
    finally:
        llama_index_span_context.uninstall(instrumentor)
        instrumentor.uninstrument()


def test_uninstall_restores_everything(openai_answers: Answers) -> None:
    originals = _stream_methods()
    hook = BaseLLM.__dict__.get("__pydantic_init_subclass__")
    handlers = list(get_dispatcher().span_handlers)
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    try:
        assert llama_index_span_context.install(instrumentor)
        assert all(hasattr(m, llama_index_span_context.MARK) for m in _stream_methods().values())
        llama_index_span_context.uninstall(instrumentor)
        assert _stream_methods() == originals
        assert BaseLLM.__dict__.get("__pydantic_init_subclass__") == hook
    finally:
        instrumentor.uninstrument()
    assert list(get_dispatcher().span_handlers) == handlers


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (VERIFIED_VERSIONS[0], True),
        (VERIFIED_VERSIONS[1], True),
        ("4.4.2", False),
        ("4.7.1", False),
        ("5.0.0", False),
        ("4.5.0", True),
        ("4.6.1", True),
        ("4.7", True),
        # A pre-release comes before its release, a post-release after it.
        ("4.7.0rc1", True),
        ("4.7.0.dev0", True),
        ("4.4.3rc1", False),
        ("4.4.3.post1", True),
        ("4.7.0.post1", False),
        ("4.7.0post1", False),
        ("4.7.0+local", False),
        ("not-a-version", False),
    ],
)
def test_only_verified_instrumentor_versions_install(installed: str, expected: bool) -> None:
    assert _verified(installed) is expected


def test_an_unverified_version_installs_nothing(
    openai_answers: Answers, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llama_index_span_context, "version", lambda name: "9.9.9")
    originals = _stream_methods()
    handlers = list(get_dispatcher().span_handlers)
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    try:
        assert not llama_index_span_context.install(instrumentor)
        assert _stream_methods() == originals
        assert [h for h in get_dispatcher().span_handlers if h not in handlers] == [
            instrumentor._span_handler
        ]
    finally:
        instrumentor.uninstrument()


def test_a_failing_lookup_never_reaches_the_caller(
    exported: Exported, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> Any:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(llama_index_span_context, "_current_llama_index_span", broken)
    assert str(_llm().chat(MESSAGES).message.content) == "Paris."
    assert str(list(_llm().stream_chat(MESSAGES))[-1].message.content) == "Paris."
    assert _nothing_current()


def test_a_failing_stream_lookup_never_reaches_the_caller(
    exported: Exported, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: Any, stream: Any) -> Any:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(llama_index_span_context._handler_class(), "take_stream_span", broken)
    assert str(list(_llm().stream_chat(MESSAGES))[-1].message.content) == "Paris."

    async def run() -> list[Any]:
        return [chunk async for chunk in await _llm().astream_chat(MESSAGES)]

    assert str(asyncio.run(run())[-1].message.content) == "Paris."


def _a_span(span_id: int = 2) -> Any:
    return trace.NonRecordingSpan(
        trace.SpanContext(1, span_id, is_remote=False, trace_flags=trace.TraceFlags(1))
    )


#: The span the handler fixture's lookup finds.
SPAN = _a_span()


@pytest.fixture
def handler(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fresh, enabled handler whose span lookup finds ``SPAN``."""
    instance = llama_index_span_context._handler_class()()
    instance.enable(True)
    monkeypatch.setattr(llama_index_span_context, "_current_llama_index_span", lambda: SPAN)
    return instance


def test_the_handler_attaches_for_an_llm_span_and_detaches_on_exit(handler: Any) -> None:
    handler.span_enter(id_="OpenAI.chat-1", instance=_llm())
    assert trace.get_current_span() is SPAN
    handler.span_exit(id_="OpenAI.chat-1", instance=_llm(), result=None)
    assert _nothing_current()


def test_the_handler_detaches_on_drop(handler: Any) -> None:
    handler.span_enter(id_="OpenAI.chat-1", instance=_llm())
    handler.span_drop(id_="OpenAI.chat-1", instance=_llm(), err=RuntimeError())
    assert _nothing_current()


def test_the_handler_leaves_spans_that_are_not_an_llms_alone(handler: Any) -> None:
    handler.span_enter(id_="Retriever.retrieve-1", instance=object())
    assert _nothing_current()
    assert not handler.busy()


def test_a_disabled_handler_attaches_nothing(handler: Any) -> None:
    handler.enable(False)
    handler.span_enter(id_="OpenAI.chat-1", instance=_llm())
    assert _nothing_current()
    assert not handler.busy()


def _detach_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "opentelemetry.context"]


def test_an_exit_in_another_thread_does_not_try_to_detach(
    handler: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # Only that the handler doesn't reset a context that isn't its own (which
    # would fail) and forgets the call. The span it attached stays current in
    # the caller's context: a known limitation (the module's "Not covered"),
    # not something this test asks for.
    handler.span_enter(id_="OpenAI.chat-1", instance=_llm())
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(
                handler.span_exit, id_="OpenAI.chat-1", instance=_llm(), result=None
            ).result()
        assert not handler.busy()
        assert _detach_errors(caplog) == []
    finally:
        otel_context.attach(otel_context.Context())


def test_an_exit_in_a_child_task_does_not_try_to_detach(
    handler: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # As above, for a task of the same thread.
    async def run() -> None:
        handler.span_enter(id_="OpenAI.achat-1", instance=_llm())

        async def child() -> None:
            handler.span_exit(id_="OpenAI.achat-1", instance=_llm(), result=None)

        await asyncio.create_task(child())

    asyncio.run(run())
    assert not handler.busy()
    assert _detach_errors(caplog) == []


def test_the_handler_never_raises(handler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # The dispatcher swallows a handler's exceptions today; the handler does
    # not rely on that.
    def broken() -> Any:
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(llama_index_span_context, "_current_llama_index_span", broken)
    handler.span_enter(id_="OpenAI.chat-1", instance=_llm())
    handler.span_exit(id_="OpenAI.chat-1", instance=_llm(), result=None)
    assert _nothing_current()


def test_a_stream_hands_its_span_over_once(handler: Any) -> None:
    def stream() -> Iterator[int]:
        yield 1

    result = stream()
    handler.span_enter(id_="OpenAI.stream_chat-1", instance=_llm())
    handler.span_exit(id_="OpenAI.stream_chat-1", instance=_llm(), result=result)
    assert handler.take_stream_span(result) is SPAN
    assert handler.take_stream_span(result) is None


def test_uninstall_keeps_a_busy_handler_until_its_call_ends(
    openai_answers: Answers,
) -> None:
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument()
    try:
        assert llama_index_span_context.install(instrumentor)
        handler = llama_index_span_context._handler
        handler._rius_entered["OpenAI.chat-1"] = llama_index_span_context._Entered(_a_span())
        llama_index_span_context.uninstall(instrumentor)
        assert handler in get_dispatcher().span_handlers
        handler.span_exit(id_="OpenAI.chat-1", instance=_llm(), result=None)
        assert not handler.busy()
    finally:
        llama_index_span_context.uninstall(instrumentor)
        instrumentor.uninstrument()
    assert llama_index_span_context._handler not in get_dispatcher().span_handlers


def test_a_failing_wrap_never_breaks_a_class_definition(
    exported: Exported, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(cls: Any) -> None:
        raise RuntimeError("wrap failed")

    monkeypatch.setattr(llama_index_span_context, "_wrap_class", broken)

    class LaterOpenAI(OpenAI):  # type: ignore[misc, valid-type]
        pass

    assert LaterOpenAI(model="gpt-4o-mini", api_key="sk-test").model == "gpt-4o-mini"
