"""``gen_ai.response.id`` recovered from ``output.value`` on LLM spans.

OpenInference emits no response-id attribute. The id (``chatcmpl-...`` for
OpenAI, ``resp_...`` for the Responses API, ``msg_...`` for Anthropic) exists
only as the top-level ``id`` of the provider object serialized under
``output.value``, and masking strips ``output.value`` under
``capture_content=False``. So the id has to be promoted at normalization,
which runs before masking, or it is lost entirely.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius import normalization as normalization_module
from rius.masking import MaskingSpanExporter
from rius.normalization import NormalizingSpanExporter, response_id_from_output_value
from rius.semconv import GEN_AI_RESPONSE_ID

_OPENAI_OUTPUT = (
    '{"id":"chatcmpl-abc","choices":[{"finish_reason":"stop","index":0,'
    '"message":{"content":"hi","role":"assistant"}}],"created":1,"model":"gpt-4o",'
    '"object":"chat.completion"}'
)


def _export(attributes: dict[str, Any], *, capture_content: bool = True) -> ReadableSpan:
    inner = InMemorySpanExporter()
    chain: Any = inner
    if not capture_content:
        chain = MaskingSpanExporter(chain, capture_content=False)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(chain)))
    with provider.get_tracer("t").start_as_current_span("ChatCompletion") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    (exported,) = inner.get_finished_spans()
    return exported


def _llm(output_value: Any, **extra: Any) -> dict[str, Any]:
    return {"openinference.span.kind": "LLM", "output.value": output_value, **extra}


# --- through the exporter ---------------------------------------------------


def test_the_id_is_promoted_and_output_value_is_untouched() -> None:
    attributes = dict(_export(_llm(_OPENAI_OUTPUT)).attributes or {})
    assert attributes[GEN_AI_RESPONSE_ID] == "chatcmpl-abc"
    assert attributes["output.value"] == _OPENAI_OUTPUT  # byte-identical


def test_the_id_survives_capture_off() -> None:
    attributes = dict(_export(_llm(_OPENAI_OUTPUT), capture_content=False).attributes or {})
    assert attributes[GEN_AI_RESPONSE_ID] == "chatcmpl-abc"
    assert "output.value" not in attributes


def test_a_native_response_id_wins() -> None:
    attributes = dict(
        _export(_llm(_OPENAI_OUTPUT, **{GEN_AI_RESPONSE_ID: "native"})).attributes or {}
    )
    assert attributes[GEN_AI_RESPONSE_ID] == "native"
    assert attributes["output.value"] == _OPENAI_OUTPUT


def test_an_operation_only_llm_span_qualifies_through_the_taxonomy() -> None:
    # A GenAI-native span names only the operation; the taxonomy rules derive
    # the LLM kind, and that is what gates the promotion.
    attributes = dict(
        _export({"gen_ai.operation.name": "chat", "output.value": _OPENAI_OUTPUT}).attributes or {}
    )
    assert attributes[GEN_AI_RESPONSE_ID] == "chatcmpl-abc"


@pytest.mark.parametrize("kind", ["TOOL", "CHAIN", "AGENT", "EMBEDDING", "RETRIEVER"])
def test_a_non_llm_span_is_left_alone(kind: str) -> None:
    # A tool's or chain's output is the user's data, and its `id` is not a
    # provider response id.
    attributes = dict(
        _export({"openinference.span.kind": kind, "output.value": '{"id":"user-1"}'}).attributes
        or {}
    )
    assert GEN_AI_RESPONSE_ID not in attributes


def test_a_span_without_a_kind_is_left_alone() -> None:
    attributes = dict(_export({"output.value": _OPENAI_OUTPUT}).attributes or {})
    assert GEN_AI_RESPONSE_ID not in attributes


# --- the function -----------------------------------------------------------


@pytest.mark.parametrize(
    ("output_value", "expected"),
    [
        (_OPENAI_OUTPUT, "chatcmpl-abc"),
        ('{"type":"message","id":"msg_1","role":"assistant"}', "msg_1"),  # not first
        ('{"id":"resp_123","object":"response"}', "resp_123"),
    ],
)
def test_the_top_level_id_is_read(output_value: str, expected: str) -> None:
    assert response_id_from_output_value(_llm(output_value)) == {GEN_AI_RESPONSE_ID: expected}


@pytest.mark.parametrize(
    "output_value",
    [
        '{"id":""}',  # empty
        '{"id":42}',  # not a string
        '{"id":null}',
        '{"message":{"id":"nested"}}',  # only nested
        '{"object":"chat.completion"}',  # no id
        '{"id":"chatcmpl-abc"',  # truncated, does not parse
        '[{"id":"x"}]',  # an array
        ' {"id":"x"}',  # does not START with "{"
        "hello",
        "",
        42,  # not a string at all
    ],
)
def test_no_usable_top_level_id_means_nothing(output_value: Any) -> None:
    assert response_id_from_output_value(_llm(output_value)) == {}


def test_no_output_value_means_nothing() -> None:
    assert response_id_from_output_value({"openinference.span.kind": "LLM"}) == {}
    assert response_id_from_output_value({}) == {}
    assert response_id_from_output_value(None) == {}


def test_a_native_id_is_never_overwritten() -> None:
    attributes = _llm(_OPENAI_OUTPUT, **{GEN_AI_RESPONSE_ID: "native"})
    assert response_id_from_output_value(attributes) == {}


class _CountingLoads:
    def __init__(self) -> None:
        self.calls = 0
        self._loads = json.loads

    def __call__(self, text: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._loads(text, *args, **kwargs)


@pytest.fixture
def counting_loads(monkeypatch: pytest.MonkeyPatch) -> _CountingLoads:
    spy = _CountingLoads()
    monkeypatch.setattr(normalization_module.json, "loads", spy)
    return spy


@pytest.mark.parametrize(
    "attributes",
    [
        _llm("plain text reply"),
        _llm('[{"id":"x"}]'),
        {"openinference.span.kind": "TOOL", "output.value": _OPENAI_OUTPUT},
        _llm(_OPENAI_OUTPUT, **{GEN_AI_RESPONSE_ID: "native"}),
    ],
)
def test_the_cheap_checks_run_before_any_parse(
    counting_loads: _CountingLoads, attributes: dict[str, Any]
) -> None:
    response_id_from_output_value(attributes)
    assert counting_loads.calls == 0


def test_output_value_is_parsed_at_most_once_per_export(counting_loads: _CountingLoads) -> None:
    _export(_llm(_OPENAI_OUTPUT))
    assert counting_loads.calls == 1


# --- end to end: the real instrumentors through init() ----------------------


@pytest.fixture
def openai_instrumentor() -> Iterator[Any]:
    oi = pytest.importorskip("openinference.instrumentation.openai")
    instrumentor = oi.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    yield instrumentor
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


@pytest.fixture
def anthropic_instrumentor() -> Iterator[Any]:
    oi = pytest.importorskip("openinference.instrumentation.anthropic")
    instrumentor = oi.AnthropicInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    yield instrumentor
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


def _sse(events: list[tuple[str | None, dict[str, Any]]], done: bool) -> str:
    body = "".join(
        (f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n"
        for name, data in events
    )
    return body + ("data: [DONE]\n\n" if done else "")


_CHAT = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_CHAT_CHUNKS: list[tuple[str | None, dict[str, Any]]] = [
    (
        None,
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}
            ],
        },
    ),
    (
        None,
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ),
]


class _ProviderServer:
    """A local HTTP server answering like a provider, streaming on request.

    A real socket rather than a mock transport: the provider SDKs pin their own
    HTTP clients (the anthropic SDK ships an httpx fork and rejects a plain
    httpx client), and a socket works whichever one they use.
    """

    def __init__(self, reply: dict[str, Any], stream_body: str) -> None:
        import http.server
        import threading

        reply_body = json.dumps(reply).encode()
        streamed = stream_body.encode()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (http.server API)
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                body, content_type = (
                    (streamed, "text/event-stream")
                    if request.get("stream")
                    else (reply_body, "application/json")
                )
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _only_llm(inner: InMemorySpanExporter) -> dict[str, Any]:
    (span,) = inner.get_finished_spans()
    return dict(span.attributes or {})


@pytest.mark.integration
@pytest.mark.parametrize(
    ("stream", "expected"), [(False, "chatcmpl-abc"), (True, "chatcmpl-stream")]
)
def test_real_openai_span_carries_the_response_id_with_capture_off(
    openai_instrumentor: Any, stream: bool, expected: str
) -> None:
    # Streaming too: the openai instrumentor serializes the ACCUMULATED
    # completion under output.value, and every chunk carries the completion's
    # own id, so the promoted id is the real one.
    openai = pytest.importorskip("openai")
    server = _ProviderServer(_CHAT, _sse(_CHAT_CHUNKS, done=True))
    inner = InMemorySpanExporter()
    client = init(
        span_exporter=inner, set_global=False, instruments=["openai"], capture_content=False
    )
    try:
        oai = openai.OpenAI(api_key="test-key", base_url=f"{server.url}/v1")
        result = oai.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=stream
        )
        if stream:
            for _ in result:
                pass
        client.flush()
    finally:
        client.shutdown()
        server.close()

    attributes = _only_llm(inner)
    assert attributes[GEN_AI_RESPONSE_ID] == expected
    assert "output.value" not in attributes


_MESSAGE = {
    "id": "msg_abc",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}

_MESSAGE_EVENTS: list[tuple[str | None, dict[str, Any]]] = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    ),
    (
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


_ANTHROPIC_KWARGS: dict[str, Any] = {
    "model": "claude-sonnet-5",
    "max_tokens": 5,
    "messages": [{"role": "user", "content": "hi"}],
}


def _anthropic_span(call: str, *, capture_content: bool) -> dict[str, Any]:
    anthropic = pytest.importorskip("anthropic")
    server = _ProviderServer(_MESSAGE, _sse(_MESSAGE_EVENTS, done=False))
    inner = InMemorySpanExporter()
    client = init(
        span_exporter=inner,
        set_global=False,
        instruments=["anthropic"],
        capture_content=capture_content,
    )
    try:
        messages = anthropic.Anthropic(api_key="test-key", base_url=server.url).messages
        if call == "create":
            messages.create(**_ANTHROPIC_KWARGS)
        elif call == "create-stream":
            for _ in messages.create(stream=True, **_ANTHROPIC_KWARGS):
                pass
        else:
            with messages.stream(**_ANTHROPIC_KWARGS) as stream:
                for _ in stream.text_stream:
                    pass
        client.flush()
    finally:
        client.shutdown()
        server.close()
    return _only_llm(inner)


@pytest.mark.integration
@pytest.mark.parametrize(("call", "expected"), [("create", "msg_abc"), ("stream", "msg_stream")])
def test_real_anthropic_span_carries_the_response_id_with_capture_off(
    anthropic_instrumentor: Any, call: str, expected: str
) -> None:
    attributes = _anthropic_span(call, capture_content=False)
    assert attributes[GEN_AI_RESPONSE_ID] == expected
    assert "output.value" not in attributes


@pytest.mark.integration
def test_real_anthropic_create_stream_has_no_output_value_and_so_no_id(
    anthropic_instrumentor: Any,
) -> None:
    # Recorded, not invented: for messages.create(stream=True) the anthropic
    # instrumentor writes no output.value at all, so there is no id to recover
    # and none is made up. Checked with capture ON, where output.value would
    # otherwise be visible.
    attributes = _anthropic_span("create-stream", capture_content=True)
    assert "output.value" not in attributes
    assert GEN_AI_RESPONSE_ID not in attributes
