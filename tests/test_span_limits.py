"""The span attribute-count limit, and keeping a drop visible when it bites.

OTel's default ``SpanLimits`` caps a span at 128 attributes and, once full,
evicts the OLDEST key for every new one. The OpenInference instrumentors write
one attribute per message field and per tool field, so an agent loop passes
128 within a handful of turns, and what goes first is exactly what they wrote
first: the request bag (model and parameters), the tool definitions, and the
system prompt. ``init()`` therefore raises the count for the provider it
builds, unless the user set the standard OTel env var, which then wins.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.client import DEFAULT_SPAN_ATTRIBUTE_COUNT_LIMIT
from rius.masking import MaskingSpanExporter
from rius.normalization import NormalizingSpanExporter
from rius.workspace import RoutingSpanExporter

_COUNT_ENV_VARS = ("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "OTEL_ATTRIBUTE_COUNT_LIMIT")


@pytest.fixture(autouse=True)
def _no_ambient_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _COUNT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _wide_span(tracer_provider: Any, n: int) -> None:
    tracer = tracer_provider.get_tracer("t")
    with tracer.start_as_current_span("wide") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        for i in range(n):
            span.set_attribute(f"custom.{i}", i)


def _init(inner: InMemorySpanExporter) -> Any:
    return init(span_exporter=inner, set_global=False, instruments=[])


# --- the limit init() applies ---------------------------------------------------


def test_init_raises_the_span_attribute_count_limit() -> None:
    inner = InMemorySpanExporter()
    client = _init(inner)
    _wide_span(client._provider, 1000)
    client.flush()

    (span,) = inner.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["custom.0"] == 0  # the oldest key was not evicted
    assert len([k for k in span.attributes if k.startswith("custom.")]) == 1000
    assert span.dropped_attributes == 0


def test_the_raised_limit_is_4096() -> None:
    assert DEFAULT_SPAN_ATTRIBUTE_COUNT_LIMIT == 4096
    client = _init(InMemorySpanExporter())
    limits = client._provider._span_limits
    assert limits.max_span_attributes == 4096


def test_init_leaves_the_value_length_limit_alone() -> None:
    # The SDK caps its own JSON attributes; OTel's length limit stays OTel's.
    client = _init(InMemorySpanExporter())
    assert client._provider._span_limits.max_span_attribute_length is None


@pytest.mark.parametrize("env_var", _COUNT_ENV_VARS)
def test_a_user_count_limit_env_var_wins(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    monkeypatch.setenv(env_var, "20")
    inner = InMemorySpanExporter()
    client = _init(inner)
    _wide_span(client._provider, 100)
    client.flush()

    (span,) = inner.get_finished_spans()
    assert span.attributes is not None
    assert len(span.attributes) <= 20
    assert client._provider._span_limits.max_span_attributes == 20


@pytest.mark.parametrize("value", ["", "  ", "abc", "-5", "1e3", "unset"])
@pytest.mark.parametrize("env_var", _COUNT_ENV_VARS)
def test_a_value_otel_does_not_honour_does_not_count_as_set(
    monkeypatch: pytest.MonkeyPatch, env_var: str, value: str
) -> None:
    # The TypeScript SDK's rule too. Left to Python OTel, a blank
    # OTEL_ATTRIBUTE_COUNT_LIMIT gives spans 128, and a non-integer or negative
    # one makes SpanLimits() raise, which would take init() down with it.
    monkeypatch.setenv(env_var, value)
    client = _init(InMemorySpanExporter())
    limits = client._provider._span_limits
    assert limits.max_span_attributes == 4096
    # Event and link attributes keep OTel's default.
    assert limits.max_attributes == 128
    assert limits.max_event_attributes == 128


@pytest.mark.parametrize(("value", "expected"), [("50", 50), (" 50 ", 50), ("+7", 7), ("0", 0)])
@pytest.mark.parametrize("env_var", _COUNT_ENV_VARS)
def test_a_value_otel_honours_wins(
    monkeypatch: pytest.MonkeyPatch, env_var: str, value: str, expected: int
) -> None:
    # Exactly the values OTel's own parser accepts: int() after strip.
    monkeypatch.setenv(env_var, value)
    client = _init(InMemorySpanExporter())
    assert client._provider._span_limits.max_span_attributes == expected


def test_an_unusable_value_is_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OTEL_ATTRIBUTE_COUNT_LIMIT", "abc")
    with caplog.at_level("WARNING", logger="rius.client"):
        _init(InMemorySpanExporter())
    assert "OTEL_ATTRIBUTE_COUNT_LIMIT" in caplog.text
    assert "4096" in caplog.text


def test_a_blank_value_is_not_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A blank entry is routine in a container env; it is not worth a warning.
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "")
    with caplog.at_level("WARNING", logger="rius.client"):
        _init(InMemorySpanExporter())
    assert "COUNT_LIMIT" not in caplog.text


def test_the_span_specific_env_var_still_beats_the_global_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OTel's own precedence, untouched: the model-specific limit wins.
    monkeypatch.setenv("OTEL_ATTRIBUTE_COUNT_LIMIT", "20")
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "30")
    client = _init(InMemorySpanExporter())
    assert client._provider._span_limits.max_span_attributes == 30


# --- a drop stays visible through the exporter chain ----------------------------


def _limited_provider(exporter: Any, limit: int) -> TracerProvider:
    provider = TracerProvider(span_limits=SpanLimits(max_span_attributes=limit))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def _only(inner: InMemorySpanExporter) -> ReadableSpan:
    (span,) = inner.get_finished_spans()
    return span


def test_normalization_carries_the_dropped_count_through() -> None:
    inner = InMemorySpanExporter()
    provider = _limited_provider(NormalizingSpanExporter(inner), 10)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("llm") as span:
        for i in range(25):
            span.set_attribute(f"custom.{i}", i)
        # Written last, so it survives the eviction and normalization rewrites
        # the span: this is the copy that used to report 0 dropped.
        span.set_attribute("llm.token_count.prompt", 5)

    exported = _only(inner)
    assert exported.attributes is not None
    assert exported.attributes["gen_ai.usage.input_tokens"] == 5
    assert "llm.token_count.prompt" not in exported.attributes
    assert exported.dropped_attributes == 16


def test_masking_carries_the_dropped_count_through() -> None:
    inner = InMemorySpanExporter()
    provider = _limited_provider(MaskingSpanExporter(inner, capture_content=False), 10)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("llm") as span:
        for i in range(25):
            span.set_attribute(f"custom.{i}", i)
        span.set_attribute("input.value", "secret")

    exported = _only(inner)
    assert exported.attributes is not None
    assert "input.value" not in exported.attributes
    assert exported.dropped_attributes == 16


def test_workspace_routing_carries_the_dropped_count_through() -> None:
    inner = InMemorySpanExporter()
    workspace_inner = InMemorySpanExporter()
    routing = RoutingSpanExporter(inner, lambda _key: workspace_inner, routes={"a": "key-a"})
    provider = _limited_provider(routing, 10)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("llm") as span:
        for i in range(25):
            span.set_attribute(f"custom.{i}", i)
        span.set_attribute("rius.workspace", "a")

    exported = _only(workspace_inner)
    assert exported.attributes is not None
    assert "rius.workspace" not in exported.attributes
    assert exported.dropped_attributes == 16


def test_a_span_that_dropped_nothing_still_reports_zero() -> None:
    inner = InMemorySpanExporter()
    provider = _limited_provider(NormalizingSpanExporter(inner), 128)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("llm") as span:
        span.set_attribute("llm.token_count.prompt", 5)

    exported = _only(inner)
    assert exported.attributes is not None
    assert exported.attributes["gen_ai.usage.input_tokens"] == 5
    assert exported.dropped_attributes == 0


# --- end to end: a realistic agent loop through the real instrumentor -----------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": f"tool_{i}",
            "description": f"Tool number {i}",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    for i in range(10)
]

_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "tool_0", "arguments": '{"q":"x"}'},
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
}


def _agent_loop(turns: int) -> list[dict[str, Any]]:
    """A conversation that has called a tool on every turn, as an agent does."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": "You are an agent."}]
    for t in range(turns):
        messages.append({"role": "user", "content": f"u{t}"})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{t}",
                        "type": "function",
                        "function": {"name": "tool_0", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{t}", "content": "r"})
        messages.append({"role": "assistant", "content": f"a{t}"})
    messages.append({"role": "user", "content": "next"})
    return messages


@pytest.fixture
def openai_instrumentor() -> Iterator[Any]:
    oi = pytest.importorskip("openinference.instrumentation.openai")
    instrumentor = oi.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    yield instrumentor
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


def _twenty_turn_span(inner: InMemorySpanExporter) -> ReadableSpan:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    client = init(span_exporter=inner, set_global=False, instruments=["openai"])
    oai = openai.OpenAI(
        api_key="test-key",
        base_url="http://mock.invalid/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=_REPLY))
        ),
    )
    oai.chat.completions.create(
        model="gpt-4o", messages=_agent_loop(20), tools=_TOOLS, temperature=0.2
    )
    client.flush()
    client.shutdown()
    return _only(inner)


@pytest.mark.integration
def test_a_twenty_turn_agent_span_keeps_model_tools_usage_and_first_messages(
    openai_instrumentor: Any,
) -> None:
    span = _twenty_turn_span(InMemorySpanExporter())
    attributes = span.attributes
    assert attributes is not None

    assert span.dropped_attributes == 0
    assert attributes["gen_ai.request.model"] == "gpt-4o"
    assert attributes["gen_ai.request.temperature"] == 0.2
    tools = json.loads(str(attributes["gen_ai.tool.definitions"]))
    assert [t["function"]["name"] for t in tools] == [f"tool_{i}" for i in range(10)]
    assert attributes["gen_ai.usage.input_tokens"] == 1200
    assert attributes["gen_ai.usage.output_tokens"] == 30
    messages = json.loads(str(attributes["gen_ai.input.messages"]))
    assert len(messages) == 1 + 20 * 4 + 1
    assert messages[0] == {
        "role": "system",
        "parts": [{"type": "text", "content": "You are an agent."}],
    }
    assert messages[1]["parts"] == [{"type": "text", "content": "u0"}]
    assert "input.value" in attributes
    assert "output.value" in attributes


@pytest.mark.integration
def test_a_user_limit_still_wins_on_the_twenty_turn_span_and_the_drop_is_visible(
    openai_instrumentor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "128")
    span = _twenty_turn_span(InMemorySpanExporter())
    attributes = span.attributes
    assert attributes is not None

    # The user's 128 applied, so the oldest keys went, and the export says so.
    assert span.dropped_attributes > 0
    assert "gen_ai.tool.definitions" not in attributes
