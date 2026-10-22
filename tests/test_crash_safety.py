"""Inputs that raised into the caller's code, or replaced the caller's exception.

The SDK's rule on bad input is to warn and skip: whatever a caller, an
instrumentor or a provider hands it, it must never raise into user code nor
swap the exception the user's own code raised for one of its own.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import math
import os
from collections.abc import Sequence
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from rius import (
    init,
    observe,
    start_as_current_generation,
    start_as_current_span,
    start_generation,
    start_span,
)
from rius._errors import record_error
from rius._provider_compaction import compactions_in_response
from rius._provider_response import is_compact_reply
from rius._serde import serialize, text_value
from rius.generation import normalize_messages
from rius.normalization import (
    NormalizingSpanExporter,
    Rule,
    response_id_from_output_value,
    to_int,
)

# --- JSON nested past the interpreter's recursion limit --------------------------
# json.loads raises RecursionError, not ValueError, on such a text. Every reader
# of output.value caught only ValueError, so the error escaped from the span
# processor into the caller's `with`, or failed the whole export batch.

_DEEP = 5000


def _deep_object(marker: str = "") -> str:
    """A JSON object nested past the recursion limit, a compaction marker in it if asked."""
    head = '{"type":"compaction","a":' if marker else '{"a":'
    return head + "[" * _DEEP + "]" * _DEEP + "}"


def test_deep_json_is_no_compaction() -> None:
    assert compactions_in_response(_deep_object(marker="compaction")).events == []


def test_deep_json_is_no_compact_reply() -> None:
    raw = '{"object":"response.compaction","a":' + "[" * _DEEP + "]" * _DEEP + "}"
    assert is_compact_reply(raw) is False


def test_deep_json_has_no_response_id() -> None:
    attributes = {"openinference.span.kind": "LLM", "output.value": _deep_object()}
    assert response_id_from_output_value(attributes) == {}


def test_deep_compaction_reply_does_not_raise_out_of_the_callers_block() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=[], heartbeat=False)
    try:
        with client.get_tracer().start_as_current_span(
            "llm",
            attributes={
                "openinference.span.kind": "LLM",
                "output.value": _deep_object(marker="compaction"),
            },
        ):
            pass
        assert client.flush()
    finally:
        client.shutdown()
    assert [s.name for s in inner.get_finished_spans()] == ["llm"]


class _Collecting(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def test_deep_json_does_not_cost_the_rest_of_the_export_batch() -> None:
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    tracer = provider.get_tracer("t")
    deep = tracer.start_span(
        "deep", attributes={"openinference.span.kind": "LLM", "output.value": _deep_object()}
    )
    deep.end()
    other = tracer.start_span("other")
    other.end()
    inner = _Collecting()
    result = NormalizingSpanExporter(inner).export([deep, other])  # type: ignore[list-item]
    assert result is SpanExportResult.SUCCESS
    assert [s.name for s in inner.spans] == ["deep", "other"]


# --- an exception whose __str__ raises ---------------------------------------------


class _Unprintable(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no str for you")


def test_record_error_survives_an_unprintable_exception(
    exported_spans: InMemorySpanExporter,
) -> None:
    span = start_span("s")
    record_error(span._span, _Unprintable())
    span.end()
    (finished,) = exported_spans.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert finished.attributes is not None
    assert finished.attributes["error.type"].endswith("_Unprintable")
    event = next(e for e in finished.events if e.name == "exception")
    assert event.attributes is not None
    assert event.attributes["exception.type"] == finished.attributes["error.type"]


def test_observe_reraises_the_callers_unprintable_exception(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe()
    def fails() -> None:
        raise _Unprintable()

    with pytest.raises(_Unprintable):
        fails()
    (finished,) = exported_spans.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["error.type"].endswith("_Unprintable")


def test_async_observe_reraises_the_callers_unprintable_exception(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe()
    async def fails() -> None:
        raise _Unprintable()

    with pytest.raises(_Unprintable):
        asyncio.run(fails())
    (finished,) = exported_spans.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["error.type"].endswith("_Unprintable")


def test_scoped_span_reraises_the_callers_unprintable_exception(
    exported_spans: InMemorySpanExporter,
) -> None:
    with pytest.raises(_Unprintable), start_as_current_span("s"):
        raise _Unprintable()
    (finished,) = exported_spans.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert finished.attributes is not None
    assert finished.attributes["error.type"].endswith("_Unprintable")
    assert any(e.name == "exception" for e in finished.events)


def test_scoped_generation_reraises_the_callers_unprintable_exception(
    exported_spans: InMemorySpanExporter,
) -> None:
    with pytest.raises(_Unprintable), start_as_current_generation("g"):
        raise _Unprintable()
    (finished,) = exported_spans.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert finished.attributes is not None
    assert finished.attributes["error.type"].endswith("_Unprintable")
    assert any(e.name == "exception" for e in finished.events)


def test_scoped_span_keeps_the_exception_event_and_status_for_a_plain_error(
    exported_spans: InMemorySpanExporter,
) -> None:
    with pytest.raises(ValueError), start_as_current_span("s"):
        raise ValueError("boom")
    (finished,) = exported_spans.get_finished_spans()
    assert finished.status.status_code == StatusCode.ERROR
    assert "boom" in (finished.status.description or "")
    assert finished.attributes is not None
    assert finished.attributes["error.type"] == "ValueError"
    events = [e for e in finished.events if e.name == "exception"]
    assert len(events) == 1  # recorded once, not by both OTel and the SDK
    assert events[0].attributes is not None
    assert events[0].attributes["exception.type"] == "ValueError"


def test_manual_handles_record_an_unprintable_exception(
    exported_spans: InMemorySpanExporter,
) -> None:
    span = start_span("s")
    span.record_exception(_Unprintable())
    span.end()
    generation = start_generation("g")
    generation.record_exception(_Unprintable())
    generation.end()
    for finished in exported_spans.get_finished_spans():
        assert finished.attributes is not None
        assert finished.attributes["error.type"].endswith("_Unprintable")


# --- malformed tool_calls in a message ----------------------------------------------


@pytest.mark.parametrize(
    "tool_calls",
    [5, "abc", [{"function": "x"}], [{"function": 3, "id": "c1"}], [None, 7]],
)
def test_malformed_tool_calls_are_skipped(tool_calls: Any) -> None:
    (message,) = normalize_messages(
        [{"role": "assistant", "content": "hi", "tool_calls": tool_calls}], "user"
    )
    assert message["role"] == "assistant"
    assert {"type": "text", "content": "hi"} in message["parts"]


def test_malformed_tool_calls_do_not_raise_out_of_a_generation(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation(
        "g", input=[{"role": "assistant", "tool_calls": [{"function": "x"}]}]
    ):
        pass
    assert [s.name for s in exported_spans.get_finished_spans()] == ["g"]


def test_a_tool_call_whose_function_is_not_a_mapping_keeps_its_id() -> None:
    (message,) = normalize_messages(
        [{"role": "assistant", "tool_calls": [{"id": "c1", "function": "x"}]}], "user"
    )
    (part,) = message["parts"]
    assert part["type"] == "tool_call"
    assert part["id"] == "c1"


# --- observe() on a callable without __qualname__ -----------------------------------


def test_observe_accepts_a_partial(exported_spans: InMemorySpanExporter) -> None:
    def add(a: int, b: int) -> int:
        return a + b

    wrapped = observe(functools.partial(add, 1))
    assert wrapped(2) == 3
    (finished,) = exported_spans.get_finished_spans()
    # Named after the wrapped function, never the partial's repr (an address).
    assert finished.name.endswith("add")


def test_observe_accepts_a_callable_instance(exported_spans: InMemorySpanExporter) -> None:
    class Adder:
        def __call__(self, a: int) -> int:
            return a + 1

    assert observe(Adder())(1) == 2
    (finished,) = exported_spans.get_finished_spans()
    assert finished.name.endswith("Adder")


# --- span limits: a valid span count beside an invalid global count -----------------


@pytest.mark.parametrize("bad", ["lots", "-1", ""])
def test_init_survives_an_invalid_global_count_beside_a_valid_span_count(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "500")
    monkeypatch.setenv("OTEL_ATTRIBUTE_COUNT_LIMIT", bad)
    client = init(span_exporter=InMemorySpanExporter(), set_global=False, instruments=[])
    try:
        # The user's valid span count still wins.
        assert client._provider._span_limits.max_span_attributes == 500
    finally:
        client.shutdown()


# --- to_int agrees with the count rule ----------------------------------------------


def _to_int(value: Any) -> Any:
    from rius.normalization import SKIP

    result = to_int([value])
    return None if result is SKIP else result


@pytest.mark.parametrize("value", [True, False, 12.9, -5, "-5", math.inf, -math.inf, math.nan])
def test_to_int_skips_what_is_not_a_token_count(value: Any) -> None:
    assert _to_int(value) is None


@pytest.mark.parametrize(("value", "expected"), [(12, 12), (12.0, 12), ("12", 12), (0, 0)])
def test_to_int_keeps_a_token_count(value: Any, expected: int) -> None:
    assert _to_int(value) == expected


def test_an_infinite_token_count_logs_no_traceback(caplog: pytest.LogCaptureFixture) -> None:
    from rius.normalization import NormalizationTable

    table = NormalizationTable(
        [Rule("llm.token_count.prompt", "gen_ai.usage.input_tokens", to_int)]
    )
    with caplog.at_level("WARNING"):
        out = table.additions({"llm.token_count.prompt": math.inf})
    assert "gen_ai.usage.input_tokens" not in out
    assert not caplog.records


# --- keyed base64 nested past the media parser's depth -------------------------------


def _image(data: str) -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


@pytest.mark.parametrize("depth", [70, 200])
def test_keyed_base64_past_the_parser_depth_is_stripped(depth: int) -> None:
    data = base64.b64encode(os.urandom(3000)).decode()
    value: Any = _image(data)
    for _ in range(depth):
        value = {"x": value}
    assert data[:64] not in serialize(value)


def test_keyed_base64_past_the_parser_depth_is_stripped_from_text() -> None:
    data = base64.b64encode(os.urandom(3000)).decode()
    # Text another instrumentation already rendered, not a value this SDK encodes.
    deep = '{"x":' * 100 + json.dumps(_image(data)) + "}" * 100
    assert data[:64] not in text_value(deep)


def test_shallow_text_keeps_a_long_word() -> None:
    word = "a" * 200
    assert word in serialize({"x": {"y": word}})


def _nested(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = {"x": value}
    return value


@pytest.mark.parametrize(("outer", "inner"), [(1, 70), (40, 30)])
def test_keyed_base64_past_the_parser_depth_inside_a_json_string_is_stripped(
    outer: int, inner: int
) -> None:
    # The parser re-parses a JSON-looking string at the depth it sits at, so
    # the depth inside the string adds to the depth around it.
    data = base64.b64encode(os.urandom(3000)).decode()
    value = _nested({"s": json.dumps(_nested(_image(data), inner))}, outer - 1)
    assert data[:64] not in serialize(value)
    assert data[:64] not in text_value(json.dumps(value))


def test_a_json_string_within_the_parser_depth_keeps_a_long_word() -> None:
    # More brackets than the parser depth in all, so the walk runs, but no
    # path, the string's own depth included, nests past it.
    word = "a" * 200
    value = {
        "a": _nested({"s": json.dumps(_nested({"w": word}, 20))}, 39),
        "b": _nested(1, 30),
    }
    assert word in serialize(value)
