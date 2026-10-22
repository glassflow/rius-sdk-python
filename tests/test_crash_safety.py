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
import time
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

# --- JSON nested too deep for the decoder ------------------------------------------
# json.loads raises RecursionError, not ValueError, on such a text. Every reader
# of output.value caught only ValueError, so the error escaped from the span
# processor into the caller's `with`, or failed the whole export batch.
#
# How deep is too deep depends on the interpreter: up to 3.11 the C decoder
# stops at sys.getrecursionlimit(), from 3.12 at the C stack's own limit, so
# 5000 levels parse there. The `too_deep` fixture makes the decoder raise on
# every version; the deep-but-readable tests check what a parsed one gives.

_DEEP = 5000


def _nest(depth: int) -> str:
    return "[" * depth + "]" * depth


def _deep_object(marker: str = "", depth: int = _DEEP) -> str:
    """A JSON object nesting ``depth`` arrays, a compaction marker in it if asked."""
    head = '{"type":"compaction","a":' if marker else '{"a":'
    return head + _nest(depth) + "}"


@pytest.fixture
def too_deep(monkeypatch: pytest.MonkeyPatch) -> None:
    """json.loads raises RecursionError, as the decoder does past its limit,
    on any text nesting ``_DEEP`` arrays."""
    real = json.loads

    def loads(s: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(s, str) and "[" * _DEEP in s:
            raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")
        return real(s, *args, **kwargs)

    monkeypatch.setattr(json, "loads", loads)


def _compact_reply(depth: int) -> str:
    """A ``POST /responses/compact`` reply with ``depth`` arrays nested in it."""
    item = {"type": "compaction", "id": "cmp_1", "encrypted_content": "x"}
    return (
        '{"id":"resp_1","object":"response.compaction","output":'
        + json.dumps([item])
        + ',"a":'
        + _nest(depth)
        + "}"
    )


@pytest.mark.usefixtures("too_deep")
def test_deep_json_is_no_compaction() -> None:
    assert compactions_in_response(_deep_object(marker="compaction")).events == []


@pytest.mark.usefixtures("too_deep")
def test_deep_json_is_no_compact_reply() -> None:
    assert is_compact_reply(_compact_reply(_DEEP)) is False


@pytest.mark.usefixtures("too_deep")
def test_deep_json_has_no_response_id() -> None:
    attributes = {"openinference.span.kind": "LLM", "output.value": _deep_object()}
    assert response_id_from_output_value(attributes) == {}


@pytest.mark.usefixtures("too_deep")
def test_deep_compaction_reply_does_not_raise_out_of_the_callers_block() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=[], heartbeat=False)
    try:
        with client.get_tracer().start_as_current_span(
            "llm",
            attributes={
                "openinference.span.kind": "LLM",
                "output.value": _compact_reply(_DEEP),
            },
        ):
            pass
        assert client.flush()
    finally:
        client.shutdown()
    assert [s.name for s in inner.get_finished_spans()] == ["llm"]


# A depth every supported decoder reads, well inside the recursion limit.
_READABLE_DEPTH = 200


def test_deep_but_readable_compact_reply_is_one() -> None:
    raw = _compact_reply(_READABLE_DEPTH)
    assert is_compact_reply(raw) is True
    found = compactions_in_response(raw)
    assert len(found.events) == 1
    assert found.compacted_context is False


def test_deep_but_readable_reply_keeps_its_response_id() -> None:
    attributes = {"openinference.span.kind": "LLM", "output.value": _compact_reply(_READABLE_DEPTH)}
    assert response_id_from_output_value(attributes) != {}


def test_json_past_every_decoders_limit_does_not_raise() -> None:
    # Deep enough that the real decoder raises on every supported version.
    raw = _compact_reply(200_000)
    assert is_compact_reply(raw) is False
    assert compactions_in_response(raw).events == []
    attributes = {"openinference.span.kind": "LLM", "output.value": raw}
    assert response_id_from_output_value(attributes) == {}


class _Collecting(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.mark.usefixtures("too_deep")
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


# --- a string left open at the end of the text --------------------------------------
#
# JSON stored in a JSON string and then cut is a string that never closes, its
# content full of escaped quotes. The depth scan reads such a string to the end
# of the text, once; the TypeScript SDK reads it the same way, so these inputs
# give the same output from either SDK.

#: Deterministic base64, long enough to be media (344 characters).
_DATA = base64.b64encode(bytes(range(256))).decode()
#: What the run fallback writes in place of a run long enough to be media.
_RUN_MARKER = "[stripped unknown unknown bytes]"


def _double_encoded_cut(size: int) -> str:
    rows = [{"id": i, "tags": ["a", "b"], "meta": {"k": "v"}} for i in range(size // 20)]
    return json.dumps({"result": json.dumps(rows)})[:size]


def _elapsed(text: str) -> float:
    start = time.perf_counter()
    text_value(text)
    return time.perf_counter() - start


# Each took tens of seconds while a failed string match restarted at every
# escaped quote; read once they take milliseconds, so a second is a wide margin.
def test_double_encoded_json_cut_inside_its_string_is_read_in_linear_time() -> None:
    assert _elapsed(_double_encoded_cut(100 * 1024)) < 1.0


@pytest.mark.parametrize("end", ["", "\\"], ids=["plain", "lone-backslash"])
def test_an_open_string_of_escaped_quotes_is_read_in_linear_time(end: str) -> None:
    assert _elapsed('"' + '\\"' * (40 * 1024) + "[" * 65 + end) < 1.0


def test_keyed_base64_deep_in_a_stringified_json_cut_short_is_stripped() -> None:
    # The cut drops the closing quote, so the inner JSON is a string left open.
    text = json.dumps({"s": json.dumps(_nested(_image(_DATA), 70))})[:-20]
    assert text_value(text) == text.replace(_DATA, _RUN_MARKER)


def test_brackets_in_a_string_left_open_count_as_its_opening_ones() -> None:
    # Its closing brackets are not counted, as in a closed string: 70 opening
    # ones are past the parser's depth, so the run fallback takes the word.
    assert text_value('["' + "[]" * 70 + "a" * 100) == '["' + "[]" * 70 + _RUN_MARKER


def test_a_string_left_open_on_a_lone_backslash_is_read_to_the_end() -> None:
    text = '["' + "{" * 70 + _DATA + "\\"
    assert text_value(text) == '["' + "{" * 70 + _RUN_MARKER + "\\"
