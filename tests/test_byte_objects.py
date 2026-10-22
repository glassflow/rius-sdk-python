"""A byte object never leaves with its content, on any path a value is written.

Every bytes, bytearray and memoryview in a value this SDK serializes is
written as the contract's marker, ``[stripped unknown <N> bytes]``, N its size
in bytes (rius-semconv ``raw_payloads.byte_objects``). One test per path a
value takes to an attribute: the span and generation helpers, ``@observe``,
the MCP tool spans, a mask's result, and the exporter's last pass over span,
event and link attributes another instrumentation set.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, NamedTuple

import pytest
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanContext

from media_contract import contract
from rius import init, observe, start_as_current_generation, start_as_current_span, start_span
from rius._serde import attribute_value, serialize, text_value
from rius.masking import MaskingSpanExporter
from rius.normalization import NormalizingSpanExporter

#: Bytes whose content would show in any leak: a PNG signature.
PNG = b"\x89PNG\r\n\x1a\n"
MARKER = "[stripped unknown 8 bytes]"
QUOTED = json.dumps(MARKER)

BYTE_OBJECTS = contract()["raw_payloads"]["byte_objects"]
PYTHON_EXAMPLES = [e for e in BYTE_OBJECTS["examples"] if e["language"] == "python"]


def _value(source: str) -> Any:
    """An example's value: a Python expression over the byte types only."""
    names = {"bytes": bytes, "bytearray": bytearray, "memoryview": memoryview}
    return eval(source, {"__builtins__": names})  # noqa: S307 - the contract's own examples


#: OpenTelemetry before 1.45 holds no mapping, nested array or byte object in an
#: attribute value: it drops the first two, and decodes a byte object to text or drops it,
#: before any exporter sees the value. No span on such a version carries one,
#: so the cases built on one apply only where OpenTelemetry holds it.
_OTEL_HOLDS_ANY_VALUE = "m" in BoundedAttributes(attributes={"m": {"k": b"v"}})


def _skip_unless_otel_holds(value: Any, *, byte_objects: bool) -> None:
    """Skip a case whose input the installed OpenTelemetry cannot hold.

    ``byte_objects`` says whether a byte object counts: it does on a path
    through OpenTelemetry's own setters, not on a hand-built event or link.
    """

    def needs(v: Any) -> bool:
        if isinstance(v, dict):
            return True
        if isinstance(v, (bytes, bytearray, memoryview)):
            return byte_objects
        if isinstance(v, (list, tuple)):
            # A nested array is no more a plain attribute value than a mapping.
            return any(isinstance(item, (list, tuple)) or needs(item) for item in v)
        return False

    if not _OTEL_HOLDS_ANY_VALUE and needs(value):
        pytest.skip("this OpenTelemetry drops or decodes the value before any exporter")


def _leaks(text: str) -> bool:
    """Whether the PNG's content shows in a text: raw, as repr, or as JSON."""
    return any(
        needle in text for needle in ("PNG", "\\x89", "\\u0089", "\x89", "137, 80", "137,80")
    )


def test_the_contract_names_the_python_byte_types() -> None:
    assert BYTE_OBJECTS["python"] == ["bytes", "bytearray", "memoryview"]
    assert PYTHON_EXAMPLES


@pytest.mark.parametrize("example", PYTHON_EXAMPLES, ids=lambda e: str(e["name"]))
def test_a_contract_example_is_written_byte_for_byte(example: dict[str, Any]) -> None:
    value = _value(example["value"])
    assert serialize(value) == example["output"]
    assert text_value(value) == example["output"]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (PNG, QUOTED),
        (bytearray(PNG), QUOTED),
        (memoryview(PNG), QUOTED),
        # Subclasses are byte objects too.
        (type("Raw", (bytes,), {})(PNG), QUOTED),
        # Nested in every container the encoder walks.
        ([[{"a": (PNG,)}]], f'[[{{"a":[{QUOTED}]}}]]'),
        ({"deep": {"er": [1, PNG, "x"]}}, f'{{"deep":{{"er":[1,{QUOTED},"x"]}}}}'),
    ],
)
def test_a_byte_object_is_the_marker_wherever_it_sits(value: Any, text: str) -> None:
    assert serialize(value) == text


def test_a_released_memoryview_has_an_unknown_size() -> None:
    view = memoryview(PNG)
    view.release()
    assert serialize({"v": view}) == '{"v":"[stripped unknown unknown bytes]"}'


def test_a_bytes_key_never_leaks() -> None:
    # A bytes key makes the JSON encoder raise; the fallback writes the value's
    # repr, with every bytes literal in it written as the marker.
    text = serialize({PNG: "image", "other": PNG})
    assert not _leaks(text)
    assert MARKER in text


@dataclass
class _Attachment:
    name: str
    data: bytes
    more: bytearray


def test_bytes_in_an_object_written_by_its_repr_never_leak() -> None:
    text = serialize({"file": _Attachment("a.png", PNG, bytearray(b'it\'s "raw"'))})
    assert not _leaks(text)
    assert "raw" not in text
    assert MARKER in text
    assert "[stripped unknown 10 bytes]" in text
    assert "a.png" in text


def test_a_repr_word_that_looks_like_bytes_is_kept() -> None:
    # Only a bytes literal is replaced: the b'x' that ends a word is the word's.
    text = serialize({"t": _Attachment("bob'x'", b"", bytearray())})
    written = (
        "_Attachment(name=\"bob'x'\", data=[stripped unknown 0 bytes], "
        "more=[stripped unknown 0 bytes])"
    )
    assert json.loads(text) == {"t": written}


# --- the paths a value takes ---


def test_span_input_and_output(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_span("s", input={"image": PNG}) as span:
        span.set_output(PNG)
    attributes = exported_spans.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["input.value"] == f'{{"image":{QUOTED}}}'
    assert attributes["output.value"] == QUOTED


def test_generation_input_and_output(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("g", input=[{"role": "user", "content": PNG}]) as gen:
        gen.set_output([{"role": "assistant", "content": [{"type": "x", "data": PNG}]}])
    attributes = exported_spans.get_finished_spans()[0].attributes
    assert attributes is not None
    for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        assert not _leaks(str(attributes[key]))
        assert MARKER in str(attributes[key])


def test_observe_arguments_and_result(exported_spans: InMemorySpanExporter) -> None:
    @observe()
    def thumbnail(image: bytes) -> bytes:
        return image

    thumbnail(PNG)
    attributes = exported_spans.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["input.value"] == f'{{"args":[{QUOTED}],"kwargs":{{}}}}'
    assert attributes["output.value"] == QUOTED


def test_mcp_tool_result() -> None:
    from rius.instrumentation_mcp import _record_result

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    span = client.get_tracer().start_span("execute_tool fake")

    class _Result:
        structured_content = {"image": PNG}
        is_error = False

    _record_result(span, _Result())
    span.end()
    client.flush()
    [finished] = inner.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["output.value"] == f'{{"image":{QUOTED}}}'


def test_mcp_call_tool_arguments() -> None:
    pytest.importorskip("mcp")
    from test_mcp_instrumentation import _connected_session, _make_server

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=["mcp"])

    async def scenario() -> None:
        async with _connected_session(_make_server()) as session:
            await session.call_tool("add", {"a": 1, "b": 2, "blob": PNG})

    # The transport refuses bytes, after the span has recorded the arguments.
    with pytest.raises(BaseException):  # noqa: B017, PT011 - whatever the transport raises
        asyncio.run(scenario())
    client.flush()
    [span] = [s for s in inner.get_finished_spans() if s.name.startswith("execute_tool")]
    assert span.attributes is not None
    assert span.attributes["input.value"] == f'{{"a":1,"b":2,"blob":{QUOTED}}}'


def test_a_mask_that_returns_bytes() -> None:
    sink = InMemorySpanExporter()
    exporter = MaskingSpanExporter(sink, capture_content=True, mask=lambda value: PNG)
    exporter.export([_span({"input.value": "secret"})])
    [out] = sink.get_finished_spans()
    assert out.attributes is not None
    assert out.attributes["input.value"] == QUOTED


# --- attributes another instrumentation set ---


def _span(
    attributes: dict[str, Any], events: tuple[Event, ...] = (), links: tuple[Link, ...] = ()
) -> ReadableSpan:
    context = SpanContext(trace_id=1, span_id=2, is_remote=False)
    return ReadableSpan(
        name="s", context=context, attributes=attributes, events=events, links=links
    )


def _export(span: ReadableSpan) -> ReadableSpan:
    sink = InMemorySpanExporter()
    NormalizingSpanExporter(sink).export([span])
    [out] = sink.get_finished_spans()
    return out


# A byte object in an attribute value, at any depth of arrays and mappings,
# is the bare marker: nothing serializes it, so no JSON quotes. An array that
# holds one beside an element that is then no string is the array's JSON
# (byte_objects.attribute_values). The Python examples of that section come
# first.
ATTRIBUTE_VALUES = BYTE_OBJECTS["attribute_values"]["examples"]
RAW_ATTRIBUTES: list[tuple[Any, Any]] = [
    *((_value(e["value"]), e["output"]) for e in ATTRIBUTE_VALUES if e["language"] == "python"),
    (PNG, MARKER),
    ((PNG, "x"), [MARKER, "x"]),
    # Beside a number, a bool, None, an array or a mapping: the array's JSON,
    # its strings stripped too.
    ((5, PNG), f"[5,{QUOTED}]"),
    ([PNG, True, None], f"[{QUOTED},true,null]"),
    ([b"a", [b"b"]], '["[stripped unknown 1 bytes]",["[stripped unknown 1 bytes]"]]'),
    (
        [{"k": PNG}, PNG, "data:image/png;base64,iVBORw0KGgo="],
        f'[{{"k":{QUOTED}}},{QUOTED},"[stripped image/png 8 bytes]"]',
    ),
    # No byte object of its own: kept as an array, its elements walked.
    ([1, [PNG]], [1, [MARKER]]),
    ({"image": PNG, "n": 1}, {"image": MARKER, "n": 1}),
    (
        {"nested": [{"image": PNG}], "deeper": [[PNG]]},
        {"nested": [{"image": MARKER}], "deeper": [[MARKER]]},
    ),
]


def test_attribute_value_examples_are_python_here_or_the_sinks() -> None:
    languages = {e["language"] for e in ATTRIBUTE_VALUES}
    assert languages == {"python", "typescript", "otlp"}
    assert any(e["language"] == "python" for e in ATTRIBUTE_VALUES)


def _as_written(value: Any) -> Any:
    """An attribute value as compared: a sequence as a list, a mapping's
    values alike."""
    if isinstance(value, (list, tuple)):
        return [_as_written(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_written(v) for k, v in value.items()}
    return value


@pytest.mark.parametrize(("value", "written"), RAW_ATTRIBUTES)
def test_a_raw_payload_byte_object(value: Any, written: Any) -> None:
    out = _export(_span({"input.value": value, "k": "v"}))
    assert out.attributes is not None
    assert _as_written(out.attributes["input.value"]) == written
    assert out.attributes["k"] == "v"


@pytest.mark.parametrize(("value", "written"), RAW_ATTRIBUTES)
def test_an_event_attribute_byte_object(value: Any, written: Any) -> None:
    _skip_unless_otel_holds(value, byte_objects=False)
    out = _export(_span({"k": "v"}, events=(Event("e", {"payload": value}, 5),)))
    assert out.events[0].attributes is not None
    assert _as_written(out.events[0].attributes["payload"]) == written


@pytest.mark.parametrize(("value", "written"), RAW_ATTRIBUTES)
def test_a_link_attribute_byte_object(value: Any, written: Any) -> None:
    _skip_unless_otel_holds(value, byte_objects=False)
    link = Link(SpanContext(trace_id=1, span_id=3, is_remote=False), {"payload": value})
    out = _export(_span({"k": "v"}, links=(link,)))
    assert out.links[0].attributes is not None
    assert _as_written(out.links[0].attributes["payload"]) == written


def test_a_span_set_attribute_byte_object_from_the_sdk_pipeline() -> None:
    _skip_unless_otel_holds(PNG, byte_objects=True)
    # The whole pipeline, as an instrumentation that sets bytes would use it.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    span = client.get_tracer().start_span("raw")
    span.set_attribute("input.value", PNG)
    span.set_attribute("output.value", [b"ab", PNG])
    span.add_event("e", {"payload": PNG})
    span.end()
    client.flush()
    [finished] = inner.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["input.value"] == MARKER
    assert list(finished.attributes["output.value"]) == ["[stripped unknown 2 bytes]", MARKER]
    assert finished.events[0].attributes == {"payload": MARKER}


#: A raw OpenTelemetry span's set_attribute (a foreign instrumentation's, or
#: one opened on the tracer directly) turns a bytearray or memoryview into a
#: tuple of its items (both are Sequences) before any exporter sees it: only
#: bytes reaches the exporter as a byte object on that path. The SDK's own
#: setters write the marker before OpenTelemetry sees the value.
_FOREIGN_HELD_AS_ITEMS = pytest.mark.xfail(
    strict=True,
    reason="a raw OpenTelemetry span holds a bytearray or memoryview as a tuple of ints",
)
PYTHON_ATTRIBUTE_VALUES = [e for e in ATTRIBUTE_VALUES if e["language"] == "python"]


def _exported(set_value: Any) -> Any:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    set_value(client)
    client.flush()
    [finished] = inner.get_finished_spans()
    assert finished.attributes is not None
    return _as_written(finished.attributes["payload"])


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(
            e,
            id=str(e["name"]),
            marks=[_FOREIGN_HELD_AS_ITEMS]
            if "bytearray" in e["value"] or "memoryview" in e["value"]
            else [],
        )
        for e in PYTHON_ATTRIBUTE_VALUES
    ],
)
def test_an_attribute_values_example_on_a_raw_opentelemetry_span(
    example: dict[str, Any],
) -> None:
    _skip_unless_otel_holds(_value(example["value"]), byte_objects=True)

    def set_value(client: Any) -> None:
        span = client.get_tracer().start_span("raw")
        span.set_attribute("payload", _value(example["value"]))
        span.end()

    assert _exported(set_value) == example["output"]


def _decode_array_strings(value: Any) -> Any:
    """``value`` with every string that is an array's JSON decoded, at any depth."""
    if isinstance(value, dict):
        return {k: _decode_array_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_array_strings(v) for v in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return value
        if isinstance(decoded, list):
            return _decode_array_strings(decoded)
    return value


def test_the_sdks_set_attribute_encodes_a_mixed_array_in_a_mapping_once(
    exported_spans: InMemorySpanExporter,
) -> None:
    observation = start_span("s")
    observation.set_attribute("payload", {"a": [b"x", 1]})
    observation.end()
    attributes = exported_spans.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["payload"] == '{"a":["[stripped unknown 1 bytes]",1]}'


@pytest.mark.parametrize("example", PYTHON_ATTRIBUTE_VALUES, ids=lambda e: str(e["name"]))
def test_an_attribute_values_example_through_the_sdks_set_attribute(
    example: dict[str, Any], exported_spans: InMemorySpanExporter
) -> None:
    _skip_unless_otel_holds(_value(example["value"]), byte_objects=False)
    # The global test provider has no exporter chain: what it holds is what
    # the SDK handed OpenTelemetry, already the marker.
    observation = start_span("s")
    observation.set_attribute("payload", _value(example["value"]))
    observation.end()
    [finished] = exported_spans.get_finished_spans()
    assert finished.attributes is not None
    written = finished.attributes["payload"]
    if isinstance(example["output"], dict):
        # The SDK's setter writes structure as JSON, as every other SDK path
        # does, where a raw OpenTelemetry span holds the mapping: one JSON
        # encoding, byte objects as markers. A raw span holds a mixed array in
        # the mapping as that array's JSON; the setter's one encoding holds it
        # as an array.
        assert isinstance(written, str)
        assert json.loads(written) == _decode_array_strings(example["output"])
    else:
        assert _as_written(written) == example["output"]


def test_the_sdks_set_attribute_leaves_other_values_as_they_are(
    exported_spans: InMemorySpanExporter,
) -> None:
    observation = start_span("s")
    values = {"s": "data:image/png;base64,iVBORw0KGgo=", "n": 3, "l": ["a", "b"], "f": 1.5}
    for key, value in values.items():
        observation.set_attribute(key, value)
    observation.end()
    attributes = exported_spans.get_finished_spans()[0].attributes
    assert attributes is not None
    # Strings are stripped by the exporter, not here.
    assert {k: _as_written(attributes[k]) for k in values} == values


class _Pair(NamedTuple):
    first: Any
    second: Any


class _Pair3(tuple):  # type: ignore[type-arg]
    """A tuple subclass whose constructor takes other arguments."""

    def __new__(cls, a: Any, b: Any, c: Any) -> _Pair3:
        return super().__new__(cls, (a, b, c))


class _Named(list):  # type: ignore[type-arg]
    """A list subclass whose constructor takes other arguments."""

    def __init__(self, name: str, items: Any) -> None:
        super().__init__(items)
        self.name = name


_SEQUENCE_SUBCLASSES = {
    "namedtuple": _Pair(PNG, "x"),
    "tuple subclass": _Pair3(PNG, "x", "y"),
    "list subclass": _Named("n", [PNG, "x"]),
}


@pytest.mark.parametrize("kind", _SEQUENCE_SUBCLASSES)
def test_a_sequence_subclass_holding_a_byte_object_never_raises_from_set_attribute(
    kind: str, exported_spans: InMemorySpanExporter
) -> None:
    observation = start_span("s")
    observation.set_attribute("payload", _SEQUENCE_SUBCLASSES[kind])
    observation.end()
    attributes = exported_spans.get_finished_spans()[0].attributes
    assert attributes is not None
    written = attributes["payload"]
    assert PNG.decode("latin-1") not in json.dumps(written, default=str)
    assert MARKER in json.dumps(written, default=str)


@pytest.mark.parametrize("kind", ["namedtuple", "tuple subclass", "list subclass"])
def test_a_homogeneous_sequence_subclass_with_a_nested_byte_object_is_rebuilt_plain(
    kind: str,
) -> None:
    holders = {
        "namedtuple": _Pair([PNG], [PNG]),
        "tuple subclass": _Pair3([PNG], [PNG], [PNG]),
        "list subclass": _Named("n", [[PNG], [PNG]]),
    }
    result = attribute_value(holders[kind])
    assert type(result) in (tuple, list)
    assert list(result) == [[MARKER]] * len(holders[kind])


def test_model_parameters_holding_byte_objects(exported_spans: InMemorySpanExporter) -> None:
    parameters = {"stop": [PNG], "seed": bytearray(PNG), "extra": {"blob": memoryview(PNG)}}
    with start_as_current_generation("g", model_parameters=parameters):
        pass
    attributes = exported_spans.get_finished_spans()[0].attributes
    assert attributes is not None
    written = {k: v for k, v in attributes.items() if "stop" in k or "seed" in k or "extra" in k}
    assert written
    assert not any(_leaks(str(v)) for v in written.values())
    assert all(MARKER in str(v) for v in written.values()), written
