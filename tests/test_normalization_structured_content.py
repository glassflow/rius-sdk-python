"""Content attributes another instrumentation sets as structured values.

The GenAI conventions now ask for ``gen_ai.input.messages``,
``gen_ai.output.messages``, ``gen_ai.system_instructions`` and
``gen_ai.tool.definitions`` to be maps and arrays rather than JSON strings,
and OpenTelemetry Python keeps such values on a span from 1.45. The exporter
writes each one as the JSON text this SDK writes for the same content, then
treats it exactly as that string: its media parts are rebuilt and stripped, so
no byte of a blob leaves. A span carrying the string and a span carrying the
structured value export the same attributes.

The spans are built directly rather than through a tracer: OpenTelemetry
before 1.45 drops a mapping attribute when it is set, and the floor CI job runs
this suite on 1.25.
"""

from __future__ import annotations

import base64
import json
import math
from types import MappingProxyType
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

from rius._serde import MAX_ATTR_CHARS, byte_object_marker
from rius.normalization import NormalizingSpanExporter
from rius.semconv import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_DEFINITIONS,
)

PNG = bytes(range(8))


def _export(attributes: dict[str, Any]) -> dict[str, Any]:
    context = SpanContext(trace_id=1, span_id=2, is_remote=False)
    span = ReadableSpan(name="chat", context=context, attributes=attributes)
    sink = InMemorySpanExporter()
    NormalizingSpanExporter(sink).export([span])
    [out] = sink.get_finished_spans()
    return dict(out.attributes or {})


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


_MESSAGES = [
    {"role": "user", "parts": [{"type": "text", "content": "Weather in Paris? <b>&</b> é"}]},
    {
        "role": "assistant",
        "parts": [
            {
                "type": "tool_call",
                "id": "call_1",
                "name": "get_weather",
                "arguments": {"city": "Paris", "days": 2},
            }
        ],
    },
]


@pytest.mark.parametrize(
    "key", [GEN_AI_INPUT_MESSAGES, GEN_AI_OUTPUT_MESSAGES, GEN_AI_SYSTEM_INSTRUCTIONS]
)
def test_a_structured_message_key_is_written_as_its_json(key: str) -> None:
    value = _MESSAGES if key != GEN_AI_SYSTEM_INSTRUCTIONS else _MESSAGES[0]["parts"]
    exported = _export({key: value})
    assert exported[key] == _json(value)


def test_structured_tool_definitions_are_written_as_their_json() -> None:
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    assert _export({GEN_AI_TOOL_DEFINITIONS: tools})[GEN_AI_TOOL_DEFINITIONS] == _json(tools)


def test_the_value_as_opentelemetry_stores_it_exports_as_the_string_does() -> None:
    # OpenTelemetry 1.45 stores a list as a tuple and keeps byte objects, so
    # the value a third-party span carries is this shape, not the one it set.
    stored = (
        {
            "role": "user",
            "parts": (
                {"type": "text", "content": "describe"},
                {"type": "blob", "modality": "image", "mime_type": "image/png", "content": PNG},
            ),
        },
    )
    as_string = _json(
        [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "describe"},
                    {
                        "type": "blob",
                        "modality": "image",
                        "mime_type": "image/png",
                        "content": base64.b64encode(PNG).decode(),
                    },
                ],
            }
        ]
    )
    structured = _export({GEN_AI_INPUT_MESSAGES: stored, "other": "x"})
    assert structured == _export({GEN_AI_INPUT_MESSAGES: as_string, "other": "x"})
    parts = json.loads(structured[GEN_AI_INPUT_MESSAGES])[0]["parts"]
    assert parts[1] == {
        "type": "blob",
        "modality": "image",
        "mime_type": "image/png",
        "byte_size": 8,
        "stripped_reason": "not_captured",
    }


def test_a_blob_in_the_system_instructions_and_tool_definitions_loses_its_bytes() -> None:
    blob = {"type": "blob", "mime_type": "image/png", "content": PNG}
    exported = _export({GEN_AI_SYSTEM_INSTRUCTIONS: [blob], GEN_AI_TOOL_DEFINITIONS: [blob]})
    encoded = base64.b64encode(PNG).decode()
    for key in (GEN_AI_SYSTEM_INSTRUCTIONS, GEN_AI_TOOL_DEFINITIONS):
        assert encoded not in exported[key]
        assert json.loads(exported[key])[0]["type"] == "blob"


def test_a_byte_object_anywhere_else_is_the_marker() -> None:
    exported = _export(
        {GEN_AI_OUTPUT_MESSAGES: [{"role": "assistant", "parts": [{"type": "x", "data": PNG}]}]}
    )
    part = json.loads(exported[GEN_AI_OUTPUT_MESSAGES])[0]["parts"][0]
    assert part == {"type": "x", "data": byte_object_marker(PNG)}


def test_a_memoryview_blob_is_read_as_its_bytes() -> None:
    blob = {"type": "blob", "mime_type": "image/png", "content": memoryview(PNG)}
    exported = _export({GEN_AI_SYSTEM_INSTRUCTIONS: [blob]})
    assert json.loads(exported[GEN_AI_SYSTEM_INSTRUCTIONS])[0]["byte_size"] == 8


@pytest.mark.parametrize(
    "key", [GEN_AI_INPUT_MESSAGES, GEN_AI_OUTPUT_MESSAGES, GEN_AI_SYSTEM_INSTRUCTIONS]
)
def test_a_released_memoryview_blob_is_not_sized_as_empty(key: str) -> None:
    # Its bytes cannot be read, so the size is unknown: never 0, which would
    # read as an empty blob once the marker's size is taken as the blob's.
    view = memoryview(PNG)
    view.release()
    blob = {"type": "blob", "mime_type": "image/png", "content": view}
    value: Any = (
        [blob] if key == GEN_AI_SYSTEM_INSTRUCTIONS else [{"role": "user", "parts": [blob]}]
    )
    exported = _export({key: value})
    payload = json.loads(exported[key])
    part = payload[0] if key == GEN_AI_SYSTEM_INSTRUCTIONS else payload[0]["parts"][0]
    assert part["type"] == "blob"
    assert "byte_size" not in part


def test_any_mapping_is_an_object() -> None:
    value = [MappingProxyType({"type": "text", "content": "hi"})]
    assert _export({GEN_AI_SYSTEM_INSTRUCTIONS: value})[GEN_AI_SYSTEM_INSTRUCTIONS] == _json(
        [{"type": "text", "content": "hi"}]
    )


def test_a_non_finite_number_inside_is_null() -> None:
    tools = [{"name": "t", "parameters": {"default": math.nan, "max": math.inf}}]
    exported = _export({GEN_AI_TOOL_DEFINITIONS: tools})
    assert (
        exported[GEN_AI_TOOL_DEFINITIONS]
        == '[{"name":"t","parameters":{"default":null,"max":null}}]'
    )


def test_an_array_of_strings_is_its_json_array() -> None:
    exported = _export({GEN_AI_SYSTEM_INSTRUCTIONS: ("be brief", "be kind")})
    assert exported[GEN_AI_SYSTEM_INSTRUCTIONS] == '["be brief","be kind"]'


def test_a_large_value_is_not_cut_at_the_attribute_cap() -> None:
    # A string under the strip budget keeps its length; so does the same
    # content as a structured value.
    text = "word " * (MAX_ATTR_CHARS // 2)
    value = [{"role": "user", "parts": [{"type": "text", "content": text}]}]
    exported = _export({GEN_AI_INPUT_MESSAGES: value})
    assert exported[GEN_AI_INPUT_MESSAGES] == _json(value)
    assert exported == _export({GEN_AI_INPUT_MESSAGES: _json(value)})


def test_a_value_that_cannot_be_encoded_never_fails_the_export() -> None:
    cycle: list[Any] = [{"type": "text", "content": "hi"}]
    cycle.append(cycle)
    exported = _export({GEN_AI_INPUT_MESSAGES: cycle, "other": "x"})
    assert isinstance(exported[GEN_AI_INPUT_MESSAGES], str)
    assert exported["other"] == "x"


def test_a_byte_key_never_fails_the_export_or_leaks_its_bytes() -> None:
    exported = _export({GEN_AI_INPUT_MESSAGES: [{b"k": PNG}]})
    assert isinstance(exported[GEN_AI_INPUT_MESSAGES], str)
    assert "\\x00\\x01" not in exported[GEN_AI_INPUT_MESSAGES]


_IMAGE = bytes(range(256)) * 40
_IMAGE_BASE64 = base64.b64encode(_IMAGE).decode("ascii")


def _cyclic_messages_with_a_blob() -> list[Any]:
    blob = {"type": "blob", "mime_type": "image/png", "content": _IMAGE}
    messages: list[Any] = [{"role": "user", "parts": [blob]}]
    messages[0]["self"] = messages
    return messages


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(_cyclic_messages_with_a_blob(), id="blob-in-a-cycle"),
        pytest.param(
            [{b"k": 1, "type": "blob", "mime_type": "image/png", "content": _IMAGE}],
            id="blob-in-a-byte-keyed-map",
        ),
        pytest.param(
            [{("a", 1): 1, "parts": [{"type": "blob", "content": _IMAGE}]}],
            id="blob-beside-a-tuple-key",
        ),
    ],
)
def test_a_blob_in_a_value_that_cannot_be_encoded_never_leaks_its_bytes(value: Any) -> None:
    exported = _export({GEN_AI_INPUT_MESSAGES: value})
    text = exported[GEN_AI_INPUT_MESSAGES]
    assert isinstance(text, str)
    assert _IMAGE_BASE64[:64] not in text
    assert "\\x00\\x01" not in text
    assert "[stripped" in text


@pytest.mark.parametrize("value", ["already text", 3, True, PNG])
def test_a_value_that_is_not_structured_is_not_reencoded(value: Any) -> None:
    exported = _export({GEN_AI_INPUT_MESSAGES: value})
    expected = byte_object_marker(PNG) if value is PNG else value
    assert exported[GEN_AI_INPUT_MESSAGES] == expected


def test_other_keys_keep_their_structured_values() -> None:
    exported = _export({"app.payload": {"a": [1, 2]}})
    assert exported["app.payload"] == {"a": [1, 2]}


def test_a_span_from_a_real_tracer_is_converted() -> None:
    probe = TracerProvider().get_tracer("probe").start_span("p", attributes={"m": {"a": 1}})
    if "m" not in (probe.attributes or {}):
        pytest.skip("this OpenTelemetry SDK drops mapping attributes (before 1.45)")
    sink = InMemorySpanExporter()
    tracer = TracerProvider().get_tracer("third.party")
    span = tracer.start_span("chat", attributes={GEN_AI_INPUT_MESSAGES: _MESSAGES})
    span.end()
    NormalizingSpanExporter(sink).export([span])  # type: ignore[list-item]
    [out] = sink.get_finished_spans()
    assert (out.attributes or {})[GEN_AI_INPUT_MESSAGES] == _json(_MESSAGES)
