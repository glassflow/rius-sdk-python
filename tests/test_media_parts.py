"""Media content becomes semconv blob / uri / file parts, and no bytes leave.

One test per row of the media contract's mapping table
(rius-semconv ``registry/media_parts.yaml``), for the paths this SDK owns: the
generation helpers (``set_input`` / ``set_output`` with raw provider dicts),
OpenInference's flattened messages, and the native semconv parts another
instrumentation writes. The rows no recorded family proves (Gemini, the OpenAI
Responses API, Anthropic url and file sources, ...) are proven here. Then the
exporter's last pass, the raw-payload strip, over every string attribute.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanContext

from media_contract import canonical_errors, contract, media_parts_in
from rius._serde import MAX_ATTR_CHARS, MAX_STRIP_CHARS, TRUNCATION_MARKER, serialize, text_value
from rius.generation import Generation, normalize_messages
from rius.normalization import (
    NormalizingSpanExporter,
    native_media_messages,
    reassemble_openinference_messages,
    strip_media,
)
from rius.semconv import GEN_AI_INPUT_MESSAGES, GEN_AI_OUTPUT_MESSAGES

PNG = "iVBORw0KGgo="  # 8 bytes
WAV = "UklGRiQAAABXQVZF"  # 12 bytes
PDF = "JVBERi0="  # 5 bytes


def _parts(content: Any, role: str = "user") -> list[dict[str, Any]]:
    """The parts a message's content becomes; every media part among them must
    be canonical, whatever the test then asserts."""
    [message] = normalize_messages([{"role": role, "content": content}], role)
    parts: list[dict[str, Any]] = message["parts"]
    for part in media_parts_in([message]):
        assert canonical_errors(part) == [], part
    return parts


# --- the contract's part examples ---

#: Each part example of the contract, made from an input its `from` describes,
#: through this SDK's message normalization.
CONTRACT_EXAMPLE_INPUTS: dict[str, dict[str, Any]] = {
    "blob_with_a_known_size": {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
    },
    "blob_whose_bytes_never_arrived": {"type": "image_url", "image_url": "__REDACTED__"},
    "data_url_that_is_not_base64": {"type": "uri", "uri": "data:text/plain,hello"},
    "data_url_that_is_not_base64_and_empty": {"type": "uri", "uri": "data:image/png,"},
    "blob_with_content_and_no_hint": {"type": "blob", "content": "iVBORw0KGgo="},
    "blob_with_content_labelled_image_for_a_pdf": {
        "type": "blob",
        "content": "JVBERi0=",
        "mime_type": "application/pdf",
        "modality": "image",
    },
    "blob_with_content_that_is_an_empty_base64_data_url": {
        "type": "blob",
        "mime_type": "image/png",
        "content": "data:;base64,",
    },
    "blob_with_no_content_and_no_stripped_reason": {"type": "blob", "modality": "audio"},
    "blob_with_content_that_is_not_a_string": {"type": "blob", "content": 5, "byte_size": 8.0},
    "uri_with_no_mime_type_and_no_modality": {
        "type": "uri",
        "uri": "https://example.com/download",
    },
    "uri_with_members_out_of_order_keeps_its_modality": {
        "uri": "https://example.com/talk",
        "modality": "image",
        "type": "uri",
        "note": "x",
    },
    "uri_with_an_invalid_mime_type": {
        "type": "uri",
        "mime_type": "text/plain;charset=x",
        "uri": "https://example.com/notes",
    },
    "uri_that_is_a_data_url": {"type": "uri", "uri": "data:image/png;base64,iVBORw0KGgo="},
    "uri_with_a_mime_type": {
        "type": "uri",
        "mime_type": "audio/mpeg",
        "uri": "https://example.com/a.mp3",
    },
    "uri_with_only_an_extension": {"type": "uri", "uri": "https://example.com/clip.mov"},
    "file_with_no_hint": {"type": "file", "file_id": "file-abc123"},
    "uri_with_a_cloud_storage_scheme": {
        "file_data": {"mime_type": "image/png", "file_uri": "gs://bucket/a.png"}
    },
    "uri_tagged_url_with_an_s3_scheme": {"type": "uri", "uri": "s3://bucket/report.pdf"},
    "uri_with_the_file_scheme_and_only_an_extension": {
        "type": "uri",
        "uri": "file:///tmp/clip.mov",
    },
    "image_url_that_is_empty": {"type": "image_url", "image_url": {"url": ""}},
    "image_url_that_is_redacted": {"type": "image_url", "image_url": {"url": "__REDACTED__"}},
    "anthropic_url_source_that_is_empty": {
        "type": "document",
        "source": {"type": "url", "url": ""},
    },
    "anthropic_file_source_that_is_empty": {
        "type": "image",
        "source": {"type": "file", "file_id": "", "media_type": "image/png"},
    },
    "input_image_with_an_empty_file_id": {"type": "input_image", "file_id": ""},
    "openai_file_with_an_empty_file_id": {
        "type": "file",
        "file": {"file_id": "", "filename": "a.png"},
    },
    "input_image_with_a_redacted_url_and_a_file_id": {
        "type": "input_image",
        "image_url": "__REDACTED__",
        "file_id": "file-abc123",
    },
}

#: Part examples only the Vercel AI SDK dialect makes, which this SDK does not
#: normalize (the TypeScript SDK and the sink do).
VERCEL_ONLY = {"vercel_file_by_url_with_no_media_type", "blob_from_an_unknown_v7_data_tag"}

PART_EXAMPLES = contract()["part_shapes"]["examples"]


def test_every_contract_part_example_is_made_here_or_is_vercel_only() -> None:
    names = {e["name"] for e in PART_EXAMPLES}
    assert names - VERCEL_ONLY == set(CONTRACT_EXAMPLE_INPUTS)
    assert names >= VERCEL_ONLY
    for example in PART_EXAMPLES:
        if example["name"] in VERCEL_ONLY:
            assert "Vercel" in example["from"], example["name"]


@pytest.mark.parametrize(
    "example",
    [e for e in PART_EXAMPLES if e["name"] in CONTRACT_EXAMPLE_INPUTS],
    ids=lambda e: str(e["name"]),
)
def test_a_contract_part_example_is_made_byte_for_byte(example: dict[str, Any]) -> None:
    [part] = _parts([CONTRACT_EXAMPLE_INPUTS[example["name"]]])
    assert json.dumps(part, separators=(",", ":")) == example["part"]


def test_every_contract_message_example_is_a_dialect_this_sdk_does_not_read() -> None:
    """The message examples pin the Vercel AI SDK dialect (ai.prompt.messages),
    which only the TypeScript SDK and the sink normalize."""
    sources = {e["source"] for e in contract()["part_shapes"].get("message_examples") or []}
    assert sources <= {"ai.prompt.messages"}


def _blob(modality: str, mime: str | None, size: int | None, reason: str) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "blob", "modality": modality}
    if mime is not None:
        part["mime_type"] = mime
    if size is not None:
        part["byte_size"] = size
    part["stripped_reason"] = reason
    return part


# --- openai.chat ---


def test_openai_image_data_url_is_a_blob_with_the_urls_mime() -> None:
    url = f"data:image/png;base64,{PNG}"
    assert _parts([{"type": "image_url", "image_url": {"url": url}}]) == [
        _blob("image", "image/png", 8, "not_captured")
    ]


def test_openai_image_https_url_is_a_uri() -> None:
    url = "https://example.com/cat.png"
    assert _parts([{"type": "image_url", "image_url": {"url": url, "detail": "low"}}]) == [
        {"type": "uri", "modality": "image", "uri": url}
    ]


def test_openai_input_audio_is_an_audio_blob_typed_by_its_format() -> None:
    assert _parts([{"type": "input_audio", "input_audio": {"data": WAV, "format": "wav"}}]) == [
        _blob("audio", "audio/wav", 12, "not_captured")
    ]


def test_openai_file_id_is_a_document_file() -> None:
    assert _parts([{"type": "file", "file": {"file_id": "file-abc"}}]) == [
        {"type": "file", "modality": "document", "file_id": "file-abc"}
    ]


def test_openai_file_data_takes_its_modality_from_the_data_url() -> None:
    file = {"file_data": f"data:application/pdf;base64,{PDF}", "filename": "a.png"}
    assert _parts([{"type": "file", "file": file}]) == [
        _blob("document", "application/pdf", 5, "not_captured")
    ]


def test_openai_file_data_without_a_mime_takes_its_modality_from_the_filename() -> None:
    file = {"file_data": PNG, "filename": "chart.PNG"}
    assert _parts([{"type": "file", "file": file}]) == [_blob("image", None, 8, "not_captured")]


def test_openai_reply_audio_is_a_blob_then_its_transcript() -> None:
    [message] = normalize_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "audio": {"id": "audio_1", "data": WAV, "transcript": "Hello."},
            }
        ],
        "assistant",
    )
    assert message == {
        "role": "assistant",
        "parts": [
            _blob("audio", None, 12, "not_captured"),
            {"type": "text", "content": "Hello."},
        ],
    }


# --- openai.responses ---


@pytest.mark.parametrize(
    ("item", "part"),
    [
        (
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{PNG}"},
            _blob("image", "image/jpeg", 8, "not_captured"),
        ),
        (
            {"type": "input_image", "image_url": "https://example.com/a.jpg", "detail": "auto"},
            {"type": "uri", "modality": "image", "uri": "https://example.com/a.jpg"},
        ),
        (
            {"type": "input_image", "file_id": "file-img"},
            {"type": "file", "modality": "image", "file_id": "file-img"},
        ),
        (
            {"type": "input_file", "file_id": "file-doc", "filename": "q3.pdf"},
            {"type": "file", "modality": "document", "file_id": "file-doc"},
        ),
        (
            {
                "type": "input_file",
                "file_data": f"data:application/pdf;base64,{PDF}",
                "filename": "q3.pdf",
            },
            _blob("document", "application/pdf", 5, "not_captured"),
        ),
        (
            {"type": "input_file", "file_url": "https://example.com/q3.pdf"},
            {"type": "uri", "modality": "document", "uri": "https://example.com/q3.pdf"},
        ),
    ],
)
def test_openai_responses_inputs(item: dict[str, Any], part: dict[str, Any]) -> None:
    assert _parts([{"type": "input_text", "text": "look"}, item])[1:] == [part]


def test_openai_responses_generated_image_is_an_image_blob() -> None:
    item = {
        "type": "image_generation_call",
        "id": "ig_1",
        "status": "completed",
        "output_format": "png",
        "result": PNG,
    }
    assert normalize_messages([item], "assistant") == [
        {"role": "assistant", "parts": [_blob("image", "image/png", 8, "not_captured")]}
    ]


# --- anthropic.messages ---


def test_anthropic_base64_image_is_a_blob_typed_by_its_media_type() -> None:
    source = {"type": "base64", "media_type": "image/png", "data": PNG}
    assert _parts([{"type": "image", "source": source}]) == [
        _blob("image", "image/png", 8, "not_captured")
    ]


def test_anthropic_base64_document_is_a_document_blob() -> None:
    source = {"type": "base64", "media_type": "application/pdf", "data": PDF}
    assert _parts([{"type": "document", "source": source}]) == [
        _blob("document", "application/pdf", 5, "not_captured")
    ]


def test_anthropic_image_labelled_with_a_non_image_mime_is_not_an_image() -> None:
    source = {"type": "base64", "media_type": "application/pdf", "data": PDF}
    assert _parts([{"type": "image", "source": source}])[0]["modality"] == "document"


@pytest.mark.parametrize(("block_type", "modality"), [("image", "image"), ("document", "document")])
def test_anthropic_url_source_is_a_uri(block_type: str, modality: str) -> None:
    source = {"type": "url", "url": "https://example.com/x"}
    assert _parts([{"type": block_type, "source": source}]) == [
        {"type": "uri", "modality": modality, "uri": "https://example.com/x"}
    ]


@pytest.mark.parametrize(("block_type", "modality"), [("image", "image"), ("document", "document")])
def test_anthropic_file_source_is_a_file(block_type: str, modality: str) -> None:
    source = {"type": "file", "file_id": "file_011"}
    assert _parts([{"type": block_type, "source": source}]) == [
        {"type": "file", "modality": modality, "file_id": "file_011"}
    ]


def test_anthropic_text_document_is_text_to_the_model() -> None:
    text_source = {"type": "text", "media_type": "text/plain", "data": "The grass is green."}
    content_source = {
        "type": "content",
        "content": [{"type": "text", "text": "One."}, {"type": "text", "text": "Two."}],
    }
    assert _parts(
        [
            {"type": "document", "source": text_source},
            {"type": "document", "source": content_source},
        ]
    ) == [
        {"type": "text", "content": "The grass is green."},
        {"type": "text", "content": "One."},
        {"type": "text", "content": "Two."},
    ]


# --- gemini.generate_content ---


@pytest.mark.parametrize("spelling", ["inline_data", "inlineData"])
def test_gemini_inline_data_is_a_blob_typed_by_its_mime(spelling: str) -> None:
    mime_key = "mime_type" if spelling == "inline_data" else "mimeType"
    part = {spelling: {mime_key: "audio/mp3", "data": WAV}}
    assert _parts([part]) == [_blob("audio", "audio/mp3", 12, "not_captured")]


@pytest.mark.parametrize("spelling", ["file_data", "fileData"])
def test_gemini_file_data_is_a_uri_typed_by_its_mime(spelling: str) -> None:
    keys = ("mime_type", "file_uri") if spelling == "file_data" else ("mimeType", "fileUri")
    uri = "gs://bucket/clip.mp4"
    part = {spelling: {keys[0]: "video/mp4", keys[1]: uri}}
    assert _parts([part]) == [
        {"type": "uri", "modality": "video", "mime_type": "video/mp4", "uri": uri}
    ]


def test_gemini_parts_in_a_parts_message_are_media_parts_too() -> None:
    [message] = normalize_messages(
        [
            {
                "role": "user",
                "parts": [
                    {"text": "what is this"},
                    {"inline_data": {"mime_type": "image/png", "data": PNG}},
                ],
            }
        ],
        "user",
    )
    assert message["parts"] == [
        {"text": "what is this"},
        _blob("image", "image/png", 8, "not_captured"),
    ]


# --- a reference dropped upstream ---


@pytest.mark.parametrize(
    ("item", "part"),
    [
        (
            {"type": "image_url", "image_url": {"url": ""}},
            _blob("image", None, None, "redacted_upstream"),
        ),
        ({"type": "image_url", "image_url": ""}, _blob("image", None, None, "redacted_upstream")),
        (
            {"type": "image_url", "image_url": {"url": "__REDACTED__"}},
            _blob("image", None, None, "redacted_upstream"),
        ),
        ({"type": "input_image", "image_url": ""}, _blob("image", None, None, "redacted_upstream")),
        ({"type": "input_image", "file_id": ""}, _blob("image", None, None, "redacted_upstream")),
        (
            {"type": "input_image", "image_url": "", "file_id": "__REDACTED__"},
            _blob("image", None, None, "redacted_upstream"),
        ),
        (
            {"type": "image", "source": {"type": "url", "url": ""}},
            _blob("image", None, None, "redacted_upstream"),
        ),
        (
            {
                "type": "document",
                "source": {"type": "url", "url": "__REDACTED__", "media_type": "application/pdf"},
            },
            _blob("document", "application/pdf", None, "redacted_upstream"),
        ),
        (
            {"type": "image", "source": {"type": "file", "file_id": ""}},
            _blob("image", None, None, "redacted_upstream"),
        ),
        (
            {"type": "document", "source": {"type": "file", "file_id": "__REDACTED__"}},
            _blob("document", None, None, "redacted_upstream"),
        ),
        (
            {"file_data": {"mime_type": "image/png", "file_uri": ""}},
            _blob("image", "image/png", None, "redacted_upstream"),
        ),
        (
            {"fileData": {"fileUri": "__REDACTED__"}},
            _blob("document", None, None, "redacted_upstream"),
        ),
        (
            {"type": "file", "file": {"file_url": "__REDACTED__", "filename": "a.png"}},
            _blob("image", None, None, "redacted_upstream"),
        ),
        (
            {"type": "input_file", "file_id": "__REDACTED__"},
            _blob("document", None, None, "redacted_upstream"),
        ),
    ],
)
def test_an_empty_or_redacted_reference_is_a_blob_redacted_upstream(
    item: dict[str, Any], part: dict[str, Any]
) -> None:
    assert _parts([item]) == [part]


@pytest.mark.parametrize(
    ("item", "part"),
    [
        (
            {"type": "input_image", "image_url": "__REDACTED__", "file_id": "file-1"},
            {"type": "file", "modality": "image", "file_id": "file-1"},
        ),
        (
            {"type": "input_image", "image_url": "", "file_id": "file-1"},
            {"type": "file", "modality": "image", "file_id": "file-1"},
        ),
        (
            {
                "type": "file",
                "file": {"file_id": "__REDACTED__", "file_data": PNG, "filename": "a.png"},
            },
            _blob("image", None, 8, "not_captured"),
        ),
        (
            {"type": "input_file", "file_id": "__REDACTED__", "file_url": "https://x/a.pdf"},
            {"type": "uri", "modality": "document", "uri": "https://x/a.pdf"},
        ),
    ],
)
def test_a_usable_reference_wins_over_a_dropped_one(
    item: dict[str, Any], part: dict[str, Any]
) -> None:
    assert _parts([item]) == [part]


@pytest.mark.parametrize(
    "file",
    [
        {"file_id": ""},
        {"file_url": ""},
        {"file_data": ""},
        {"file_id": "", "file_url": "", "filename": "a.wav"},
        {"file_data": "", "file_id": "__REDACTED__", "filename": "a.wav"},
        {"file_data": "__REDACTED__", "filename": "a.wav"},
    ],
)
def test_an_openai_file_whose_references_are_all_dropped_is_a_redacted_blob(
    file: dict[str, Any],
) -> None:
    """Never a part of type file without a file_id, as the chat item, whose own
    type is file, would be if it stayed as it came."""
    modality = "audio" if "filename" in file else "document"
    redacted = _blob(modality, None, None, "redacted_upstream")
    assert _parts([{"type": "file", "file": file}]) == [redacted]
    assert _parts([{"type": "input_file", **file}]) == [redacted]


def test_an_openai_file_with_no_reference_at_all_stays_as_it_came() -> None:
    assert _parts([{"type": "input_file", "filename": "a.png"}]) == [
        {"type": "input_file", "filename": "a.png"}
    ]


def test_a_native_uri_or_file_keeps_an_empty_reference() -> None:
    """A native part is no provider item: its empty uri or file_id is kept."""
    assert _parts([{"type": "uri", "uri": ""}, {"type": "file", "file_id": ""}]) == [
        {"type": "uri", "modality": "document", "uri": ""},
        {"type": "file", "modality": "document", "file_id": ""},
    ]


# --- a URL of any scheme ---


@pytest.mark.parametrize(
    ("item", "part"),
    [
        (
            {"type": "image_url", "image_url": {"url": "gs://bucket/a.png"}},
            {"type": "uri", "modality": "image", "uri": "gs://bucket/a.png"},
        ),
        (
            {"type": "input_image", "image_url": "s3://bucket/a.jpg"},
            {"type": "uri", "modality": "image", "uri": "s3://bucket/a.jpg"},
        ),
        (
            {"type": "document", "source": {"type": "url", "url": "file:///tmp/a.pdf"}},
            {"type": "uri", "modality": "document", "uri": "file:///tmp/a.pdf"},
        ),
        (
            {"file_data": {"file_uri": "s3://bucket/clip.webm"}},
            {"type": "uri", "modality": "video", "uri": "s3://bucket/clip.webm"},
        ),
        (
            {"type": "uri", "uri": "gs://bucket/a.wav"},
            {"type": "uri", "modality": "audio", "uri": "gs://bucket/a.wav"},
        ),
    ],
)
def test_a_url_of_any_scheme_but_data_keeps_its_link(
    item: dict[str, Any], part: dict[str, Any]
) -> None:
    assert _parts([item]) == [part]


# --- semconv.native ---


def test_a_native_blob_with_content_is_stripped() -> None:
    blob = {"type": "blob", "mime_type": "image/png", "modality": "image", "content": PNG}
    [message] = normalize_messages([{"role": "user", "parts": [blob]}], "user")
    assert message["parts"] == [_blob("image", "image/png", 8, "not_captured")]


def test_a_stripped_blob_is_left_as_it_is() -> None:
    blob = _blob("image", "image/png", 8, "not_captured")
    [message] = normalize_messages([{"role": "user", "parts": [dict(blob)]}], "user")
    assert message["parts"] == [blob]


def test_a_native_blob_mislabelled_image_gets_the_modality_of_its_mime() -> None:
    blob = {"type": "blob", "modality": "image", "mime_type": "application/pdf", "content": PDF}
    [message] = normalize_messages([{"role": "user", "parts": [blob]}], "user")
    assert message["parts"] == [_blob("document", "application/pdf", 5, "not_captured")]


def _native(part: dict[str, Any]) -> str:
    """A native part after normalization, as the bytes it is written as."""
    [message] = normalize_messages([{"role": "user", "parts": [part]}], "user")
    return json.dumps(message["parts"], separators=(",", ":"))


def test_a_native_uri_missing_a_modality_is_rebuilt_with_it_in_shape_order() -> None:
    uri = {"uri": "https://example.com/a.png", "mime_type": "IMAGE/png", "type": "uri", "x": 1}
    assert _native(uri) == (
        '[{"type":"uri","modality":"image","mime_type":"IMAGE/png",'
        '"uri":"https://example.com/a.png"}]'
    )


def test_a_native_file_missing_a_modality_is_rebuilt_with_it() -> None:
    file = {"type": "file", "mime_type": "audio/wav", "file_id": "f-1"}
    assert (
        _native(file)
        == '[{"type":"file","modality":"audio","mime_type":"audio/wav","file_id":"f-1"}]'
    )


def test_a_native_uri_keeps_the_modality_it_has() -> None:
    uri = {"type": "uri", "modality": "image", "mime_type": "application/pdf", "uri": "u"}
    assert _native(uri) == json.dumps([uri], separators=(",", ":"))


@pytest.mark.parametrize(
    ("part", "written"),
    [
        (
            {"type": "uri", "uri": "https://x"},
            '[{"type":"uri","modality":"document","uri":"https://x"}]',
        ),
        (
            {"uri": "u", "mime_type": "application/pdf", "modality": "image", "type": "uri"},
            '[{"type":"uri","modality":"image","mime_type":"application/pdf","uri":"u"}]',
        ),
        (
            {"type": "uri", "mime_type": "text/plain;charset=x", "uri": "u"},
            '[{"type":"uri","modality":"document","uri":"u"}]',
        ),
        (
            {"type": "uri", "uri": "https://example.com/clip.mov?sig=1"},
            '[{"type":"uri","modality":"video","uri":"https://example.com/clip.mov?sig=1"}]',
        ),
        (
            {"type": "file", "file_id": "file-abc123", "name": "x"},
            '[{"type":"file","modality":"document","file_id":"file-abc123"}]',
        ),
        (
            {"stripped_reason": "not_captured", "type": "blob", "byte_size": 3, "x": None},
            '[{"type":"blob","modality":"document","byte_size":3,"stripped_reason":"not_captured"}]',
        ),
        (
            {"type": "blob", "modality": "image", "byte_size": True},
            '[{"type":"blob","modality":"image","stripped_reason":"redacted_upstream"}]',
        ),
    ],
)
def test_every_native_media_part_is_rebuilt_into_the_canonical_shape(
    part: dict[str, Any], written: str
) -> None:
    assert _native(part) == written


@pytest.mark.parametrize(
    "part",
    [
        {"type": "uri", "uri": 3},
        {"modality": "image", "type": "uri", "x": 1},
        {"type": "file", "mime_type": "text/plain;charset=x"},
        {"file_id": None, "type": "file", "modality": "audio"},
    ],
)
def test_a_native_uri_or_file_with_nothing_to_point_to_is_kept_as_it_came(
    part: dict[str, Any],
) -> None:
    assert _native(part) == json.dumps([part], separators=(",", ":"))


def test_a_native_blob_with_content_and_no_hint_is_a_document_with_no_bytes() -> None:
    blob = {"type": "blob", "content": PNG, "extra": True}
    assert _native(blob) == (
        '[{"type":"blob","modality":"document","byte_size":8,"stripped_reason":"not_captured"}]'
    )


def test_an_invalid_mime_is_omitted_never_null() -> None:
    blob = {"type": "blob", "modality": "image", "mime_type": "png", "content": PNG}
    assert _native(blob) == (
        '[{"type":"blob","modality":"image","byte_size":8,"stripped_reason":"not_captured"}]'
    )


@pytest.mark.parametrize(
    ("filename", "modality"),
    [
        ("a.webm", "video"),
        ("a.MOV", "video"),
        ("a.flac", "audio"),
        ("a.txt", "document"),
        ("a", "document"),
    ],
)
def test_a_filename_names_the_modality_of_file_data(filename: str, modality: str) -> None:
    parts = _parts([{"type": "file", "file": {"file_data": PNG, "filename": filename}}])
    assert parts == [_blob(modality, None, 8, "not_captured")]


def test_a_gemini_part_with_no_mime_is_a_document() -> None:
    assert _parts([{"inline_data": {"data": PNG}}]) == [_blob("document", None, 8, "not_captured")]


def test_a_native_uri_holding_a_data_url_is_a_blob() -> None:
    uri = {"type": "uri", "modality": "image", "uri": f"data:image/png;base64,{PNG}"}
    [message] = normalize_messages([{"role": "user", "parts": [uri]}], "user")
    assert message["parts"] == [_blob("image", "image/png", 8, "not_captured")]


# --- what is unknown passes through ---


def test_an_unknown_typed_part_passes_through_unchanged() -> None:
    part = {"type": "video_frame", "data": {"frame": 3}}
    assert _parts([part]) == [part]


def test_an_image_url_part_is_never_wrapped_in_a_text_part() -> None:
    parts = _parts([{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}])
    assert parts == [_blob("image", "image/png", 0, "not_captured")]


@pytest.mark.parametrize(
    ("url", "part"),
    [
        ("data:text/plain,hi", _blob("document", "text/plain", None, "not_captured")),
        ("data:None;url,None", _blob("image", None, None, "redacted_upstream")),
        ("data:image/png,", _blob("image", "image/png", None, "redacted_upstream")),
    ],
)
def test_a_data_url_that_is_not_base64_is_a_blob_with_no_size(
    url: str, part: dict[str, Any]
) -> None:
    assert _parts([{"type": "image_url", "image_url": {"url": url}}]) == [part]


# --- the generation helpers end to end ---


def test_set_input_records_the_blob_and_no_bytes(exported_spans: InMemorySpanExporter) -> None:
    from rius import start_as_current_generation

    big = "iVBORw0KGgo" + "A" * 100_000 + "="
    with start_as_current_generation(model="m") as generation:
        assert isinstance(generation, Generation)
        generation.set_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"inline data:image/png;base64,{PNG} too"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": big},
                        },
                    ],
                }
            ]
        )
    [span] = exported_spans.get_finished_spans()
    messages = span.attributes[GEN_AI_INPUT_MESSAGES]
    assert isinstance(messages, str)
    assert json.loads(messages) == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "inline [stripped image/png 8 bytes] too"},
                _blob("image", "image/png", 75_008, "not_captured"),
            ],
        }
    ]
    sizes = json.loads(str(span.attributes["rius.context.sizes"]))
    assert sizes["input_messages"][0]["parts"] == [
        {"type": "text", "bytes": len(b"inline  too")},
        {"type": "blob", "modality": "image", "byte_size": 8},
        {"type": "blob", "modality": "image", "byte_size": 75_008},
    ]


# --- the strip before the cap ---


def test_a_request_with_a_large_image_stays_whole_json() -> None:
    big = "A" * (MAX_ATTR_CHARS * 4)
    request = {
        "model": "claude",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": big},
                    }
                ],
            }
        ],
    }
    text = serialize(request)
    assert not text.endswith(TRUNCATION_MARKER)
    assert json.loads(text)["messages"][0]["content"][0]["source"]["data"] == (
        f"[stripped image/png {len(big) * 3 // 4} bytes]"
    )


def test_a_string_input_is_stripped_before_it_is_cut() -> None:
    text = "x" * 10 + f"data:image/png;base64,{'A' * (MAX_ATTR_CHARS * 2)}" + " end"
    assert (
        text_value(text) == "x" * 10 + f"[stripped image/png {MAX_ATTR_CHARS * 3 // 2} bytes] end"
    )


# --- the exporter ---


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


def test_openinference_images_become_media_parts() -> None:
    prefix = "llm.input_messages.0.message."
    attributes = {
        prefix + "role": "user",
        prefix + "contents.0.message_content.type": "image",
        prefix + "contents.0.message_content.image.image.url": f"data:image/png;base64,{PNG}",
        prefix + "contents.1.message_content.type": "image",
        prefix + "contents.1.message_content.image.image.url": "https://example.com/cat.png",
        prefix + "contents.2.message_content.type": "image",
        prefix + "contents.2.message_content.image.image.url": "__REDACTED__",
    }
    rebuilt = reassemble_openinference_messages(attributes)
    assert rebuilt is not None
    assert json.loads(rebuilt[GEN_AI_INPUT_MESSAGES])[0]["parts"] == [
        _blob("image", "image/png", 8, "not_captured"),
        {"type": "uri", "modality": "image", "uri": "https://example.com/cat.png"},
        _blob("image", None, None, "redacted_upstream"),
    ]


@pytest.mark.parametrize("url", [None, 7, ["https://example.com/cat.png"]])
def test_an_image_item_whose_url_is_not_a_string_makes_no_media_part(url: Any) -> None:
    from rius.normalization import _content_part

    item = {"type": "image", "image.image.url": url}
    part = _content_part(item, {})
    assert part is not None
    assert part["type"] == "text"
    assert "uri" not in part


@pytest.mark.parametrize(
    ("member_order", "parts"),
    [
        (
            '{"type":"text","uri":"https://example.com/a.png","type":"uri"}',
            [{"type": "uri", "modality": "image", "uri": "https://example.com/a.png"}],
        ),
        (
            '{"type":"uri","uri":"https://example.com/a.png","type":"text","content":"hi"}',
            [{"type": "text", "uri": "https://example.com/a.png", "content": "hi"}],
        ),
    ],
)
def test_a_duplicate_part_type_is_read_as_its_last_member(
    member_order: str, parts: list[dict[str, Any]]
) -> None:
    raw = '[{"role":"user","parts":[' + member_order + "]}]"
    out = _export(_span({GEN_AI_INPUT_MESSAGES: raw}))
    assert out.attributes is not None
    assert json.loads(str(out.attributes[GEN_AI_INPUT_MESSAGES]))[0]["parts"] == parts


def test_native_messages_with_blob_content_are_rewritten() -> None:
    raw = json.dumps(
        [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "What is in this image?"},
                    {"type": "blob", "mime_type": "image/png", "modality": "image", "content": PNG},
                ],
            }
        ],
        separators=(",", ":"),
    )
    rebuilt = native_media_messages({GEN_AI_INPUT_MESSAGES: raw})
    assert rebuilt is not None
    assert rebuilt[GEN_AI_INPUT_MESSAGES] == (
        '[{"role":"user","parts":[{"type":"text","content":"What is in this image?"},'
        '{"type":"blob","modality":"image","mime_type":"image/png","byte_size":8,'
        '"stripped_reason":"not_captured"}]}]'
    )


@pytest.mark.parametrize(
    ("size", "written"),
    [("8.0", ',"byte_size":8'), ("8e0", ',"byte_size":8'), ("8.5", ""), ("9007199254740992", "")],
)
def test_a_native_blob_keeps_a_size_that_is_a_safe_whole_count(size: str, written: str) -> None:
    raw = '[{"type":"blob","byte_size":' + size + ',"stripped_reason":"not_captured"}]'
    rebuilt = native_media_messages({"gen_ai.system_instructions": raw})
    assert rebuilt is not None
    assert rebuilt["gen_ai.system_instructions"] == (
        '[{"type":"blob","modality":"document"' + written + ',"stripped_reason":"not_captured"}]'
    )


def test_a_native_part_is_rebuilt_in_place_and_every_other_byte_kept() -> None:
    raw = (
        '[ {"role": "user", "parts": [ {"type": "blob", "content": "QUJD",'
        ' "mime_type": "image/png"} ] } ]'
    )
    rebuilt = native_media_messages({GEN_AI_INPUT_MESSAGES: raw})
    assert rebuilt is not None
    assert rebuilt[GEN_AI_INPUT_MESSAGES] == (
        '[ {"role": "user", "parts": [ {"type":"blob","modality":"image",'
        '"mime_type":"image/png","byte_size":3,"stripped_reason":"not_captured"} ] } ]'
    )


def test_system_instructions_are_rebuilt_in_place_and_every_other_byte_kept() -> None:
    raw = '[ {"type": "text", "content": "Be brief."} ,\n  {"type": "blob", "content": "QUJD"} ]'
    rebuilt = native_media_messages({"gen_ai.system_instructions": raw})
    assert rebuilt is not None
    assert rebuilt["gen_ai.system_instructions"] == (
        '[ {"type": "text", "content": "Be brief."} ,\n  {"type":"blob","modality":"document",'
        '"byte_size":3,"stripped_reason":"not_captured"} ]'
    )


def test_a_canonical_part_written_with_spaces_is_rewritten_compact() -> None:
    raw = '[ {"type": "file", "modality": "document", "file_id": "f-1"} ]'
    rebuilt = native_media_messages({"gen_ai.system_instructions": raw})
    assert rebuilt is not None
    assert rebuilt["gen_ai.system_instructions"] == (
        '[ {"type":"file","modality":"document","file_id":"f-1"} ]'
    )


@pytest.mark.parametrize(
    "raw",
    [
        # already canonical, spaced only around it
        '[\n  {"role": "user", "parts": [ {"type":"uri","modality":"image",'
        '"uri":"https://example.com/a.png"} ]}\n]',
        # a number spelled 1e2 elsewhere, and a non-ASCII escape kept as written
        '[{"role":"user","n":1e2,"parts":[{"type":"text","content":"caf\\u00e9"},'
        '{"type":"file","modality":"document","file_id":"f-1"}]}]',
    ],
)
def test_a_value_whose_parts_are_canonical_is_kept_byte_for_byte(raw: str) -> None:
    assert native_media_messages({GEN_AI_INPUT_MESSAGES: raw}) is None


def test_only_the_rebuilt_part_is_rewritten_and_its_neighbours_keep_their_bytes() -> None:
    raw = (
        '[{"role":"user", "parts":[ {"type":"text","content":"caf\\u00e9 \\/"} ,'
        ' {"uri": "https://x/a.mov?sig=1", "type": "uri", "extra": [1, 2]} ]},'
        ' {"role":"assistant","parts":[]}]'
    )
    rebuilt = native_media_messages({GEN_AI_OUTPUT_MESSAGES: raw})
    assert rebuilt is not None
    assert rebuilt[GEN_AI_OUTPUT_MESSAGES] == (
        '[{"role":"user", "parts":[ {"type":"text","content":"caf\\u00e9 \\/"} ,'
        ' {"type":"uri","modality":"video","uri":"https://x/a.mov?sig=1"} ]},'
        ' {"role":"assistant","parts":[]}]'
    )


def test_a_rebuilt_part_is_written_with_the_minimal_escapes() -> None:
    raw = '[{"type": "file", "file_id": "f\\u00e9\\u2028\\u0001\\"", "mime_type": "x"}]'
    rebuilt = native_media_messages({"gen_ai.system_instructions": raw})
    assert rebuilt is not None
    assert rebuilt["gen_ai.system_instructions"] == (
        '[{"type":"file","modality":"document","file_id":"fé \\u0001\\""}]'
    )


def test_a_part_type_written_with_an_escape_is_still_rebuilt() -> None:
    raw = '[{"type": "\\u0062lob", "content": "QUJD"}]'
    rebuilt = native_media_messages({"gen_ai.system_instructions": raw})
    assert rebuilt is not None
    assert rebuilt["gen_ai.system_instructions"] == (
        '[{"type":"blob","modality":"document","byte_size":3,"stripped_reason":"not_captured"}]'
    )


@pytest.mark.parametrize(("levels", "rebuilt"), [(500, True), (1000, True), (1001, False)])
def test_parts_are_rebuilt_up_to_a_thousand_levels_and_a_deeper_value_is_left(
    levels: int, rebuilt: bool
) -> None:
    # The message array, the message, its parts and the tool call are four.
    arguments = "[" * (levels - 4) + "]" * (levels - 4)
    raw = (
        '[{"role":"user","parts":[{"type":"blob","modality":"image","content":"QUFB"},'
        '{"type":"tool_call","arguments":' + arguments + "}]}]"
    )
    out = native_media_messages({GEN_AI_INPUT_MESSAGES: raw})
    if not rebuilt:
        assert out is None
        return
    assert out is not None
    assert out[GEN_AI_INPUT_MESSAGES] == raw.replace(
        '{"type":"blob","modality":"image","content":"QUFB"}',
        '{"type":"blob","modality":"image","byte_size":3,"stripped_reason":"not_captured"}',
    )


def test_a_value_past_the_part_limit_still_exports_and_is_stripped() -> None:
    arguments = "[" * 997 + "]" * 997
    raw = (
        '[{"role":"user","parts":[{"type":"uri","uri":"data:image/png;base64,QUFB"},'
        '{"type":"tool_call","arguments":' + arguments + "}]}]"
    )
    out = _export(_span({GEN_AI_INPUT_MESSAGES: raw}))
    assert out.attributes is not None
    assert str(out.attributes[GEN_AI_INPUT_MESSAGES]).startswith(
        '[{"role":"user","parts":[{"type":"uri","uri":"[stripped image/png 3 bytes]"}'
    )


def test_a_duplicated_parts_member_is_read_as_its_last() -> None:
    blob = '{"type":"blob","content":"QUJD"}'
    rebuilt = '{"type":"blob","modality":"document","byte_size":3,"stripped_reason":"not_captured"}'
    first = '[{"role":"user","parts":[' + blob + '],"parts":[{"type":"text","content":"hi"}]}]'
    assert native_media_messages({GEN_AI_INPUT_MESSAGES: first}) is None
    last = '[{"role":"user","parts":[' + blob + '],"parts":[' + blob + "]}]"
    out = native_media_messages({GEN_AI_INPUT_MESSAGES: last})
    assert out is not None
    assert out[GEN_AI_INPUT_MESSAGES] == (
        '[{"role":"user","parts":[' + blob + '],"parts":[' + rebuilt + "]}]"
    )


@pytest.mark.parametrize("parts_only", [False, True])
def test_a_value_twenty_thousand_levels_deep_is_left_as_it_came(parts_only: bool) -> None:
    raw = '[{"type":"blob","content":"QUFB","parts":' + "[" * 20000 + "]" * 20000 + "}]"
    key = "gen_ai.system_instructions" if parts_only else GEN_AI_INPUT_MESSAGES
    assert native_media_messages({key: raw}) is None
    out = _export(_span({key: raw}))
    assert out.attributes is not None


def test_native_messages_without_media_are_left_byte_for_byte() -> None:
    raw = '[{"role": "user", "parts": [{"type": "text", "content": "a blob of text"}]}]'
    assert native_media_messages({GEN_AI_OUTPUT_MESSAGES: raw}) is None


def test_system_instructions_are_parts_and_are_rewritten() -> None:
    raw = '[{"type":"blob","modality":"image","content":"' + PNG + '"}]'
    rebuilt = native_media_messages({"gen_ai.system_instructions": raw})
    assert rebuilt is not None
    assert json.loads(rebuilt["gen_ai.system_instructions"]) == [
        _blob("image", None, 8, "not_captured")
    ]


def test_the_strip_reaches_every_string_attribute_and_array_element() -> None:
    url = f"data:image/png;base64,{PNG}"
    stripped = strip_media({"input.value": f'{{"url":"{url}"}}', "tags": ("ok", url), "n": 3})
    assert stripped == {
        "input.value": '{"url":"[stripped image/png 8 bytes]"}',
        "tags": ("ok", "[stripped image/png 8 bytes]"),
        "n": 3,
    }


def _past_the_budget() -> str:
    # A third-party request of several MB: an Anthropic image source.
    data = "iVBORw0KGgo" * (3 * MAX_STRIP_CHARS // 11)
    source = {"type": "base64", "media_type": "image/png", "data": data}
    return json.dumps({"messages": [{"role": "user", "content": [{"source": source}]}]})


def _stripped_past_the_budget(text: Any) -> None:
    assert isinstance(text, str)
    assert text.endswith(TRUNCATION_MARKER)
    assert len(text) <= MAX_ATTR_CHARS + len(TRUNCATION_MARKER)
    assert "[stripped unknown unknown bytes]" in text
    assert "iVBORw0KGgo" * 6 not in text


def test_a_third_party_value_past_the_budget_is_stripped_by_the_fallback() -> None:
    raw = _past_the_budget()
    assert len(raw) > MAX_STRIP_CHARS
    out = _export(_span({"input.value": raw, "tags": ("ok", raw)}))
    assert out.attributes is not None
    _stripped_past_the_budget(out.attributes["input.value"])
    tags = out.attributes["tags"]
    assert isinstance(tags, tuple) and tags[0] == "ok"
    _stripped_past_the_budget(tags[1])


def test_nothing_to_strip_returns_none() -> None:
    assert strip_media({"input.value": "hello", "n": 1}) is None


def test_the_exporter_strips_raw_payloads_in_place() -> None:
    raw = (
        '{"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "input_audio", '
        f'"input_audio": {{"data": "{WAV}", "format": "wav"}}}}]}}]}}'
    )
    out = _export(_span({"input.value": raw, "output.value": "plain"}))
    assert out.attributes is not None
    assert out.attributes["input.value"] == raw.replace(
        f'"{WAV}"', '"[stripped audio/wav 12 bytes]"'
    )
    assert out.attributes["output.value"] == "plain"


def test_the_exporter_strips_event_and_link_attributes() -> None:
    url = f"data:image/png;base64,{PNG}"
    event = Event("exception", {"exception.message": f"bad image {url}"}, 5)
    link = Link(SpanContext(trace_id=1, span_id=3, is_remote=False), {"note": url})
    out = _export(_span({"k": "v"}, events=(event,), links=(link,)))
    assert out.events[0].attributes == {
        "exception.message": "bad image [stripped image/png 8 bytes]"
    }
    assert out.events[0].timestamp == 5
    assert out.links[0].attributes == {"note": "[stripped image/png 8 bytes]"}


def test_a_stripped_event_or_link_keeps_its_dropped_attribute_count() -> None:
    url = f"data:image/png;base64,{PNG}"
    attributes = BoundedAttributes(maxlen=1, attributes={"lost": 1, "note": url})
    assert attributes.dropped == 1
    event = Event("exception", attributes, 5)
    link = Link(SpanContext(trace_id=1, span_id=3, is_remote=False), attributes)
    out = _export(_span({"k": "v"}, events=(event,), links=(link,)))
    assert out.events[0].attributes == {"note": "[stripped image/png 8 bytes]"}
    assert out.events[0].dropped_attributes == 1
    assert out.links[0].attributes == {"note": "[stripped image/png 8 bytes]"}
    assert out.links[0].dropped_attributes == 1


def test_a_span_with_no_media_is_exported_as_it_came() -> None:
    span = _span({"input.value": '{"a": "data"}'})
    assert _export(span) is span
