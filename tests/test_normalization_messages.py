"""OpenInference's flattened messages reassembled into gen_ai.input/output.messages.

OpenInference writes one attribute per message field
(``llm.input_messages.0.message.role``, ``...tool_calls.0.tool_call.id``, ...).
The SDK's own generations write one JSON array in the GenAI role/parts shape,
and so does the sink when it normalizes foreign traffic. The SDK reassembles
at export so the wire is uniform, and it must write exactly the bytes the sink
would for the same input: a span normalized by either side has to be
indistinguishable downstream.

``tests/fixtures/openinference_messages.json`` is the shared contract. It is
consumed unchanged by the TypeScript SDK's port, and every case marked
``sink_identical`` is checked against the sink's ``reassembleMessages``
(all of them, today). The Anthropic cases are shapes read from the
openinference-instrumentation-anthropic code, Python and JS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius._serde import MAX_ATTR_CHARS, TRUNCATION_MARKER
from rius.normalization import NormalizingSpanExporter, reassemble_openinference_messages
from rius.semconv import GEN_AI_INPUT_MESSAGES, GEN_AI_OUTPUT_MESSAGES

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "openinference_messages.json").read_text()
)
_CASES = _FIXTURE["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_fixture(case: dict[str, Any]) -> None:
    attributes = dict(case["attributes"])
    rebuilt = reassemble_openinference_messages(attributes)
    out = attributes if rebuilt is None else rebuilt
    assert out == case["expected"]
    # Byte-identical, not merely equal once parsed: compare the raw strings.
    for key in (GEN_AI_INPUT_MESSAGES, GEN_AI_OUTPUT_MESSAGES):
        if key in case["expected"]:
            assert out[key] == case["expected"][key]
    # The input mapping is never mutated; the exporter works on copies.
    assert attributes == case["attributes"]


def test_fixture_covers_the_ticket_cases() -> None:
    names = " ".join(c["name"] for c in _CASES)
    for needle in ("user", "system", "tool call", "response", "multimodal"):
        assert needle in names


def test_nothing_to_reassemble_returns_none() -> None:
    assert reassemble_openinference_messages(None) is None
    assert reassemble_openinference_messages({}) is None
    assert reassemble_openinference_messages({"gen_ai.request.model": "gpt-4o"}) is None


def test_both_families_native_returns_none() -> None:
    attributes = {
        GEN_AI_INPUT_MESSAGES: "[]",
        GEN_AI_OUTPUT_MESSAGES: "[]",
        "llm.input_messages.0.message.role": "user",
        "llm.output_messages.0.message.role": "assistant",
    }
    assert reassemble_openinference_messages(attributes) is None


def test_a_non_string_value_leaves_the_family_untouched() -> None:
    """OpenInference writes every message field as a string. A family holding
    anything else is not one we can vouch for: it is left as it came, and it
    is content by prefix already, so nothing escapes under masking."""
    attributes: dict[str, Any] = {
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": 42,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "yo",
    }
    out = reassemble_openinference_messages(attributes)
    assert out is not None
    assert GEN_AI_INPUT_MESSAGES not in out
    assert out["llm.input_messages.0.message.content"] == 42
    assert out["llm.input_messages.0.message.role"] == "user"
    assert out[GEN_AI_OUTPUT_MESSAGES] == (
        '[{"role":"assistant","parts":[{"type":"text","content":"yo"}]}]'
    )


def test_a_non_string_contents_value_leaves_the_family_untouched() -> None:
    attributes: dict[str, Any] = {
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.contents.0.message_content.type": "text",
        "llm.input_messages.0.message.contents.0.message_content.text": ["a"],
    }
    assert reassemble_openinference_messages(attributes) is None


def test_the_attribute_cap_applies_like_native_messages() -> None:
    attributes = {
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "x" * (MAX_ATTR_CHARS * 2),
    }
    out = reassemble_openinference_messages(attributes)
    assert out is not None
    value = out[GEN_AI_INPUT_MESSAGES]
    assert value.endswith(TRUNCATION_MARKER)
    assert len(value) == MAX_ATTR_CHARS + len(TRUNCATION_MARKER)
    assert value.startswith('[{"role":"user","parts":[{"type":"text","content":"xxx')
    assert not [k for k in out if k.startswith("llm.input_messages.")]


# --- through the exporter ---------------------------------------------------


def _raw_span(attributes: dict[str, Any]) -> ReadableSpan:
    """A finished span from a bare provider, never touched by normalization."""
    collected = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collected))
    provider.get_tracer("test").start_span("ChatCompletion", attributes=attributes).end()
    (finished,) = collected.get_finished_spans()
    return finished


def _openai_chat() -> dict[str, Any]:
    return {
        "openinference.span.kind": "LLM",
        "llm.provider": "openai",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "hi",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "yo",
    }


def test_the_exporter_reassembles_and_deletes_the_flattened_keys() -> None:
    inner = InMemorySpanExporter()
    NormalizingSpanExporter(inner).export([_raw_span(_openai_chat())])
    exported = dict(inner.get_finished_spans()[0].attributes or {})
    assert exported[GEN_AI_INPUT_MESSAGES] == (
        '[{"role":"user","parts":[{"type":"text","content":"hi"}]}]'
    )
    assert exported[GEN_AI_OUTPUT_MESSAGES] == (
        '[{"role":"assistant","parts":[{"type":"text","content":"yo"}]}]'
    )
    assert not [k for k in exported if k.startswith(("llm.input_messages", "llm.output_messages"))]
    # The table still ran alongside.
    assert exported["gen_ai.provider.name"] == "openai"


@pytest.mark.parametrize("capture_content", [True, False])
def test_reassembled_messages_are_content(capture_content: bool) -> None:
    """Through ``init()``'s real chain: normalization runs before masking, so
    the reassembled keys are what masking sees, and under
    ``capture_content=False`` they are stripped exactly like native ones."""
    collected = InMemorySpanExporter()
    client = init(
        span_exporter=collected,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        capture_content=capture_content,
    )
    try:
        client.get_tracer().start_span("ChatCompletion", attributes=_openai_chat()).end()
        client.flush()
        (span,) = collected.get_finished_spans()
    finally:
        client.shutdown()
    exported = dict(span.attributes or {})
    assert not [k for k in exported if k.startswith(("llm.input_messages", "llm.output_messages"))]
    if capture_content:
        assert json.loads(exported[GEN_AI_INPUT_MESSAGES]) == [
            {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
        ]
    else:
        assert GEN_AI_INPUT_MESSAGES not in exported
        assert GEN_AI_OUTPUT_MESSAGES not in exported
        assert "hi" not in json.dumps(exported)
