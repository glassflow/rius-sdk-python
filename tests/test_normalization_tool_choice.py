"""``tool_choice`` promoted out of the OpenInference request bag.

Context attribution prices the Anthropic tool-use preamble from whether a
request forced a tool call, and reads that from ``rius.request.tool_choice``.
The whole ``llm.invocation_parameters`` bag is content, so with
``capture_content=False`` a ``tool_choice`` left inside it would be dropped
with it. Normalization runs before masking and moves it out.

The cases live in ``tests/fixtures/tool_choice_cases.json``, which the
TypeScript SDK carries byte for byte, and whose expected values are the sink's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.normalization import DEFAULT_TABLE, NormalizingSpanExporter, NormalizingSpanProcessor
from rius.semconv import LLM_INVOCATION_PARAMETERS, RIUS_REQUEST_TOOL_CHOICE

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "tool_choice_cases.json").read_text(encoding="utf-8")
)["cases"]


def _normalize(attributes: dict[str, Any]) -> dict[str, Any]:
    out = DEFAULT_TABLE.normalize(attributes)
    return dict(attributes) if out is None else dict(out)


def _assert_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert set(actual) == set(expected)
    for key, want in expected.items():
        if key == LLM_INVOCATION_PARAMETERS:
            # Parsed: how the leftover bag is spelled is the serializer's business.
            assert json.loads(actual[key]) == want
        else:
            assert actual[key] == want, key


def test_the_key_is_the_vendor_one() -> None:
    assert RIUS_REQUEST_TOOL_CHOICE == "rius.request.tool_choice"


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_shared_case(case: dict[str, Any]) -> None:
    _assert_matches(_normalize(case["input"]), case["expected"])


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_never_writes_a_gen_ai_request_key_for_it(case: dict[str, Any]) -> None:
    """The GenAI conventions define no ``tool_choice``."""
    assert "gen_ai.request.tool_choice" not in _normalize(case["input"])


@pytest.mark.parametrize("case", _CASES, ids=[case["name"] for case in _CASES])
def test_is_idempotent(case: dict[str, Any]) -> None:
    """The processor maps at start and the exporter maps again at export."""
    once = _normalize(case["input"])
    assert _normalize(once) == once


def test_the_value_is_compacted_explicitly_whatever_the_serializer_does() -> None:
    """Separators and escaping are fixed here, not inherited from ``_serde``."""
    out = _normalize({LLM_INVOCATION_PARAMETERS: '{"tool_choice": {"type": "any"}, "user": "u"}'})
    assert out[RIUS_REQUEST_TOOL_CHOICE] == '{"type":"any"}'
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"user": "u"}


def test_on_a_bare_provider_the_key_is_on_the_live_span_and_the_member_leaves_at_export() -> None:
    """Start-time mapping is add-only, so the key rides a pending snapshot the
    way a native one does; the exporter then takes the member out of the bag."""
    collected = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(NormalizingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(collected)))
    span = provider.get_tracer("test").start_span(
        "ChatCompletion",
        attributes={
            LLM_INVOCATION_PARAMETERS: json.dumps(
                {"tool_choice": {"type": "tool", "name": "get_weather"}, "user": "u"}
            )
        },
    )
    live = dict(span.attributes or {})  # type: ignore[attr-defined]
    span.end()
    (exported,) = collected.get_finished_spans()

    assert live[RIUS_REQUEST_TOOL_CHOICE] == '{"type":"tool","name":"get_weather"}'
    attributes = dict(exported.attributes or {})
    assert attributes[RIUS_REQUEST_TOOL_CHOICE] == '{"type":"tool","name":"get_weather"}'
    assert json.loads(attributes[LLM_INVOCATION_PARAMETERS]) == {"user": "u"}


@pytest.mark.parametrize("capture_content", [True, False])
def test_it_survives_capture_content_off_through_init(capture_content: bool) -> None:
    """Through ``init()``'s real chain: normalization runs before masking, so
    the promoted key is out of the bag by the time masking drops the bag."""
    choice = {"type": "tool", "name": "sentinel_tool_7f3a"}
    collected = InMemorySpanExporter()
    client = init(
        span_exporter=collected,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        capture_content=capture_content,
    )
    try:
        client.get_tracer().start_span(
            "ChatCompletion",
            attributes={
                "openinference.span.kind": "LLM",
                LLM_INVOCATION_PARAMETERS: json.dumps(
                    {"tool_choice": choice, "user": "SECRET-USER"}
                ),
            },
        ).end()
        client.flush()
        (span,) = collected.get_finished_spans()
    finally:
        client.shutdown()

    exported = dict(span.attributes or {})
    assert exported[RIUS_REQUEST_TOOL_CHOICE] == '{"type":"tool","name":"sentinel_tool_7f3a"}'
    assert "gen_ai.request.tool_choice" not in exported
    if capture_content:
        assert json.loads(exported[LLM_INVOCATION_PARAMETERS]) == {"user": "SECRET-USER"}
    else:
        assert LLM_INVOCATION_PARAMETERS not in exported
        assert "SECRET" not in json.dumps(exported)


def test_a_lone_surrogate_mode_survives_otlp_encoding() -> None:
    """Kept unpaired, the mode could not be encoded as UTF-8, and the OTLP
    exporter would drop the attribute (it logs and skips it); as U+FFFD it
    reaches the wire."""
    from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans

    collected = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(collected)))
    with provider.get_tracer("t").start_as_current_span("llm") as span:
        span.set_attribute(LLM_INVOCATION_PARAMETERS, '{"tool_choice": "a\\ud800b"}')

    request = encode_spans(collected.get_finished_spans())
    wire = {
        kv.key: kv.value.string_value
        for kv in request.resource_spans[0].scope_spans[0].spans[0].attributes
    }
    assert wire[RIUS_REQUEST_TOOL_CHOICE] == "a�b"
