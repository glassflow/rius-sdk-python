"""Two GenAI-shaped usage spellings mapped onto the canonical counts.

``gen_ai.usage.cache_creation.input_tokens`` is the cache-write count's name
before the upstream rename to ``cache_write``. pydantic-ai and ``@ai-sdk/otel``
still emit it. ``gen_ai.usage.details.reasoning_tokens`` is where pydantic-ai
writes OpenAI's reasoning count. Third-party spans passing through the SDK
must leave it canonical, exactly as the sink maps them.

Each is its own rule AFTER the OpenInference rule for the same target, not an
extra source on it: a multi-source rule converts only the first source
present, so an OpenInference count that won't parse would block a good
alternate.
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.normalization import DEFAULT_TABLE
from rius.semconv import (
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
)

CACHE_CREATION = "gen_ai.usage.cache_creation.input_tokens"
DETAILS_REASONING = "gen_ai.usage.details.reasoning_tokens"

# (alternate spelling, OpenInference spelling for the same target, target)
SPELLINGS = [
    (
        CACHE_CREATION,
        "llm.token_count.prompt_details.cache_write",
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    ),
    (
        DETAILS_REASONING,
        "llm.token_count.completion_details.reasoning",
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    ),
]
ALTERNATES = [(alternate, target) for alternate, _, target in SPELLINGS]


def _normalize(attributes: dict[str, Any]) -> dict[str, Any]:
    out = DEFAULT_TABLE.normalize(attributes)
    return dict(attributes) if out is None else out


@pytest.mark.parametrize(("source", "target"), ALTERNATES)
def test_the_spelling_is_mapped_and_deleted(source: str, target: str) -> None:
    assert _normalize({source: 7}) == {target: 7}


@pytest.mark.parametrize(("source", "target"), ALTERNATES)
def test_native_wins_and_the_source_still_goes(source: str, target: str) -> None:
    assert _normalize({source: 7, target: 99}) == {target: 99}


@pytest.mark.parametrize(("source", "target"), ALTERNATES)
def test_zero_is_a_real_count(source: str, target: str) -> None:
    out = _normalize({source: 0})
    assert out == {target: 0}


@pytest.mark.parametrize(("source", "target"), ALTERNATES)
def test_a_string_count_is_parsed_as_a_count(source: str, target: str) -> None:
    out = _normalize({source: "12"})
    assert out == {target: 12}
    assert isinstance(out[target], int)


@pytest.mark.parametrize(("source", "target"), ALTERNATES)
def test_an_unparseable_count_is_dropped(source: str, target: str) -> None:
    assert _normalize({source: "lots"}) == {}


@pytest.mark.parametrize(("alternate", "openinference", "target"), SPELLINGS)
def test_the_openinference_count_wins_when_both_parse(
    alternate: str, openinference: str, target: str
) -> None:
    assert _normalize({openinference: 5, alternate: 7}) == {target: 5}


@pytest.mark.parametrize(("alternate", "openinference", "target"), SPELLINGS)
def test_an_unparseable_openinference_count_does_not_block_the_alternate(
    alternate: str, openinference: str, target: str
) -> None:
    """Why these are separate rules and not extra sources on the OpenInference one."""
    assert _normalize({openinference: "lots", alternate: 7}) == {target: 7}


def test_a_pydantic_ai_span_leaves_the_sdk_canonical() -> None:
    """End to end through ``init()``'s real exporter chain."""
    exporter = InMemorySpanExporter()
    client = init(span_exporter=exporter, set_global=False, service_name="test-svc", instruments=[])
    try:
        with client.get_tracer().start_as_current_span(
            "chat gpt-4o",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.usage.input_tokens": 100,
                CACHE_CREATION: 40,
                DETAILS_REASONING: 0,
            },
        ):
            pass
        client.flush()
        (span,) = exporter.get_finished_spans()
    finally:
        client.shutdown()
    attributes = dict(span.attributes or {})
    assert attributes[GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS] == 40
    assert attributes[GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] == 0
    assert CACHE_CREATION not in attributes
    assert DETAILS_REASONING not in attributes
