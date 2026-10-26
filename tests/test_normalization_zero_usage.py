"""Zero usage breakdowns leave every span at export.

The GenAI conventions recommend the usage breakdowns (the cache and reasoning
subsets and the per-modality counts) only when available and non-zero, and
still report the input and output totals at 0. A zero breakdown goes, whether
the producer wrote it under a canonical key or the table mapped it there, and
a mapped one's source goes with it.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

from rius.normalization import USAGE_BREAKDOWN_SUFFIXES, NormalizingSpanExporter

BREAKDOWNS = [
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_write.input_tokens",
    "gen_ai.usage.reasoning.output_tokens",
    "gen_ai.usage.text.input_tokens",
    "gen_ai.usage.text.output_tokens",
    "gen_ai.usage.text.cache_read.input_tokens",
    "gen_ai.usage.image.input_tokens",
    "gen_ai.usage.image.output_tokens",
    "gen_ai.usage.image.cache_read.input_tokens",
    "gen_ai.usage.audio.input_tokens",
    "gen_ai.usage.audio.output_tokens",
    "gen_ai.usage.audio.cache_read.input_tokens",
]


def _exported(attributes: dict[str, Any]) -> dict[str, Any]:
    context = SpanContext(trace_id=1, span_id=2, is_remote=False)
    sink = InMemorySpanExporter()
    NormalizingSpanExporter(sink).export(
        [ReadableSpan(name="s", context=context, attributes=attributes)]
    )
    [out] = sink.get_finished_spans()
    return dict(out.attributes or {})


def test_the_breakdowns_are_the_twelve_canonical_ones() -> None:
    assert sorted("gen_ai.usage." + suffix for suffix in USAGE_BREAKDOWN_SUFFIXES) == sorted(
        BREAKDOWNS
    )


@pytest.mark.parametrize("key", BREAKDOWNS)
def test_a_native_zero_breakdown_is_dropped(key: str) -> None:
    assert _exported({key: 0}) == {}


@pytest.mark.parametrize("key", BREAKDOWNS)
def test_a_native_breakdown_that_is_not_zero_stays(key: str) -> None:
    assert _exported({key: 3}) == {key: 3}


def test_zero_totals_stay() -> None:
    totals = {"gen_ai.usage.input_tokens": 0, "gen_ai.usage.output_tokens": 0}
    assert _exported(totals) == totals


def test_a_mapped_zero_breakdown_goes_and_its_source_with_it() -> None:
    assert _exported(
        {
            "llm.token_count.prompt": 0,
            "llm.token_count.completion": 0,
            "llm.token_count.prompt_details.cache_read": 0,
            "llm.token_count.prompt_details.cache_write": 0,
            "llm.token_count.completion_details.reasoning": 0,
            "llm.token_count.prompt_details.audio": 0,
            "llm.token_count.completion_details.audio": 0,
        }
    ) == {"gen_ai.usage.input_tokens": 0, "gen_ai.usage.output_tokens": 0}


def test_a_parsed_string_zero_is_dropped() -> None:
    assert _exported({"llm.token_count.prompt_details.cache_read": "0"}) == {}


@pytest.mark.parametrize("value", ["0", 0.5, False])
def test_only_the_number_zero_is_dropped(value: Any) -> None:
    key = "gen_ai.usage.cache_read.input_tokens"
    assert _exported({key: value}) == {key: value}


@pytest.mark.parametrize("value", [0.0, -0.0])
def test_a_zero_float_is_dropped(value: float) -> None:
    assert _exported({"gen_ai.usage.cache_read.input_tokens": value}) == {}


def test_nan_is_no_zero_and_stays() -> None:
    key = "gen_ai.usage.cache_read.input_tokens"
    exported = _exported({key: math.nan})
    assert list(exported) == [key]
    assert math.isnan(exported[key])


def test_a_zero_under_a_key_that_is_no_breakdown_stays() -> None:
    key = "gen_ai.usage.details.audio_tokens"
    assert _exported({key: 0}) == {key: 0}


def test_a_zero_breakdown_of_a_runs_usage_moved_aside_is_dropped() -> None:
    assert _exported(
        {
            "rius.aggregated_usage.input_tokens": 0,
            "rius.aggregated_usage.cache_read.input_tokens": 0,
            "rius.aggregated_usage.reasoning.output_tokens": 4,
        }
    ) == {
        "rius.aggregated_usage.input_tokens": 0,
        "rius.aggregated_usage.reasoning.output_tokens": 4,
    }
