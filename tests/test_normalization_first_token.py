"""Normalization of the OpenInference first-token signal.

The source is a span EVENT, not an attribute, so it does not go through the
rule table: ``NormalizingSpanExporter`` rewrites it directly. See the
``normalization`` module for the shape and why the exporter is the component
that can do it.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.normalization import (
    OPENINFERENCE_FIRST_TOKEN_EVENT,
    NormalizingSpanExporter,
)
from rius.semconv import (
    GEN_AI_FIRST_TOKEN_EVENT,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
)

# A second apart, in nanoseconds, so the derived value is unambiguous in
# seconds (1.5) and would be 1_500_000_000 or 1500 under the wrong unit.
_START_NS = 1_700_000_000_000_000_000
_FIRST_TOKEN_NS = _START_NS + 1_500_000_000


def _raw_span(
    *,
    attributes: dict[str, Any] | None = None,
    events: list[tuple[str, int]] | None = None,
) -> ReadableSpan:
    """A finished span from a bare OTel provider, never touched by normalization.

    Not through ``init()``: its exporter chain already normalizes, so a span
    collected there would arrive carrying ``gen_ai.first_token`` and every
    assertion below would pass without the export step doing anything. This
    input carries only what an instrumentor emits.
    """
    collected = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collected))
    try:
        span = provider.get_tracer("test").start_span(
            "chat", attributes=attributes or {}, start_time=_START_NS
        )
        for name, timestamp in events or []:
            span.add_event(name, timestamp=timestamp)
        span.end()
        (finished,) = collected.get_finished_spans()
    finally:
        provider.shutdown()
    # The input is exactly what was asked for: nothing derived, nothing renamed.
    assert _events(finished) == list(events or [])
    assert dict(finished.attributes or {}) == dict(attributes or {})
    return finished


def _export(span: ReadableSpan) -> ReadableSpan:
    inner = InMemorySpanExporter()
    NormalizingSpanExporter(inner).export([span])
    return inner.get_finished_spans()[0]


def _events(span: ReadableSpan) -> list[tuple[str, int | None]]:
    return [(event.name, event.timestamp) for event in span.events]


def test_an_openinference_streaming_span_becomes_indistinguishable_from_a_native_one() -> None:
    raw = _raw_span(
        attributes={"gen_ai.request.model": "gpt-4o"},
        events=[(OPENINFERENCE_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS)],
    )
    exported = _export(raw)
    assert _events(exported) == [(GEN_AI_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS)]
    assert dict(exported.attributes or {}) == {
        **dict(raw.attributes or {}),
        GEN_AI_REQUEST_STREAM: True,
        GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK: 1.5,
    }


def test_the_derived_duration_is_seconds_since_span_start() -> None:
    """Pinned unit: the native path emits seconds (float), so this must too."""
    raw = _raw_span(events=[(OPENINFERENCE_FIRST_TOKEN_EVENT, _START_NS + 250_000_000)])
    exported = _export(raw)
    value = (exported.attributes or {})[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK]
    assert isinstance(value, float)
    assert value == 0.25


def test_a_span_already_carrying_the_canonical_form_is_untouched() -> None:
    """Native wins, exactly as it does for attributes."""
    raw = _raw_span(
        attributes={
            GEN_AI_REQUEST_STREAM: True,
            GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK: 0.125,
        },
        events=[(GEN_AI_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS)],
    )
    exported = _export(raw)
    assert exported is raw
    assert _events(exported) == [(GEN_AI_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS)]
    assert (exported.attributes or {})[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] == 0.125


def test_native_attributes_are_not_overwritten_by_the_derived_ones() -> None:
    raw = _raw_span(
        attributes={GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK: 0.125},
        events=[(OPENINFERENCE_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS)],
    )
    exported = _export(raw)
    assert (exported.attributes or {})[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] == 0.125
    assert (exported.attributes or {})[GEN_AI_REQUEST_STREAM] is True


def test_a_duplicate_canonical_event_is_not_added_but_the_source_still_goes() -> None:
    raw = _raw_span(
        events=[
            (GEN_AI_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS),
            (OPENINFERENCE_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS + 1),
        ],
    )
    exported = _export(raw)
    assert _events(exported) == [(GEN_AI_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS)]


def test_a_non_streaming_span_gains_nothing() -> None:
    raw = _raw_span(attributes={"gen_ai.request.model": "gpt-4o"})
    exported = _export(raw)
    assert exported is raw
    assert GEN_AI_REQUEST_STREAM not in (exported.attributes or {})
    assert GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK not in (exported.attributes or {})


def test_an_unrelated_event_is_left_alone() -> None:
    raw = _raw_span(events=[("exception", _FIRST_TOKEN_NS)])
    exported = _export(raw)
    assert exported is raw
    assert _events(exported) == [("exception", _FIRST_TOKEN_NS)]


def test_other_events_are_preserved_in_order() -> None:
    raw = _raw_span(
        events=[
            ("before", _START_NS + 1),
            (OPENINFERENCE_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS),
            ("after", _FIRST_TOKEN_NS + 1),
        ],
    )
    exported = _export(raw)
    assert _events(exported) == [
        ("before", _START_NS + 1),
        (GEN_AI_FIRST_TOKEN_EVENT, _FIRST_TOKEN_NS),
        ("after", _FIRST_TOKEN_NS + 1),
    ]


def test_nothing_is_mutated_in_place() -> None:
    """A ReadableSpan shares its event list with every processor on the provider."""
    other = InMemorySpanExporter()
    exporter = InMemorySpanExporter()
    client = init(span_exporter=exporter, set_global=False, service_name="test-svc", instruments=[])
    try:
        client._provider.add_span_processor(SimpleSpanProcessor(other))
        with client.get_tracer().start_as_current_span("chat") as span:
            span.add_event(OPENINFERENCE_FIRST_TOKEN_EVENT)
        client.flush()
        (seen,) = other.get_finished_spans()
        (normalized,) = exporter.get_finished_spans()
    finally:
        client.shutdown()
    assert [event.name for event in seen.events] == [OPENINFERENCE_FIRST_TOKEN_EVENT]
    assert GEN_AI_REQUEST_STREAM not in (seen.attributes or {})
    assert [event.name for event in normalized.events] == [GEN_AI_FIRST_TOKEN_EVENT]
    assert (normalized.attributes or {})[GEN_AI_REQUEST_STREAM] is True


def test_it_is_wired_into_init() -> None:
    exporter = InMemorySpanExporter()
    client = init(span_exporter=exporter, set_global=False, service_name="test-svc", instruments=[])
    try:
        with client.get_tracer().start_as_current_span("chat") as span:
            span.add_event(OPENINFERENCE_FIRST_TOKEN_EVENT)
        client.flush()
        (finished,) = exporter.get_finished_spans()
    finally:
        client.shutdown()
    assert [event.name for event in finished.events] == [GEN_AI_FIRST_TOKEN_EVENT]
    assert (finished.attributes or {})[GEN_AI_REQUEST_STREAM] is True
    assert (finished.attributes or {})[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] >= 0.0


# --- the upstream guard -----------------------------------------------------
#
# The whole mapping hangs on one free-text English string that upstream has no
# constant for and can reword in a patch release. These assert against the
# INSTALLED instrumentor, not against our own constant — comparing our literal
# to our literal would prove nothing. Both openinference instrumentors are in
# the `dev` dependency group, so this runs on every ordinary `uv run pytest`
# and turns a silent data gap into a build error on the next `uv lock`.


def test_the_openai_instrumentor_still_emits_the_event_name_we_map() -> None:
    openai_stream = pytest.importorskip("openinference.instrumentation.openai._stream")
    source = inspect.getsource(openai_stream._Stream._process_chunk)
    assert f'add_event("{OPENINFERENCE_FIRST_TOKEN_EVENT}")' in source, (
        "openinference-instrumentation-openai no longer emits "
        f"{OPENINFERENCE_FIRST_TOKEN_EVENT!r} from _Stream._process_chunk. The event was "
        "renamed or moved upstream: read the new _process_chunk, update "
        "OPENINFERENCE_FIRST_TOKEN_EVENT in rius/normalization.py, and check whether the "
        "SHAPE changed too — we rely on the event carrying no attributes and no explicit "
        "timestamp, so the SDK stamps it at chunk arrival."
    )


def test_the_anthropic_instrumentor_still_emits_no_first_token_event() -> None:
    """Documented gap: Anthropic ships the add_event helper but never calls it.

    If this fails, that is GOOD news — the signal now exists, and an
    auto-instrumented Anthropic streaming span can be made indistinguishable
    from a native one too. Check the name it uses and widen the mapping.
    """
    anthropic_stream = pytest.importorskip("openinference.instrumentation.anthropic._stream")
    source = inspect.getsource(anthropic_stream)
    assert "add_event" not in source, (
        "openinference-instrumentation-anthropic now calls add_event from its stream "
        "wrapper. If it is a first-token marker, add its name alongside "
        "OPENINFERENCE_FIRST_TOKEN_EVENT in rius/normalization.py."
    )
