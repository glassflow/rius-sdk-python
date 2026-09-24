"""``error.type`` derived from the exception event on auto-instrumented spans.

A third-party instrumentor that fails a span records an ``exception`` event
and an ERROR status but sets no ``error.type``, which the GenAI conventions
make Conditionally Required on a failed span. The source is an EVENT, so this
does not go through the rule table: ``NormalizingSpanExporter`` derives it,
the same way it handles the first-token event.
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from rius import init
from rius.normalization import NormalizingSpanExporter, error_type_from_exception_event
from rius.semconv import ERROR_TYPE, EXCEPTION_EVENT, EXCEPTION_TYPE


def _raw_span(
    *,
    status: StatusCode = StatusCode.UNSET,
    attributes: dict[str, Any] | None = None,
    events: list[tuple[str, dict[str, Any]]] | None = None,
) -> ReadableSpan:
    """A finished span from a bare OTel provider, never touched by normalization.

    Not through ``init()``: its exporter chain already normalizes, so a span
    collected there would arrive with ``error.type`` set and every assertion
    below would pass without the derivation doing anything.
    """
    collected = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collected))
    span = provider.get_tracer("test").start_span("chat", attributes=attributes or {})
    for name, event_attributes in events or []:
        span.add_event(name, attributes=event_attributes)
    span.set_status(Status(status))
    span.end()
    (finished,) = collected.get_finished_spans()
    assert ERROR_TYPE not in (finished.attributes or {}) or ERROR_TYPE in (attributes or {})
    return finished


def _export(span: ReadableSpan) -> ReadableSpan:
    inner = InMemorySpanExporter()
    NormalizingSpanExporter(inner).export([span])
    return inner.get_finished_spans()[0]


def _exception(type_: Any) -> tuple[str, dict[str, Any]]:
    return EXCEPTION_EVENT, {EXCEPTION_TYPE: type_, "exception.message": "boom"}


def test_the_semconv_spellings_are_the_otel_ones() -> None:
    # record_exception() writes exactly these; a typo here silently disables
    # the whole derivation.
    assert EXCEPTION_EVENT == "exception"
    assert EXCEPTION_TYPE == "exception.type"


def test_a_failed_span_takes_error_type_from_its_exception_event_verbatim() -> None:
    # Spelled exactly as the event spells it, module path and all, so one span
    # never carries two spellings of the same failure.
    raw = _raw_span(status=StatusCode.ERROR, events=[_exception("openai.RateLimitError")])
    exported = _export(raw)
    assert dict(exported.attributes or {}) == {
        **dict(raw.attributes or {}),
        ERROR_TYPE: "openai.RateLimitError",
    }
    # The event itself is untouched: masking owns what leaves it.
    assert [(e.name, dict(e.attributes or {})) for e in exported.events] == [
        (e.name, dict(e.attributes or {})) for e in raw.events
    ]


def test_an_error_with_no_exception_event_gets_nothing() -> None:
    # Pinned on purpose: this is the case most likely to be "helpfully" filled
    # in later. An ERROR status with no exception is not classifiable, and a
    # guessed value is worse than an absent one.
    raw = _raw_span(status=StatusCode.ERROR, events=[("some other event", {"k": "v"})])
    exported = _export(raw)
    assert ERROR_TYPE not in (exported.attributes or {})
    # Nothing changed, so the span is passed through, not copied.
    assert exported is raw


@pytest.mark.parametrize("status", [StatusCode.UNSET, StatusCode.OK])
def test_a_handled_exception_on_a_span_that_did_not_fail_is_left_alone(
    status: StatusCode,
) -> None:
    # A recorded-and-handled exception is not a failed span.
    raw = _raw_span(status=status, events=[_exception("ValueError")])
    exported = _export(raw)
    assert ERROR_TYPE not in (exported.attributes or {})
    assert exported is raw


def test_a_native_error_type_is_never_overwritten() -> None:
    raw = _raw_span(
        status=StatusCode.ERROR,
        attributes={ERROR_TYPE: "tool_error"},
        events=[_exception("ValueError")],
    )
    exported = _export(raw)
    assert (exported.attributes or {})[ERROR_TYPE] == "tool_error"
    # Held by the derivation itself, not only by the exporter's setdefault:
    # the function is the contract the sink and the TypeScript SDK mirror.
    assert error_type_from_exception_event(raw) == {}


def test_the_first_usable_exception_event_names_the_failure() -> None:
    # The same choice as the TypeScript SDK and the sink, so the three agree
    # on a span that recorded more than one exception.
    raw = _raw_span(
        status=StatusCode.ERROR,
        events=[_exception(""), _exception("TimeoutError"), _exception("ValueError")],
    )
    assert error_type_from_exception_event(raw) == {ERROR_TYPE: "TimeoutError"}


@pytest.mark.parametrize("bad", ["", 42])
def test_an_exception_event_without_a_usable_type_gives_nothing(bad: Any) -> None:
    raw = _raw_span(status=StatusCode.ERROR, events=[_exception(bad)])
    assert error_type_from_exception_event(raw) == {}


def test_error_type_and_table_rules_combine_on_one_span() -> None:
    # The derived key joins whatever the rule table produced; neither pass
    # discards the other's work.
    raw = _raw_span(
        status=StatusCode.ERROR,
        attributes={"openinference.span.kind": "LLM", "llm.provider": "openai"},
        events=[_exception("openai.APIError")],
    )
    attributes = dict(_export(raw).attributes or {})
    assert attributes[ERROR_TYPE] == "openai.APIError"
    assert attributes["gen_ai.provider.name"] == "openai"
    assert "llm.provider" not in attributes


@pytest.mark.parametrize("capture_content", [True, False])
def test_an_auto_instrumented_failure_through_init_carries_error_type(
    capture_content: bool,
) -> None:
    """End to end through ``init()``'s real exporter chain.

    A raw tracer span that raises is what a third-party instrumentor produces:
    the OTel SDK records the exception event and the ERROR status, and nobody
    sets ``error.type``. ``exception.type`` survives masking, so the derived
    key must survive with content capture off too — it is the class, never the
    message.
    """
    collected = InMemorySpanExporter()
    client = init(
        span_exporter=collected,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        capture_content=capture_content,
    )
    try:
        with (
            pytest.raises(KeyError),
            client.get_tracer().start_as_current_span(
                "tool", attributes={"openinference.span.kind": "TOOL"}
            ),
        ):
            raise KeyError("secret-key-name")
        client.flush()
        (span,) = collected.get_finished_spans()
    finally:
        client.shutdown()
    assert span.status.status_code is StatusCode.ERROR
    assert (span.attributes or {})[ERROR_TYPE] == "KeyError"
