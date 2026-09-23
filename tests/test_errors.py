"""``error.type`` spelling shared by every span that records an exception."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from rius._errors import error_type


class _Custom(RuntimeError):
    pass


def test_builtin_exceptions_stay_bare() -> None:
    assert error_type(ValueError("x")) == "ValueError"
    assert error_type(KeyError("k")) == "KeyError"


def test_user_exceptions_are_module_qualified() -> None:
    assert error_type(_Custom()) == f"{_Custom.__module__}.{_Custom.__qualname__}"


def test_nested_classes_keep_their_qualname() -> None:
    class Inner(Exception):
        pass

    assert error_type(Inner()).endswith(".Inner")
    assert Inner.__module__ in error_type(Inner())


# --- record_exception on the manual handles ---
# The scoped helpers set error.type for the caller. The manual handles from
# start_span / start_generation cannot: nothing observes the caller's failure,
# so the handle has to be told. Without it a manual span carried the exception
# event and ERROR status but no error.type, and grouping on error type missed it.


def test_manual_span_record_exception_matches_the_scoped_path(
    exported_spans: InMemorySpanExporter,
) -> None:
    from rius import start_as_current_span, start_span

    obs = start_span("manual")
    try:
        raise ValueError("boom")
    except ValueError as exc:
        obs.record_exception(exc)
    obs.end()

    with pytest.raises(ValueError, match="boom"), start_as_current_span("scoped"):
        raise ValueError("boom")

    spans = {s.name: s for s in exported_spans.get_finished_spans()}
    for span in spans.values():
        assert span.status.status_code == StatusCode.ERROR
        event = next(e for e in span.events if e.name == "exception")
        # One span must never carry two spellings of the same fact.
        assert span.attributes["error.type"] == event.attributes["exception.type"]
    assert spans["manual"].attributes["error.type"] == "ValueError"


def test_manual_generation_record_exception_sets_error_type(
    exported_spans: InMemorySpanExporter,
) -> None:
    from rius import start_generation

    gen = start_generation("chat", model="gpt-4o", provider="openai")
    try:
        raise TimeoutError("upstream took too long")
    except TimeoutError as exc:
        gen.record_exception(exc)
    gen.end()

    span = exported_spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    event = next(e for e in span.events if e.name == "exception")
    assert span.attributes["error.type"] == event.attributes["exception.type"] == "TimeoutError"


def test_manual_record_exception_is_module_qualified_for_user_exceptions(
    exported_spans: InMemorySpanExporter,
) -> None:
    from rius import start_span

    class SinkUnavailable(RuntimeError):
        pass

    obs = start_span("manual")
    try:
        raise SinkUnavailable("details that must not leak into error.type")
    except SinkUnavailable as exc:
        obs.record_exception(exc)
    obs.end()

    attrs = exported_spans.get_finished_spans()[0].attributes
    expected = f"{SinkUnavailable.__module__}.{SinkUnavailable.__qualname__}"
    assert attrs["error.type"] == expected
