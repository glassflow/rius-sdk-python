"""``error.type`` spelling shared by every span that records an exception."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

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


# --- one status description and one cancellation policy across every helper ---
# Every error path writes the bare message as the status description (the
# scoped helpers used to leave OTel's "ValueError: boom"), and a BaseException
# that is not an Exception (cancellation, KeyboardInterrupt, SystemExit) is not
# an error on any of them: no exception event, UNSET status, no error.type.


def _scoped_span(exc: BaseException) -> None:
    from rius import start_as_current_span

    with start_as_current_span("scoped"):
        raise exc


def _scoped_generation(exc: BaseException) -> None:
    from rius import start_as_current_generation

    with start_as_current_generation("scoped", model="gpt-4o", provider="openai"):
        raise exc


def _observe_sync(exc: BaseException) -> None:
    from rius import observe

    @observe(name="scoped")
    def fn() -> None:
        raise exc

    fn()


def _observe_async(exc: BaseException) -> None:
    from rius import observe

    @observe(name="scoped")
    async def fn() -> None:
        raise exc

    coro = fn()
    try:
        coro.send(None)
    finally:
        coro.close()


def _observe_generator(exc: BaseException) -> None:
    from rius import observe

    @observe(name="scoped")
    def fn() -> Iterator[int]:
        yield 1
        raise exc

    list(fn())


def _manual_span(exc: BaseException) -> None:
    from rius import start_span

    obs = start_span("scoped")
    obs.record_exception(exc)
    obs.end()
    raise exc


def _manual_generation(exc: BaseException) -> None:
    from rius import start_generation

    gen = start_generation("scoped", model="gpt-4o", provider="openai")
    gen.record_exception(exc)
    gen.end()
    raise exc


_ERROR_PATHS = [
    _scoped_span,
    _scoped_generation,
    _observe_sync,
    _observe_async,
    _observe_generator,
    _manual_span,
    _manual_generation,
]
_HELPERS = _ERROR_PATHS[:5]


@pytest.mark.parametrize("path", _ERROR_PATHS, ids=lambda f: f.__name__)
def test_every_error_path_writes_the_bare_message_as_the_status_description(
    exported_spans: InMemorySpanExporter, path: Callable[[BaseException], None]
) -> None:
    with pytest.raises(ValueError, match="boom"):
        path(ValueError("boom"))

    (span,) = exported_spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"
    assert span.attributes["error.type"] == "ValueError"


@pytest.mark.parametrize("path", _HELPERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(
    "exc", [asyncio.CancelledError(), KeyboardInterrupt()], ids=["cancelled", "interrupt"]
)
def test_cancellation_is_not_an_error_on_any_helper(
    exported_spans: InMemorySpanExporter,
    path: Callable[[BaseException], None],
    exc: BaseException,
) -> None:
    with pytest.raises(type(exc)):
        path(exc)

    (span,) = exported_spans.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    assert not [e for e in span.events if e.name == "exception"]
    assert "error.type" not in (span.attributes or {})
