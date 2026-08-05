"""The `@observe` decorator: trace the developer's own functions.

Wraps sync, async, generator, and async-generator functions (and methods),
recording timing, inputs/outputs, and exceptions as an OpenTelemetry span.
Spans nest automatically via OTel context propagation and are exported through
whatever provider `init()` configured (a no-op if the SDK was never initialized).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, overload

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from . import __version__
from ._serde import serialize
from .semconv import INPUT_VALUE, OUTPUT_VALUE, TRACER_NAME, SpanKind, set_span_kind

F = TypeVar("F", bound=Callable[..., Any])


def _serialize_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return serialize({"args": args, "kwargs": kwargs})


def _record_exception(span: trace.Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


@overload
def observe(func: F) -> F: ...


@overload
def observe(
    *,
    name: str | None = ...,
    capture_input: bool = ...,
    capture_output: bool = ...,
    kind: SpanKind = ...,
) -> Callable[[F], F]: ...


def observe(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
    kind: SpanKind = SpanKind.CHAIN,
) -> Any:
    """Decorate a function so each call is traced as a span.

    Usable bare (``@observe``) or parameterized (``@observe(name=..., ...)``).
    Supports sync functions, ``async def`` functions, generators, and async
    generators; for generators the span covers the whole iteration and the
    tracing context is attached only around each step. Exceptions are
    recorded with ERROR status and always re-raised.

    Args:
        func: The decorated function (filled in by bare ``@observe`` usage).
        name: Span name; defaults to the function's ``__qualname__``.
        capture_input: Record call arguments as JSON in ``input.value``.
        capture_output: Record the return value as JSON in ``output.value``.
        kind: Span taxonomy (``openinference.span.kind``); default ``CHAIN``.

    Returns:
        The wrapped function (or a decorator, when used parameterized).
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or fn.__qualname__

        def _set_input(span: trace.Span, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if capture_input:
                span.set_attribute(INPUT_VALUE, _serialize_inputs(args, kwargs))

        def _set_output(span: trace.Span, result: Any) -> None:
            if capture_output:
                span.set_attribute(OUTPUT_VALUE, serialize(result))

        if inspect.isasyncgenfunction(fn):
            # Generators interleave with caller code between yields, so the span
            # is only attached around each step — never leaked into the caller.

            @functools.wraps(fn)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = trace.get_tracer(TRACER_NAME, __version__)
                span = tracer.start_span(span_name)
                set_span_kind(span, kind)
                _set_input(span, args, kwargs)
                agen = fn(*args, **kwargs)
                try:
                    while True:
                        token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            item = await agen.__anext__()
                        except StopAsyncIteration:
                            break
                        finally:
                            otel_context.detach(token)
                        yield item
                except Exception as exc:
                    _record_exception(span, exc)
                    raise
                finally:
                    span.end()

            return async_gen_wrapper

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = trace.get_tracer(TRACER_NAME, __version__)
                with tracer.start_as_current_span(
                    span_name, record_exception=False, set_status_on_exception=False
                ) as span:
                    set_span_kind(span, kind)
                    _set_input(span, args, kwargs)
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        _record_exception(span, exc)
                        raise
                    _set_output(span, result)
                    return result

            return async_wrapper

        if inspect.isgeneratorfunction(fn):
            # See the async-generator note: attach only around each step.

            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = trace.get_tracer(TRACER_NAME, __version__)
                span = tracer.start_span(span_name)
                set_span_kind(span, kind)
                _set_input(span, args, kwargs)
                gen = fn(*args, **kwargs)
                try:
                    while True:
                        token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            item = next(gen)
                        except StopIteration:
                            break
                        finally:
                            otel_context.detach(token)
                        yield item
                except Exception as exc:
                    _record_exception(span, exc)
                    raise
                finally:
                    span.end()

            return gen_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(TRACER_NAME, __version__)
            with tracer.start_as_current_span(
                span_name, record_exception=False, set_status_on_exception=False
            ) as span:
                set_span_kind(span, kind)
                _set_input(span, args, kwargs)
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    _record_exception(span, exc)
                    raise
                _set_output(span, result)
                return result

        return sync_wrapper

    if func is not None:
        return decorate(func)
    return decorate
