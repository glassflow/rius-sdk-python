"""The `@observe` decorator: trace the developer's own functions.

Wraps sync, async, generator, and async-generator functions (and methods),
recording timing, inputs/outputs, and exceptions as an OpenTelemetry span.
Spans nest automatically via OTel context propagation and are exported through
whatever provider `init()` configured (a no-op if the SDK was never initialized).
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, overload

from opentelemetry import context as otel_context
from opentelemetry import trace

from ._agent import resolve_agent_name
from ._errors import record_error
from ._serde import serialize
from ._tracer import sdk_tracer
from .semconv import (
    INPUT_VALUE,
    OUTPUT_VALUE,
    SpanKind,
    kind_attributes,
    otel_span_kind,
)

F = TypeVar("F", bound=Callable[..., Any])


def _serialize_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return serialize({"args": args, "kwargs": kwargs})


def _resolve_tool_name(
    tool_name: str | None, name: str | None, qualname: str, kind: SpanKind
) -> str:
    """The tool's identity: explicit name, else the span name, else the qualname.

    The middle rung is why this is a function rather than an ``or`` chain. Every
    comparable decorator in the ecosystem resolves tool identity as "explicit
    name, else function name", and a custom name on a tool-kind decorator names
    the tool rather than merely labelling the span: OpenInference's ``@tool``,
    LangSmith's ``@traceable(run_type="tool")`` and Traceloop's ``@tool`` all
    feed one value to both. Dropping the middle rung would silently rename the
    tool of every existing caller who passed ``name=``, and ``gen_ai.tool.name``
    is a Required metric dimension, so the rename splits their histogram series
    with no error anywhere.

    It does leave ``name`` ambiguous: a caller labelling one span for a
    waterfall also renames the tool. That is what the warning is for. Passing
    ``tool_name`` explicitly silences it and is the shape this will converge on.
    """
    if tool_name is not None:
        return tool_name
    if kind is SpanKind.TOOL and name is not None:
        warnings.warn(
            f"observe(name={name!r}, kind=TOOL) is naming the tool as well as the span; "
            f"gen_ai.tool.name will be {name!r}. Pass tool_name= to set them separately. "
            "A future major release will stop deriving the tool name from the span name.",
            DeprecationWarning,
            stacklevel=3,
        )
        return name
    return qualname


@overload
def observe(func: F) -> F: ...


@overload
def observe(
    *,
    name: str | None = ...,
    capture_input: bool = ...,
    capture_output: bool = ...,
    kind: SpanKind = ...,
    tool_name: str | None = ...,
    data_source_id: str | None = ...,
    top_k: int | None = ...,
    agent_name: str | None = ...,
    agent_id: str | None = ...,
) -> Callable[[F], F]: ...


def observe(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
    kind: SpanKind = SpanKind.CHAIN,
    tool_name: str | None = None,
    data_source_id: str | None = None,
    top_k: int | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
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
        tool_name: The tool's own name (``gen_ai.tool.name``), used only when
            ``kind`` is ``TOOL``. Unset, it falls back to ``name`` and then to
            the function's ``__qualname__``. Separate from ``name`` because the
            span name may carry an operation prefix and the tool name must not;
            the fallback through ``name`` exists so an existing caller is not
            silently renamed, and warns so the two can be separated.
        data_source_id: The index, collection or knowledge base a ``RETRIEVER``
            span searched (``gen_ai.data_source.id``). Ignored for other kinds.
        top_k: How many documents the retrieval asked for
            (``gen_ai.retrieval.top_k``). Ignored for other kinds. What came
            back is not a decorator argument: it is only known once the
            function returns, so record it from the return value instead.
        agent_name: The agent an ``AGENT`` span invokes
            (``gen_ai.agent.name``). Unset, it falls back to the agent name
            ``init()`` was given, and is never taken from the function or the
            span name: a function name is not an agent's identity. Ignored for
            other kinds.
        agent_id: The agent's stable identifier (``gen_ai.agent.id``), where
            the caller has one. Never invented.

    Returns:
        The wrapped function (or a decorator, when used parameterized).
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or fn.__qualname__
        span_tool_name = _resolve_tool_name(tool_name, name, fn.__qualname__, kind)

        def _kind_attributes() -> dict[str, str | int]:
            # Resolved per CALL, not per decoration: init() usually runs after
            # the module defining the decorated function is imported, so the
            # configured agent name is not yet published at decoration time.
            return kind_attributes(
                kind,
                span_tool_name,
                data_source_id=data_source_id,
                top_k=top_k,
                agent_name=resolve_agent_name(agent_name, kind),
                agent_id=agent_id,
            )

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
                tracer = sdk_tracer()
                span = tracer.start_span(
                    span_name,
                    kind=otel_span_kind(kind),
                    attributes=_kind_attributes(),
                )
                _set_input(span, args, kwargs)
                agen = fn(*args, **kwargs)
                # A transparent proxy: send() and throw() from the caller reach
                # the inner generator, and closing the wrapper closes it, so
                # its finally blocks run now rather than at garbage collection.
                pending: tuple[str, Any] = ("send", None)
                try:
                    while True:
                        token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            if pending[0] == "send":
                                item = await agen.asend(pending[1])
                            else:
                                item = await agen.athrow(pending[1])
                        except StopAsyncIteration:
                            break
                        finally:
                            otel_context.detach(token)
                        try:
                            pending = ("send", (yield item))
                        except GeneratorExit:
                            raise
                        except BaseException as thrown:  # the caller's athrow()
                            pending = ("throw", thrown)
                except Exception as exc:
                    record_error(span, exc)
                    raise
                finally:
                    await agen.aclose()
                    span.end()

            return async_gen_wrapper

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = sdk_tracer()
                with tracer.start_as_current_span(
                    span_name,
                    kind=otel_span_kind(kind),
                    attributes=_kind_attributes(),
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    _set_input(span, args, kwargs)
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        record_error(span, exc)
                        raise
                    _set_output(span, result)
                    return result

            return async_wrapper

        if inspect.isgeneratorfunction(fn):
            # See the async-generator note: attach only around each step.

            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = sdk_tracer()
                span = tracer.start_span(
                    span_name,
                    kind=otel_span_kind(kind),
                    attributes=_kind_attributes(),
                )
                _set_input(span, args, kwargs)
                gen = fn(*args, **kwargs)
                pending: tuple[str, Any] = ("send", None)
                try:
                    while True:
                        token = otel_context.attach(trace.set_span_in_context(span))
                        try:
                            if pending[0] == "send":
                                item = gen.send(pending[1])
                            else:
                                item = gen.throw(pending[1])
                        except StopIteration:
                            break
                        finally:
                            otel_context.detach(token)
                        try:
                            pending = ("send", (yield item))
                        except GeneratorExit:
                            raise
                        except BaseException as thrown:  # the caller's throw()
                            pending = ("throw", thrown)
                except Exception as exc:
                    record_error(span, exc)
                    raise
                finally:
                    gen.close()
                    span.end()

            return gen_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = sdk_tracer()
            with tracer.start_as_current_span(
                span_name,
                kind=otel_span_kind(kind),
                attributes=_kind_attributes(),
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                _set_input(span, args, kwargs)
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    record_error(span, exc)
                    raise
                _set_output(span, result)
                return result

        return sync_wrapper

    if func is not None:
        return decorate(func)
    return decorate
