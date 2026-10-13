"""Nest the provider call under LiteLLM's span on streaming calls.

openinference-instrumentation-litellm starts a streaming call's span with
``start_span`` and calls the wrapped litellm function without making that span
current. LiteLLM calls the provider SDK inside that function, so the openai
instrumentor's ``ChatCompletion`` span for a streamed call sees no current
span and starts a trace of its own. Non-streamed calls nest, because those
branches use ``start_as_current_span``.

Two things follow for rius users. A streamed LiteLLM call shows up in two
traces. And the usage dedup in :mod:`rius.aggregated_usage`, which relies on
the OpenAI span being the LiteLLM span's child, cannot see it, so the call's
usage is counted twice.

This module is a stopgap until the instrumentor makes the span current itself.
It hooks two private details of the instrumentor: its ``_tracer`` attribute,
proxied so a stream span it starts is noted for the current call, and the
``original_litellm_funcs`` table its wrappers call through, whose stream
entries then run with that span current. Because those are private, it only
installs on the instrumentor versions it was checked against, and it steps
aside on its own when the instrumentor already made the span current. It never
raises into the caller's code.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from opentelemetry import trace

logger = logging.getLogger(__name__)

#: The instrumentor releases whose stream branches were read and found to call
#: the wrapped function without making the span current. A later release may
#: have fixed it, or reshaped the internals this module hooks; it is left alone.
VERIFIED_VERSIONS = ("0.1.34", "0.1.50")

#: The wrapped litellm functions whose stream branch starts its span without
#: making it current, keyed to the span name each one starts.
_STREAM_FUNCS = {
    "completion": "completion",
    "acompletion": "acompletion",
    "responses": "responses",
    "aresponses": "aresponses",
}

#: The span the instrumentor just started for this call, by span name. Set by
#: the tracer proxy, taken by the wrapped function that runs right after it.
_STARTED: contextvars.ContextVar[tuple[str, Any] | None] = contextvars.ContextVar(
    "rius_litellm_started_span", default=None
)

_MARK = "__rius_stream_context__"


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def _verified(installed: str) -> bool:
    low, high = (_version_tuple(v) for v in VERIFIED_VERSIONS)
    return low <= _version_tuple(installed) <= high


class _TracerProxy:
    """The instrumentor's tracer, noting each stream span it starts."""

    def __init__(self, tracer: Any) -> None:
        self._rius_tracer = tracer

    def start_span(self, name: str, *args: Any, **kwargs: Any) -> Any:
        span = self._rius_tracer.start_span(name, *args, **kwargs)
        if name in _STREAM_FUNCS.values():
            _STARTED.set((name, span))
        return span

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._rius_tracer, attribute)


def _take_started(span_name: str) -> Any:
    started = _STARTED.get()
    if started is None:
        return None
    _STARTED.set(None)
    name, span = started
    return span if name == span_name else None


def _needs_context(span: Any) -> bool:
    # None: no stream span was started for this call. Current already: the
    # instrumentor made it current itself (a fixed release), nothing to do.
    return span is not None and trace.get_current_span() is not span


def _wrap(original: Any, span_name: str) -> Any:
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            span: Any = _take_started(span_name) if kwargs.get("stream") else None
            if not _needs_context(span):
                return await original(*args, **kwargs)
            with trace.use_span(
                span, end_on_exit=False, record_exception=False, set_status_on_exception=False
            ):
                return await original(*args, **kwargs)

        setattr(run_async, _MARK, original)
        return run_async

    @functools.wraps(original)
    def run(*args: Any, **kwargs: Any) -> Any:
        span: Any = _take_started(span_name) if kwargs.get("stream") else None
        if not _needs_context(span):
            return original(*args, **kwargs)
        with trace.use_span(
            span, end_on_exit=False, record_exception=False, set_status_on_exception=False
        ):
            return original(*args, **kwargs)

    setattr(run, _MARK, original)
    return run


def install(instrumentor: Any) -> bool:
    """Make ``instrumentor``'s streaming calls run under their span.

    Returns whether it installed. Call it after the instrumentor was
    instrumented; calling it again re-installs over the instrumentor's current
    state without stacking.
    """
    try:
        try:
            installed = version("openinference-instrumentation-litellm")
        except PackageNotFoundError:
            return False
        if not _verified(installed):
            logger.debug(
                "openinference-instrumentation-litellm %s is outside the versions "
                "checked for the stream span context (%s to %s); leaving it alone",
                installed,
                *VERIFIED_VERSIONS,
            )
            return False
        tracer = getattr(instrumentor, "_tracer", None)
        table = getattr(type(instrumentor), "original_litellm_funcs", None)
        if tracer is None or not isinstance(table, dict):
            return False
        if not isinstance(tracer, _TracerProxy):
            instrumentor._tracer = _TracerProxy(tracer)
        for func_name, span_name in _STREAM_FUNCS.items():
            original = table.get(func_name)
            if original is None:
                continue
            # Already ours (a re-init), or restored from ours by uninstrument
            # and recorded again as the "original": wrap the real function.
            original = getattr(original, _MARK, original)
            table[func_name] = _wrap(original, span_name)
        return True
    except Exception:
        logger.debug("could not install the litellm stream span context", exc_info=True)
        return False


def uninstall(instrumentor: Any) -> None:
    """Undo :func:`install` on ``instrumentor``, leaving its own state."""
    try:
        tracer = getattr(instrumentor, "_tracer", None)
        if isinstance(tracer, _TracerProxy):
            instrumentor._tracer = tracer._rius_tracer
        table = getattr(type(instrumentor), "original_litellm_funcs", None)
        if isinstance(table, dict):
            for func_name, func in list(table.items()):
                table[func_name] = getattr(func, _MARK, func)
    except Exception:
        logger.debug("could not uninstall the litellm stream span context", exc_info=True)
