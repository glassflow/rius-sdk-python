"""Nest the provider call under LangChain's chat model span.

openinference-instrumentation-langchain never makes its spans current: it
builds them from LangChain's callbacks, and its code explains why it does not
attach a context there (a callback that fails before the matching one runs
would leave the context attached for good). So a chat model's provider call,
traced by the provider's own instrumentor (openai, anthropic), sees no current
span: its span is a root span in a trace of its own.

Two things follow for rius users. One model call shows up in two traces. And
the usage dedup in :mod:`rius.aggregated_usage`, which relies on the provider
span being the LangChain span's child, cannot see it, so the call's usage is
counted twice.

This module makes the LangChain span current inside the model call itself,
where entering and leaving the context happen in the same frame, so nothing
can be left attached:

* ``BaseChatModel._generate_with_cache`` and ``_agenerate_with_cache`` (every
  non-streamed call, sync and async) receive the call's run manager, whose run
  id names the span; the call runs with that span current.
* ``BaseChatModel.stream`` (sync) creates its run inside the generator, so the
  instrumentor's ``_start_trace`` is wrapped to make the new chat model span
  current for the rest of that step only, and every later step runs under it.
  The callback runs inline in the caller's thread for a sync stream; when it
  does not (an executor thread), nothing is attached.

``astream`` is not covered: its callbacks run in an executor with a copied
context, so there is no frame that both sees the new span and runs the
provider call. A streamed async LangChain call is still counted twice.

It hooks private details of the instrumentor (its ``_tracer``, the tracer's
``get_span`` and ``_start_trace``), so it only installs on the instrumentor
versions it was checked against. It steps aside when the span is already
current, and never raises into the caller's code.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace

logger = logging.getLogger(__name__)

#: The instrumentor releases whose tracer internals were read and found as
#: this module expects. A later release may have reshaped them; it is left
#: alone.
VERIFIED_VERSIONS = ("0.1.67", "0.1.79")

_MARK = "__rius_span_context__"

#: The instrumentor's tracer the hooks look spans up in. Set by install().
_tracer: Any = None


class _PendingStream:
    """One sync stream's chat model span, found when its run starts."""

    __slots__ = ("span", "thread", "token")

    def __init__(self) -> None:
        self.span: Any = None
        self.thread = threading.get_ident()
        self.token: Any = None


#: The stream whose step is running, so its run's start can attach the span.
_PENDING: contextvars.ContextVar[_PendingStream | None] = contextvars.ContextVar(
    "rius_langchain_pending_stream", default=None
)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def _verified(installed: str) -> bool:
    low, high = (_version_tuple(v) for v in VERIFIED_VERSIONS)
    return low <= _version_tuple(installed) <= high


def _span_of(run_manager: Any) -> Any:
    run_id = getattr(run_manager, "run_id", None)
    tracer = _tracer
    if run_id is None or tracer is None:
        return None
    try:
        return tracer.get_span(run_id)
    except Exception:
        return None


def _needs_context(span: Any) -> bool:
    return span is not None and trace.get_current_span() is not span


def _under(span: Any) -> Any:
    return trace.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    )


def _run_manager(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    # (self, messages, stop, run_manager, ...): run_manager is passed by
    # keyword by LangChain, positionally by nobody we know of.
    return kwargs.get("run_manager", args[3] if len(args) > 3 else None)


def _wrap_generate(original: Any) -> Any:
    @functools.wraps(original)
    def run(*args: Any, **kwargs: Any) -> Any:
        span = _span_of(_run_manager(args, kwargs))
        if not _needs_context(span):
            return original(*args, **kwargs)
        with _under(span):
            return original(*args, **kwargs)

    setattr(run, _MARK, original)
    return run


def _wrap_agenerate(original: Any) -> Any:
    @functools.wraps(original)
    async def run(*args: Any, **kwargs: Any) -> Any:
        span = _span_of(_run_manager(args, kwargs))
        if not _needs_context(span):
            return await original(*args, **kwargs)
        with _under(span):
            return await original(*args, **kwargs)

    setattr(run, _MARK, original)
    return run


def _wrap_stream(original: Any) -> Any:
    @functools.wraps(original)
    def run(*args: Any, **kwargs: Any) -> Iterator[Any]:
        steps = original(*args, **kwargs)
        pending = _PendingStream()
        try:
            while True:
                if pending.span is None:
                    # The run, and with it the span, starts inside this step;
                    # the start hook attaches it for the rest of the step.
                    marker = _PENDING.set(pending)
                    try:
                        item = next(steps)
                    except StopIteration:
                        return
                    finally:
                        if pending.token is not None:
                            otel_context.detach(pending.token)
                            pending.token = None
                        _PENDING.reset(marker)
                elif _needs_context(pending.span):
                    with _under(pending.span):
                        try:
                            item = next(steps)
                        except StopIteration:
                            return
                else:
                    try:
                        item = next(steps)
                    except StopIteration:
                        return
                yield item
        finally:
            steps.close()

    setattr(run, _MARK, original)
    return run


def _wrap_start_trace(original: Any, tracer: Any) -> Any:
    @functools.wraps(original)
    def run(run_record: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(run_record, *args, **kwargs)
        try:
            pending = _PENDING.get()
            if (
                pending is not None
                and pending.span is None
                and pending.thread == threading.get_ident()
                and getattr(run_record, "run_type", None) == "llm"
            ):
                span = tracer.get_span(run_record.id)
                if span is not None:
                    pending.span = span
                    if _needs_context(span):
                        pending.token = otel_context.attach(trace.set_span_in_context(span))
        except Exception:
            logger.debug("could not attach the langchain stream span", exc_info=True)
        return result

    setattr(run, _MARK, original)
    return run


def _chat_model_class() -> Any:
    from langchain_core.language_models.chat_models import BaseChatModel

    return BaseChatModel


_HOOKS = {
    "_generate_with_cache": _wrap_generate,
    "_agenerate_with_cache": _wrap_agenerate,
    "stream": _wrap_stream,
}


def install(instrumentor: Any) -> bool:
    """Make LangChain chat model calls run under their span.

    Returns whether it installed. Call it after the instrumentor was
    instrumented; calling it again re-installs without stacking.
    """
    global _tracer
    try:
        try:
            installed = version("openinference-instrumentation-langchain")
        except PackageNotFoundError:
            return False
        if not _verified(installed):
            logger.debug(
                "openinference-instrumentation-langchain %s is outside the versions "
                "checked for the chat model span context (%s to %s); leaving it alone",
                installed,
                *VERIFIED_VERSIONS,
            )
            return False
        tracer = getattr(instrumentor, "_tracer", None)
        if tracer is None or not callable(getattr(tracer, "get_span", None)):
            return False
        start_trace = getattr(tracer, "_start_trace", None)
        if not callable(start_trace):
            return False
        model = _chat_model_class()
        if not all(callable(getattr(model, name, None)) for name in _HOOKS):
            return False
        for name, wrap in _HOOKS.items():
            current = model.__dict__.get(name)
            if current is None:
                continue
            setattr(model, name, wrap(getattr(current, _MARK, current)))
        tracer._start_trace = _wrap_start_trace(getattr(start_trace, _MARK, start_trace), tracer)
        _tracer = tracer
        return True
    except Exception:
        logger.debug("could not install the langchain chat model span context", exc_info=True)
        return False


def uninstall(instrumentor: Any) -> None:
    """Undo :func:`install`, leaving the instrumentor's own state."""
    global _tracer
    try:
        tracer = getattr(instrumentor, "_tracer", None)
        hooked = getattr(tracer, "__dict__", {}).get("_start_trace")
        if hooked is not None and hasattr(hooked, _MARK):
            del tracer.__dict__["_start_trace"]
        model = _chat_model_class()
        for name in _HOOKS:
            current = model.__dict__.get(name)
            if current is not None and hasattr(current, _MARK):
                setattr(model, name, getattr(current, _MARK))
        _tracer = None
    except Exception:
        logger.debug("could not uninstall the langchain chat model span context", exc_info=True)
