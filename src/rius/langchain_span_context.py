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
* ``BaseChatModel.stream`` and ``astream`` (and the chains that stream through
  them) start their run inside the generator. Each step that may start it is
  marked, and LangChain's public ``CallbackManager.on_chat_model_start`` /
  ``AsyncCallbackManager.on_chat_model_start``, which the generator calls in
  its own frame (only the handlers may run elsewhere), make the new span
  current for the rest of that step. The stream wrapper takes it down again
  before the step's chunk reaches the caller. The provider call starts in that
  step, and a span's parent is fixed when it starts, so later steps need
  nothing.

Not covered: ``BaseLLM`` text-completion models (``langchain_openai.OpenAI``
and other non-chat LLM wrappers). Their calls are still counted twice.

It relies on private details on both sides. In the instrumentor, its
tracer's ``get_span``, so it only installs on the instrumentor versions it was
checked against. In langchain-core, ``BaseChatModel._generate_with_cache`` and
``_agenerate_with_cache`` and the ``run_manager`` keyword they are called
with; langchain-core is the user's own dependency, so a release that renames
them is not caught here: install() finds nothing to wrap and returns False, or
the run manager is not found, and the call runs un-nested. The
upstream-stopgaps workflow runs these tests against the newest releases. It
steps aside when the span is already current, attaches only in the thread and
asyncio task whose step is running, and never raises into the caller's code.
"""

from __future__ import annotations

import contextvars
import functools
import logging
from collections.abc import AsyncIterator, Iterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace

from ._stopgap import MARK, needs_context, owner, under, verified

logger = logging.getLogger(__name__)

#: The instrumentor releases whose tracer internals were read and found as
#: this module expects. A later release may have reshaped them; it is left
#: alone.
VERIFIED_VERSIONS = ("0.1.67", "0.1.80")

#: The instrumentor's tracer the hooks look spans up in. Set by install().
_tracer: Any = None


class _PendingStream:
    """One stream step that may start the chat model's run.

    ``owner`` is the thread and asyncio task running the step: a context
    copied elsewhere (an executor thread, a child task) also sees the marker,
    and an attach there would land in a context the step cannot detach from.
    """

    __slots__ = ("owner", "span", "token")

    def __init__(self) -> None:
        self.owner = owner()
        self.span: Any = None
        self.token: Any = None


#: The stream step that is running, so its run's start can attach the span.
_PENDING: contextvars.ContextVar[_PendingStream | None] = contextvars.ContextVar(
    "rius_langchain_pending_stream", default=None
)


def _verified(installed: str) -> bool:
    return verified(installed, VERIFIED_VERSIONS)


def _span_of(run_manager: Any) -> Any:
    run_id = getattr(run_manager, "run_id", None)
    tracer = _tracer
    if run_id is None or tracer is None:
        return None
    try:
        return tracer.get_span(run_id)
    except Exception:
        return None


def _run_manager(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    # (self, messages, stop, run_manager, ...): run_manager is passed by
    # keyword by LangChain, positionally by nobody we know of.
    return kwargs.get("run_manager", args[3] if len(args) > 3 else None)


def _wrap_generate(original: Any) -> Any:
    @functools.wraps(original)
    def run(*args: Any, **kwargs: Any) -> Any:
        span = _span_of(_run_manager(args, kwargs))
        if not needs_context(span):
            return original(*args, **kwargs)
        with under(span):
            return original(*args, **kwargs)

    setattr(run, MARK, original)
    return run


def _wrap_agenerate(original: Any) -> Any:
    @functools.wraps(original)
    async def run(*args: Any, **kwargs: Any) -> Any:
        span = _span_of(_run_manager(args, kwargs))
        if not needs_context(span):
            return await original(*args, **kwargs)
        with under(span):
            return await original(*args, **kwargs)

    setattr(run, MARK, original)
    return run


def _end_step(pending: _PendingStream, marker: Any) -> None:
    if pending.token is not None:
        otel_context.detach(pending.token)
        pending.token = None
    _PENDING.reset(marker)


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
                        _end_step(pending, marker)
                else:
                    try:
                        item = next(steps)
                    except StopIteration:
                        return
                yield item
        finally:
            steps.close()

    setattr(run, MARK, original)
    return run


def _wrap_astream(original: Any) -> Any:
    @functools.wraps(original)
    async def run(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        steps = original(*args, **kwargs)
        pending = _PendingStream()
        try:
            while True:
                if pending.span is None:
                    marker = _PENDING.set(pending)
                    try:
                        item = await steps.__anext__()
                    except StopAsyncIteration:
                        return
                    finally:
                        _end_step(pending, marker)
                else:
                    try:
                        item = await steps.__anext__()
                    except StopAsyncIteration:
                        return
                yield item
        finally:
            await steps.aclose()

    setattr(run, MARK, original)
    return run


def _attach_started(run_managers: Any) -> None:
    """After a chat model run starts: make its span current for the step."""
    try:
        pending = _PENDING.get()
        if pending is None or pending.span is not None or pending.owner != owner():
            return
        span = _span_of(run_managers[0])
        if span is None:
            return
        pending.span = span
        if needs_context(span):
            pending.token = otel_context.attach(trace.set_span_in_context(span))
    except Exception:
        logger.debug("could not attach the langchain stream span", exc_info=True)


def _wrap_on_chat_model_start(original: Any) -> Any:
    @functools.wraps(original)
    def run(*args: Any, **kwargs: Any) -> Any:
        run_managers = original(*args, **kwargs)
        _attach_started(run_managers)
        return run_managers

    setattr(run, MARK, original)
    return run


def _wrap_aon_chat_model_start(original: Any) -> Any:
    @functools.wraps(original)
    async def run(*args: Any, **kwargs: Any) -> Any:
        run_managers = await original(*args, **kwargs)
        _attach_started(run_managers)
        return run_managers

    setattr(run, MARK, original)
    return run


def _hooked_classes() -> list[tuple[Any, dict[str, Any]]]:
    from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
    from langchain_core.language_models.chat_models import BaseChatModel

    return [
        (
            BaseChatModel,
            {
                "_generate_with_cache": _wrap_generate,
                "_agenerate_with_cache": _wrap_agenerate,
                "stream": _wrap_stream,
                "astream": _wrap_astream,
            },
        ),
        (CallbackManager, {"on_chat_model_start": _wrap_on_chat_model_start}),
        (AsyncCallbackManager, {"on_chat_model_start": _wrap_aon_chat_model_start}),
    ]


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
        hooked = _hooked_classes()
        if not all(callable(getattr(cls, name, None)) for cls, hooks in hooked for name in hooks):
            return False
        for cls, hooks in hooked:
            for name, wrap in hooks.items():
                current = cls.__dict__.get(name)
                if current is None:
                    continue
                setattr(cls, name, wrap(getattr(current, MARK, current)))
        _tracer = tracer
        return True
    except Exception:
        logger.debug("could not install the langchain chat model span context", exc_info=True)
        return False


def uninstall(instrumentor: Any) -> None:
    """Undo :func:`install`, leaving the instrumentor's own state."""
    global _tracer
    try:
        for cls, hooks in _hooked_classes():
            for name in hooks:
                current = cls.__dict__.get(name)
                if current is not None and hasattr(current, MARK):
                    setattr(cls, name, getattr(current, MARK))
        _tracer = None
    except Exception:
        logger.debug("could not uninstall the langchain chat model span context", exc_info=True)
