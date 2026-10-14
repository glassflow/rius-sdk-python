"""Nest the provider call under LlamaIndex's LLM span.

openinference-instrumentation-llama-index builds its spans from LlamaIndex's
instrumentation dispatcher: its span handler's ``new_span`` starts each one
with ``start_span`` and never makes it current. So an LLM's provider call,
traced by the provider's own instrumentor (openai, anthropic), sees no current
span: its span is a root span in a trace of its own.

Two things follow for rius users. One model call shows up in two traces. And
the usage dedup in :mod:`rius.aggregated_usage`, which relies on the provider
span being the LlamaIndex span's child, cannot see it, so the call's usage is
counted twice.

This module makes the LlamaIndex span current around the provider call, in
two places:

* A span handler of its own, through LlamaIndex's public instrumentation API
  (``BaseSpanHandler``, ``Dispatcher.add_span_handler``), runs after the
  instrumentor's. For an LLM span (one whose instance is a ``BaseLLM``) it
  makes the span current on ``span_enter`` and takes it down on ``span_exit``
  or ``span_drop``. The dispatcher calls those in its own frame, around the
  method (in the same coroutine for an async one), so every non-streamed call
  (``chat``, ``complete``, ``achat``, ``acomplete`` and the methods built on
  them) runs with its span current and nothing outlives the call. The span is
  the instrumentor's, found with its public ``get_current_span()``.
* A streamed call returns its generator before any request is made: the
  request starts on the first step, after the dispatcher's frame has ended,
  and LlamaIndex runs no hook in that step. So the four stream methods
  (``stream_chat``, ``stream_complete``, ``astream_chat``,
  ``astream_complete``) are wrapped on every ``BaseLLM`` subclass, including
  ones defined later (through pydantic's ``__pydantic_init_subclass__`` on
  ``BaseLLM``). The span handler hands over the span of the stream it saw
  returned, and the wrapper runs each step of that stream with the span
  current, taking it down before the chunk reaches the caller.

Entering and leaving happen in one frame per step, and the span handler only
takes a span down in the thread and asyncio task that put it up, so nothing
can be left attached or leak into another thread or task.

It relies on the instrumentor's behaviour rather than its private symbols: it
starts an LLM span in ``new_span``, before the method runs, keeps a streamed
call's span open until the stream ends, and ``get_current_span()`` finds it.
So it only installs on the instrumentor versions it was checked against. The
LlamaIndex side (the dispatcher calling span handlers around the method, the
stream methods returning the stream) is the user's own dependency; the
upstream-stopgaps workflow runs these tests against the newest releases. It
steps aside when the span is already current and never raises into the
caller's code.

Not covered: ``MultiModalLLM`` models, and spans that are not an LLM's (an
embedding's, a retriever's), whose provider calls still start their own trace.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
import weakref
from collections.abc import AsyncIterator, Iterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace

logger = logging.getLogger(__name__)

#: The instrumentor releases whose span handling was read and found as this
#: module expects. A later release may have reshaped it; it is left alone.
VERIFIED_VERSIONS = ("4.4.3", "4.6.0")

#: The methods whose result is a stream that makes the provider request later.
_STREAM_METHODS = ("stream_chat", "stream_complete", "astream_chat", "astream_complete")

_MARK = "__rius_span_context__"
_SUBCLASS_HOOK = "__pydantic_init_subclass__"

#: The handler install() added to LlamaIndex's root dispatcher, kept across
#: re-installs so a call in flight still finds the handler that attached it.
_handler: Any = None
#: The hook install() set on BaseLLM for subclasses defined later.
_subclass_hook: Any = None


def _owner() -> tuple[int, Any]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def _verified(installed: str) -> bool:
    low, high = (_version_tuple(v) for v in VERIFIED_VERSIONS)
    return low <= _version_tuple(installed) <= high


def _needs_context(span: Any) -> bool:
    return span is not None and trace.get_current_span() is not span


def _under(span: Any) -> Any:
    return trace.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    )


def _current_llama_index_span() -> Any:
    from openinference.instrumentation.llama_index import get_current_span

    return get_current_span()


class _Entered:
    """An LLM span the handler saw start, and what it attached for it."""

    __slots__ = ("owner", "span", "token")

    def __init__(self, span: Any) -> None:
        self.span = span
        self.owner = _owner()
        self.token: Any = None


@functools.cache
def _handler_class() -> Any:
    from llama_index.core.base.llms.base import BaseLLM
    from llama_index.core.instrumentation.span_handlers import BaseSpanHandler

    class _SpanContextHandler(BaseSpanHandler[Any]):  # type: ignore[misc]
        """Makes an LLM span current while its method runs.

        Never keeps spans of its own: ``new_span`` and the ``prepare_*``
        methods return None, and the dispatcher's notices are taken directly.
        """

        def __init__(self) -> None:
            super().__init__()
            self._rius_lock = threading.Lock()
            self._rius_enabled = False
            self._rius_entered: dict[str, _Entered] = {}
            # A stream's span, by the stream the method returned.
            self._rius_streams: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()

        @classmethod
        def class_name(cls) -> str:
            return "RiusSpanContextHandler"

        def new_span(self, *args: Any, **kwargs: Any) -> None:
            return None

        def prepare_to_exit_span(self, *args: Any, **kwargs: Any) -> None:
            return None

        def prepare_to_drop_span(self, *args: Any, **kwargs: Any) -> None:
            return None

        def span_enter(self, id_: str, *args: Any, **kwargs: Any) -> None:
            try:
                if not isinstance(kwargs.get("instance"), BaseLLM):
                    return
                span = _current_llama_index_span()
                if span is None:
                    return
                entered = _Entered(span)
                with self._rius_lock:
                    if not self._rius_enabled or id_ in self._rius_entered:
                        return
                    if _needs_context(span):
                        entered.token = otel_context.attach(trace.set_span_in_context(span))
                    self._rius_entered[id_] = entered
            except Exception:
                logger.debug("could not attach the llama-index span", exc_info=True)

        def span_exit(self, id_: str, *args: Any, **kwargs: Any) -> None:
            entered = self._leave(id_)
            result = kwargs.get("result")
            if entered is not None and (inspect.isgenerator(result) or inspect.isasyncgen(result)):
                with self._rius_lock:
                    self._rius_streams[result] = entered.span

        def span_drop(self, id_: str, *args: Any, **kwargs: Any) -> None:
            self._leave(id_)

        def _leave(self, id_: str) -> _Entered | None:
            with self._rius_lock:
                entered = self._rius_entered.pop(id_, None)
            if entered is not None and entered.token is not None:
                if entered.owner == _owner():
                    otel_context.detach(entered.token)
                else:
                    # Not the frame that attached it (a sync method whose
                    # result is a future ends in the event loop): the
                    # context is not ours to reset, and a detach would fail.
                    logger.debug("llama-index span %s ended elsewhere", id_)
            return entered

        def take_stream_span(self, stream: Any) -> Any:
            if not (inspect.isgenerator(stream) or inspect.isasyncgen(stream)):
                return None
            with self._rius_lock:
                return self._rius_streams.pop(stream, None)

        def busy(self) -> bool:
            with self._rius_lock:
                return bool(self._rius_entered)

        def enable(self, enabled: bool) -> None:
            with self._rius_lock:
                self._rius_enabled = enabled

    return _SpanContextHandler


def _stream_span(stream: Any) -> Any:
    handler = _handler
    if handler is None:
        return None
    try:
        return handler.take_stream_span(stream)
    except Exception:
        logger.debug("could not find the llama-index stream span", exc_info=True)
        return None


def _steps_under(span: Any, stream: Iterator[Any]) -> Iterator[Any]:
    try:
        while True:
            with _under(span):
                try:
                    item = next(stream)
                except StopIteration:
                    return
            yield item
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


async def _asteps_under(span: Any, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    try:
        while True:
            with _under(span):
                try:
                    item = await stream.__anext__()
                except StopAsyncIteration:
                    return
            yield item
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


def _wrap_stream(original: Any) -> Any:
    @functools.wraps(original)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        stream = original.__get__(self, type(self))(*args, **kwargs)
        span = _stream_span(stream)
        if span is None:
            return stream
        return _steps_under(span, stream)

    setattr(run, _MARK, original)
    return run


def _wrap_astream(original: Any) -> Any:
    @functools.wraps(original)
    async def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        stream = await original.__get__(self, type(self))(*args, **kwargs)
        span = _stream_span(stream)
        if span is None:
            return stream
        return _asteps_under(span, stream)

    setattr(run, _MARK, original)
    return run


def _wrap_class(cls: Any) -> None:
    for name in _STREAM_METHODS:
        current = cls.__dict__.get(name)
        if current is None:
            continue
        original = getattr(current, _MARK, current)
        wrap = _wrap_astream if name.startswith("a") else _wrap_stream
        setattr(cls, name, wrap(original))


def _unwrap_class(cls: Any) -> None:
    for name in _STREAM_METHODS:
        current = cls.__dict__.get(name)
        if current is not None and hasattr(current, _MARK):
            setattr(cls, name, getattr(current, _MARK))


def _subclasses(cls: Any) -> list[Any]:
    found: list[Any] = []
    pending = [cls]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub not in found:
                found.append(sub)
                pending.append(sub)
    return found


def _install_subclass_hook(base: Any) -> None:
    global _subclass_hook
    if base.__dict__.get(_SUBCLASS_HOOK) is not None:
        # Ours already, or someone else's on BaseLLM itself: leave it (and
        # with someone else's, any class defined later) alone.
        return

    def hook(cls: Any, /, **kwargs: Any) -> None:
        super(base, cls).__pydantic_init_subclass__(**kwargs)
        try:
            _wrap_class(cls)
        except Exception:
            logger.debug("could not wrap the llama-index LLM %r", cls, exc_info=True)

    _subclass_hook = classmethod(hook)
    setattr(base, _SUBCLASS_HOOK, _subclass_hook)


def _place_handler(dispatcher: Any, handler: Any) -> None:
    """Put ``handler`` last, so it runs after the instrumentor's handler."""
    others = [h for h in dispatcher.span_handlers if h is not handler]
    dispatcher.span_handlers[:] = [*others, handler]


def install(instrumentor: Any) -> bool:
    """Make LlamaIndex LLM calls run under their span.

    Returns whether it installed. Call it after the instrumentor was
    instrumented; calling it again re-installs without stacking.
    """
    global _handler
    try:
        try:
            installed = version("openinference-instrumentation-llama-index")
        except PackageNotFoundError:
            return False
        if not _verified(installed):
            logger.debug(
                "openinference-instrumentation-llama-index %s is outside the versions "
                "checked for the LLM span context (%s to %s); leaving it alone",
                installed,
                *VERIFIED_VERSIONS,
            )
            return False
        if getattr(instrumentor, "_event_handler", None) is None:
            # The legacy callback handler, or not instrumented: no span handler.
            return False
        from llama_index.core.base.llms.base import BaseLLM
        from llama_index.core.instrumentation import get_dispatcher

        if _handler is None:
            _handler = _handler_class()()
        _place_handler(get_dispatcher(), _handler)
        for cls in _subclasses(BaseLLM):
            _wrap_class(cls)
        _install_subclass_hook(BaseLLM)
        _handler.enable(True)
        return True
    except Exception:
        logger.debug("could not install the llama-index LLM span context", exc_info=True)
        return False


def uninstall(instrumentor: Any) -> None:
    """Undo :func:`install`, leaving the instrumentor's own state.

    A call in flight keeps the handler until it ends, so the span it made
    current is still taken down: the handler is only disabled then.
    """
    global _subclass_hook
    try:
        from llama_index.core.base.llms.base import BaseLLM
        from llama_index.core.instrumentation import get_dispatcher

        if _subclass_hook is not None:
            if BaseLLM.__dict__.get(_SUBCLASS_HOOK) is _subclass_hook:
                delattr(BaseLLM, _SUBCLASS_HOOK)
            _subclass_hook = None
        for cls in _subclasses(BaseLLM):
            _unwrap_class(cls)
        handler = _handler
        if handler is not None:
            handler.enable(False)
            if not handler.busy():
                dispatcher = get_dispatcher()
                dispatcher.span_handlers[:] = [
                    h for h in dispatcher.span_handlers if h is not handler
                ]
    except Exception:
        logger.debug("could not uninstall the llama-index LLM span context", exc_info=True)
