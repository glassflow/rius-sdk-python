"""``error.type`` for a raised exception, shared by every span helper.

The GenAI conventions make ``error.type`` Conditionally Required on any span
that ends in an error and want it low-cardinality: the exception class, never
the message (which echoes request content and is unbounded).
"""

from __future__ import annotations

import logging

from opentelemetry.trace import Span, Status, StatusCode

from .semconv import ERROR_TYPE, EXCEPTION_EVENT, EXCEPTION_TYPE

logger = logging.getLogger(__name__)

#: The message for an exception whose ``__str__`` raises, as ``traceback``
#: writes it for the same exception.
UNPRINTABLE_MESSAGE = "<exception str() failed>"


def error_type(exc: BaseException) -> str:
    """The exception class, module-qualified unless it is a builtin.

    This is the same spelling the span's own exception event uses for
    ``exception.type``, so the two never disagree on one span.
    """
    cls = type(exc)
    if cls.__module__ in ("builtins", "__main__"):
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def record_error(span: Span, exc: BaseException) -> None:
    """Record a failure on ``span`` as the exception event, ERROR status and ``error.type``.

    One function so the three signals cannot diverge: a span that carries an
    exception event but no ``error.type`` is invisible to any grouping on that
    key, which is exactly the inconsistency this exists to prevent.

    Never raises: the caller is about to re-raise ``exc``, and an exception
    whose ``__str__`` raises must still reach it rather than the SDK's error.
    """
    message = _message(exc)
    try:
        span.record_exception(exc)
    except Exception:
        # OpenTelemetry's own record_exception calls str(exc). Write the
        # event without it, so the span still carries one.
        _record_exception_event(span, exc)
    span.set_status(Status(StatusCode.ERROR, message))
    span.set_attribute(ERROR_TYPE, error_type(exc))


def _message(exc: BaseException) -> str:
    """``str(exc)``, or a fixed placeholder when the exception's ``__str__`` raises."""
    try:
        return str(exc)
    except Exception:
        return UNPRINTABLE_MESSAGE


def _record_exception_event(span: Span, exc: BaseException) -> None:
    # Only the type: the registry names the event and that attribute, and the
    # type is what a grouping reads. The message is what this exception lacks.
    try:
        span.add_event(EXCEPTION_EVENT, {EXCEPTION_TYPE: error_type(exc)})
    except Exception:
        logger.warning("could not record %s on the span", error_type(exc), exc_info=True)
