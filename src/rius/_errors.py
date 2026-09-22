"""``error.type`` for a raised exception, shared by every span helper.

The GenAI conventions make ``error.type`` Conditionally Required on any span
that ends in an error and want it low-cardinality: the exception class, never
the message (which echoes request content and is unbounded).
"""

from __future__ import annotations

from opentelemetry.trace import Span, Status, StatusCode

from .semconv import ERROR_TYPE


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
    """
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.set_attribute(ERROR_TYPE, error_type(exc))
