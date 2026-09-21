"""``error.type`` for a raised exception, shared by every span helper.

The GenAI conventions make ``error.type`` Conditionally Required on any span
that ends in an error and want it low-cardinality: the exception class, never
the message (which echoes request content and is unbounded).
"""

from __future__ import annotations


def error_type(exc: BaseException) -> str:
    """The exception class, module-qualified unless it is a builtin.

    This is the same spelling the span's own exception event uses for
    ``exception.type``, so the two never disagree on one span.
    """
    cls = type(exc)
    if cls.__module__ in ("builtins", "__main__"):
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"
