"""Value serialization for span attributes: JSON with a safe fallback and a bound.

The bound is 32 KB of serialized text, shared with the TypeScript SDK so a
payload lands the same size whichever SDK produced it. It bounds the WORK as
well as the output: encoding stops once the cap is reached rather than
rendering a multi-megabyte payload and then cutting it, which the 2026-09
review measured at 200x the cost on large tool results.
"""

from __future__ import annotations

import json
from typing import Any

MAX_ATTR_CHARS = 32 * 1024
TRUNCATION_MARKER = "…(truncated)"

_encoder = json.JSONEncoder(default=repr)


def truncate(text: str) -> str:
    """Cut ``text`` at the attribute cap, marking the cut."""
    if len(text) > MAX_ATTR_CHARS:
        return text[:MAX_ATTR_CHARS] + TRUNCATION_MARKER
    return text


def serialize(value: Any) -> str:
    """Serialize a value to a bounded string suitable for a span attribute."""
    try:
        return _bounded_dumps(value)
    except Exception:
        # default=repr can itself raise (broken __repr__, detached ORM proxies);
        # tracing must never crash the host over an unserializable value.
        try:
            return truncate(repr(value))
        except Exception:
            return f"<unserializable {type(value).__name__}>"


def _bounded_dumps(value: Any) -> str:
    # iterencode without _one_shot takes the pure-Python path, which yields
    # chunks lazily as it walks the value; the C encoder renders everything
    # first. A small payload costs a few extra microseconds this way; a large
    # one stops being walked at the cap instead of being rendered whole.
    chunks: list[str] = []
    size = 0
    for chunk in _encoder.iterencode(value, _one_shot=False):
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_ATTR_CHARS:
            return "".join(chunks)[:MAX_ATTR_CHARS] + TRUNCATION_MARKER
    return "".join(chunks)
