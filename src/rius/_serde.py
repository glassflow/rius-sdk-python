"""Value serialization for span attributes: JSON with a safe fallback and truncation."""

from __future__ import annotations

import json
from typing import Any

MAX_ATTR_CHARS = 8192


def serialize(value: Any) -> str:
    """Serialize a value to a bounded string suitable for a span attribute."""
    try:
        text = json.dumps(value, default=repr)
    except Exception:
        # default=repr can itself raise (broken __repr__, detached ORM proxies);
        # tracing must never crash the host over an unserializable value.
        try:
            text = repr(value)
        except Exception:
            text = f"<unserializable {type(value).__name__}>"
    if len(text) > MAX_ATTR_CHARS:
        text = text[:MAX_ATTR_CHARS] + "…(truncated)"
    return text
