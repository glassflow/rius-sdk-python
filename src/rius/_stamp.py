"""One span processor for "stamp a context value as an attribute at span start".

Sessions, users and workspaces all work the same way: a scope puts a value in
the OTel context, and a processor copies it onto every span at ``on_start``
so pending snapshots (built at start) carry it too. This is the shared
mechanism; each identity module keeps its own context key, attribute name and
default policy.
"""

from __future__ import annotations

from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import Span, SpanProcessor


class ContextStampProcessor(SpanProcessor):
    """Copy the context value under ``key`` to attribute ``attribute`` at start.

    The active scope wins; otherwise ``default`` applies; with neither, the
    attribute is not set.
    """

    def __init__(self, key: str, attribute: str, default: str | None = None) -> None:
        self._key = key
        self._attribute = attribute
        self._default = default

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        value = otel_context.get_value(self._key, context=parent_context)
        stamped = value if isinstance(value, str) else self._default
        if stamped is not None:
            span.set_attribute(self._attribute, stamped)

    def on_end(self, span: Any) -> None:  # pragma: no cover - nothing to do
        pass

    def shutdown(self) -> None:  # pragma: no cover - nothing to hold
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True
