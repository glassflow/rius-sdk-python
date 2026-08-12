"""Export-stage PII controls: content opt-out + a redaction mask.

A ``SpanExporter`` wrapper that, before spans leave the process, either strips
content attributes (``capture_content=False``) or applies a caller-supplied
``mask``. It runs on every span it sees, including third-party
instrumentation, so it's a single client-side choke point for sensitive data.

Sanitization works on **copies**: a ``ReadableSpan`` shares its attribute dict
by reference with every processor on the provider, so mutating it in place
would rewrite what other exporters see (and race with their iteration).

Fail-closed guarantees: a mask that raises, returns ``None``, or returns a
value OTel can't encode never leaks the original; the attribute is dropped
(or the return value serialized), and the rest of the batch is delivered.
"""

from __future__ import annotations

import copy
import inspect
import logging
from collections.abc import Callable, Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link

from ._serde import serialize
from .semconv import (
    CONTENT_ATTRIBUTE_PREFIXES,
    CONTENT_ATTRIBUTE_SUFFIXES,
    CONTENT_ATTRIBUTES,
)

logger = logging.getLogger(__name__)

Mask = Callable[..., Any]

_PRIMITIVES = (str, bool, int, float, bytes)


def _accepts_key(mask: Mask) -> bool:
    """True if the mask can receive the attribute key as ``key=`` keyword."""
    try:
        parameters = inspect.signature(mask).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        or (p.name == "key" and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD))
        for p in parameters
    )


def _is_content_key(key: str) -> bool:
    return (
        key in CONTENT_ATTRIBUTES
        or key.startswith(CONTENT_ATTRIBUTE_PREFIXES)
        or key.endswith(CONTENT_ATTRIBUTE_SUFFIXES)
    )


# record_exception() writes the provider's error string, and providers echo
# the rejected request into it, so these two carry the same content the
# attribute strip removes. exception.type deliberately stays: failures remain
# visible and classifiable with content capture off (same policy as the
# TypeScript SDK).
_EXCEPTION_EVENT_NAME = "exception"
_EXCEPTION_CONTENT_KEYS = frozenset({"exception.message", "exception.stacktrace"})


class MaskingSpanExporter(SpanExporter):
    """Strip or redact content attributes before delegating to ``inner``."""

    def __init__(
        self,
        inner: SpanExporter,
        *,
        capture_content: bool = True,
        mask: Mask | None = None,
    ) -> None:
        self._inner = inner
        self._capture_content = capture_content
        self._mask = mask
        self._mask_accepts_key = mask is not None and _accepts_key(mask)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not self._capture_content or self._mask is not None:
            spans = [self._sanitized(span) for span in spans]
        return self._inner.export(spans)

    def _sanitized(self, span: ReadableSpan) -> ReadableSpan:
        # The privacy boundary is the whole span: content rides attributes,
        # event attributes (OTel GenAI's event-based shape, record_exception's
        # provider-echoed error strings), and link attributes alike.
        new_attributes = self._sanitize_mapping(span.attributes)
        new_events = self._sanitize_events(span.events)
        new_links = self._sanitize_links(span.links)
        if new_attributes is None and new_events is None and new_links is None:
            return span
        sanitized = copy.copy(span)
        if new_attributes is not None:
            sanitized._attributes = new_attributes
        if new_events is not None:
            sanitized._events = new_events
        if new_links is not None:
            sanitized._links = new_links
        return sanitized

    def _sanitize_events(self, events: Sequence[Event]) -> tuple[Event, ...] | None:
        if not events:
            return None
        changed = False
        out: list[Event] = []
        for event in events:
            extra = _EXCEPTION_CONTENT_KEYS if event.name == _EXCEPTION_EVENT_NAME else frozenset()
            new_attributes = self._sanitize_mapping(event.attributes, extra_content_keys=extra)
            if new_attributes is None:
                out.append(event)
                continue
            changed = True
            out.append(Event(event.name, new_attributes, event.timestamp))
        return tuple(out) if changed else None

    def _sanitize_links(self, links: Sequence[Link]) -> tuple[Link, ...] | None:
        if not links:
            return None
        changed = False
        out: list[Link] = []
        for link in links:
            new_attributes = self._sanitize_mapping(link.attributes)
            if new_attributes is None:
                out.append(link)
                continue
            changed = True
            out.append(Link(link.context, new_attributes))
        return tuple(out) if changed else None

    def _sanitize_mapping(
        self,
        attributes: Any,
        *,
        extra_content_keys: frozenset[str] = frozenset(),
    ) -> dict[str, Any] | None:
        """Sanitized copy of an attribute mapping, or None when unchanged."""
        if not attributes:
            return None
        keys = [key for key in attributes if _is_content_key(key) or key in extra_content_keys]
        if not keys:
            return None

        new_attributes = dict(attributes)
        for key in keys:
            if not self._capture_content:
                del new_attributes[key]
                continue
            assert self._mask is not None  # guarded in export()
            try:
                if self._mask_accepts_key:
                    raw = self._mask(new_attributes[key], key=key)
                else:
                    raw = self._mask(new_attributes[key])
                masked = self._safe_value(raw)
            except Exception:
                # Fail closed: a broken mask must neither leak the unmasked
                # value nor take down the whole batch.
                masked = None
                logger.warning(
                    "mask callable raised for attribute %r; value dropped",
                    key,
                    exc_info=True,
                )
            if masked is None:
                del new_attributes[key]
            else:
                new_attributes[key] = masked
        return new_attributes

    @staticmethod
    def _safe_value(value: Any) -> Any:
        """Coerce a mask's return into something OTel can encode, or None to drop.

        BoundedAttributes-style cleaning silently refuses invalid values, which
        would leave the ORIGINAL in place, so we validate ourselves.
        """
        if value is None or isinstance(value, _PRIMITIVES):
            return value
        try:
            return serialize(value)
        except Exception:
            return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()
