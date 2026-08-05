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

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

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
        attributes = span.attributes
        if not attributes:
            return span
        keys = [key for key in attributes if _is_content_key(key)]
        if not keys:
            return span

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

        sanitized = copy.copy(span)
        sanitized._attributes = new_attributes
        return sanitized

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
