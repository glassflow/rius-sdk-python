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
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link, Status

from ._serde import serialize
from .semconv import (
    CONTENT_ATTRIBUTE_PREFIXES,
    CONTENT_ATTRIBUTE_SUFFIXES,
    CONTENT_ATTRIBUTES,
    INVOCATION_PARAMETERS_CONTENT_MEMBERS,
    LLM_INVOCATION_PARAMETERS,
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


def _redact_invocation_parameters(value: Any) -> str | None:
    """The tools/functions members removed, the rest kept; None = drop it all.

    ``llm.invocation_parameters`` is not wholly content — sampling parameters
    are identity — but the litellm and langchain instrumentations embed the
    request's tool definitions inside it. Whenever sanitization runs (content
    capture off, or a mask installed), those members must not leave the
    process. An unparseable payload is dropped whole: it might hide tool
    definitions, and unreadable is exactly when a pass-through is wrong.
    """
    if not isinstance(value, str):
        return None
    try:
        parameters = json.loads(value)
    except ValueError:
        return None
    if not isinstance(parameters, dict):
        return None
    if not any(member in parameters for member in INVOCATION_PARAMETERS_CONTENT_MEMBERS):
        return value  # nothing sensitive; keep byte-identical
    for member in INVOCATION_PARAMETERS_CONTENT_MEMBERS:
        parameters.pop(member, None)
    return serialize(parameters)


# The namespace the OpenInference Vercel transform mirrors untranslated
# attributes into. A content key comes through twice there, and the mirrored
# copy is content exactly when the original is; the TypeScript SDK's sentinel
# test found metadata.gen_ai.system_instructions carrying the system prompt
# past capture_content=False. Mirrored here for parity: the transform is a
# TypeScript package today, but the rule costs nothing and the wire is shared.
_METADATA_PREFIX = "metadata."


def _is_content_key(key: str) -> bool:
    if (
        key in CONTENT_ATTRIBUTES
        or key.startswith(CONTENT_ATTRIBUTE_PREFIXES)
        or key.endswith(CONTENT_ATTRIBUTE_SUFFIXES)
    ):
        return True
    return key.startswith(_METADATA_PREFIX) and _is_content_key(key[len(_METADATA_PREFIX) :])


# record_exception() writes the provider's error string, and providers echo
# the rejected request into it, so these two carry the same content the
# attribute strip removes. exception.type deliberately stays: failures remain
# visible and classifiable with content capture off (same policy as the
# TypeScript SDK).
_EXCEPTION_EVENT_NAME = "exception"
_EXCEPTION_CONTENT_KEYS = frozenset({"exception.message", "exception.stacktrace"})
# The status description is the same string once more: observe()/MCP write
# str(exc) into it, and providers echo the rejected request. Not an attribute,
# so it needs its own pass; this is the key= a mask sees for it.
_STATUS_DESCRIPTION_KEY = "status.description"


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
        new_status = self._sanitize_status(span.status)
        if (
            new_attributes is None
            and new_events is None
            and new_links is None
            and new_status is None
        ):
            return span
        sanitized = copy.copy(span)
        if new_attributes is not None:
            sanitized._attributes = new_attributes
        if new_events is not None:
            sanitized._events = new_events
        if new_links is not None:
            sanitized._links = new_links
        if new_status is not None:
            sanitized._status = new_status
        return sanitized

    def _sanitize_status(self, status: Status | None) -> Status | None:
        """Status with its description dropped or masked, or None when unchanged.

        The code always survives: a failure stays visible and classifiable
        with content capture off, the same policy as ``exception.type``.
        """
        if status is None or not status.description:
            return None
        if not self._capture_content:
            return Status(status.status_code)
        assert self._mask is not None  # guarded in export()
        try:
            if self._mask_accepts_key:
                raw = self._mask(status.description, key=_STATUS_DESCRIPTION_KEY)
            else:
                raw = self._mask(status.description)
            masked = self._safe_value(raw)
        except Exception:
            masked = None
            logger.warning(
                "mask callable raised for the status description; value dropped", exc_info=True
            )
        if masked is None:
            return Status(status.status_code)
        return Status(status.status_code, str(masked))

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

        # Partial redaction, not the strip/mask below: the key mixes identity
        # (sampling params) with content (embedded tool definitions).
        invocation = attributes.get(LLM_INVOCATION_PARAMETERS)
        redacted_invocation = (
            _redact_invocation_parameters(invocation) if invocation is not None else None
        )
        invocation_changed = invocation is not None and redacted_invocation != invocation

        if not keys and not invocation_changed:
            return None

        new_attributes = dict(attributes)
        if invocation_changed:
            if redacted_invocation is None:
                del new_attributes[LLM_INVOCATION_PARAMETERS]
            else:
                new_attributes[LLM_INVOCATION_PARAMETERS] = redacted_invocation
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
