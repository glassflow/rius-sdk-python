"""Credential headers never leave the process, whatever the content settings.

A header is not content: masking every header would hide harmless ones
(content-type, user-agent), and with ``capture_content=True`` and no mask a
credential would still ship. So a key naming a credential header is dropped
from every span by an exporter that is always in the chain, and pending
snapshots go through the same chain.

Works on copies, as masking does: a ``ReadableSpan`` shares its attribute
mapping with every processor on the provider. Each rebuilt span, event and
link keeps the OTel SDK's dropped-attributes count.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link

from ._attributes import rebuilt_item_attributes, replacement_attributes
from .semconv import (
    CREDENTIAL_HEADER_NAME_SUFFIXES,
    CREDENTIAL_HEADER_NAMES,
    HEADER_ATTRIBUTE_SEGMENTS,
)

_NAMES = frozenset(CREDENTIAL_HEADER_NAMES)


def is_credential_header_key(key: str) -> bool:
    """True when ``key`` holds a credential header.

    A key segment from HEADER_ATTRIBUTE_SEGMENTS (``ai.request.headers.<name>``,
    ``http.request.header.<name>``), followed by a name in
    CREDENTIAL_HEADER_NAMES or ending in one of CREDENTIAL_HEADER_NAME_SUFFIXES.
    Compared lower-cased with "_" read as "-": the AI SDK keeps the caller's
    spelling on some operations, and OpenTelemetry Python's HTTP
    instrumentations write ``x_api_key``. The name is what follows the
    segment's last occurrence.
    """
    lowered = key.lower()
    for segment in HEADER_ATTRIBUTE_SEGMENTS:
        at = lowered.rfind(segment)
        if at < 0:
            continue
        name = lowered[at + len(segment) :].replace("_", "-")
        if name in _NAMES or name.endswith(CREDENTIAL_HEADER_NAME_SUFFIXES):
            return True
    return False


def _without_credentials(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A copy without the credential headers, or None when there are none."""
    if not attributes or not any(is_credential_header_key(k) for k in attributes):
        return None
    return {k: v for k, v in attributes.items() if not is_credential_header_key(k)}


class CredentialHeaderDroppingExporter(SpanExporter):
    """Drops credential headers from span, event and link attributes."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._dropped(span) for span in spans])

    @staticmethod
    def _dropped(span: ReadableSpan) -> ReadableSpan:
        attributes = _without_credentials(span.attributes)
        events: list[Event] | None = None
        for i, event in enumerate(span.events):
            kept = _without_credentials(event.attributes)
            if kept is None:
                continue
            events = events if events is not None else list(span.events)
            events[i] = Event(event.name, rebuilt_item_attributes(event, kept), event.timestamp)
        links: list[Link] | None = None
        for i, link in enumerate(span.links):
            kept = _without_credentials(link.attributes)
            if kept is None:
                continue
            links = links if links is not None else list(span.links)
            links[i] = Link(link.context, rebuilt_item_attributes(link, kept))
        if attributes is None and events is None and links is None:
            return span
        dropped = copy.copy(span)
        if attributes is not None:
            dropped._attributes = replacement_attributes(span, attributes)
        if events is not None:
            dropped._events = tuple(events)
        if links is not None:
            dropped._links = tuple(links)
        return dropped

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()
