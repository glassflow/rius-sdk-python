"""Swapping a span's attributes on an export-time copy without hiding a drop.

The exporters that rewrite a span (normalization, masking, workspace routing)
work on a ``copy.copy`` of it and assign a plain dict to ``_attributes``. The
OTel SDK reads ``dropped_attributes`` off that object, and only a
``BoundedAttributes`` carries a count, so a plain dict reports 0. A span that
hit the attribute-count limit would then leave the process claiming it lost
nothing, which is exactly when the count matters. The same holds for a span's
events and links, which the media strip rebuilds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from opentelemetry.attributes import BoundedAttributes


class _Dropping(Protocol):
    """A span, event or link: anything that carries attributes and may drop some."""

    @property
    def attributes(self) -> Any: ...


def _dropped_count(item: _Dropping) -> int:
    # OTel SDKs before 1.25 (events) and 1.27 (links) have no
    # ``dropped_attributes`` property; the count lives only on the
    # BoundedAttributes the SDK built.
    count = getattr(item, "dropped_attributes", None)
    if count is None:
        attributes = item.attributes
        count = attributes.dropped if isinstance(attributes, BoundedAttributes) else 0
    return int(count)


def replacement_attributes(span: _Dropping, attributes: Mapping[str, Any]) -> Mapping[str, Any]:
    """``attributes``, holding the original's dropped count if it had one.

    One that dropped nothing keeps the plain mapping, so the common case
    costs nothing. Only one that already lost keys pays for the rebuild.
    The count is the OTel SDK's own and is carried unchanged: keys a rewrite
    removed on purpose (a mapped source, a masked value) were not dropped.
    """
    dropped = _dropped_count(span)
    if not dropped:
        return attributes
    return _carrying(attributes, dropped)


def rebuilt_item_attributes(
    item: _Dropping | None, attributes: Mapping[str, Any]
) -> BoundedAttributes:
    """Attributes for an event or link rebuilt at export time.

    Always a ``BoundedAttributes``, unlike ``replacement_attributes``: the
    OTLP encoder of OTel before 1.25 reads ``attributes.dropped`` straight off
    every event and link, so a plain dict there fails the whole batch. ``item``
    is the event or link being replaced, whose dropped count is carried; pass
    ``None`` for a new one.
    """
    return _carrying(attributes, 0 if item is None else _dropped_count(item))


def _carrying(attributes: Mapping[str, Any], dropped: int) -> BoundedAttributes:
    carried = BoundedAttributes(attributes=attributes, immutable=True)
    carried.dropped = dropped
    return carried
