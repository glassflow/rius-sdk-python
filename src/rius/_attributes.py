"""Swapping a span's attributes on an export-time copy without hiding a drop.

The exporters that rewrite a span (normalization, masking, workspace routing)
work on a ``copy.copy`` of it and assign a plain dict to ``_attributes``. The
OTel SDK reads ``dropped_attributes`` off that object, and only a
``BoundedAttributes`` carries a count, so a plain dict reports 0. A span that
hit the attribute-count limit would then leave the process claiming it lost
nothing, which is exactly when the count matters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import ReadableSpan


def replacement_attributes(span: ReadableSpan, attributes: Mapping[str, Any]) -> Mapping[str, Any]:
    """``attributes``, holding the original span's dropped count if it had one.

    A span that dropped nothing keeps the plain mapping, so the common case
    costs nothing. Only a span that already lost keys pays for the rebuild.
    The count is the OTel SDK's own and is carried unchanged: keys a rewrite
    removed on purpose (a mapped source, a masked value) were not dropped.
    """
    dropped = span.dropped_attributes
    if not dropped:
        return attributes
    carried = BoundedAttributes(attributes=attributes, immutable=True)
    carried.dropped = dropped
    return carried
