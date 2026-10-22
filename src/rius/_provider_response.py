"""Reading the provider's response object back out of an LLM span's ``output.value``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._serde import loads_object
from .semconv import OUTPUT_MIME_TYPE, OUTPUT_VALUE

JSON_MIME_TYPE = "application/json"

#: What ``POST /responses/compact`` answers with. That call is the compaction
#: itself, not a request that ran on a compacted context.
COMPACT_ENDPOINT_OBJECT = "response.compaction"


def response_json(attributes: Mapping[str, Any]) -> str | None:
    """``output.value`` when it can be the provider's JSON response object.

    ``output.mime_type``, when present, must be ``application/json``: the JS
    instrumentors record a streamed reply's TEXT there as text/plain, and a
    JSON-mode reply's ``{"id": ...}`` is then the model's words. Exact match,
    as in the TypeScript SDK; an absent type is still taken. A value that does
    not start with ``{`` is never parsed.
    """
    if OUTPUT_MIME_TYPE in attributes and attributes[OUTPUT_MIME_TYPE] != JSON_MIME_TYPE:
        return None
    raw = attributes.get(OUTPUT_VALUE)
    if not isinstance(raw, str) or not raw.startswith("{"):
        return None
    return raw


def is_compact_reply(raw: str) -> bool:
    """Whether ``raw`` is the reply of ``POST /responses/compact``."""
    if COMPACT_ENDPOINT_OBJECT not in raw:
        return False
    payload = loads_object(raw)
    return payload is not None and payload.get("object") == COMPACT_ENDPOINT_OBJECT
