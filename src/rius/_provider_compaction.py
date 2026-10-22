"""Compactions a provider reports inside its response object.

Anthropic's context editing (``context_management.applied_edits``) and
compaction blocks, and OpenAI's compaction items, are facts about the request
that only the provider's serialized response holds. The OpenInference
instrumentors record that object whole under ``output.value`` and nothing
else, so the compactions are read back out of it, at export, as the attribute
sets of ``rius.compaction`` events.

Every shape is read defensively. The text is another party's JSON, a
different SDK version may add or drop a field, and a value of the wrong type
is not worth losing a span over: it is left out of the event, and a block or
edit that cannot be told apart from a failure is not reported at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple

from ._provider_response import COMPACT_ENDPOINT_OBJECT
from ._serde import text_value
from .compaction import dropped_tokens
from .semconv import (
    RIUS_COMPACTION_EXECUTOR,
    RIUS_COMPACTION_EXECUTOR_PROVIDER,
    RIUS_COMPACTION_ID,
    RIUS_COMPACTION_KIND,
    RIUS_COMPACTION_KIND_CLEAR_THINKING,
    RIUS_COMPACTION_KIND_CLEAR_TOOL_RESULTS,
    RIUS_COMPACTION_KIND_SUMMARIZE,
    RIUS_COMPACTION_SUMMARY_BYTES,
    RIUS_COMPACTION_SUMMARY_TEXT,
    RIUS_COMPACTION_TOKENS_AFTER,
    RIUS_COMPACTION_TOKENS_BEFORE,
    RIUS_COMPACTION_TOKENS_DROPPED,
    RIUS_COMPACTION_TOOL_RESULTS_CLEARED,
)

#: Present in every response that reports a compaction, which a response that
#: reports none almost never contains, so the common reply is never parsed. A
#: reply that merely mentions one of them is parsed and finds nothing.
_MARKERS = ('"compaction"', '"applied_edits"')

#: Anthropic names an edit by family and date (``clear_tool_uses_20250919``),
#: so a newer date of a known family is still that family.
_EDIT_KIND_BY_PREFIX = (
    ("clear_tool_uses", RIUS_COMPACTION_KIND_CLEAR_TOOL_RESULTS),
    ("clear_thinking", RIUS_COMPACTION_KIND_CLEAR_THINKING),
)

_COMPACTION_TYPE = "compaction"


class ProviderCompactions(NamedTuple):
    """What a response reports: the events, and whether the call ran on a compacted context.

    The reply of ``POST /responses/compact`` reports the compaction it
    performed, but it is the call that read the whole context, so it is not
    flagged.
    """

    events: list[dict[str, Any]]
    compacted_context: bool


_NONE = ProviderCompactions([], False)


def compactions_in_response(raw: str) -> ProviderCompactions:
    """The ``rius.compaction`` event attributes a provider response reports.

    ``raw`` is the text of a JSON object. One event per compaction, in the
    order the response lists them; none when it reports none.
    """
    if not any(marker in raw for marker in _MARKERS):
        return _NONE
    try:
        payload = json.loads(raw)
    except ValueError:
        return _NONE
    if not isinstance(payload, dict):
        return _NONE
    if payload.get("object") == COMPACT_ENDPOINT_OBJECT:
        return ProviderCompactions(list(_compact_endpoint_events(payload)), False)
    events = [*_edit_events(payload), *_summary_events(payload), *_item_events(payload)]
    return ProviderCompactions(events, bool(events))


def _provider_event(kind: str) -> dict[str, Any]:
    return {RIUS_COMPACTION_KIND: kind, RIUS_COMPACTION_EXECUTOR: RIUS_COMPACTION_EXECUTOR_PROVIDER}


def _members(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, list):
        for member in value:
            if isinstance(member, dict):
                yield member


def _size(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _set(event: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        event[key] = value


def _identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _edit_events(payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    management = payload.get("context_management")
    edits = management.get("applied_edits") if isinstance(management, dict) else None
    for edit in _members(edits):
        edit_type = edit.get("type")
        kind = next(
            (
                kind
                for prefix, kind in _EDIT_KIND_BY_PREFIX
                if isinstance(edit_type, str) and edit_type.startswith(prefix)
            ),
            None,
        )
        if kind is None:
            continue
        event = _provider_event(kind)
        _set(event, RIUS_COMPACTION_TOKENS_DROPPED, _size(edit.get("cleared_input_tokens")))
        if kind == RIUS_COMPACTION_KIND_CLEAR_TOOL_RESULTS:
            _set(event, RIUS_COMPACTION_TOOL_RESULTS_CLEARED, _size(edit.get("cleared_tool_uses")))
        yield event


def _summary_events(payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    usage = payload.get("usage")
    iterations = list(_members(usage.get("iterations") if isinstance(usage, dict) else None))
    # The compaction iteration is the one that read the context being
    # compacted. The size left behind is what the last sampling iteration read.
    before_sizes = [
        _size(iteration.get("input_tokens"))
        for iteration in iterations
        if iteration.get("type") == _COMPACTION_TYPE
    ]
    sampled = [it for it in iterations if it.get("type") != _COMPACTION_TYPE]
    after = _size(sampled[-1].get("input_tokens")) if sampled else None
    blocks = [b for b in _members(payload.get("content")) if b.get("type") == _COMPACTION_TYPE]
    for position, block in enumerate(blocks):
        summary = block.get("content")
        # No summary is a compaction that failed; the server treats the block
        # as a no-op, so nothing was compacted.
        if not isinstance(summary, str):
            continue
        before = before_sizes[position] if position < len(before_sizes) else None
        event = _provider_event(RIUS_COMPACTION_KIND_SUMMARIZE)
        _set(event, RIUS_COMPACTION_TOKENS_BEFORE, before)
        _set(event, RIUS_COMPACTION_TOKENS_AFTER, after)
        _set(event, RIUS_COMPACTION_TOKENS_DROPPED, dropped_tokens(before, after))
        event[RIUS_COMPACTION_SUMMARY_BYTES] = len(summary.encode("utf-8", "replace"))
        event[RIUS_COMPACTION_SUMMARY_TEXT] = text_value(summary)
        _set(event, RIUS_COMPACTION_ID, _identifier(block.get("id")))
        yield event


def _item_events(payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for item in _members(payload.get("output")):
        if item.get("type") != _COMPACTION_TYPE:
            continue
        event = _provider_event(RIUS_COMPACTION_KIND_SUMMARIZE)
        _set(event, RIUS_COMPACTION_ID, _identifier(item.get("id")))
        yield event


def _compact_endpoint_events(payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    usage = payload.get("usage")
    before = _size(usage.get("input_tokens")) if isinstance(usage, dict) else None
    for event in _item_events(payload):
        _set(event, RIUS_COMPACTION_TOKENS_BEFORE, before)
        yield event
