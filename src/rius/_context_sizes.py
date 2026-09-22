"""Per-part byte sizes of a generation's context (``rius.context.sizes``).

Every generation span carries one compact JSON attribute describing the
UTF-8 byte size of each piece of context the SDK serialized: each tool
definition, each input and output message part, and where the prompt-cache
marker sits. The backend uses it to attribute ``gen_ai.usage.input_tokens``
across those parts.

Sizes are computed from the NORMALIZED messages BEFORE the content
attributes are truncated at their cap, and the attribute is not content:
it survives ``capture_content=False`` and ``mask`` untouched, so the
attribution stays correct when the messages themselves do not reach the
backend in full. The wire shape is shared with the TypeScript SDK; both must
produce byte-identical strings for the same input (see the parity fixture).

Wire shape, version 1 (only the present keys, in this order; compact JSON,
readable through its keys rather than whitespace)::

    {
        "version": 1,
        "tool_definitions": [{"name": str | null, "bytes": int}, ...],
        "input_messages": [{"role": str, "parts": [part, ...]}, ...],
        "output_messages": [{"role": str, "parts": [part, ...]}, ...],
        "cache_marker": int,
        "folded": {
            "messages": int,
            "system_bytes": int,
            "user_bytes": int,
            "assistant_bytes": int,
            "tools": [{"tool": str | null, "bytes": int}, ...],
            "multimodal_parts": int,
        },
    }

``cache_marker`` is the index, into the original input list, of the last
message with a non-null ``cache_control`` part. ``folded`` summarizes the
input messages older than the detail window (the last ``DETAIL_WINDOW``
messages keep per-part detail); the window halves while the whole string
exceeds ``MAX_SIZES_BYTES``, and the attribute is never truncated.

A part is ``{"type": "text", "bytes"}``, ``{"type": "tool_call", "tool",
"bytes"}``, ``{"type": "tool_call_response", "tool", "bytes"}`` or, for any
other part type, ``{"type": <type>}`` with no size. Roles are the literal
role strings of the normalized messages; ``tool`` is the tool NAME (resolved
through the call id for responses), ``null`` when unknown.

This function runs on the user's request path, so it never raises: any
unexpected shape yields a best-effort entry instead.
"""

from __future__ import annotations

import json
from typing import Any

DETAIL_WINDOW = 50
MAX_SIZES_BYTES = 8192
SIZES_VERSION = 1

# Text bytes of folded messages are bucketed by role: system-side, assistant
# and everything else (user, tool results, unknown roles) as user-side.
_SYSTEM_ROLES = ("system", "developer")
_ASSISTANT_ROLES = ("assistant",)


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=repr)


def _canonical_bytes(value: Any) -> int:
    """UTF-8 byte length of ``value``'s compact JSON; ``0`` if it cannot be encoded."""
    try:
        return len(_dumps(value).encode())
    except Exception:
        return 0


def _text_bytes(content: Any) -> int:
    if isinstance(content, str):
        return len(content.encode())
    return _canonical_bytes(content)


def _tool_name(tool: Any) -> str | None:
    """``function.name`` (OpenAI shape), else ``name`` (Anthropic / generic), else None."""
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        name: str = function["name"]
        return name
    name_value = tool.get("name")
    return name_value if isinstance(name_value, str) else None


def _role(message: Any) -> str:
    role = message.get("role") if isinstance(message, dict) else None
    return role if isinstance(role, str) else str(role)


def _parts(message: Any) -> list[Any]:
    if isinstance(message, dict):
        parts = message.get("parts")
        if isinstance(parts, list):
            return parts
    return []


def _part_type(part: Any) -> str:
    if isinstance(part, dict):
        part_type = part.get("type")
        if isinstance(part_type, str):
            return part_type
    return "unknown"


def _call_names(*message_lists: list[Any] | None) -> dict[str, str | None]:
    """Tool-call id -> tool name over every message, first occurrence wins.

    Response parts only carry the call id; the name comes from the call part
    with that id, wherever it is (input first, then output).
    """
    names: dict[str, str | None] = {}
    for messages in message_lists:
        for message in messages or ():
            for part in _parts(message):
                if _part_type(part) == "tool_call" and isinstance(part.get("id"), str):
                    name = part.get("name")
                    names.setdefault(part["id"], name if isinstance(name, str) else None)
    return names


def _part_entry(part: Any, call_names: dict[str, str | None]) -> dict[str, Any]:
    part_type = _part_type(part)
    if part_type == "text":
        return {"type": "text", "bytes": _text_bytes(part.get("content"))}
    if part_type == "tool_call":
        name = part.get("name")
        tool = name if isinstance(name, str) else None
        return {"type": "tool_call", "tool": tool, "bytes": _canonical_bytes(part)}
    if part_type == "tool_call_response":
        call_id = part.get("id")
        tool = call_names.get(call_id) if isinstance(call_id, str) else None
        return {"type": "tool_call_response", "tool": tool, "bytes": _canonical_bytes(part)}
    return {"type": part_type}


def _message_entry(message: Any, call_names: dict[str, str | None]) -> dict[str, Any]:
    return {
        "role": _role(message),
        "parts": [_part_entry(part, call_names) for part in _parts(message)],
    }


def _has_cache_marker(message: Any) -> bool:
    return any(
        isinstance(part, dict) and part.get("cache_control") is not None for part in _parts(message)
    )


def _fold(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate message entries that fall outside the detail window."""
    text = {"system_bytes": 0, "user_bytes": 0, "assistant_bytes": 0}
    tool_bytes: dict[str | None, int] = {}
    multimodal = 0
    for entry in entries:
        role = entry["role"]
        if role in _SYSTEM_ROLES:
            bucket = "system_bytes"
        elif role in _ASSISTANT_ROLES:
            bucket = "assistant_bytes"
        else:
            bucket = "user_bytes"
        for part in entry["parts"]:
            if part["type"] == "text":
                text[bucket] += part["bytes"]
            elif part["type"] in ("tool_call", "tool_call_response"):
                tool_bytes[part["tool"]] = tool_bytes.get(part["tool"], 0) + part["bytes"]
            else:
                multimodal += 1
    folded: dict[str, Any] = {"messages": len(entries)}
    for key in ("system_bytes", "user_bytes", "assistant_bytes"):
        if text[key]:
            folded[key] = text[key]
    if tool_bytes:
        folded["tools"] = [{"tool": tool, "bytes": size} for tool, size in tool_bytes.items()]
    if multimodal:
        folded["multimodal_parts"] = multimodal
    return folded


def _compose(
    tool_entries: list[dict[str, Any]] | None,
    input_entries: list[dict[str, Any]] | None,
    output_entries: list[dict[str, Any]] | None,
    cache_marker: int | None,
    window: int,
) -> str:
    sizes: dict[str, Any] = {"version": SIZES_VERSION}
    if tool_entries is not None:
        sizes["tool_definitions"] = tool_entries
    folded: dict[str, Any] | None = None
    if input_entries is not None:
        if len(input_entries) > window:
            cut = len(input_entries) - window
            folded = _fold(input_entries[:cut])
            sizes["input_messages"] = input_entries[cut:]
        else:
            sizes["input_messages"] = input_entries
    if output_entries is not None:
        sizes["output_messages"] = output_entries
    if cache_marker is not None:
        sizes["cache_marker"] = cache_marker
    if folded is not None:
        sizes["folded"] = folded
    return _dumps(sizes)


def _context_sizes(
    tools: list[Any] | None,
    input_messages: list[Any] | None,
    output_messages: list[Any] | None,
) -> str:
    if not isinstance(input_messages, list):
        input_messages = None
    if not isinstance(output_messages, list):
        output_messages = None
    call_names = _call_names(input_messages, output_messages)

    tool_entries = None
    if tools is not None:
        tool_entries = [
            {"name": _tool_name(tool), "bytes": _canonical_bytes(tool)} for tool in tools
        ]
    input_entries = None
    cache_marker = None
    if input_messages is not None:
        input_entries = [_message_entry(message, call_names) for message in input_messages]
        for index, message in enumerate(input_messages):
            if _has_cache_marker(message):
                cache_marker = index
    output_entries = None
    if output_messages is not None:
        output_entries = [_message_entry(message, call_names) for message in output_messages]

    # Detail is kept for the last DETAIL_WINDOW input messages; older ones are
    # folded into one summary. If the result still exceeds the byte cap the
    # window halves until it fits or nothing is left to fold, so the attribute
    # itself is never cut mid-JSON.
    window = DETAIL_WINDOW
    while True:
        result = _compose(tool_entries, input_entries, output_entries, cache_marker, window)
        if window == 0 or len(result.encode()) <= MAX_SIZES_BYTES:
            return result
        window //= 2


def context_sizes(
    tools: list[Any] | None,
    input_messages: list[Any] | None,
    output_messages: list[Any] | None,
) -> str:
    """Compact JSON describing the byte size of every context part.

    Args:
        tools: The tool definitions as passed by the caller (any provider
            shape), or ``None`` when none were set.
        input_messages: NORMALIZED input messages (``{"role", "parts"}``),
            or ``None`` when none were set.
        output_messages: NORMALIZED output messages, or ``None``.

    Returns:
        The ``rius.context.sizes`` attribute value. Never raises.
    """
    try:
        return _context_sizes(tools, input_messages, output_messages)
    except Exception:
        return _dumps({"version": SIZES_VERSION})
