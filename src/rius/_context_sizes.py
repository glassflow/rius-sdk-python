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

Wire shape, version 1 (only the present keys, in this order)::

    {"v": 1,
     "t": [[name | null, bytes], ...],            # tool definitions
     "i": [[role, part, ...], ...],               # input messages
     "o": [[role, part, ...], ...],               # output messages
     "c": <index into i of the last cached message>,
     "f": {"n", "s", "u", "a", "t", "m"}}         # folded older input

A part is a bare integer (text bytes), ``["c", tool, bytes]`` (tool call),
``["r", tool, bytes]`` (tool call response) or ``["m", type]`` (any other
part type, unsized). ``tool`` is an index into ``t`` when the name matches
a definition, else the name, else ``null``.

This function runs on the user's request path, so it never raises: any
unexpected shape yields a best-effort entry instead.
"""

from __future__ import annotations

import json
from typing import Any

DETAIL_WINDOW = 50
MAX_SIZES_BYTES = 8192
SIZES_VERSION = 1

_ROLE_CODES = {
    "system": "s",
    "developer": "s",
    "user": "u",
    "assistant": "a",
    "tool": "t",
    "function": "t",
}

ToolRef = int | str | None


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


def _role_code(message: Any) -> str:
    role = message.get("role") if isinstance(message, dict) else None
    if isinstance(role, str):
        return _ROLE_CODES.get(role, role)
    return str(role)


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


class _Resolver:
    """Resolves tool names to ``t`` indexes and tool-call ids to tool names."""

    def __init__(self, tools: list[Any] | None, *message_lists: list[Any] | None) -> None:
        self._index_by_name: dict[str, int] = {}
        for index, tool in enumerate(tools or ()):
            name = _tool_name(tool)
            if name is not None:
                self._index_by_name.setdefault(name, index)
        # Response parts only carry the call id; the name comes from the call
        # part with that id, wherever it is (input first, then output).
        self._name_by_call_id: dict[str, Any] = {}
        for messages in message_lists:
            for message in messages or ():
                for part in _parts(message):
                    if _part_type(part) == "tool_call" and isinstance(part.get("id"), str):
                        self._name_by_call_id.setdefault(part["id"], part.get("name"))

    def ref(self, name: Any) -> ToolRef:
        if isinstance(name, str):
            return self._index_by_name.get(name, name)
        return None

    def ref_for_call_id(self, call_id: Any) -> ToolRef:
        if isinstance(call_id, str):
            return self.ref(self._name_by_call_id.get(call_id))
        return None


def _part_entry(part: Any, resolver: _Resolver) -> Any:
    part_type = _part_type(part)
    if part_type == "text":
        return _text_bytes(part.get("content"))
    if part_type == "tool_call":
        return ["c", resolver.ref(part.get("name")), _canonical_bytes(part)]
    if part_type == "tool_call_response":
        return ["r", resolver.ref_for_call_id(part.get("id")), _canonical_bytes(part)]
    return ["m", part_type]


def _message_entry(message: Any, resolver: _Resolver) -> list[Any]:
    return [_role_code(message), *(_part_entry(part, resolver) for part in _parts(message))]


def _has_cache_marker(message: Any) -> bool:
    return any(
        isinstance(part, dict) and part.get("cache_control") is not None for part in _parts(message)
    )


def _fold(entries: list[list[Any]]) -> dict[str, Any]:
    """Aggregate message entries that fall outside the detail window."""
    text: dict[str, int] = {"s": 0, "u": 0, "a": 0}
    tool_bytes: dict[ToolRef, int] = {}
    media = 0
    for entry in entries:
        role = entry[0]
        # Only system and assistant text is tracked separately; every other
        # role (user, tool results, uncoded roles) counts as user-side text.
        bucket = role if role in ("s", "a") else "u"
        for part in entry[1:]:
            if isinstance(part, int):
                text[bucket] += part
            elif part[0] in ("c", "r"):
                tool_bytes[part[1]] = tool_bytes.get(part[1], 0) + part[2]
            else:
                media += 1
    folded: dict[str, Any] = {"n": len(entries)}
    for key in ("s", "u", "a"):
        if text[key]:
            folded[key] = text[key]
    if tool_bytes:
        folded["t"] = [[ref, size] for ref, size in tool_bytes.items()]
    if media:
        folded["m"] = media
    return folded


def _compose(
    tool_entries: list[list[Any]] | None,
    input_entries: list[list[Any]] | None,
    output_entries: list[list[Any]] | None,
    cache_index: int | None,
    window: int,
) -> str:
    sizes: dict[str, Any] = {"v": SIZES_VERSION}
    if tool_entries is not None:
        sizes["t"] = tool_entries
    folded: dict[str, Any] | None = None
    if input_entries is not None:
        if len(input_entries) > window:
            cut = len(input_entries) - window
            folded = _fold(input_entries[:cut])
            sizes["i"] = input_entries[cut:]
        else:
            sizes["i"] = input_entries
    if output_entries is not None:
        sizes["o"] = output_entries
    if cache_index is not None:
        sizes["c"] = cache_index
    if folded is not None:
        sizes["f"] = folded
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
    resolver = _Resolver(tools, input_messages, output_messages)

    tool_entries = None
    if tools is not None:
        tool_entries = [[_tool_name(tool), _canonical_bytes(tool)] for tool in tools]
    input_entries = None
    cache_index = None
    if input_messages is not None:
        input_entries = [_message_entry(message, resolver) for message in input_messages]
        for index, message in enumerate(input_messages):
            if _has_cache_marker(message):
                cache_index = index
    output_entries = None
    if output_messages is not None:
        output_entries = [_message_entry(message, resolver) for message in output_messages]

    # Detail is kept for the last DETAIL_WINDOW input messages; older ones are
    # folded into one summary. If the result still exceeds the byte cap the
    # window halves until it fits or nothing is left to fold, so the attribute
    # itself is never cut mid-JSON.
    window = DETAIL_WINDOW
    while True:
        result = _compose(tool_entries, input_entries, output_entries, cache_index, window)
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
        return _dumps({"v": SIZES_VERSION})
