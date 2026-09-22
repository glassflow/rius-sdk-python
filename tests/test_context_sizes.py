"""Unit tests for the pure ``rius.context.sizes`` computation."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from rius._context_sizes import (
    DETAIL_WINDOW,
    MAX_SIZES_BYTES,
    SIZES_VERSION,
    context_sizes,
)
from rius.generation import normalize_messages


def _sizes(
    tools: list[Any] | None = None,
    input_messages: list[Any] | None = None,
    output_messages: list[Any] | None = None,
) -> dict[str, Any]:
    """Normalize raw messages like the generation handle does and decode the result."""
    normalized_in = normalize_messages(input_messages, "user") if input_messages else None
    normalized_out = normalize_messages(output_messages, "assistant") if output_messages else None
    result = context_sizes(tools, normalized_in, normalized_out)
    decoded: dict[str, Any] = json.loads(result)
    return decoded


def _canonical_bytes(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


def _text(size: int) -> dict[str, Any]:
    return {"type": "text", "bytes": size}


def _message(role: str, *parts: Any) -> dict[str, Any]:
    return {"role": role, "parts": list(parts)}


# --- constants and shape ---


def test_constants() -> None:
    assert DETAIL_WINDOW == 50
    assert MAX_SIZES_BYTES == 8192
    assert SIZES_VERSION == 1


def test_empty_arguments_yield_only_version() -> None:
    assert context_sizes(None, None, None) == '{"version":1}'


def test_output_is_compact_json_with_fixed_key_order() -> None:
    raw = context_sizes(
        [{"name": "a"}],
        normalize_messages([{"role": "user", "content": "x"}], "user"),
        normalize_messages(["y"], "assistant"),
    )
    assert " " not in raw
    assert list(json.loads(raw).keys()) == [
        "version",
        "tool_definitions",
        "input_messages",
        "output_messages",
    ]


def test_sections_present_only_when_set() -> None:
    assert "tool_definitions" not in _sizes(input_messages=["hi"])
    assert "output_messages" not in _sizes(input_messages=["hi"])
    assert "input_messages" not in _sizes(output_messages=["hi"])
    assert _sizes(tools=[])["tool_definitions"] == []  # an empty list is still "set"


# --- tools ---


def test_tool_names_openai_anthropic_and_unnamed() -> None:
    openai_tool = {"type": "function", "function": {"name": "query", "parameters": {}}}
    anthropic_tool = {"name": "status", "input_schema": {}}
    unnamed_tool = {"type": "web_search", "max_uses": 3}
    sizes = _sizes(tools=[openai_tool, anthropic_tool, unnamed_tool])
    assert sizes["tool_definitions"] == [
        {"name": "query", "bytes": _canonical_bytes(openai_tool)},
        {"name": "status", "bytes": _canonical_bytes(anthropic_tool)},
        {"name": None, "bytes": _canonical_bytes(unnamed_tool)},
    ]


def test_non_dict_tool_is_measured_with_null_name() -> None:
    assert _sizes(tools=["plain"])["tool_definitions"] == [
        {"name": None, "bytes": _canonical_bytes("plain")}
    ]


# --- roles and text parts ---


def test_roles_are_literal_strings() -> None:
    messages = [
        {"role": "system", "content": "a"},
        {"role": "developer", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "tool", "content": "e"},
        {"role": "function", "content": "f"},
        {"role": "moderator", "content": "g"},
        {"role": 7, "content": "h"},
    ]
    roles = [entry["role"] for entry in _sizes(input_messages=messages)["input_messages"]]
    assert roles == [
        "system",
        "developer",
        "user",
        "assistant",
        "tool",
        "function",
        "moderator",
        "7",
    ]


def test_text_part_is_utf8_byte_length_not_character_count() -> None:
    text = "状態を確認して 🚨"
    assert len(text) != len(text.encode())
    assert _sizes(input_messages=[text])["input_messages"] == [
        _message("user", _text(len(text.encode())))
    ]


def test_non_string_text_content_uses_canonical_bytes() -> None:
    message = {"role": "user", "parts": [{"type": "text", "content": {"k": "v", "n": [1, 2]}}]}
    assert _sizes(input_messages=[message])["input_messages"] == [
        _message("user", _text(_canonical_bytes({"k": "v", "n": [1, 2]})))
    ]


def test_message_with_several_parts_lists_each() -> None:
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "cde"}],
    }
    assert _sizes(input_messages=[message])["input_messages"] == [
        _message("user", _text(2), _text(3))
    ]


# --- tool calls and responses ---


def test_tool_call_carries_the_tool_name_not_an_index() -> None:
    tools = [{"name": "first"}, {"function": {"name": "second"}}]
    message = {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "function": {"name": "second", "arguments": "{}"}}],
    }
    sizes = _sizes(tools=tools, input_messages=[message])
    part = {"type": "tool_call", "id": "c1", "name": "second", "arguments": "{}"}
    assert sizes["input_messages"] == [
        _message(
            "assistant", {"type": "tool_call", "tool": "second", "bytes": _canonical_bytes(part)}
        )
    ]


def test_tool_call_to_an_unlisted_tool_still_carries_its_name() -> None:
    message = {
        "role": "assistant",
        "content": "x",
        "tool_calls": [{"id": "c1", "function": {"name": "mystery", "arguments": "{}"}}],
    }
    sizes = _sizes(tools=[{"name": "known"}], input_messages=[message])
    assert sizes["input_messages"][0]["parts"][1]["tool"] == "mystery"


def test_tool_call_without_name_uses_null() -> None:
    message = {"role": "assistant", "parts": [{"type": "tool_call", "id": "c1"}]}
    part = _sizes(input_messages=[message])["input_messages"][0]["parts"][0]
    assert part["type"] == "tool_call"
    assert part["tool"] is None


def test_tool_call_with_non_string_name_uses_null() -> None:
    message = {"role": "assistant", "parts": [{"type": "tool_call", "id": "c1", "name": 3}]}
    assert _sizes(input_messages=[message])["input_messages"][0]["parts"][0]["tool"] is None


def test_tool_call_response_resolves_name_via_id() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "status"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    sizes = _sizes(input_messages=messages)
    response_part = {"type": "tool_call_response", "id": "c1", "response": "ok"}
    assert sizes["input_messages"][1] == _message(
        "tool",
        {"type": "tool_call_response", "tool": "status", "bytes": _canonical_bytes(response_part)},
    )


def test_tool_call_response_with_unknown_id_is_null() -> None:
    messages = [{"role": "tool", "tool_call_id": "nope", "content": "ok"}]
    part = _sizes(input_messages=messages)["input_messages"][0]["parts"][0]
    assert part["type"] == "tool_call_response"
    assert part["tool"] is None


def test_tool_call_response_without_id_is_null() -> None:
    message = {"role": "tool", "parts": [{"type": "tool_call_response", "response": "x"}]}
    assert _sizes(input_messages=[message])["input_messages"][0]["parts"][0]["tool"] is None


def test_tool_call_id_map_spans_input_then_output() -> None:
    # A response in the output can refer to a call made in the input, and a
    # response in the input can refer to a call that only appears in the output
    # (the map is built over everything before any part is measured).
    input_messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "alpha"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "from output call"},
    ]
    output_messages = [
        {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "beta"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "from input call"},
    ]
    sizes = _sizes(input_messages=input_messages, output_messages=output_messages)
    assert sizes["input_messages"][1]["parts"][0]["tool"] == "beta"
    assert sizes["output_messages"][1]["parts"][0]["tool"] == "alpha"


def test_input_call_wins_over_output_call_for_same_id() -> None:
    input_messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "alpha"}}]},
    ]
    output_messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "beta"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    sizes = _sizes(input_messages=input_messages, output_messages=output_messages)
    assert sizes["output_messages"][1]["parts"][0]["tool"] == "alpha"


def test_orphan_tool_result_without_id_is_a_text_part() -> None:
    # The normalizer turns a role=tool message without tool_call_id into a plain
    # text part, so it is measured as text under the literal "tool" role.
    messages = [{"role": "tool", "content": "orphan"}]
    assert _sizes(input_messages=messages)["input_messages"] == [
        _message("tool", _text(len(b"orphan")))
    ]


# --- non-text parts ---


def test_non_text_parts_are_marked_by_type_without_size() -> None:
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "audio", "data": "..."},
        ],
    }
    assert _sizes(input_messages=[message])["input_messages"] == [
        _message("user", _text(4), {"type": "image_url"}, {"type": "audio"})
    ]


def test_part_with_missing_or_non_string_type_is_unknown() -> None:
    message = {"role": "user", "parts": [{"foo": 1}, {"type": 3}, "not a dict"]}
    assert _sizes(input_messages=[message])["input_messages"] == [
        _message("user", {"type": "unknown"}, {"type": "unknown"}, {"type": "unknown"})
    ]


def test_message_without_parts_list_has_empty_parts() -> None:
    assert _sizes(input_messages=[{"role": "user", "parts": "bogus"}])["input_messages"] == [
        _message("user")
    ]


# --- cache marker ---


def test_cache_marker_points_at_last_message_with_cache_control() -> None:
    messages = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "a", "cache_control": {"type": "e"}}],
        },
        {"role": "user", "parts": [{"type": "text", "content": "b"}]},
        {
            "role": "user",
            "parts": [{"type": "text", "content": "c", "cache_control": {"type": "e"}}],
        },
        {"role": "user", "parts": [{"type": "text", "content": "d"}]},
    ]
    sizes = _sizes(input_messages=messages)
    assert sizes["cache_marker"] == 2
    assert list(sizes.keys()) == ["version", "input_messages", "cache_marker"]


def test_cache_marker_ignores_null_cache_control_and_output() -> None:
    input_messages = [
        {"role": "user", "parts": [{"type": "text", "content": "a", "cache_control": None}]}
    ]
    output_messages = [
        {"role": "assistant", "parts": [{"type": "text", "content": "b", "cache_control": {}}]}
    ]
    sizes = _sizes(input_messages=input_messages, output_messages=output_messages)
    assert "cache_marker" not in sizes


def test_cache_marker_on_non_text_part_counts() -> None:
    messages = [
        {"role": "user", "parts": [{"type": "image", "cache_control": {"type": "ephemeral"}}]},
    ]
    assert _sizes(input_messages=messages)["cache_marker"] == 0


# --- folding ---


def _conversation(count: int) -> list[dict[str, Any]]:
    """``count`` messages cycling system / user / assistant(+call) / tool / image."""
    messages: list[dict[str, Any]] = []
    for index in range(count):
        kind = index % 5
        if kind == 0:
            messages.append({"role": "system", "content": "s" * 10})
        elif kind == 1:
            messages.append({"role": "user", "content": "u" * 20})
        elif kind == 2:
            messages.append(
                {
                    "role": "assistant",
                    "content": "a" * 30,
                    "tool_calls": [
                        {"id": f"c{index}", "function": {"name": "search", "arguments": "{}"}}
                    ],
                }
            )
        elif kind == 3:
            messages.append({"role": "tool", "tool_call_id": f"c{index - 1}", "content": "r" * 40})
        else:
            messages.append(
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
            )
    return messages


def test_no_folding_at_exactly_the_detail_window() -> None:
    sizes = _sizes(input_messages=_conversation(DETAIL_WINDOW))
    assert len(sizes["input_messages"]) == DETAIL_WINDOW
    assert "folded" not in sizes


def test_folding_keeps_the_last_window_and_aggregates_the_rest() -> None:
    total = DETAIL_WINDOW + 10
    messages = _conversation(total)
    sizes = _sizes(tools=[{"name": "search"}], input_messages=messages)
    assert len(sizes["input_messages"]) == DETAIL_WINDOW
    assert list(sizes.keys()) == ["version", "tool_definitions", "input_messages", "folded"]

    # The folded prefix is messages 0..9: two of each kind.
    folded = sizes["folded"]
    assert list(folded.keys()) == [
        "messages",
        "system_bytes",
        "user_bytes",
        "assistant_bytes",
        "tools",
        "multimodal_parts",
    ]
    assert folded["messages"] == 10
    assert folded["system_bytes"] == 2 * 10
    assert folded["user_bytes"] == 2 * 20
    assert folded["assistant_bytes"] == 2 * 30
    assert folded["multimodal_parts"] == 2
    call_part_2 = {"type": "tool_call", "id": "c2", "name": "search", "arguments": "{}"}
    call_part_7 = {"type": "tool_call", "id": "c7", "name": "search", "arguments": "{}"}
    resp_part_3 = {"type": "tool_call_response", "id": "c2", "response": "r" * 40}
    resp_part_8 = {"type": "tool_call_response", "id": "c7", "response": "r" * 40}
    expected = sum(
        _canonical_bytes(p) for p in (call_part_2, call_part_7, resp_part_3, resp_part_8)
    )
    assert folded["tools"] == [{"tool": "search", "bytes": expected}]

    # The detail window is the tail, unchanged in shape.
    assert sizes["input_messages"][0] == _message("system", _text(10))  # message 10


def test_folded_summary_omits_zero_keys_and_keeps_message_count() -> None:
    messages = [{"role": "user", "content": "x"} for _ in range(DETAIL_WINDOW + 3)]
    folded = _sizes(input_messages=messages)["folded"]
    assert folded == {"messages": 3, "user_bytes": 3}


def test_folded_literal_and_tool_role_text_counts_as_user() -> None:
    messages = [{"role": "narrator", "content": "abcd"}, {"role": "tool", "content": "xy"}] + [
        {"role": "user", "content": "x"} for _ in range(DETAIL_WINDOW)
    ]
    assert _sizes(input_messages=messages)["folded"] == {"messages": 2, "user_bytes": 6}


def test_folded_developer_text_counts_as_system() -> None:
    messages = [{"role": "developer", "content": "abc"}] + [
        {"role": "user", "content": "x"} for _ in range(DETAIL_WINDOW)
    ]
    assert _sizes(input_messages=messages)["folded"] == {"messages": 1, "system_bytes": 3}


def test_folded_tool_sums_keep_first_seen_order_with_null_as_its_own_bucket() -> None:
    messages = [
        {"role": "assistant", "parts": [{"type": "tool_call", "id": "1", "name": "zeta"}]},
        {"role": "assistant", "parts": [{"type": "tool_call", "id": "2"}]},
        {"role": "assistant", "parts": [{"type": "tool_call", "id": "3", "name": "zeta"}]},
    ] + [{"role": "user", "content": "x"} for _ in range(DETAIL_WINDOW)]
    folded = _sizes(input_messages=messages)["folded"]
    zeta = _canonical_bytes({"type": "tool_call", "id": "1", "name": "zeta"}) + _canonical_bytes(
        {"type": "tool_call", "id": "3", "name": "zeta"}
    )
    unnamed = _canonical_bytes({"type": "tool_call", "id": "2"})
    assert folded["tools"] == [{"tool": "zeta", "bytes": zeta}, {"tool": None, "bytes": unnamed}]


def test_cache_marker_keeps_original_index_after_folding() -> None:
    messages: list[Any] = [{"role": "user", "content": "x"} for _ in range(DETAIL_WINDOW + 5)]
    cached = {"type": "text", "content": "x", "cache_control": {"type": "ephemeral"}}
    messages[2] = {"role": "user", "parts": [cached]}
    sizes = _sizes(input_messages=messages)
    assert sizes["cache_marker"] == 2
    assert sizes["folded"]["messages"] == 5


def test_output_messages_are_never_folded() -> None:
    sizes = _sizes(output_messages=[{"role": "assistant", "content": "x"}] * (DETAIL_WINDOW + 5))
    assert len(sizes["output_messages"]) == DETAIL_WINDOW + 5
    assert "folded" not in sizes


def test_refold_loop_shrinks_window_until_under_the_byte_cap() -> None:
    # ~600 messages, each with many tool parts, so even the last 50 messages in
    # full detail exceed 8 KB and the window must halve.
    messages: list[dict[str, Any]] = []
    for index in range(600):
        parts = [
            {"type": "tool_call", "id": f"c{index}-{n}", "name": f"tool_{n}", "arguments": "{}"}
            for n in range(12)
        ]
        messages.append({"role": "assistant", "parts": parts})
    raw = context_sizes(None, normalize_messages(messages, "user"), None)
    assert len(raw.encode()) <= MAX_SIZES_BYTES
    sizes = json.loads(raw)
    assert sizes["version"] == 1
    assert 0 < len(sizes["input_messages"]) < DETAIL_WINDOW
    assert sizes["folded"]["messages"] == 600 - len(sizes["input_messages"])


def test_refold_loop_bottoms_out_at_window_zero_without_truncating() -> None:
    # A single message that is too wide to ever fit: the window reaches 0, all
    # messages fold, and the attribute is still valid untruncated JSON.
    parts = [
        {"type": "tool_call", "id": f"c{n}", "name": f"tool_{n}", "arguments": "{}"}
        for n in range(1500)
    ]
    normalized = normalize_messages([{"role": "assistant", "parts": parts}], "user")
    sizes = json.loads(context_sizes(None, normalized, None))
    assert sizes["input_messages"] == []
    assert sizes["folded"]["messages"] == 1
    assert len(sizes["folded"]["tools"]) == 1500


# --- robustness ---


def test_never_raises_on_garbage_shapes() -> None:
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    raw = context_sizes(
        [None, 3, Weird()],
        [None, 3, {"role": None, "parts": [None, {"type": "text", "content": Weird()}]}],
        "not a list",  # deliberately the wrong type
    )
    sizes = json.loads(raw)
    assert sizes["version"] == 1
    assert len(sizes["tool_definitions"]) == 3


# --- cross-SDK parity ---

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_parity_fixture_matches_expected_output() -> None:
    # The same input fixture is run through the TypeScript SDK; both must
    # produce this exact string. On a first run the expected file is written
    # so it can be committed and diffed against the other SDK's result.
    fixture = json.loads((_FIXTURES / "context_sizes_input.json").read_text(encoding="utf-8"))
    result = context_sizes(
        fixture["tools"],
        normalize_messages(fixture["input"], "user"),
        normalize_messages(fixture["output"], "assistant"),
    )
    expected_path = _FIXTURES / "context_sizes_expected.json"
    if not expected_path.exists():
        expected_path.write_text(result + "\n", encoding="utf-8")
    assert result == expected_path.read_text(encoding="utf-8").rstrip("\n")
