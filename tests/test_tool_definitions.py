"""Tool definitions recorded in the GenAI conventions' shape.

A ``function`` tool must be ``{"type": "function", "name", "description"?,
"parameters"?}``: the conventions' schema no longer lets a ``function`` tool
that misses that shape fall back to the generic definition. A provider's
built-in tool keeps its own fields, with its ``type`` prefixed by the
provider name (``anthropic.web_search_20250305``).
"""

from __future__ import annotations

from typing import Any

import pytest

from rius._tool_definitions import semconv_tool_definitions

_PARAMETERS = {"type": "object", "properties": {"city": {"type": "string"}}}


def test_openai_nested_function_is_flattened() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": _PARAMETERS,
                "strict": True,
            },
        }
    ]
    assert semconv_tool_definitions(tools, "openai") == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": _PARAMETERS,
            "strict": True,
        }
    ]


def test_anthropic_custom_tool_becomes_a_function_with_parameters() -> None:
    tools = [{"name": "search_kb", "description": "Search", "input_schema": _PARAMETERS}]
    assert semconv_tool_definitions(tools, "anthropic") == [
        {
            "type": "function",
            "name": "search_kb",
            "description": "Search",
            "parameters": _PARAMETERS,
        }
    ]


def test_anthropic_explicit_custom_type_becomes_a_function() -> None:
    tools = [{"type": "custom", "name": "search_kb", "input_schema": _PARAMETERS}]
    assert semconv_tool_definitions(tools, "anthropic") == [
        {"type": "function", "name": "search_kb", "parameters": _PARAMETERS}
    ]


def test_legacy_openai_function_without_type_becomes_a_function() -> None:
    tools = [{"name": "get_weather", "parameters": _PARAMETERS}]
    assert semconv_tool_definitions(tools, None) == [
        {"type": "function", "name": "get_weather", "parameters": _PARAMETERS}
    ]


def test_flat_function_with_camel_case_input_schema_names_it_parameters() -> None:
    # The Vercel AI SDK's function tool spells the schema inputSchema.
    tools = [{"type": "function", "name": "get_weather", "inputSchema": _PARAMETERS}]
    assert semconv_tool_definitions(tools, "openai") == [
        {"type": "function", "name": "get_weather", "parameters": _PARAMETERS}
    ]


def test_semconv_function_tool_is_unchanged() -> None:
    tool = {"type": "function", "name": "get_weather", "parameters": _PARAMETERS}
    assert semconv_tool_definitions([tool], "openai") == [tool]


def test_provider_built_in_tool_type_is_prefixed() -> None:
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    assert semconv_tool_definitions(tools, "anthropic") == [
        {"type": "anthropic.web_search_20250305", "name": "web_search", "max_uses": 5}
    ]


def test_built_in_tool_without_a_name_gets_none_invented() -> None:
    assert semconv_tool_definitions([{"type": "web_search"}], "openai") == [
        {"type": "openai.web_search"}
    ]


def test_built_in_tool_is_left_unprefixed_without_a_provider() -> None:
    tool = {"type": "web_search_20250305", "name": "web_search"}
    assert semconv_tool_definitions([tool], None) == [tool]
    assert semconv_tool_definitions([tool], "") == [tool]


def test_an_already_namespaced_type_is_not_prefixed_again() -> None:
    tool = {"type": "anthropic.web_search_20250305", "name": "web_search"}
    assert semconv_tool_definitions([tool], "anthropic") == [tool]
    # A provider whose name has dots of its own does not stack a second prefix.
    assert semconv_tool_definitions([tool], "azure.ai.openai") == [tool]


def test_vercel_provider_tool_takes_its_namespaced_id_as_type() -> None:
    tool = {
        "type": "provider-defined",
        "id": "anthropic.web_search_20250305",
        "name": "web_search",
        "args": {},
    }
    assert semconv_tool_definitions([tool], "anthropic") == [
        {**tool, "type": "anthropic.web_search_20250305"}
    ]


@pytest.mark.parametrize("tool", ["plain", 3, None, ["x"], {"description": "no name or type"}])
def test_anything_unreadable_is_kept_as_it_is(tool: Any) -> None:
    assert semconv_tool_definitions([tool], "openai") == [tool]


def test_normalizing_twice_equals_once() -> None:
    tools = [
        {"type": "function", "function": {"name": "a", "parameters": _PARAMETERS}},
        {"name": "b", "input_schema": _PARAMETERS},
        {"type": "web_search_20250305", "name": "web_search"},
    ]
    once = semconv_tool_definitions(tools, "anthropic")
    assert semconv_tool_definitions(once, "anthropic") == once


def test_the_caller_list_is_not_mutated() -> None:
    tools = [{"name": "b", "input_schema": _PARAMETERS}, {"type": "web_search"}]
    semconv_tool_definitions(tools, "openai")
    assert tools == [{"name": "b", "input_schema": _PARAMETERS}, {"type": "web_search"}]
