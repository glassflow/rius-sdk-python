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


def test_bedrock_tool_spec_becomes_a_function_with_its_json_schema() -> None:
    # Bedrock Converse wraps the function in toolSpec and the schema in
    # inputSchema.json.
    tools = [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Weather for a city",
                "inputSchema": {"json": _PARAMETERS},
            }
        }
    ]
    assert semconv_tool_definitions(tools, "aws.bedrock") == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": _PARAMETERS,
        }
    ]


def test_bedrock_tool_spec_without_a_name_or_a_cache_point_is_kept() -> None:
    nameless = {"toolSpec": {"inputSchema": {"json": _PARAMETERS}}}
    cache_point = {"cachePoint": {"type": "default"}}
    assert semconv_tool_definitions([nameless, cache_point], "aws.bedrock") == [
        nameless,
        cache_point,
    ]


@pytest.mark.parametrize(
    ("container", "schema"),
    [("function_declarations", "parameters"), ("functionDeclarations", "parametersJsonSchema")],
)
def test_gemini_declarations_become_one_function_tool_each(container: str, schema: str) -> None:
    tools = [
        {
            container: [
                {"name": "get_weather", "description": "Weather", schema: _PARAMETERS},
                {"name": "get_time"},
            ]
        },
        {"name": "lookup", "input_schema": _PARAMETERS},
    ]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Weather",
            "parameters": _PARAMETERS,
        },
        {"type": "function", "name": "get_time"},
        {"type": "function", "name": "lookup", "parameters": _PARAMETERS},
    ]


def test_gemini_snake_case_json_schema_is_named_parameters() -> None:
    tools = [{"function_declarations": [{"name": "f", "parameters_json_schema": _PARAMETERS}]}]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [
        {"type": "function", "name": "f", "parameters": _PARAMETERS}
    ]


def test_gemini_built_ins_beside_declarations_are_kept_without_a_name() -> None:
    # A built-in has no name of its own: inventing one would record a tool
    # that is offered but never called by that name.
    tools = [
        {"function_declarations": [{"name": "f"}], "google_search": {}},
        {"code_execution": {}},
    ]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [
        {"type": "function", "name": "f"},
        {"google_search": {}},
        {"code_execution": {}},
    ]


def test_gemini_declaration_without_a_name_is_kept_as_given() -> None:
    tools = [{"functionDeclarations": [{"description": "no name"}, "odd"]}]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [{"description": "no name"}, "odd"]


def test_gemini_container_that_is_not_a_list_is_kept() -> None:
    tool = {"function_declarations": {"name": "f"}}
    assert semconv_tool_definitions([tool], "gcp.gemini") == [tool]


@pytest.mark.parametrize("tool", ["plain", 3, None, ["x"], {"description": "no name or type"}])
def test_anything_unreadable_is_kept_as_it_is(tool: Any) -> None:
    assert semconv_tool_definitions([tool], "openai") == [tool]


def test_normalizing_twice_equals_once() -> None:
    tools = [
        {"type": "function", "function": {"name": "a", "parameters": _PARAMETERS}},
        {"name": "b", "input_schema": _PARAMETERS},
        {"type": "web_search_20250305", "name": "web_search"},
        {"toolSpec": {"name": "c", "inputSchema": {"json": _PARAMETERS}}},
        {"functionDeclarations": [{"name": "d", "parametersJsonSchema": _PARAMETERS}]},
        {"function_declarations": [{"name": "e"}], "google_search": {}},
    ]
    once = semconv_tool_definitions(tools, "anthropic")
    assert semconv_tool_definitions(once, "anthropic") == once


def test_the_caller_list_is_not_mutated() -> None:
    tools = [{"name": "b", "input_schema": _PARAMETERS}, {"type": "web_search"}]
    semconv_tool_definitions(tools, "openai")
    assert tools == [{"name": "b", "input_schema": _PARAMETERS}, {"type": "web_search"}]


# --- second-pass stability: every output is already in the conventions' shape


def test_every_list_valued_declarations_key_expands() -> None:
    tools = [{"function_declarations": [{"name": "a"}], "functionDeclarations": [{"name": "b"}]}]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [
        {"type": "function", "name": "a"},
        {"type": "function", "name": "b"},
    ]


def test_a_declarations_spelling_that_is_not_a_list_stays_in_the_rest() -> None:
    tools = [{"function_declarations": [{"name": "a"}], "functionDeclarations": "odd"}]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [
        {"type": "function", "name": "a"},
        {"functionDeclarations": "odd"},
    ]


@pytest.mark.parametrize("spelling", ["input_schema", "inputSchema"])
def test_a_declaration_schema_in_any_spelling_is_named_parameters(spelling: str) -> None:
    tools = [{"function_declarations": [{"name": "a", spelling: _PARAMETERS}]}]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [
        {"type": "function", "name": "a", "parameters": _PARAMETERS}
    ]


def test_a_nameless_declaration_takes_the_single_tool_rule() -> None:
    tools = [{"function_declarations": [{"type": "web_search"}]}]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [{"type": "gcp.gemini.web_search"}]


def test_the_rest_of_a_gemini_tool_takes_the_single_tool_rule() -> None:
    tools = [{"function_declarations": [{"name": "a"}], "name": "x"}]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [
        {"type": "function", "name": "a"},
        {"type": "function", "name": "x"},
    ]


def test_a_nested_function_is_not_lifted_over_a_top_level_name() -> None:
    named = {"name": "x", "function": {"name": "y"}}
    typed = {"type": "function", "name": "x", "function": {"name": "y"}}
    assert semconv_tool_definitions([named, typed], "openai") == [
        {"type": "function", "name": "x", "function": {"name": "y"}},
        typed,
    ]


def test_a_null_parameters_gives_way_to_a_schema_spelling() -> None:
    # A Python model_dump() of a google-genai FunctionDeclaration.
    tools = [
        {
            "function_declarations": [
                {"name": "f", "parameters": None, "parameters_json_schema": _PARAMETERS}
            ]
        },
        {"name": "g", "parameters": None, "input_schema": _PARAMETERS},
    ]
    assert semconv_tool_definitions(tools, "gcp.gemini") == [
        {"type": "function", "name": "f", "parameters": _PARAMETERS},
        {"type": "function", "name": "g", "parameters": _PARAMETERS},
    ]


def test_a_null_parameters_without_another_schema_is_kept() -> None:
    tool = {"type": "function", "name": "f", "parameters": None}
    assert semconv_tool_definitions([tool], "openai") == [tool]


def test_an_empty_type_gets_no_provider_prefix() -> None:
    tool = {"type": "", "size": 1}
    assert semconv_tool_definitions([tool], "openai") == [tool]


def test_mcp_tool_headers_are_dropped() -> None:
    # OpenAI Responses' MCP tool carries the server's request headers, which
    # can hold an Authorization credential.
    tool = {
        "type": "mcp",
        "server_label": "docs",
        "server_url": "https://mcp.example.com",
        "headers": {"Authorization": "Bearer secret"},
        "require_approval": "never",
    }
    assert semconv_tool_definitions([tool], "openai") == [
        {
            "type": "openai.mcp",
            "server_label": "docs",
            "server_url": "https://mcp.example.com",
            "require_approval": "never",
        }
    ]
    namespaced = {"type": "openai.mcp", "server_label": "docs", "headers": {"X-Key": "k"}}
    assert semconv_tool_definitions([namespaced], None) == [
        {"type": "openai.mcp", "server_label": "docs"}
    ]
    assert semconv_tool_definitions([{"type": "mcp", "headers": {}}], None) == [{"type": "mcp"}]
