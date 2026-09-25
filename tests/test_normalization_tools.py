"""Tool identity and tool definitions on OpenInference spans.

Two different mechanisms, because the two facts are different categories:

* ``gen_ai.tool.name`` is IDENTITY (it rides pending snapshots), so it is a
  plain table rule from ``tool.name`` and is applied at span start as well as
  at export. There is deliberately NO fallback to the span name.
* ``gen_ai.tool.definitions`` is CONTENT, and its source is an indexed family
  (``llm.tools.N.tool.json_schema``) the exact-key rule table cannot express,
  so ``NormalizingSpanExporter`` reassembles it at export, like the event
  passes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.normalization import (
    DEFAULT_TABLE,
    NormalizingSpanExporter,
    normalize_tool_definitions,
)
from rius.semconv import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_TOOL_NAME,
    LLM_INVOCATION_PARAMETERS,
    RIUS_SPAN_PENDING,
)

# What openinference-instrumentation-openai 0.1.52 records for a chat request
# with function tools: _request_attributes_extractor.py:99-101 POPS `tools`
# out of the request bag and writes each one as
# llm.tools.{i}.tool.json_schema = safe_json_dumps(tool).
_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}},
    },
]

# openinference-instrumentation-anthropic 2.1.4, _wrappers.py:567-569: the
# request's tools, one json_schema per index, in Anthropic's own shape
# (top-level name / input_schema) — which must survive verbatim, not be
# rewritten into OpenAI's.
_ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def _indexed(tools: list[dict[str, Any]]) -> dict[str, str]:
    return {f"llm.tools.{i}.tool.json_schema": json.dumps(tool) for i, tool in enumerate(tools)}


def _raw_span(attributes: dict[str, Any], name: str = "ChatCompletion") -> ReadableSpan:
    """A finished span from a bare OTel provider, never touched by normalization.

    Not through ``init()``: its exporter chain already normalizes, so a span
    collected there would arrive pre-normalized and prove nothing.
    """
    collected = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collected))
    provider.get_tracer("test").start_span(name, attributes=attributes).end()
    (finished,) = collected.get_finished_spans()
    return finished


def _export(span: ReadableSpan) -> ReadableSpan:
    inner = InMemorySpanExporter()
    NormalizingSpanExporter(inner).export([span])
    return inner.get_finished_spans()[0]


def _exported_attributes(
    attributes: dict[str, Any], name: str = "ChatCompletion"
) -> dict[str, Any]:
    return dict(_export(_raw_span(attributes, name)).attributes or {})


def _definitions(attributes: dict[str, Any]) -> Any:
    return json.loads(attributes[GEN_AI_TOOL_DEFINITIONS])


# --- gen_ai.tool.definitions from llm.tools.N.tool.json_schema -----------


def test_openai_function_tools_become_one_definitions_array() -> None:
    raw = {
        "openinference.span.kind": "LLM",
        "llm.provider": "openai",
        LLM_INVOCATION_PARAMETERS: json.dumps({"model": "gpt-4o", "tool_choice": "auto"}),
        **_indexed(_OPENAI_TOOLS),
    }
    out = _exported_attributes(raw)
    assert _definitions(out) == _OPENAI_TOOLS
    assert out == {
        "openinference.span.kind": "LLM",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_PROVIDER_NAME: "openai",
        GEN_AI_REQUEST_MODEL: "gpt-4o",
        LLM_INVOCATION_PARAMETERS: '{"tool_choice":"auto"}',
        GEN_AI_TOOL_DEFINITIONS: out[GEN_AI_TOOL_DEFINITIONS],
    }
    assert not [key for key in out if key.startswith("llm.tools")]


def test_anthropic_input_schema_tools_survive_verbatim() -> None:
    raw = {
        "openinference.span.kind": "LLM",
        "llm.provider": "anthropic",
        "llm.request.model_name": "claude-sonnet-4-5",
        **_indexed(_ANTHROPIC_TOOLS),
    }
    out = _exported_attributes(raw)
    assert _definitions(out) == _ANTHROPIC_TOOLS
    assert not [key for key in out if key.startswith("llm.tools")]


def test_the_definitions_are_spelled_like_the_native_path() -> None:
    # One serializer for both paths, so a normalized span and a native one
    # carrying the same tools put the same string on the wire.
    from rius._serde import serialize

    out = normalize_tool_definitions(_indexed(_ANTHROPIC_TOOLS))
    assert out is not None
    assert out[GEN_AI_TOOL_DEFINITIONS] == serialize(_ANTHROPIC_TOOLS)


def test_indices_are_ordered_numerically_not_lexically() -> None:
    tools = [{"name": f"tool_{i}"} for i in range(12)]
    # Deliberately inserted out of order: attribute order is not index order.
    attributes = dict(reversed(list(_indexed(tools).items())))
    out = normalize_tool_definitions(attributes)
    assert out is not None
    assert _definitions(out) == tools


def test_a_native_definitions_key_wins_and_the_indexed_keys_still_go() -> None:
    attributes = {GEN_AI_TOOL_DEFINITIONS: '[{"name": "native"}]', **_indexed(_OPENAI_TOOLS)}
    assert normalize_tool_definitions(attributes) == {
        GEN_AI_TOOL_DEFINITIONS: '[{"name": "native"}]'
    }


def test_an_unparseable_schema_leaves_the_whole_family_alone() -> None:
    # Reassembly is only total when every schema reads back as JSON. Rather
    # than drop one tool from the array, leave the source untouched: it is
    # already content by prefix, so nothing escapes, and nothing is lost.
    attributes = {
        "llm.tools.0.tool.json_schema": json.dumps({"name": "ok"}),
        "llm.tools.1.tool.json_schema": "{not json",
    }
    assert normalize_tool_definitions(attributes) is None


def test_other_keys_under_llm_tools_are_not_touched() -> None:
    attributes = {**_indexed([{"name": "a"}]), "llm.tools.0.tool.something_else": "x"}
    out = normalize_tool_definitions(attributes)
    assert out is not None
    assert out["llm.tools.0.tool.something_else"] == "x"
    assert "llm.tools.0.tool.json_schema" not in out


def test_a_span_without_tools_is_unchanged() -> None:
    assert normalize_tool_definitions({"openinference.span.kind": "LLM"}) is None
    assert normalize_tool_definitions({}) is None
    raw = _raw_span({"openinference.span.kind": "LLM", GEN_AI_OPERATION_NAME: "chat"})
    assert _export(raw) is raw


# --- gen_ai.tool.definitions from the request bag ------------------------


def test_bag_tools_are_promoted_when_llm_tools_is_absent() -> None:
    # litellm and langchain leave the request's tools inside the bag.
    raw = {
        "openinference.span.kind": "LLM",
        LLM_INVOCATION_PARAMETERS: json.dumps(
            {"model": "gpt-4o", "tool_choice": "auto", "tools": _OPENAI_TOOLS}
        ),
    }
    out = _exported_attributes(raw)
    assert _definitions(out) == _OPENAI_TOOLS
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"tool_choice": "auto"}
    assert out[GEN_AI_REQUEST_MODEL] == "gpt-4o"


def test_legacy_bag_functions_are_promoted() -> None:
    # OpenAI's pre-tools `functions` parameter. The openai instrumentor pops
    # `tools` out of the bag but leaves `functions` in it.
    functions = [{"name": "get_weather", "parameters": {"type": "object"}}]
    out = normalize_tool_definitions(
        {LLM_INVOCATION_PARAMETERS: json.dumps({"functions": functions, "temperature": 0.1})}
    )
    assert out is not None
    assert _definitions(out) == functions
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"temperature": 0.1}


def test_bag_tools_and_functions_are_both_kept_in_one_array() -> None:
    tools = [{"type": "function", "function": {"name": "a"}}]
    functions = [{"name": "b"}]
    out = normalize_tool_definitions(
        {LLM_INVOCATION_PARAMETERS: json.dumps({"tools": tools, "functions": functions})}
    )
    assert out is not None
    assert _definitions(out) == tools + functions
    # Nothing else was in the bag, so the bag goes rather than riding as "{}".
    assert LLM_INVOCATION_PARAMETERS not in out


def test_bag_tools_are_left_alone_when_llm_tools_is_present() -> None:
    bag = json.dumps({"tools": [{"name": "from-bag"}], "tool_choice": "auto"})
    out = normalize_tool_definitions(
        {LLM_INVOCATION_PARAMETERS: bag, **_indexed([{"name": "from-llm-tools"}])}
    )
    assert out is not None
    assert _definitions(out) == [{"name": "from-llm-tools"}]
    assert out[LLM_INVOCATION_PARAMETERS] == bag


def test_a_native_definitions_key_wins_over_the_bag_and_the_member_still_goes() -> None:
    out = normalize_tool_definitions(
        {
            GEN_AI_TOOL_DEFINITIONS: '[{"name": "native"}]',
            LLM_INVOCATION_PARAMETERS: json.dumps({"tools": [{"name": "bag"}], "seed": 1}),
        }
    )
    assert out == {
        GEN_AI_TOOL_DEFINITIONS: '[{"name": "native"}]',
        LLM_INVOCATION_PARAMETERS: '{"seed":1}',
    }


@pytest.mark.parametrize(
    "bag",
    [
        "{not json",
        json.dumps(["tools"]),
        json.dumps({"tools": "not-a-list"}),
        json.dumps({"temperature": 0.2}),
    ],
    ids=["unparseable", "not-an-object", "tools-not-a-list", "no-tool-members"],
)
def test_a_bag_with_no_usable_tool_list_is_left_alone(bag: str) -> None:
    assert normalize_tool_definitions({LLM_INVOCATION_PARAMETERS: bag}) is None


# --- the content policy, end to end --------------------------------------


@pytest.mark.parametrize("capture_content", [True, False])
@pytest.mark.parametrize("route", ["llm.tools", "bag"])
def test_promoted_definitions_are_content(capture_content: bool, route: str) -> None:
    """Through ``init()``'s real chain: normalization runs before masking, so
    the promoted key is what masking sees, and it is stripped under
    ``capture_content=False`` exactly like a native ``gen_ai.tool.definitions``."""
    secret = [{"name": "lookup", "description": "SECRET-PROMPT-ENGINEERING"}]
    attributes: dict[str, Any] = {"openinference.span.kind": "LLM"}
    if route == "llm.tools":
        attributes.update(_indexed(secret))
    else:
        attributes[LLM_INVOCATION_PARAMETERS] = json.dumps({"tools": secret, "seed": 7})

    collected = InMemorySpanExporter()
    client = init(
        span_exporter=collected,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        capture_content=capture_content,
    )
    try:
        client.get_tracer().start_span("ChatCompletion", attributes=attributes).end()
        client.flush()
        (span,) = collected.get_finished_spans()
    finally:
        client.shutdown()
    exported = dict(span.attributes or {})
    assert not [key for key in exported if key.startswith("llm.tools")]
    if capture_content:
        assert _definitions(exported) == secret
    else:
        assert GEN_AI_TOOL_DEFINITIONS not in exported
        assert "SECRET" not in json.dumps(exported)
        if route == "bag":
            # The knob the table promoted is identity and survives.
            assert exported["gen_ai.request.seed"] == 7


# --- gen_ai.tool.name from tool.name --------------------------------------


def test_tool_name_is_mapped_and_the_source_deleted() -> None:
    raw = {
        "openinference.span.kind": "TOOL",
        "tool.name": "get_weather",
        "tool.description": "Weather for a city",
        "tool.parameters": '{"type": "object"}',
    }
    out = DEFAULT_TABLE.normalize(raw)
    assert out == {
        "openinference.span.kind": "TOOL",
        GEN_AI_OPERATION_NAME: "execute_tool",
        GEN_AI_TOOL_NAME: "get_weather",
        # Content, already on the masking allowlist; not this ticket's to move.
        "tool.description": "Weather for a city",
        "tool.parameters": '{"type": "object"}',
    }


def test_a_native_tool_name_wins_and_the_source_still_goes() -> None:
    out = DEFAULT_TABLE.normalize({GEN_AI_TOOL_NAME: "native", "tool.name": "other"})
    assert out == {GEN_AI_TOOL_NAME: "native"}


def test_a_tool_span_without_tool_name_does_not_borrow_the_span_name() -> None:
    """Deliberately NO fallback to the span name.

    The native path stopped deriving the tool name from the span name because
    a span name is not a tool name, and a wrong tool name silently groups
    unrelated calls, which is worse than an absent one. A third-party span is
    no different: nothing guarantees its name is the tool's. Do not "helpfully"
    add the fallback here.
    """
    out = _exported_attributes({"openinference.span.kind": "TOOL"}, name="get_weather")
    assert GEN_AI_TOOL_NAME not in out


@pytest.mark.parametrize("bad", ["", 42])
def test_a_tool_name_that_is_not_a_name_maps_to_nothing(bad: Any) -> None:
    out = DEFAULT_TABLE.normalize({"tool.name": bad})
    assert out == {}


def test_a_tool_name_set_at_start_reaches_the_pending_snapshot() -> None:
    # Identity, like the native gen_ai.tool.name: the start-time processor
    # adds it, so a still-running third-party tool call is named in the live
    # view.
    collected = InMemorySpanExporter()
    client = init(
        span_exporter=collected,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        partial_spans=True,
    )
    try:
        span = client.get_tracer().start_span(
            "tool", attributes={"openinference.span.kind": "TOOL", "tool.name": "get_weather"}
        )
        client.flush()
        span.end()
        client.flush()
        spans = collected.get_finished_spans()
    finally:
        client.shutdown()
    (pending,) = [s for s in spans if (s.attributes or {}).get(RIUS_SPAN_PENDING)]
    (final,) = [s for s in spans if not (s.attributes or {}).get(RIUS_SPAN_PENDING)]
    assert (pending.attributes or {})[GEN_AI_TOOL_NAME] == "get_weather"
    assert (final.attributes or {})[GEN_AI_TOOL_NAME] == "get_weather"
    assert "tool.name" not in (final.attributes or {})
