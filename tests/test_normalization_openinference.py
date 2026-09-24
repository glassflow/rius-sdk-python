"""The shipped OpenInference table: ``llm.*`` mapped onto the canonical wire.

Golden-style: a realistic raw attribute set from a bundled instrumentor goes
in, the whole canonical set comes out and is asserted as a WHOLE — an
unintended extra key or a silently dropped one fails here, which is the point
when every rule deletes its source.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.normalization import DEFAULT_TABLE, openinference_invocation_parameters
from rius.semconv import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    LLM_INVOCATION_PARAMETERS,
)


def _normalize(attributes: dict[str, Any]) -> dict[str, Any]:
    out = DEFAULT_TABLE.normalize(attributes)
    return dict(attributes) if out is None else out


# --- golden sets ------------------------------------------------------------


def test_openai_chat_span() -> None:
    """openinference-instrumentation-openai 0.1.52, chat.completions.create.

    ``llm.model_name`` comes from the RESPONSE object there
    (_response_attributes_extractor.py:96), so it is deliberately unmapped and
    rides through untouched; the request model comes out of the invocation
    parameters, which keep ``model`` (_request_attributes_extractor.py:95).
    """
    raw = {
        "openinference.span.kind": "LLM",
        "llm.provider": "openai",
        "llm.system": "openai",
        "llm.model_name": "gpt-4o-2024-08-06",
        "llm.invocation_parameters": json.dumps(
            {
                "model": "gpt-4o",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 256,
                "n": 1,
                "stop": "\n",
                "stream": False,
                "seed": 42,
                "tool_choice": "auto",
            }
        ),
        "llm.input_messages.0.message.role": "user",
        "llm.token_count.prompt": 1024,
        "llm.token_count.completion": 64,
        "llm.token_count.prompt_details.cache_read": 512,
        "llm.token_count.completion_details.reasoning": 16,
        "llm.token_count.total": 1088,
        "llm.finish_reason": "stop",
    }

    assert _normalize(raw) == {
        # kept, not consumed: the taxonomy rules derive the operation FROM it
        # and both keys must be present on every span
        "openinference.span.kind": "LLM",
        GEN_AI_OPERATION_NAME: "chat",
        # untouched: not mapped by any rule
        "llm.system": "openai",
        "llm.model_name": "gpt-4o-2024-08-06",
        "llm.input_messages.0.message.role": "user",
        "llm.token_count.total": 1088,
        GEN_AI_RESPONSE_FINISH_REASONS: ["stop"],
        # mapped
        GEN_AI_PROVIDER_NAME: "openai",
        GEN_AI_REQUEST_MODEL: "gpt-4o",
        "gen_ai.request.temperature": 0.7,
        "gen_ai.request.top_p": 0.9,
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.choice.count": 1,
        "gen_ai.request.stop_sequences": ["\n"],
        "gen_ai.request.stream": False,
        "gen_ai.request.seed": 42,
        # the members no canonical key covers stay in the blob, which is the
        # key masking knows how to redact
        LLM_INVOCATION_PARAMETERS: '{"tool_choice": "auto"}',
        GEN_AI_USAGE_INPUT_TOKENS: 1024,
        GEN_AI_USAGE_OUTPUT_TOKENS: 64,
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 512,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: 16,
    }


def test_anthropic_messages_span() -> None:
    """openinference-instrumentation-anthropic 2.1.4, messages.create.

    The only bundled instrumentor that emits the unambiguous
    ``llm.request.model_name`` / ``llm.response.model_name`` pair
    (_wrappers.py:595, :603).
    """
    raw = {
        "openinference.span.kind": "LLM",
        "llm.provider": "anthropic",
        "llm.system": "anthropic",
        "llm.model_name": "claude-sonnet-4-5-20250929",
        "llm.request.model_name": "claude-sonnet-4-5",
        "llm.response.model_name": "claude-sonnet-4-5-20250929",
        "llm.invocation_parameters": json.dumps(
            {"max_tokens": 1024, "temperature": 1.0, "stop_sequences": ["END"]}
        ),
        "llm.token_count.prompt": 1536,
        "llm.token_count.completion": 128,
        "llm.token_count.prompt_details.cache_read": 1024,
        "llm.token_count.prompt_details.cache_write": 256,
        "llm.token_count.total": 1664,
    }

    assert _normalize(raw) == {
        "openinference.span.kind": "LLM",
        GEN_AI_OPERATION_NAME: "chat",
        "llm.system": "anthropic",
        "llm.model_name": "claude-sonnet-4-5-20250929",
        "llm.token_count.total": 1664,
        GEN_AI_PROVIDER_NAME: "anthropic",
        GEN_AI_REQUEST_MODEL: "claude-sonnet-4-5",
        GEN_AI_RESPONSE_MODEL: "claude-sonnet-4-5-20250929",
        "gen_ai.request.max_tokens": 1024,
        "gen_ai.request.temperature": 1.0,
        "gen_ai.request.stop_sequences": ["END"],
        # every member mapped: the blob is gone rather than left empty
        GEN_AI_USAGE_INPUT_TOKENS: 1536,
        GEN_AI_USAGE_OUTPUT_TOKENS: 128,
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 1024,
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: 256,
    }


def test_anthropic_input_tokens_are_not_summed_again() -> None:
    """``llm.token_count.prompt`` is ALREADY cache-inclusive in every bundled
    instrumentor, so summing it with the cache counts would double-count.

    anthropic _utils.py:29, llama-index _callback.py:684, langchain
    _tracer.py:1154, litellm via litellm's own Anthropic transformation
    (chat/transformation.py:2358), openai by the provider's own definition of
    ``prompt_tokens``.
    """
    out = _normalize(
        {
            "llm.token_count.prompt": 1536,
            "llm.token_count.prompt_details.cache_read": 1024,
            "llm.token_count.prompt_details.cache_write": 256,
        }
    )
    assert out[GEN_AI_USAGE_INPUT_TOKENS] == 1536


# --- the contract, per family ----------------------------------------------


@pytest.mark.parametrize(
    ("source", "target", "source_value", "native_value"),
    [
        ("llm.provider", GEN_AI_PROVIDER_NAME, "openai", "anthropic"),
        ("llm.request.model_name", GEN_AI_REQUEST_MODEL, "a", "native"),
        ("llm.response.model_name", GEN_AI_RESPONSE_MODEL, "a", "native"),
        ("llm.token_count.prompt", GEN_AI_USAGE_INPUT_TOKENS, 1, 99),
        ("llm.token_count.completion", GEN_AI_USAGE_OUTPUT_TOKENS, 1, 99),
        (
            "llm.token_count.prompt_details.cache_read",
            GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
            1,
            99,
        ),
        (
            "llm.token_count.prompt_details.cache_write",
            GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
            1,
            99,
        ),
        (
            "llm.token_count.completion_details.reasoning",
            GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
            1,
            99,
        ),
    ],
)
def test_native_wins_and_the_source_goes(
    source: str, target: str, source_value: Any, native_value: Any
) -> None:
    out = _normalize({source: source_value, target: native_value})
    assert out == {target: native_value}


@pytest.mark.parametrize(
    ("source", "target", "value"),
    [
        ("llm.provider", GEN_AI_PROVIDER_NAME, "openai"),
        ("llm.request.model_name", GEN_AI_REQUEST_MODEL, "gpt-4o"),
        ("llm.response.model_name", GEN_AI_RESPONSE_MODEL, "gpt-4o"),
        ("llm.token_count.prompt", GEN_AI_USAGE_INPUT_TOKENS, 7),
        ("llm.token_count.completion", GEN_AI_USAGE_OUTPUT_TOKENS, 7),
        ("llm.token_count.prompt_details.cache_read", GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, 7),
        ("llm.token_count.prompt_details.cache_write", GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS, 7),
        (
            "llm.token_count.completion_details.reasoning",
            GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
            7,
        ),
    ],
)
def test_source_is_mapped_and_deleted(source: str, target: str, value: Any) -> None:
    assert _normalize({source: value}) == {target: value}


@pytest.mark.parametrize(
    "key",
    [
        "llm.model_name",
        "llm.system",
        "llm.token_count.total",
        "llm.token_count.prompt_details.audio",
        "llm.token_count.prompt_details.cache_input",
        "llm.token_count.completion_details.audio",
        "llm.cost.total",
        "llm.tools.0.tool.json_schema",
        "llm.input_messages.0.message.content",
    ],
)
def test_deliberately_unmapped_key_survives(key: str) -> None:
    assert _normalize({key: "x"}) == {key: "x"}


def test_a_string_token_count_is_coerced() -> None:
    assert _normalize({"llm.token_count.prompt": "12"})[GEN_AI_USAGE_INPUT_TOKENS] == 12


def test_an_unparseable_token_count_maps_to_nothing() -> None:
    out = _normalize({"llm.token_count.prompt": "lots"})
    assert out == {}


# --- provider ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("source_value", "expected"),
    [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        # registry spelling differs from OpenInference's enum value
        ("mistralai", "mistral_ai"),
        ("xai", "x_ai"),
        # no unambiguous registry value: verbatim, because the enum is open
        ("azure", "azure"),
        ("aws", "aws"),
        ("google", "google"),
        ("ollama", "ollama"),
    ],
)
def test_provider_spellings(source_value: str, expected: str) -> None:
    assert _normalize({"llm.provider": source_value}) == {GEN_AI_PROVIDER_NAME: expected}


@pytest.mark.parametrize(
    ("source_value", "expected"),
    [
        ("openai", "openai"),
        ("vertex_ai", "gcp.vertex_ai"),
        ("gemini", "gcp.gemini"),
        ("az.ai.inference", "azure.ai.inference"),
        ("az.ai.openai", "azure.ai.openai"),
        ("xai", "x_ai"),
    ],
)
def test_legacy_gen_ai_system_is_mapped_and_never_emitted(source_value: str, expected: str) -> None:
    assert _normalize({"gen_ai.system": source_value}) == {GEN_AI_PROVIDER_NAME: expected}


def test_llm_provider_wins_over_legacy_gen_ai_system() -> None:
    out = _normalize({"llm.provider": "azure", "gen_ai.system": "openai"})
    assert out == {GEN_AI_PROVIDER_NAME: "azure"}


# --- invocation parameters --------------------------------------------------


def test_unparseable_invocation_parameters_survive_untouched() -> None:
    raw = {LLM_INVOCATION_PARAMETERS: "not json"}
    assert _normalize(raw) == raw


def test_non_object_invocation_parameters_survive_untouched() -> None:
    raw = {LLM_INVOCATION_PARAMETERS: "[1, 2]"}
    assert _normalize(raw) == raw


def test_tool_definitions_stay_inside_the_blob() -> None:
    """litellm and langchain embed the request's tools there. The TABLE
    leaves them: their canonical key is content and must not be written at
    span start, which is when the table also runs. The exporter's
    ``normalize_tool_definitions`` promotes them (tests/test_normalization_tools.py).
    """
    raw = {
        LLM_INVOCATION_PARAMETERS: json.dumps(
            {"temperature": 0, "tools": [{"name": "secret_tool"}]}
        )
    }
    out = _normalize(raw)
    assert out["gen_ai.request.temperature"] == 0
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"tools": [{"name": "secret_tool"}]}


def test_max_tokens_wins_over_max_completion_tokens() -> None:
    out = _normalize(
        {LLM_INVOCATION_PARAMETERS: json.dumps({"max_tokens": 10, "max_completion_tokens": 20})}
    )
    assert out["gen_ai.request.max_tokens"] == 10
    # the member that did not map is not deleted either
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"max_completion_tokens": 20}


def test_max_completion_tokens_alone_maps() -> None:
    out = _normalize({LLM_INVOCATION_PARAMETERS: json.dumps({"max_completion_tokens": 20})})
    assert out == {"gen_ai.request.max_tokens": 20}


def test_a_wrongly_typed_member_is_left_in_the_blob() -> None:
    out = _normalize({LLM_INVOCATION_PARAMETERS: json.dumps({"temperature": "warm"})})
    assert "gen_ai.request.temperature" not in out
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"temperature": "warm"}


def test_a_native_request_key_wins_and_the_member_still_goes() -> None:
    out = _normalize(
        {
            "gen_ai.request.temperature": 0.1,
            LLM_INVOCATION_PARAMETERS: json.dumps({"temperature": 0.9, "user": "u"}),
        }
    )
    assert out["gen_ai.request.temperature"] == 0.1
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"user": "u"}


def test_expansion_is_idempotent() -> None:
    """The processor maps at start and the exporter maps again at end."""
    once = _normalize({LLM_INVOCATION_PARAMETERS: json.dumps({"temperature": 0.5, "user": "u"})})
    assert _normalize(once) == once


def test_the_blob_is_returned_unchanged_when_nothing_maps() -> None:
    raw = json.dumps({"user": "u"})
    assert openinference_invocation_parameters(raw) == {LLM_INVOCATION_PARAMETERS: raw}


# --- the shipped table, end to end -----------------------------------------


def test_the_shipped_table_normalizes_a_span() -> None:
    exporter = InMemorySpanExporter()
    client = init(span_exporter=exporter, set_global=False, service_name="test-svc", instruments=[])
    try:
        with client.get_tracer().start_as_current_span(
            "chat",
            attributes={
                "llm.provider": "mistralai",
                "llm.token_count.prompt": 10,
                "llm.invocation_parameters": json.dumps({"model": "m", "user": "u"}),
            },
        ):
            pass
        client.flush()
        (span,) = exporter.get_finished_spans()
    finally:
        client.shutdown()
    assert span.attributes[GEN_AI_PROVIDER_NAME] == "mistral_ai"
    assert span.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GEN_AI_REQUEST_MODEL] == "m"
    assert json.loads(span.attributes[LLM_INVOCATION_PARAMETERS]) == {"user": "u"}
    assert "llm.provider" not in span.attributes
    assert "llm.token_count.prompt" not in span.attributes


def test_tool_definitions_left_in_the_blob_are_still_stripped() -> None:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        capture_content=False,
    )
    try:
        with client.get_tracer().start_as_current_span(
            "chat",
            attributes={
                "llm.invocation_parameters": json.dumps(
                    {"temperature": 0, "tools": [{"name": "secret_tool"}]}
                )
            },
        ):
            pass
        client.flush()
        (span,) = exporter.get_finished_spans()
    finally:
        client.shutdown()
    assert span.attributes["gen_ai.request.temperature"] == 0
    assert "secret_tool" not in span.attributes.get(LLM_INVOCATION_PARAMETERS, "")


# --- finish reasons: verbatim, by decision ----------------------------------


def test_a_scalar_finish_reason_becomes_a_one_element_array() -> None:
    """The source is a scalar and the canonical key is an array, one entry per
    generation."""
    assert _normalize({"llm.finish_reason": "stop"}) == {GEN_AI_RESPONSE_FINISH_REASONS: ["stop"]}


def test_a_list_valued_finish_reason_passes_through_as_a_list() -> None:
    """An instrumentor that already reports one reason per choice must not be
    double-wrapped into [["stop", "length"]]."""
    assert _normalize({"llm.finish_reason": ["stop", "length"]}) == {
        GEN_AI_RESPONSE_FINISH_REASONS: ["stop", "length"]
    }


@pytest.mark.parametrize(
    "provider_value",
    ["tool_calls", "function_call", "tool_use", "end_turn", "STOP", "max_tokens"],
)
def test_the_provider_value_is_never_rewritten(provider_value: str) -> None:
    """The decision this rule exists to encode.

    The registry defines the key as a free-form string array with no enum, so
    there is no vocabulary to conform to. OpenInference's own converter
    lowercases and folds tool_calls/function_call into tool_call; we do not,
    because that would replace what OpenAI actually returned with a string
    neither the provider nor the conventions use, while leaving Anthropic's
    end_turn and tool_use untouched. Fidelity lost, nothing unified.

    If this test is ever changed to expect a folded value, the change belongs
    in RIUS-917 first, and it has to bind the native path too.
    """
    out = _normalize({"llm.finish_reason": provider_value})
    assert out[GEN_AI_RESPONSE_FINISH_REASONS] == [provider_value]


def test_a_native_finish_reason_wins_and_the_source_still_goes() -> None:
    out = _normalize({"llm.finish_reason": "stop", GEN_AI_RESPONSE_FINISH_REASONS: ["length"]})
    assert out[GEN_AI_RESPONSE_FINISH_REASONS] == ["length"]
    assert "llm.finish_reason" not in out
