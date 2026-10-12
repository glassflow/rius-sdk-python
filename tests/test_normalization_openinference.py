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
    GEN_AI_REQUEST_MAX_TOKENS,
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
        # not a convention key, and not content either: attribution reads it,
        # so it leaves the bag, and the bag, now empty, goes with it
        "rius.request.tool_choice": "auto",
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


@pytest.mark.parametrize(
    ("source_value", "expected"),
    [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("cohere", "cohere"),
        ("mistralai", "mistral_ai"),
        ("vertexai", "gcp.vertex_ai"),
    ],
)
def test_llm_system_is_the_provider_when_the_host_is_unknown(
    source_value: str, expected: str
) -> None:
    """An OpenAI client pointed at a gateway (Vercel AI Gateway, OpenRouter, a
    LiteLLM proxy) gets no ``llm.provider``: the openai instrumentor sets it
    only for hosts it recognizes. ``llm.system`` names the API the client
    speaks, which is what the registry asks of the provider: the
    instrumentation's best knowledge, a discriminator for the telemetry
    flavor, allowed to differ from whoever served the request behind a proxy.

    ``llm.system`` is kept: on Azure it differs from the provider, so it is a
    fact of its own and not a duplicate.
    """
    out = _normalize({"llm.system": source_value})
    assert out == {"llm.system": source_value, GEN_AI_PROVIDER_NAME: expected}


def test_llm_provider_wins_over_llm_system() -> None:
    """Azure OpenAI: the host names the provider, the system names the API."""
    out = _normalize({"llm.provider": "azure", "llm.system": "openai"})
    assert out == {"llm.system": "openai", GEN_AI_PROVIDER_NAME: "azure"}


def test_legacy_gen_ai_system_wins_over_llm_system() -> None:
    out = _normalize({"gen_ai.system": "az.ai.openai", "llm.system": "openai"})
    assert out == {"llm.system": "openai", GEN_AI_PROVIDER_NAME: "azure.ai.openai"}


def test_native_provider_wins_over_llm_system() -> None:
    attributes = {GEN_AI_PROVIDER_NAME: "groq", "llm.system": "openai"}
    assert _normalize(attributes) == attributes


# --- routed model names -----------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        # LiteLLM records the model as routed: the prefix is the provider.
        ("openai", "openai/gpt-4o-mini", "gpt-4o-mini"),
        ("anthropic", "anthropic/claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
        # compared under the registry's spelling
        ("vertex_ai", "vertex_ai/gemini-2.5-flash", "gemini-2.5-flash"),
        # a prefix naming anyone else is part of the name
        ("openai", "openai/google/gemini-2.5-flash", "google/gemini-2.5-flash"),
        ("openai", "google/gemini-2.5-flash", "google/gemini-2.5-flash"),
        ("anthropic", "openai/gpt-4o-mini", "openai/gpt-4o-mini"),
        # nothing after the slash, or no slash
        ("openai", "openai/", "openai/"),
        ("openai", "gpt-4o-mini", "gpt-4o-mini"),
    ],
)
def test_a_routing_prefix_naming_the_provider_is_dropped(
    provider: str, model: str, expected: str
) -> None:
    raw = {"llm.provider": provider, LLM_INVOCATION_PARAMETERS: json.dumps({"model": model})}
    assert _normalize(raw)[GEN_AI_REQUEST_MODEL] == expected


def test_a_routed_model_keeps_its_prefix_without_a_provider() -> None:
    raw = {LLM_INVOCATION_PARAMETERS: json.dumps({"model": "openai/gpt-4o-mini"})}
    assert _normalize(raw)[GEN_AI_REQUEST_MODEL] == "openai/gpt-4o-mini"


def test_a_native_provider_strips_a_mapped_model() -> None:
    raw = {
        GEN_AI_PROVIDER_NAME: "openai",
        LLM_INVOCATION_PARAMETERS: json.dumps({"model": "openai/gpt-4o-mini"}),
    }
    assert _normalize(raw)[GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"


def test_a_native_model_is_never_stripped() -> None:
    """The producer named it; that name is its own choice."""
    attributes = {GEN_AI_PROVIDER_NAME: "openai", GEN_AI_REQUEST_MODEL: "openai/gpt-4o-mini"}
    assert _normalize(attributes) == attributes


def test_the_start_time_additions_already_strip() -> None:
    """The processor writes additions onto the live span at start; by export
    the model counts as native, so the start-time value must already be bare."""
    raw = {
        "llm.provider": "openai",
        LLM_INVOCATION_PARAMETERS: json.dumps({"model": "openai/gpt-4o-mini"}),
    }
    assert DEFAULT_TABLE.additions(raw)[GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"


# --- invocation parameters --------------------------------------------------


def test_llama_index_model_metadata_gives_the_model_and_max_tokens() -> None:
    """LlamaIndex fills the bag with the model's metadata, not the request's
    parameters: the model is ``model_name`` and the output budget is
    ``num_output``. The rest of the metadata stays in the bag."""
    bag = {
        "context_window": 128000,
        "num_output": 64,
        "is_chat_model": True,
        "model_name": "gpt-4o-mini",
        "system_role": "system",
    }
    out = _normalize({LLM_INVOCATION_PARAMETERS: json.dumps(bag)})
    assert out[GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert out[GEN_AI_REQUEST_MAX_TOKENS] == 64
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {
        "context_window": 128000,
        "is_chat_model": True,
        "system_role": "system",
    }


def test_llama_index_unset_output_budget_is_not_a_max_tokens() -> None:
    """``num_output: -1`` is LlamaIndex saying no budget was set."""
    bag = {"num_output": -1, "model_name": "gpt-4o-mini"}
    out = _normalize({LLM_INVOCATION_PARAMETERS: json.dumps(bag)})
    assert GEN_AI_REQUEST_MAX_TOKENS not in out
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"num_output": -1}


def test_model_wins_over_a_different_model_name_which_stays() -> None:
    out = _normalize(
        {LLM_INVOCATION_PARAMETERS: json.dumps({"model": "gpt-4o", "model_name": "gpt-4o-x"})}
    )
    assert out[GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"model_name": "gpt-4o-x"}


def test_a_second_spelling_with_the_same_value_is_a_duplicate() -> None:
    """LangChain's bag has ``model_name`` next to ``model``, same value."""
    bag = {"model": "gpt-4o", "model_name": "gpt-4o", "_type": "openai-chat"}
    out = _normalize({LLM_INVOCATION_PARAMETERS: json.dumps(bag)})
    assert out[GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"_type": "openai-chat"}


def test_normalizing_the_bag_twice_is_normalizing_it_once() -> None:
    """The SDK normalizes at span start and again at export."""
    bag = {"model": "gpt-4o", "model_name": "gpt-4o", "num_output": -1, "max_tokens": 5}
    once = _normalize({LLM_INVOCATION_PARAMETERS: json.dumps(bag)})
    assert _normalize(once) == once


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


# --- langchain's stream member ---------------------------------------------
#
# LangChain serializes the MODEL object's settings into the bag, so `stream`
# is the model's default, not what the call did: ChatOpenAI.stream() records
# "stream": false (openinference-instrumentation-langchain 0.1.79 and
# @arizeai/openinference-instrumentation-langchain 4.1.4, rius-semconv
# recordings, scenario chat_stream). It is never promoted, and stays in the bag.


def test_a_python_langchain_bag_never_sets_stream() -> None:
    """The Python integration marks its bag with LangChain's `_type`."""
    out = _normalize(
        {
            LLM_INVOCATION_PARAMETERS: json.dumps(
                {"model": "gpt-4o-mini", "stream": False, "_type": "openai-chat"}
            )
        }
    )
    assert out[GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert "gen_ai.request.stream" not in out
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"stream": False, "_type": "openai-chat"}


def test_a_js_langchain_bag_never_sets_stream() -> None:
    """The JS integration's bag has no marker; its run metadata (ls_*) does."""
    metadata = json.dumps({"ls_provider": "openai", "ls_integration": "langchain_chat_model"})
    out = _normalize(
        {
            "metadata": metadata,
            LLM_INVOCATION_PARAMETERS: json.dumps({"model": "gpt-4o-mini", "stream": False}),
        }
    )
    assert out[GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert "gen_ai.request.stream" not in out
    assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"stream": False}
    # Read, not consumed.
    assert out["metadata"] == metadata


def test_a_request_body_bag_still_sets_stream() -> None:
    """Instrumentors that record the request body report what the call did."""
    out = _normalize(
        {
            "metadata": json.dumps({"tenant": "acme"}),
            LLM_INVOCATION_PARAMETERS: json.dumps({"model": "gpt-4o-mini", "stream": True}),
        }
    )
    assert out["gen_ai.request.stream"] is True
    assert LLM_INVOCATION_PARAMETERS not in out


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
