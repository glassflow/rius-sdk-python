"""The request's ``system`` member promoted out of the OpenInference request bag.

The OpenInference Anthropic instrumentation for JavaScript records the request
body minus ``messages`` as ``llm.invocation_parameters``, so Anthropic's
top-level ``system`` prompt, a string or a block list, stays in the bag and
never becomes a message. Context attribution then books nearly the whole
prompt as unattributed. The Python instrumentor writes it as a message
already. The TypeScript SDK (``promoteSystemInstruction``) and the sink
(``bagSystemPrompt``) apply this same rule, so the three store the same bytes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius.normalization import NormalizingSpanExporter, promote_system_instruction
from rius.semconv import GEN_AI_INPUT_MESSAGES, LLM_INVOCATION_PARAMETERS

USER = {"role": "user", "parts": [{"type": "text", "content": "hi"}]}


def _js(value: Any) -> str:
    """``value`` the way ``JSON.stringify`` writes it."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def normalized(attributes: dict[str, Any]) -> dict[str, Any]:
    """The span's attributes after the shipped exporter's normalization."""
    sink = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(sink)))
    provider.get_tracer("t").start_span("llm", attributes=attributes).end()
    provider.shutdown()
    (span,) = sink.get_finished_spans()
    return dict(span.attributes or {})


class TestStringSystemPrompt:
    def test_becomes_the_first_input_message_and_leaves_the_bag(self) -> None:
        out = normalized(
            {
                GEN_AI_INPUT_MESSAGES: _js([USER]),
                LLM_INVOCATION_PARAMETERS: _js({"system": "Be brief.", "metadata": {"a": 1}}),
            }
        )
        assert out[GEN_AI_INPUT_MESSAGES] == _js(
            [{"role": "system", "parts": [{"type": "text", "content": "Be brief."}]}, USER]
        )
        assert json.loads(out[LLM_INVOCATION_PARAMETERS]) == {"metadata": {"a": 1}}

    def test_drops_a_bag_it_empties(self) -> None:
        out = normalized(
            {
                GEN_AI_INPUT_MESSAGES: _js([USER]),
                LLM_INVOCATION_PARAMETERS: _js({"system": "Be brief."}),
            }
        )
        assert LLM_INVOCATION_PARAMETERS not in out

    def test_is_promoted_from_the_flattened_openinference_messages_too(self) -> None:
        out = normalized(
            {
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "hi",
                LLM_INVOCATION_PARAMETERS: _js({"system": "Be brief."}),
            }
        )
        assert json.loads(out[GEN_AI_INPUT_MESSAGES]) == [
            {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
            USER,
        ]

    def test_creates_the_messages_when_the_span_has_none(self) -> None:
        out = normalized({LLM_INVOCATION_PARAMETERS: _js({"system": "Be brief."})})
        assert out[GEN_AI_INPUT_MESSAGES] == _js(
            [{"role": "system", "parts": [{"type": "text", "content": "Be brief."}]}]
        )


class TestBlockListSystemPrompt:
    def test_one_system_message_with_a_text_part_per_text_block(self) -> None:
        out = normalized(
            {
                GEN_AI_INPUT_MESSAGES: _js([USER]),
                LLM_INVOCATION_PARAMETERS: _js(
                    {
                        "max_tokens": 5,
                        "system": [
                            {"type": "text", "text": "You are an agent."},
                            {
                                "type": "text",
                                "text": "Long cached context.",
                                "cache_control": {"type": "ephemeral"},
                            },
                        ],
                    }
                ),
            }
        )
        assert json.loads(out[GEN_AI_INPUT_MESSAGES]) == [
            {
                "role": "system",
                "parts": [
                    {"type": "text", "content": "You are an agent."},
                    {"type": "text", "content": "Long cached context."},
                ],
            },
            USER,
        ]
        # max_tokens was promoted by the table, system by this pass: nothing is left.
        assert LLM_INVOCATION_PARAMETERS not in out
        assert out["gen_ai.request.max_tokens"] == 5

    def test_any_other_block_is_its_compact_json_keys_sorted_at_every_depth(self) -> None:
        out = normalized(
            {
                LLM_INVOCATION_PARAMETERS: _js(
                    {
                        "system": [
                            {"type": "image", "source": {"url": "u", "type": "url"}},
                            {"type": "text", "text": 7},
                            "bare",
                        ]
                    }
                )
            }
        )
        (system,) = json.loads(out[GEN_AI_INPUT_MESSAGES])
        assert system["parts"] == [
            {"type": "text", "content": '{"source":{"type":"url","url":"u"},"type":"image"}'},
            {"type": "text", "content": '{"text":7,"type":"text"}'},
            {"type": "text", "content": '"bare"'},
        ]

    def test_escapes_nothing_json_does_not_require(self) -> None:
        out = normalized({LLM_INVOCATION_PARAMETERS: _js({"system": [{"type": "x", "v": "<&>é"}]})})
        (system,) = json.loads(out[GEN_AI_INPUT_MESSAGES])
        assert system["parts"][0]["content"] == '{"type":"x","v":"<&>é"}'

    def test_line_separators_stay_raw_and_a_lone_surrogate_is_replaced(self) -> None:
        # The bag as JSON.stringify writes a lone surrogate: escaped.
        bag = '{"system":[{"v":"a b \\ud800c"}]}'
        out = normalized({LLM_INVOCATION_PARAMETERS: bag})
        raw = out[GEN_AI_INPUT_MESSAGES]
        assert " " in raw
        assert "\\u2028" not in raw
        (system,) = json.loads(raw)
        assert system["parts"][0]["content"] == '{"v":"a b �c"}'


class TestLeftAlone:
    def test_when_the_span_already_has_a_system_message(self) -> None:
        messages = _js([{"role": "system", "parts": [{"type": "text", "content": "native"}]}, USER])
        bag = _js({"system": "Be brief."})
        out = normalized({GEN_AI_INPUT_MESSAGES: messages, LLM_INVOCATION_PARAMETERS: bag})
        assert out[GEN_AI_INPUT_MESSAGES] == messages
        assert out[LLM_INVOCATION_PARAMETERS] == bag

    @pytest.mark.parametrize(
        "system",
        [{"a": 1}, 3, None, "", []],
        ids=["an object", "a number", "null", "an empty string", "an empty list"],
    )
    def test_when_system_is_not_a_prompt(self, system: Any) -> None:
        bag = _js({"system": system})
        out = normalized({GEN_AI_INPUT_MESSAGES: _js([USER]), LLM_INVOCATION_PARAMETERS: bag})
        assert out[GEN_AI_INPUT_MESSAGES] == _js([USER])
        assert out[LLM_INVOCATION_PARAMETERS] == bag

    @pytest.mark.parametrize("messages", ['{"a":1}', "not json"])
    def test_when_the_messages_are_not_a_json_array(self, messages: str) -> None:
        bag = _js({"system": "Be brief."})
        out = normalized({GEN_AI_INPUT_MESSAGES: messages, LLM_INVOCATION_PARAMETERS: bag})
        assert out[GEN_AI_INPUT_MESSAGES] == messages
        assert out[LLM_INVOCATION_PARAMETERS] == bag

    @pytest.mark.parametrize("bag", ["[1]", "{"])
    def test_when_the_bag_is_not_a_json_object(self, bag: str) -> None:
        assert promote_system_instruction({LLM_INVOCATION_PARAMETERS: bag}) is None

    def test_when_there_is_no_bag(self) -> None:
        assert promote_system_instruction({}) is None


def test_normalizing_twice_changes_nothing() -> None:
    once = normalized(
        {
            GEN_AI_INPUT_MESSAGES: _js([USER]),
            LLM_INVOCATION_PARAMETERS: _js({"system": "Be brief.", "metadata": {"a": 1}}),
        }
    )
    assert normalized(once) == once
