import json
import time
from typing import Any

import pytest
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind as OtelSpanKind
from opentelemetry.trace import StatusCode

from rius import start_as_current_generation, start_generation
from rius.generation import Generation

# --- context manager: start_as_current_generation ---


def test_cm_is_llm_kind(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["gen_ai.operation.name"] == "chat"


def test_cm_span_name(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("my-llm-call"):
        pass
    assert exported_spans.get_finished_spans()[0].name == "my-llm-call"


def test_cm_model_and_input(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation(
        "chat", model="gpt-4o", input=[{"role": "user", "content": "hi"}]
    ):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert "hi" in attrs["gen_ai.input.messages"]


def test_cm_provider(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat", provider="openai"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.provider.name"] == "openai"
    assert "gen_ai.system" not in attrs  # legacy key must not be emitted


def test_cm_output_and_usage(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat", model="gpt-4o") as gen:
        gen.set_output([{"role": "assistant", "content": "hello"}])
        gen.set_usage(input_tokens=10, output_tokens=5)
        gen.set_response_model("gpt-4o-2026-05")
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "hello" in attrs["gen_ai.output.messages"]
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.response.model"] == "gpt-4o-2026-05"


def test_cm_usage_cache_tokens(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_usage(
            input_tokens=10,
            output_tokens=202,
            cache_read_input_tokens=11579,
            cache_write_input_tokens=12694,
        )
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 11579
    assert attrs["gen_ai.usage.cache_write.input_tokens"] == 12694


def test_cm_usage_cache_tokens_omitted_are_absent(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_usage(input_tokens=10, output_tokens=5)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "gen_ai.usage.cache_read.input_tokens" not in attrs
    assert "gen_ai.usage.cache_write.input_tokens" not in attrs


def test_cm_usage_cache_tokens_zero_recorded(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_usage(cache_read_input_tokens=0, cache_write_input_tokens=0)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 0
    assert attrs["gen_ai.usage.cache_write.input_tokens"] == 0


def test_cm_usage_reasoning_tokens(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_usage(output_tokens=900, reasoning_output_tokens=700)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 700


def test_cm_usage_reasoning_tokens_omitted_absent(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_usage(input_tokens=10, output_tokens=5)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "gen_ai.usage.reasoning.output_tokens" not in attrs


def test_cm_usage_reasoning_tokens_zero_recorded(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_usage(reasoning_output_tokens=0)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 0


def test_anthropic_input_tokens_summed_with_cache(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat", provider="anthropic") as gen:
        gen.set_usage(
            input_tokens=10,
            output_tokens=202,
            cache_read_input_tokens=11579,
            cache_write_input_tokens=12694,
        )
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 10 + 11579 + 12694
    # the subset attributes stay as reported
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 11579
    assert attrs["gen_ai.usage.cache_write.input_tokens"] == 12694


def test_anthropic_input_tokens_summed_case_insensitive(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat", provider="Anthropic") as gen:
        gen.set_usage(input_tokens=100, cache_read_input_tokens=50)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 150


def test_anthropic_input_tokens_unchanged_without_cache(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat", provider="anthropic") as gen:
        gen.set_usage(input_tokens=100, output_tokens=5)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 100


def test_non_anthropic_input_tokens_never_summed(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat", provider="openai") as gen:
        gen.set_usage(input_tokens=1000, cache_read_input_tokens=400)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 1000


def test_no_provider_input_tokens_never_summed(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_usage(input_tokens=1000, cache_read_input_tokens=400)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 1000


def test_anthropic_caches_without_input_emit_no_total(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat", provider="anthropic") as gen:
        gen.set_usage(cache_read_input_tokens=400, cache_write_input_tokens=100)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "gen_ai.usage.input_tokens" not in attrs


def test_cm_finish_reasons_list(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_finish_reasons(["stop", "length"])
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.response.finish_reasons"] == ("stop", "length")


def test_cm_finish_reasons_single_string(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_finish_reasons("stop")
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)


def test_cm_operation_override(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("completion", operation="text_completion"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.operation.name"] == "text_completion"
    assert attrs["openinference.span.kind"] == "LLM"


def test_manual_operation_override(exported_spans: InMemorySpanExporter) -> None:
    gen = start_generation("completion", operation="text_completion")
    gen.end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.operation.name"] == "text_completion"


def test_cm_reasoning_level(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat", model="o4-mini", reasoning_level="high"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.reasoning.level"] == "high"


def test_cm_reasoning_level_omitted_absent(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat", model="o4-mini"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "gen_ai.request.reasoning.level" not in attrs


def test_manual_reasoning_level(exported_spans: InMemorySpanExporter) -> None:
    gen = start_generation("chat", reasoning_level="low")
    gen.end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.reasoning.level"] == "low"


def test_cm_model_parameters(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation(
        "chat", model_parameters={"temperature": 0.7, "max_tokens": 256}
    ):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["gen_ai.request.max_tokens"] == 256


# --- spec message shape: gen_ai.*.messages must be role/parts arrays ---


def _input_messages(exported_spans: InMemorySpanExporter) -> list:
    return json.loads(exported_spans.get_finished_spans()[0].attributes["gen_ai.input.messages"])


def _output_messages(exported_spans: InMemorySpanExporter) -> list:
    return json.loads(exported_spans.get_finished_spans()[0].attributes["gen_ai.output.messages"])


def test_bare_string_input_wrapped_as_user_message(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat", input="raw prompt"):
        pass
    assert _input_messages(exported_spans) == [
        {"role": "user", "parts": [{"type": "text", "content": "raw prompt"}]}
    ]


def test_bare_string_output_wrapped_as_assistant_message(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_output("Hello!")
    assert _output_messages(exported_spans) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Hello!"}]}
    ]


def test_openai_style_messages_converted_to_role_parts(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation(
        "chat",
        input=[
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ],
    ):
        pass
    assert _input_messages(exported_spans) == [
        {"role": "system", "parts": [{"type": "text", "content": "be nice"}]},
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]},
    ]


def test_already_conformant_messages_pass_through(
    exported_spans: InMemorySpanExporter,
) -> None:
    conformant = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
    with start_as_current_generation("chat", input=conformant):
        pass
    assert _input_messages(exported_spans) == conformant


def test_openai_tool_call_message_converted_to_tool_call_part(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat") as gen:
        gen.set_output(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                }
            ]
        )
    assert _output_messages(exported_spans) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                }
            ],
        }
    ]


def test_tool_response_message_converted_to_tool_call_response_part(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation(
        "chat", input=[{"role": "tool", "tool_call_id": "call_1", "content": "22C"}]
    ):
        pass
    assert _input_messages(exported_spans) == [
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call_1", "response": "22C"}],
        }
    ]


def test_multimodal_content_list_converted_to_parts(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation(
        "chat", input=[{"role": "user", "content": [{"type": "text", "text": "describe this"}]}]
    ):
        pass
    assert _input_messages(exported_spans) == [
        {"role": "user", "parts": [{"type": "text", "content": "describe this"}]}
    ]


def test_non_dict_message_falls_back_to_serialized_text_part(
    exported_spans: InMemorySpanExporter,
) -> None:
    class FrameworkMessage:
        def __repr__(self) -> str:
            return "FrameworkMessage(hello)"

    with start_as_current_generation("chat", input=[FrameworkMessage()]):
        pass
    (message,) = _input_messages(exported_spans)
    assert message["role"] == "user"
    (part,) = message["parts"]
    assert part["type"] == "text"
    assert "FrameworkMessage" in part["content"]


# --- manual lifecycle: start_generation / update / end ---


def test_manual_generation(exported_spans: InMemorySpanExporter) -> None:
    gen = start_generation("chat", model="gpt-4o")
    gen.update(output=[{"role": "assistant", "content": "hi"}])
    gen.set_usage(input_tokens=1, output_tokens=2)
    assert exported_spans.get_finished_spans() == ()  # not ended
    gen.end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert "hi" in attrs["gen_ai.output.messages"]
    assert attrs["gen_ai.usage.input_tokens"] == 1


# --- record_first_token: the TTFT anchor for streaming ---


def _first_token_events(span: ReadableSpan) -> list[Event]:
    return [e for e in span.events if e.name == "gen_ai.first_token"]


def test_record_first_token_adds_event_between_start_and_end(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat") as gen:
        gen.record_first_token()
    span = exported_spans.get_finished_spans()[0]
    (event,) = _first_token_events(span)
    assert span.start_time <= event.timestamp <= span.end_time


def test_record_first_token_is_idempotent(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        gen.record_first_token()
        gen.record_first_token()
        gen.record_first_token()
    span = exported_spans.get_finished_spans()[0]
    assert len(_first_token_events(span)) == 1


def test_record_first_token_after_end_is_noop(exported_spans: InMemorySpanExporter) -> None:
    gen = start_generation("chat")
    gen.end()
    gen.record_first_token()  # must neither raise nor record
    span = exported_spans.get_finished_spans()[0]
    assert _first_token_events(span) == []


def test_record_first_token_on_manual_generation(exported_spans: InMemorySpanExporter) -> None:
    gen = start_generation("chat", model="gpt-4o")
    gen.record_first_token()
    gen.end()
    span = exported_spans.get_finished_spans()[0]
    assert len(_first_token_events(span)) == 1


def test_record_first_token_sets_time_to_first_chunk_and_stream(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat") as gen:
        time.sleep(0.01)
        gen.record_first_token()
    span = exported_spans.get_finished_spans()[0]
    ttfc = span.attributes["gen_ai.response.time_to_first_chunk"]
    assert isinstance(ttfc, float)
    assert ttfc > 0
    assert span.attributes["gen_ai.request.stream"] is True
    assert len(_first_token_events(span)) == 1  # the event is kept, not replaced


def test_time_to_first_chunk_matches_event_offset(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat") as gen:
        time.sleep(0.02)
        gen.record_first_token()
    span = exported_spans.get_finished_spans()[0]
    (event,) = _first_token_events(span)
    from_event = (event.timestamp - span.start_time) / 1e9
    assert abs(span.attributes["gen_ai.response.time_to_first_chunk"] - from_event) < 0.005


def test_second_record_first_token_does_not_change_time_to_first_chunk(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat") as gen:
        gen.record_first_token()
        first = gen._span.attributes["gen_ai.response.time_to_first_chunk"]  # type: ignore[attr-defined]
        time.sleep(0.02)
        gen.record_first_token()
    span = exported_spans.get_finished_spans()[0]
    assert span.attributes["gen_ai.response.time_to_first_chunk"] == first


def test_generation_without_first_token_has_no_stream_attributes(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation("chat"):
        pass
    span = exported_spans.get_finished_spans()[0]
    assert _first_token_events(span) == []
    assert "gen_ai.response.time_to_first_chunk" not in span.attributes
    assert "gen_ai.request.stream" not in span.attributes


def test_record_first_token_on_span_without_start_time_skips_attribute() -> None:
    class _Recording:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.attributes: dict[str, Any] = {}

        def is_recording(self) -> bool:
            return True

        def add_event(self, name: str, timestamp: int | None = None) -> None:
            self.events.append(name)

        def set_attribute(self, key: str, value: Any) -> None:
            self.attributes[key] = value

    span = _Recording()
    gen = Generation(span)  # type: ignore[arg-type]
    gen.record_first_token()  # must not raise
    assert span.events == ["gen_ai.first_token"]
    assert "gen_ai.response.time_to_first_chunk" not in span.attributes
    assert span.attributes["gen_ai.request.stream"] is True


# --- tool definitions ---


def test_cm_tools_kwarg_records_definitions_verbatim(
    exported_spans: InMemorySpanExporter,
) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    with start_as_current_generation("chat", tools=tools):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert json.loads(attrs["gen_ai.tool.definitions"]) == tools


def test_set_tool_definitions_accepts_anthropic_shape(
    exported_spans: InMemorySpanExporter,
) -> None:
    # Verbatim on purpose: provider tool formats differ (OpenAI nests under
    # "function", Anthropic uses top-level name/input_schema) and the backend
    # reads names and sizes from either shape.
    tools = [
        {
            "name": "search_kb",
            "description": "Search the knowledge base",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]
    gen = start_generation("chat", model="claude-sonnet-5", provider="anthropic")
    gen.set_tool_definitions(tools)
    gen.end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert json.loads(attrs["gen_ai.tool.definitions"]) == tools


def test_tools_omitted_attribute_absent(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "gen_ai.tool.definitions" not in attrs


# --- error.type ---
# GenAI semconv: error.type is Conditionally Required on inference spans that
# end in an error; the class only (low cardinality), never the message.


def test_cm_exception_sets_error_type(exported_spans: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError, match="boom"), start_as_current_generation("chat"):
        raise ValueError("boom")
    span = exported_spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)
    assert span.attributes["error.type"] == "ValueError"


def test_cm_error_type_is_module_qualified_for_user_exceptions(
    exported_spans: InMemorySpanExporter,
) -> None:
    class ProviderDown(RuntimeError):
        pass

    with pytest.raises(ProviderDown), start_as_current_generation("chat"):
        raise ProviderDown()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["error.type"] == f"{ProviderDown.__module__}.{ProviderDown.__qualname__}"
    assert "boom" not in attrs["error.type"]


def test_cm_success_sets_no_error_type(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat"):
        pass
    assert "error.type" not in exported_spans.get_finished_spans()[0].attributes


# --- the OTel SpanKind FIELD ---


def test_generation_spans_are_client_kind(exported_spans: InMemorySpanExporter) -> None:
    """Inference calls a remote model: the GenAI inference convention says CLIENT."""
    start_generation("manual").end()
    with start_as_current_generation("scoped"):
        pass
    kinds = {s.name: s.kind for s in exported_spans.get_finished_spans()}
    assert kinds == {"manual": OtelSpanKind.CLIENT, "scoped": OtelSpanKind.CLIENT}


# --- rius.context.sizes ---


def _sizes_of(span: ReadableSpan) -> dict[str, Any]:
    assert span.attributes is not None
    decoded: dict[str, Any] = json.loads(str(span.attributes["rius.context.sizes"]))
    return decoded


def test_every_generation_span_carries_context_sizes(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("chat"):
        pass
    gen = start_generation("chat2")
    gen.end()
    for span in exported_spans.get_finished_spans():
        assert _sizes_of(span) == {"version": 1}


def test_context_sizes_combine_input_output_and_tools(
    exported_spans: InMemorySpanExporter,
) -> None:
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    with start_as_current_generation("chat", input=[{"role": "system", "content": "abc"}]) as gen:
        # Computed once, at span end: while the block runs only the seed is set.
        assert _sizes_of_live(gen) == {"version": 1}
        gen.set_output(
            [
                {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}],
                }
            ]
        )
        gen.set_tool_definitions(tools)
    sizes = _sizes_of(exported_spans.get_finished_spans()[0])
    call_part = {"type": "tool_call", "id": "c1", "name": "lookup", "arguments": "{}"}
    call_bytes = len(json.dumps(call_part, separators=(",", ":")).encode())
    tool_bytes = len(json.dumps(tools[0], separators=(",", ":")).encode())
    assert sizes == {
        "version": 1,
        "tool_definitions": [{"name": "lookup", "bytes": tool_bytes}],
        "input_messages": [{"role": "system", "parts": [{"type": "text", "bytes": 3}]}],
        "output_messages": [
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "bytes": 2},
                    {"type": "tool_call", "tool": "lookup", "bytes": call_bytes},
                ],
            }
        ],
    }


def _sizes_of_live(gen: Generation) -> dict[str, Any]:
    attributes = getattr(gen._span, "attributes", None)
    assert attributes is not None
    decoded: dict[str, Any] = json.loads(str(attributes["rius.context.sizes"]))
    return decoded


def test_context_sizes_are_the_full_size_when_content_is_truncated(
    exported_spans: InMemorySpanExporter,
) -> None:
    text = "é" * 40_000  # 40 K chars, 80 K bytes: well past the 32 K content cap
    with start_as_current_generation("chat") as gen:
        gen.set_output(text)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs is not None
    assert str(attrs["gen_ai.output.messages"]).endswith("…(truncated)")
    assert _sizes_of(exported_spans.get_finished_spans()[0])["output_messages"] == [
        {"role": "assistant", "parts": [{"type": "text", "bytes": 80_000}]}
    ]


def test_context_sizes_survive_capture_content_false() -> None:
    from rius import init

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    gen = Generation(client.get_tracer().start_span("chat"))
    gen.set_input("secret prompt")
    gen.end()
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs is not None
    assert "gen_ai.input.messages" not in attrs
    assert json.loads(str(attrs["rius.context.sizes"])) == {
        "version": 1,
        "input_messages": [{"role": "user", "parts": [{"type": "text", "bytes": 13}]}],
    }


def test_context_sizes_describe_unmasked_text() -> None:
    from rius import init

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    gen = Generation(client.get_tracer().start_span("chat"))
    gen.set_input("a much longer secret prompt")
    gen.end()
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs is not None
    assert attrs["gen_ai.input.messages"] == "***"
    assert json.loads(str(attrs["rius.context.sizes"])) == {
        "version": 1,
        "input_messages": [{"role": "user", "parts": [{"type": "text", "bytes": 27}]}],
    }


def test_context_sizes_do_not_change_the_message_attributes(
    exported_spans: InMemorySpanExporter,
) -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    with start_as_current_generation("chat", input=messages) as gen:
        gen.set_output("yo")
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs is not None
    assert json.loads(str(attrs["gen_ai.input.messages"])) == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]
    assert json.loads(str(attrs["gen_ai.output.messages"])) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "yo"}]}
    ]


# --- default span names: "{operation} {model}" when no name is given ---


def test_cm_names_a_generation_after_the_operation_and_the_request_model(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation(model="gpt-4o"):
        pass
    assert exported_spans.get_finished_spans()[0].name == "chat gpt-4o"


def test_manual_names_a_generation_after_the_operation_and_the_request_model(
    exported_spans: InMemorySpanExporter,
) -> None:
    start_generation(model="gpt-4o").end()
    assert exported_spans.get_finished_spans()[0].name == "chat gpt-4o"


def test_generation_without_a_model_is_the_bare_operation(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation():
        pass
    start_generation().end()
    assert [s.name for s in exported_spans.get_finished_spans()] == ["chat", "chat"]


def test_generation_name_uses_the_resolved_operation(
    exported_spans: InMemorySpanExporter,
) -> None:
    """The operation is overridable per generation, so the name composes from
    what was resolved rather than from a hardcoded "chat"."""
    with start_as_current_generation(operation="embeddings", model="text-embedding-3-small"):
        pass
    span = exported_spans.get_finished_spans()[0]
    assert span.name == "embeddings text-embedding-3-small"
    assert span.attributes["gen_ai.operation.name"] == "embeddings"


def test_generation_name_never_uses_the_response_model(
    exported_spans: InMemorySpanExporter,
) -> None:
    """The response model can arrive long after the span (and its pending
    snapshot) started; only the request model is known at creation."""
    generation = start_generation(model="gpt-4o")
    generation.set_response_model("gpt-4o-2024-08-06")
    generation.end()
    assert exported_spans.get_finished_spans()[0].name == "chat gpt-4o"


def test_an_explicit_generation_name_always_wins(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation("summarize", model="gpt-4o"):
        pass
    start_generation("summarize", model="gpt-4o").end()
    assert [s.name for s in exported_spans.get_finished_spans()] == ["summarize", "summarize"]


# --- gen_ai.output.type (request-shaped) and gen_ai.response.id (post-call) ---


def test_output_type_is_set_at_creation(exported_spans: InMemorySpanExporter) -> None:
    """The requested output type is known before the call, so it is a creation
    argument and reaches pending snapshots."""
    with start_as_current_generation(model="gpt-4o", output_type="json"):
        pass
    start_generation(model="gpt-4o", output_type="json").end()
    for span in exported_spans.get_finished_spans():
        assert span.attributes is not None
        assert span.attributes["gen_ai.output.type"] == "json"


def test_an_unset_output_type_is_absent(exported_spans: InMemorySpanExporter) -> None:
    with start_as_current_generation(model="gpt-4o"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs is not None
    assert "gen_ai.output.type" not in attrs


def test_an_unrecognised_output_type_is_recorded_verbatim(
    exported_spans: InMemorySpanExporter,
) -> None:
    """The conventions list text/json/image/speech, but the enum is theirs to
    extend and the caller's string is the caller's truth: rejecting a value we
    do not recognise would lose data over a spec we do not control."""
    with start_as_current_generation(model="gpt-4o", output_type="video"):
        pass
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs is not None
    assert attrs["gen_ai.output.type"] == "video"


def test_the_output_type_does_not_change_the_span_name(
    exported_spans: InMemorySpanExporter,
) -> None:
    with start_as_current_generation(model="gpt-4o", output_type="json"):
        pass
    assert exported_spans.get_finished_spans()[0].name == "chat gpt-4o"


def test_response_id_is_recorded_after_the_call(exported_spans: InMemorySpanExporter) -> None:
    """The provider's completion id only exists once the call returns, so it
    is a setter next to set_response_model rather than a creation argument."""
    with start_as_current_generation(model="gpt-4o") as generation:
        generation.set_response_id("chatcmpl-123")
    manual = start_generation(model="gpt-4o")
    manual.set_response_id("chatcmpl-456")
    manual.end()
    ids = [s.attributes["gen_ai.response.id"] for s in exported_spans.get_finished_spans()]  # type: ignore[index]
    assert ids == ["chatcmpl-123", "chatcmpl-456"]


def test_neither_key_is_content() -> None:
    """A completion id and an output modality say nothing about what was said,
    so both survive capture_content=False."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter as Exporter,
    )

    from rius import init

    inner = Exporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("gen_ai.output.type", "json")
        span.set_attribute("gen_ai.response.id", "chatcmpl-123")
        span.set_attribute("gen_ai.input.messages", "secret")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs is not None
    assert attrs["gen_ai.output.type"] == "json"
    assert attrs["gen_ai.response.id"] == "chatcmpl-123"
    assert "gen_ai.input.messages" not in attrs


# --- model_parameters normalization (RIUS-926) ---


def _params(exported_spans: InMemorySpanExporter, **parameters: Any) -> Any:
    start_generation("chat", model_parameters=parameters).end()
    return exported_spans.get_finished_spans()[0].attributes


def test_spec_parameter_lands_under_gen_ai_request(
    exported_spans: InMemorySpanExporter,
) -> None:
    attrs = _params(exported_spans, temperature=0.7)
    assert attrs["gen_ai.request.temperature"] == 0.7


def test_unrecognised_parameter_lands_under_rius_request(
    exported_spans: InMemorySpanExporter,
) -> None:
    """Our own namespace, not gen_ai.request.<key>: OTel's naming guidance
    forbids extending a semantic-convention namespace with application keys."""
    attrs = _params(exported_spans, my_custom_knob=3)
    assert attrs["rius.request.my_custom_knob"] == 3
    assert "gen_ai.request.my_custom_knob" not in attrs


def test_provider_spelling_lands_under_the_canonical_name_only(
    exported_spans: InMemorySpanExporter,
) -> None:
    """A recognised provider spelling normalises to ONE key, the canonical
    one; the provider spelling is not also emitted."""
    attrs = _params(exported_spans, max_completion_tokens=256)
    assert attrs["gen_ai.request.max_tokens"] == 256
    assert "gen_ai.request.max_completion_tokens" not in attrs
    assert "rius.request.max_completion_tokens" not in attrs


def test_openai_stop_normalises_to_stop_sequences(
    exported_spans: InMemorySpanExporter,
) -> None:
    attrs = _params(exported_spans, stop=["a", "b"])
    assert attrs["gen_ai.request.stop_sequences"] == ("a", "b")
    assert "gen_ai.request.stop" not in attrs


def test_top_logprobs_is_not_top_k(exported_spans: InMemorySpanExporter) -> None:
    """The spec's note on gen_ai.request.top_k says OpenAI's top_logprobs
    MUST NOT be reported as top_k; it is not a sampling parameter."""
    attrs = _params(exported_spans, top_logprobs=5)
    assert attrs["rius.request.top_logprobs"] == 5
    assert "gen_ai.request.top_k" not in attrs


def test_nested_parameter_values_are_json_encoded(
    exported_spans: InMemorySpanExporter,
) -> None:
    """OTel stores scalars and homogeneous scalar arrays only. Anything else
    is JSON-encoded rather than dropped: a response_format the model was
    actually sent is worth keeping, even as a string."""
    attrs = _params(
        exported_spans,
        response_format={"type": "json_object"},
        mixed=[1, "a"],
    )
    assert attrs["rius.request.response_format"] == '{"type": "json_object"}'
    assert attrs["rius.request.mixed"] == '[1, "a"]'


def test_none_valued_parameters_are_skipped(exported_spans: InMemorySpanExporter) -> None:
    attrs = _params(exported_spans, temperature=None, seed=None)
    assert "gen_ai.request.temperature" not in attrs
    assert "gen_ai.request.seed" not in attrs


def test_unrecognised_key_is_untouched_apart_from_the_prefix(
    exported_spans: InMemorySpanExporter,
) -> None:
    attrs = _params(exported_spans, **{"Weird.Key-1": "x"})
    assert attrs["rius.request.Weird.Key-1"] == "x"


def test_a_tools_model_parameter_lands_on_a_maskable_key(
    exported_spans: InMemorySpanExporter,
) -> None:
    """`tools` is not spec-defined, so it lands in rius.request.* — and that
    key must be one masking recognises as content (see test_masking.py)."""
    from rius.semconv import CONTENT_ATTRIBUTES

    attrs = _params(exported_spans, tools=[{"name": "get_weather"}])
    assert "rius.request.tools" in attrs
    assert "rius.request.tools" in CONTENT_ATTRIBUTES


# --- model_parameters: canonical keys carry only the value shape they define ---
#
# The native path goes through the SAME per-key guard as the normalizer's
# llm.invocation_parameters rule, so one parameter has one shape on the wire
# whichever way it arrived. A value that fails its key's guard is still a
# parameter the model was sent, so it lands under rius.request.<spelling>
# unchanged rather than being dropped.


def test_a_lone_stop_string_is_wrapped_into_a_list(exported_spans: InMemorySpanExporter) -> None:
    """gen_ai.request.stop_sequences is a string array; OpenAI accepts a lone
    string for `stop`, and it means the one-element list."""
    attrs = _params(exported_spans, stop="END")
    assert attrs["gen_ai.request.stop_sequences"] == ("END",)
    assert "rius.request.stop" not in attrs


def test_a_lone_encoding_format_is_wrapped_into_a_list(
    exported_spans: InMemorySpanExporter,
) -> None:
    attrs = _params(exported_spans, encoding_format="float")
    assert attrs["gen_ai.request.encoding_formats"] == ("float",)


def test_a_string_list_on_an_array_key_passes_through(
    exported_spans: InMemorySpanExporter,
) -> None:
    attrs = _params(exported_spans, stop=["a", "b"], embedding_types=["float", "int8"])
    assert attrs["gen_ai.request.stop_sequences"] == ("a", "b")
    assert attrs["gen_ai.request.encoding_formats"] == ("float", "int8")


class _Encoded:
    """A value recorded JSON-encoded; compared parsed, so the assertion does
    not pin the encoder's whitespace."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and json.loads(other) == self.value

    def __repr__(self) -> str:
        return f"_Encoded({self.value!r})"


@pytest.mark.parametrize(
    ("spelling", "value", "recorded"),
    [
        # numbers: a numeric string is not a number, and bool is not either
        ("temperature", "0.2", "0.2"),
        ("top_p", True, True),
        ("frequency_penalty", [0.1], (0.1,)),
        ("presencePenalty", "high", "high"),
        # counts: an object is JSON-encoded under OUR key, not the numeric one
        ("max_tokens", {"limit": 5}, _Encoded({"limit": 5})),
        ("top_k", "40", "40"),
        ("seed", False, False),
        ("n", "2", "2"),
        # text: the empty string is not a value, nor is a number
        ("model", "", ""),
        ("model", 4, 4),
        ("reasoning_effort", 3, 3),
        ("previous_response_id", ["r1"], ("r1",)),
        ("starting_after", 9, 9),
        # string arrays: a list of non-strings is not a string array
        ("stop", [["a"], ["b"]], _Encoded([["a"], ["b"]])),
        ("stop_sequences", [1, 2], (1, 2)),
        ("encoding_format", 7, 7),
        # flag
        ("stream", "yes", "yes"),
        ("stream", 1, 1),
    ],
)
def test_a_value_that_fails_its_keys_guard_lands_under_rius_request(
    exported_spans: InMemorySpanExporter, spelling: str, value: Any, recorded: Any
) -> None:
    from rius.semconv import request_attribute_key

    attrs = _params(exported_spans, **{spelling: value})
    assert request_attribute_key(spelling) not in attrs
    assert recorded == attrs[f"rius.request.{spelling}"]


def test_a_numeric_parameter_is_recorded_as_the_normalizer_records_it(
    exported_spans: InMemorySpanExporter,
) -> None:
    """The guard's output, not the caller's value: a temperature is a double,
    a token limit an int, as they are when llm.invocation_parameters is
    promoted."""
    attrs = _params(exported_spans, temperature=1, max_tokens=256.0)
    assert attrs["gen_ai.request.temperature"] == 1.0
    assert isinstance(attrs["gen_ai.request.temperature"], float)
    assert attrs["gen_ai.request.max_tokens"] == 256
    assert isinstance(attrs["gen_ai.request.max_tokens"], int)


# --- model_parameters: two spellings of one parameter, the first one wins ---


def test_two_spellings_of_one_parameter_the_first_one_wins(
    exported_spans: InMemorySpanExporter,
) -> None:
    """First in the spelling table's precedence, as in every rule of the
    normalizer's table. The losing spelling is still a parameter the caller
    sent, so it is kept under our own namespace rather than dropped."""
    attrs = _params(exported_spans, max_tokens=100, max_completion_tokens=200)
    assert attrs["gen_ai.request.max_tokens"] == 100
    assert attrs["rius.request.max_completion_tokens"] == 200


def test_collision_precedence_does_not_depend_on_the_callers_order(
    exported_spans: InMemorySpanExporter,
) -> None:
    attrs = _params(exported_spans, max_output_tokens=300, max_completion_tokens=200, maxTokens=1)
    assert attrs["gen_ai.request.max_tokens"] == 1
    assert attrs["rius.request.max_completion_tokens"] == 200
    assert attrs["rius.request.max_output_tokens"] == 300


def test_a_spelling_that_fails_its_guard_does_not_claim_the_key(
    exported_spans: InMemorySpanExporter,
) -> None:
    """ "First to PRODUCE a value", as the invocation-parameters rule says: a
    wrongly-shaped first spelling leaves the key to the next one."""
    attrs = _params(exported_spans, max_tokens="100", max_output_tokens=200)
    assert attrs["gen_ai.request.max_tokens"] == 200
    assert attrs["rius.request.max_tokens"] == "100"


def test_the_model_argument_still_beats_a_model_parameter(
    exported_spans: InMemorySpanExporter,
) -> None:
    start_generation("chat", model="gpt-4o", model_parameters={"model": "other"}).end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.model"] == "gpt-4o"


@pytest.mark.parametrize("spelling", ["temperature", "max_tokens", "seed"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_fails_a_numeric_guard(
    exported_spans: InMemorySpanExporter, spelling: str, value: float
) -> None:
    """Not a number a model can be sent, and int() of one raises: the guard
    rejects it (as the TypeScript SDK's does) instead of failing the call."""
    attrs = _params(exported_spans, **{spelling: value})
    assert f"gen_ai.request.{spelling}" not in attrs
    assert f"rius.request.{spelling}" in attrs
