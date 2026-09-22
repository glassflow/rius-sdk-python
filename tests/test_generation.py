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
        assert _sizes_of_live(gen) == {
            "version": 1,
            "input_messages": [{"role": "system", "parts": [{"type": "text", "bytes": 3}]}],
        }
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
    with client.get_tracer().start_as_current_span("chat") as span:
        Generation(span).set_input("secret prompt")
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
    with client.get_tracer().start_as_current_span("chat") as span:
        Generation(span).set_input("a much longer secret prompt")
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
