import json

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import start_as_current_generation, start_generation

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
