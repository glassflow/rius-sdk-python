"""Compactions a provider reports in its response, recovered from ``output.value``.

OpenInference writes nothing about them: Anthropic's context editing and
compaction blocks, and OpenAI's compaction items, exist only inside the
serialized provider object an LLM span carries under ``output.value``. They
are lifted at export, before masking, into one ``rius.compaction`` event each
plus ``gen_ai.conversation.compacted`` on the span.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from anthropic.types.beta import BetaMessage
from openai.types.responses import Response
from openai.types.responses.compacted_response import CompactedResponse
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius import normalization as normalization_module
from rius._provider_compaction import compactions_in_response
from rius._serde import MAX_ATTR_CHARS, TRUNCATION_MARKER
from rius.masking import MaskingSpanExporter
from rius.normalization import NormalizingSpanExporter, compactions_from_output_value
from rius.semconv import (
    GEN_AI_CONVERSATION_COMPACTED,
    RIUS_COMPACTION_EVENT,
    RIUS_COMPACTION_EXECUTOR,
    RIUS_COMPACTION_ID,
    RIUS_COMPACTION_KIND,
    RIUS_COMPACTION_SUMMARY_BYTES,
    RIUS_COMPACTION_SUMMARY_TEXT,
    RIUS_COMPACTION_TOKENS_AFTER,
    RIUS_COMPACTION_TOKENS_BEFORE,
    RIUS_COMPACTION_TOKENS_DROPPED,
    RIUS_COMPACTION_TOOL_RESULTS_CLEARED,
)

_SUMMARY = "The user is migrating a billing service; the tests pass."


def _usage(input_tokens: int = 10) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": 5,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _clear_tool_uses(tokens: int = 4000, uses: int = 3) -> dict[str, Any]:
    return {
        "type": "clear_tool_uses_20250919",
        "cleared_input_tokens": tokens,
        "cleared_tool_uses": uses,
    }


def _clear_thinking(tokens: int = 900, turns: int = 2) -> dict[str, Any]:
    return {
        "type": "clear_thinking_20251015",
        "cleared_input_tokens": tokens,
        "cleared_thinking_turns": turns,
    }


def _iteration(kind: str, input_tokens: int) -> dict[str, Any]:
    return {"type": kind, **_usage(input_tokens)}


def _anthropic_body(
    *,
    content: list[dict[str, Any]] | None = None,
    edits: list[dict[str, Any]] | None = None,
    iterations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    usage = _usage()
    if iterations is not None:
        usage["iterations"] = iterations
    body: dict[str, Any] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": content if content is not None else [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }
    if edits is not None:
        body["context_management"] = {"applied_edits": edits}
    return body


def _anthropic(**parts: Any) -> str:
    """The JSON the instrumentor records: a real beta message, dumped."""
    return BetaMessage.model_validate(_anthropic_body(**parts)).model_dump_json()


def _compaction_block(content: str | None = _SUMMARY) -> dict[str, Any]:
    return {"type": "compaction", "content": content}


def _text_item(text: str = "hi") -> dict[str, Any]:
    return {
        "id": "msg_item",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _compaction_item(item_id: str = "cmp_1") -> dict[str, Any]:
    return {"id": item_id, "type": "compaction", "encrypted_content": "opaque"}


_OPENAI_USAGE = {
    "input_tokens": 1,
    "output_tokens": 1,
    "total_tokens": 2,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


def _openai(output: list[dict[str, Any]]) -> str:
    return Response.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-5",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": _OPENAI_USAGE,
        }
    ).model_dump_json()


def _export(
    attributes: dict[str, Any],
    *,
    capture_content: bool = True,
    mask: Callable[[Any], Any] | None = None,
) -> ReadableSpan:
    inner = InMemorySpanExporter()
    chain: Any = inner
    if not capture_content or mask is not None:
        chain = MaskingSpanExporter(chain, capture_content=capture_content, mask=mask)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(chain)))
    with provider.get_tracer("t").start_as_current_span("call") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    (exported,) = inner.get_finished_spans()
    return exported


def _llm(output_value: Any, **extra: Any) -> dict[str, Any]:
    return {"openinference.span.kind": "LLM", "output.value": output_value, **extra}


def _events(span: ReadableSpan) -> list[dict[str, Any]]:
    return [dict(e.attributes or {}) for e in span.events if e.name == RIUS_COMPACTION_EVENT]


# --- Anthropic context editing -----------------------------------------------


def test_a_tool_use_clearing_edit_is_an_event_and_marks_the_span() -> None:
    span = _export(_llm(_anthropic(edits=[_clear_tool_uses(4000, 3)])))
    assert _events(span) == [
        {
            RIUS_COMPACTION_KIND: "clear_tool_results",
            RIUS_COMPACTION_EXECUTOR: "provider",
            RIUS_COMPACTION_TOKENS_DROPPED: 4000,
            RIUS_COMPACTION_TOOL_RESULTS_CLEARED: 3,
        }
    ]
    assert (span.attributes or {})[GEN_AI_CONVERSATION_COMPACTED] is True


def test_a_thinking_clearing_edit_has_no_tool_result_count() -> None:
    span = _export(_llm(_anthropic(edits=[_clear_thinking(900)])))
    assert _events(span) == [
        {
            RIUS_COMPACTION_KIND: "clear_thinking",
            RIUS_COMPACTION_EXECUTOR: "provider",
            RIUS_COMPACTION_TOKENS_DROPPED: 900,
        }
    ]
    assert (span.attributes or {})[GEN_AI_CONVERSATION_COMPACTED] is True


def test_every_applied_edit_is_its_own_event_in_order() -> None:
    span = _export(_llm(_anthropic(edits=[_clear_thinking(), _clear_tool_uses()])))
    assert [e[RIUS_COMPACTION_KIND] for e in _events(span)] == [
        "clear_thinking",
        "clear_tool_results",
    ]


@pytest.mark.parametrize(
    ("edit_type", "kind"),
    [
        ("clear_tool_uses_20990101", "clear_tool_results"),
        ("clear_thinking_20990101", "clear_thinking"),
    ],
)
def test_a_later_dated_edit_type_is_read_by_its_family(edit_type: str, kind: str) -> None:
    edit = {"type": edit_type, "cleared_input_tokens": 7}
    body = json.dumps(_anthropic_body(edits=[edit]))
    assert [e[RIUS_COMPACTION_KIND] for e in _events(_export(_llm(body)))] == [kind]


def test_an_edit_of_an_unknown_family_reports_nothing() -> None:
    edit = {"type": "summon_the_kraken_20990101", "cleared_input_tokens": 7}
    span = _export(_llm(json.dumps(_anthropic_body(edits=[edit]))))
    assert _events(span) == []
    assert GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {})


def test_an_edit_without_a_cleared_size_omits_the_key() -> None:
    edit = {"type": "clear_tool_uses_20250919", "cleared_tool_uses": 2}
    (event,) = _events(_export(_llm(json.dumps(_anthropic_body(edits=[edit])))))
    assert RIUS_COMPACTION_TOKENS_DROPPED not in event
    assert event[RIUS_COMPACTION_TOOL_RESULTS_CLEARED] == 2


@pytest.mark.parametrize("size", ["many", True, -5, 1.5, None])
def test_a_size_that_is_not_a_count_is_left_out(size: Any) -> None:
    edit = {
        "type": "clear_tool_uses_20250919",
        "cleared_input_tokens": size,
        "cleared_tool_uses": size,
    }
    (event,) = _events(_export(_llm(json.dumps(_anthropic_body(edits=[edit])))))
    assert RIUS_COMPACTION_TOKENS_DROPPED not in event
    assert RIUS_COMPACTION_TOOL_RESULTS_CLEARED not in event


def test_the_event_is_stamped_when_the_call_ended() -> None:
    span = _export(_llm(_anthropic(edits=[_clear_tool_uses()])))
    (event,) = [e for e in span.events if e.name == RIUS_COMPACTION_EVENT]
    assert event.timestamp == span.end_time


def test_a_response_with_no_edits_reports_nothing() -> None:
    span = _export(_llm(_anthropic(edits=[])))
    assert _events(span) == []
    assert GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {})


# --- Anthropic compaction blocks ----------------------------------------------


def test_a_compaction_block_is_a_summarize_event_sized_from_its_iterations() -> None:
    body = _anthropic(
        content=[_compaction_block(), {"type": "text", "text": "hi"}],
        iterations=[_iteration("compaction", 9000), _iteration("message", 400)],
    )
    span = _export(_llm(body))
    assert _events(span) == [
        {
            RIUS_COMPACTION_KIND: "summarize",
            RIUS_COMPACTION_EXECUTOR: "provider",
            RIUS_COMPACTION_TOKENS_BEFORE: 9000,
            RIUS_COMPACTION_TOKENS_AFTER: 400,
            RIUS_COMPACTION_TOKENS_DROPPED: 8600,
            RIUS_COMPACTION_SUMMARY_BYTES: len(_SUMMARY.encode()),
            RIUS_COMPACTION_SUMMARY_TEXT: _SUMMARY,
        }
    ]
    assert (span.attributes or {})[GEN_AI_CONVERSATION_COMPACTED] is True


def test_a_compaction_block_without_iterations_carries_no_sizes() -> None:
    (event,) = _events(_export(_llm(_anthropic(content=[_compaction_block()]))))
    assert event[RIUS_COMPACTION_KIND] == "summarize"
    for key in (RIUS_COMPACTION_TOKENS_BEFORE, RIUS_COMPACTION_TOKENS_AFTER):
        assert key not in event
    assert RIUS_COMPACTION_TOKENS_DROPPED not in event


def test_the_message_after_the_compaction_is_not_the_compaction_itself() -> None:
    # With only the compaction iteration there is no "after" to report; the
    # last iteration is that same compaction, not a size of what it left.
    body = _anthropic(content=[_compaction_block()], iterations=[_iteration("compaction", 9000)])
    (event,) = _events(_export(_llm(body)))
    assert event[RIUS_COMPACTION_TOKENS_BEFORE] == 9000
    assert RIUS_COMPACTION_TOKENS_AFTER not in event
    assert RIUS_COMPACTION_TOKENS_DROPPED not in event


def test_a_context_that_grew_has_no_dropped_count() -> None:
    body = _anthropic(
        content=[_compaction_block()],
        iterations=[_iteration("compaction", 100), _iteration("message", 400)],
    )
    (event,) = _events(_export(_llm(body)))
    assert event[RIUS_COMPACTION_TOKENS_AFTER] == 400
    assert RIUS_COMPACTION_TOKENS_DROPPED not in event


def test_a_compaction_block_never_gets_an_invented_id() -> None:
    (event,) = _events(_export(_llm(_anthropic(content=[_compaction_block()]))))
    assert RIUS_COMPACTION_ID not in event


def test_a_block_that_carries_an_id_passes_it_on() -> None:
    block = {**_compaction_block(), "id": "cmp_native"}
    body = json.dumps(_anthropic_body(content=[block]))
    (event,) = _events(_export(_llm(body)))
    assert event[RIUS_COMPACTION_ID] == "cmp_native"


def test_the_summary_size_is_in_utf8_bytes() -> None:
    summary = "héllo ✓"
    (event,) = _events(_export(_llm(_anthropic(content=[_compaction_block(summary)]))))
    assert event[RIUS_COMPACTION_SUMMARY_BYTES] == len(summary.encode("utf-8")) == 10


def test_a_long_summary_is_capped_like_any_attribute_but_sized_whole() -> None:
    summary = "x" * (MAX_ATTR_CHARS + 500)
    (event,) = _events(_export(_llm(_anthropic(content=[_compaction_block(summary)]))))
    assert event[RIUS_COMPACTION_SUMMARY_TEXT] == "x" * MAX_ATTR_CHARS + TRUNCATION_MARKER
    assert event[RIUS_COMPACTION_SUMMARY_BYTES] == len(summary)


def test_a_failed_compaction_block_with_no_summary_is_not_a_compaction() -> None:
    # The SDK type: content None means the compaction produced no summary and
    # the server treats the block as a no-op.
    span = _export(_llm(_anthropic(content=[_compaction_block(None)])))
    assert _events(span) == []
    assert GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {})


def test_an_edit_and_a_compaction_block_are_both_reported() -> None:
    body = _anthropic(content=[_compaction_block()], edits=[_clear_thinking()])
    assert [e[RIUS_COMPACTION_KIND] for e in _events(_export(_llm(body)))] == [
        "clear_thinking",
        "summarize",
    ]


# --- privacy ------------------------------------------------------------------


def test_the_summary_text_is_exported_when_content_is_captured() -> None:
    (event,) = _events(_export(_llm(_anthropic(content=[_compaction_block()]))))
    assert event[RIUS_COMPACTION_SUMMARY_TEXT] == _SUMMARY


def test_capture_off_strips_the_summary_text_but_keeps_its_size() -> None:
    span = _export(_llm(_anthropic(content=[_compaction_block()])), capture_content=False)
    (event,) = _events(span)
    assert RIUS_COMPACTION_SUMMARY_TEXT not in event
    assert event[RIUS_COMPACTION_SUMMARY_BYTES] == len(_SUMMARY.encode())
    assert event[RIUS_COMPACTION_KIND] == "summarize"
    assert (span.attributes or {})[GEN_AI_CONVERSATION_COMPACTED] is True
    assert "output.value" not in (span.attributes or {})


def test_a_user_mask_sees_the_summary_text() -> None:
    def mask(value: Any) -> Any:
        return str(value).replace("billing", "[redacted]")

    span = _export(_llm(_anthropic(content=[_compaction_block()])), mask=mask)
    (event,) = _events(span)
    assert event[RIUS_COMPACTION_SUMMARY_TEXT] == _SUMMARY.replace("billing", "[redacted]")


# --- OpenAI -------------------------------------------------------------------


def test_an_openai_compaction_item_is_a_summarize_event_with_its_id() -> None:
    span = _export(_llm(_openai([_compaction_item("cmp_9"), _text_item()])))
    assert _events(span) == [
        {
            RIUS_COMPACTION_KIND: "summarize",
            RIUS_COMPACTION_EXECUTOR: "provider",
            RIUS_COMPACTION_ID: "cmp_9",
        }
    ]
    assert (span.attributes or {})[GEN_AI_CONVERSATION_COMPACTED] is True


def test_a_response_without_a_compaction_item_reports_nothing() -> None:
    span = _export(_llm(_openai([_text_item()])))
    assert _events(span) == []
    assert GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {})


def test_the_compact_endpoint_reply_is_not_marked() -> None:
    # responses.compact is the compaction call itself, recorded by the OpenAI
    # instrumentor as an LLM span whose output.value holds the compacted
    # window. It is not a call that ran on a compacted context.
    reply = CompactedResponse.model_validate(
        {
            "id": "resp_c",
            "object": "response.compaction",
            "created_at": 1,
            "output": [_compaction_item()],
            "usage": _OPENAI_USAGE,
        }
    ).model_dump_json()
    span = _export(_llm(reply))
    assert _events(span) == []
    assert GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {})


# --- what never qualifies ------------------------------------------------------


@pytest.mark.parametrize("kind", ["TOOL", "CHAIN", "AGENT", "RETRIEVER"])
def test_a_non_llm_span_carrying_the_same_json_gets_nothing(kind: str) -> None:
    body = _anthropic(content=[_compaction_block()], edits=[_clear_thinking()])
    span = _export({"openinference.span.kind": kind, "output.value": body})
    assert _events(span) == []
    assert GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {})


def test_an_operation_only_llm_span_qualifies_through_the_taxonomy() -> None:
    body = _anthropic(edits=[_clear_thinking()])
    span = _export({"gen_ai.operation.name": "chat", "output.value": body})
    assert len(_events(span)) == 1


def test_a_non_json_mime_type_is_skipped() -> None:
    body = _anthropic(edits=[_clear_thinking()])
    span = _export(_llm(body, **{"output.mime_type": "text/plain"}))
    assert _events(span) == []


def test_an_llm_span_without_a_compaction_gets_nothing() -> None:
    span = _export(_llm(_anthropic()))
    assert _events(span) == []
    assert GEN_AI_CONVERSATION_COMPACTED not in (span.attributes or {})


def test_the_model_saying_the_word_is_not_a_compaction() -> None:
    reply = _anthropic(content=[{"type": "text", "text": 'a "compaction" and "applied_edits"'}])
    span = _export(_llm(reply))
    assert _events(span) == []


# --- native wins ----------------------------------------------------------------


def test_a_native_compacted_flag_is_not_overwritten() -> None:
    body = _anthropic(edits=[_clear_thinking()])
    span = _export(_llm(body, **{GEN_AI_CONVERSATION_COMPACTED: False}))
    assert (span.attributes or {})[GEN_AI_CONVERSATION_COMPACTED] is False


def test_a_span_that_already_has_a_compaction_event_is_left_as_it_is() -> None:
    inner = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(inner)))
    with provider.get_tracer("t").start_as_current_span("call") as span:
        for key, value in _llm(_anthropic(edits=[_clear_thinking()])).items():
            span.set_attribute(key, value)
        span.add_event(RIUS_COMPACTION_EVENT, {RIUS_COMPACTION_KIND: "truncate"})
    (exported,) = inner.get_finished_spans()
    assert [e[RIUS_COMPACTION_KIND] for e in _events(exported)] == ["truncate"]


def test_output_value_is_untouched() -> None:
    body = _anthropic(edits=[_clear_thinking()])
    assert (_export(_llm(body)).attributes or {})["output.value"] == body


# --- the hot path -----------------------------------------------------------------


class _CountingLoads:
    def __init__(self) -> None:
        self.calls = 0
        self._loads = json.loads

    def __call__(self, text: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._loads(text, *args, **kwargs)


@pytest.fixture
def counting_loads(monkeypatch: pytest.MonkeyPatch) -> Iterator[_CountingLoads]:
    spy = _CountingLoads()
    monkeypatch.setattr(normalization_module.json, "loads", spy)
    yield spy


@pytest.mark.parametrize(
    "attributes",
    [
        _llm("plain text reply"),
        _llm('[{"type":"compaction"}]'),
        _llm(' {"type":"compaction"}'),
        _llm(42),
        {"openinference.span.kind": "TOOL", "output.value": '{"type":"compaction"}'},
        _llm('{"type":"compaction"}', **{"output.mime_type": "text/plain"}),
        {"openinference.span.kind": "LLM"},
    ],
)
def test_the_cheap_checks_run_before_any_parse(
    counting_loads: _CountingLoads, attributes: dict[str, Any]
) -> None:
    assert compactions_from_output_value(attributes) == []
    assert counting_loads.calls == 0


def test_a_response_with_no_compaction_marker_is_never_parsed_for_one(
    counting_loads: _CountingLoads,
) -> None:
    assert compactions_from_output_value(_llm(_anthropic())) == []
    assert counting_loads.calls == 0


@pytest.mark.parametrize(
    "output_value",
    ['{"compaction": ', '{"type":"compaction"', '{"applied_edits": [}', '{"compaction"}'],
)
def test_a_marker_in_text_that_does_not_parse_means_nothing(output_value: str) -> None:
    assert compactions_from_output_value(_llm(output_value)) == []


# --- end to end: the real instrumentors through init() ---------------------------


class _Provider:
    """A local HTTP server replying as a provider does, with the body per path.

    A real socket: the provider SDKs pin their own HTTP clients, and a socket
    works whichever one they use.
    """

    def __init__(self, replies: dict[str, Any], stream_body: str = "") -> None:
        import http.server
        import threading

        encoded = {path: json.dumps(body).encode() for path, body in replies.items()}
        streamed = stream_body.encode()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (http.server API)
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                body, content_type = (
                    (streamed, "text/event-stream")
                    if request.get("stream")
                    else (encoded[self.path.split("?")[0]], "application/json")
                )
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _instrumentor(module: str, name: str) -> Iterator[Any]:
    oi = pytest.importorskip(module)
    instrumentor = getattr(oi, name)()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    yield instrumentor
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


@pytest.fixture
def openai_instrumentor() -> Iterator[Any]:
    yield from _instrumentor("openinference.instrumentation.openai", "OpenAIInstrumentor")


@pytest.fixture
def anthropic_instrumentor() -> Iterator[Any]:
    yield from _instrumentor("openinference.instrumentation.anthropic", "AnthropicInstrumentor")


def _spans(inner: InMemorySpanExporter) -> list[ReadableSpan]:
    return list(inner.get_finished_spans())


@pytest.mark.integration
@pytest.mark.parametrize("capture_content", [True, False])
def test_real_anthropic_beta_message_reports_its_compaction(
    anthropic_instrumentor: Any, capture_content: bool
) -> None:
    anthropic = pytest.importorskip("anthropic")
    reply = _anthropic_body(
        content=[_compaction_block(), {"type": "text", "text": "hi"}],
        edits=[_clear_tool_uses(4000, 3)],
        iterations=[_iteration("compaction", 9000), _iteration("message", 400)],
    )
    server = _Provider({"/v1/messages": reply})
    inner = InMemorySpanExporter()
    client = init(
        span_exporter=inner,
        set_global=False,
        instruments=["anthropic"],
        capture_content=capture_content,
    )
    try:
        anthropic.Anthropic(api_key="test-key", base_url=server.url).beta.messages.create(
            model="claude-sonnet-5",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        client.flush()
    finally:
        client.shutdown()
        server.close()

    (span,) = _spans(inner)
    assert [e[RIUS_COMPACTION_KIND] for e in _events(span)] == ["clear_tool_results", "summarize"]
    summary = _events(span)[1]
    assert summary[RIUS_COMPACTION_TOKENS_BEFORE] == 9000
    assert (RIUS_COMPACTION_SUMMARY_TEXT in summary) is capture_content
    assert summary[RIUS_COMPACTION_SUMMARY_BYTES] == len(_SUMMARY.encode())
    assert (span.attributes or {})[GEN_AI_CONVERSATION_COMPACTED] is True


@pytest.mark.integration
def test_real_anthropic_beta_stream_reports_its_compaction(anthropic_instrumentor: Any) -> None:
    # The streamed path serializes the final snapshot, which carries the
    # message_delta's context_management and usage.iterations.
    anthropic = pytest.importorskip("anthropic")
    start = _anthropic_body(content=[])
    start["stop_reason"] = None
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": start}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "compaction", "content": None},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "compaction_delta", "content": _SUMMARY},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "context_management": {"applied_edits": [_clear_thinking(900)]},
                "usage": {
                    "output_tokens": 5,
                    "iterations": [_iteration("compaction", 9000), _iteration("message", 400)],
                },
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    server = _Provider({}, stream_body=body)
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=["anthropic"])
    try:
        messages = anthropic.Anthropic(api_key="test-key", base_url=server.url).beta.messages
        with messages.stream(
            model="claude-sonnet-5",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        ) as stream:
            for _ in stream:
                pass
        client.flush()
    finally:
        client.shutdown()
        server.close()

    (span,) = _spans(inner)
    assert [e[RIUS_COMPACTION_KIND] for e in _events(span)] == ["clear_thinking", "summarize"]
    assert _events(span)[1][RIUS_COMPACTION_TOKENS_AFTER] == 400


@pytest.mark.integration
def test_real_openai_responses_reports_a_compaction_item_and_not_the_compact_call(
    openai_instrumentor: Any,
) -> None:
    openai = pytest.importorskip("openai")
    compacted = {
        "id": "resp_c",
        "object": "response.compaction",
        "created_at": 1,
        "output": [_compaction_item()],
        "usage": _OPENAI_USAGE,
    }
    server = _Provider(
        {
            "/v1/responses": json.loads(_openai([_compaction_item("cmp_7"), _text_item()])),
            "/v1/responses/compact": compacted,
        }
    )
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, instruments=["openai"])
    try:
        oai = openai.OpenAI(api_key="test-key", base_url=f"{server.url}/v1")
        oai.responses.create(model="gpt-5", input="hi")
        oai.responses.compact(model="gpt-5", input="hi")
        client.flush()
    finally:
        client.shutdown()
        server.close()

    by_name = {span.name: span for span in _spans(inner)}
    created = by_name["Response"]
    assert [e[RIUS_COMPACTION_ID] for e in _events(created)] == ["cmp_7"]
    assert (created.attributes or {})[GEN_AI_CONVERSATION_COMPACTED] is True
    compact_call = by_name["CompactedResponse"]
    assert _events(compact_call) == []
    assert GEN_AI_CONVERSATION_COMPACTED not in (compact_call.attributes or {})


@pytest.mark.parametrize(
    "raw",
    ['["compaction"]', '"applied_edits"', "null", "7", '[{"type": "compaction", "content": "x"}]'],
)
def test_a_json_value_that_is_not_an_object_reports_nothing(raw: str) -> None:
    assert compactions_in_response(raw) == []
