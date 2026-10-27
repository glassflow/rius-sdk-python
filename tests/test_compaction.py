"""``mark_compaction``: an app that shrinks its own history says so.

The span it records is the compaction (client executor); the next model call
in the same context is flagged ``gen_ai.conversation.compacted`` once, so a
reader can tell the call that ran on the shrunk context.
"""

from __future__ import annotations

import contextvars
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rius
from rius import _tracer, init, mark_compaction
from rius._serde import MAX_ATTR_CHARS, TRUNCATION_MARKER
from rius.compaction import OPERATION_COMPACTION, CompactedContextProcessor
from rius.masking import MaskingSpanExporter
from rius.normalization import NormalizingSpanExporter
from rius.semconv import (
    GEN_AI_CONVERSATION_COMPACTED,
    GEN_AI_OPERATION_NAME,
    OPENINFERENCE_SPAN_KIND,
    RIUS_COMPACTION_EXECUTOR,
    RIUS_COMPACTION_KIND,
    RIUS_COMPACTION_MESSAGES_COMPACTED,
    RIUS_COMPACTION_MESSAGES_KEPT,
    RIUS_COMPACTION_SUMMARY_BYTES,
    RIUS_COMPACTION_SUMMARY_TEXT,
    RIUS_COMPACTION_TOKENS_AFTER,
    RIUS_COMPACTION_TOKENS_BEFORE,
    RIUS_COMPACTION_TOKENS_DROPPED,
    RIUS_COMPACTION_TRIGGER,
    RIUS_SPAN_PENDING,
    SpanKind,
    kind_for_operation,
)


class _Started(SpanProcessor):
    """Records the attributes each span had the moment it started."""

    def __init__(self) -> None:
        self.at_start: dict[str, dict[str, Any]] = {}

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        self.at_start[span.name] = dict(span.attributes or {})


class _Pipeline:
    def __init__(self, *, capture_content: bool = True) -> None:
        self.exporter = InMemorySpanExporter()
        self.started = _Started()
        sink: Any = self.exporter
        if not capture_content:
            sink = MaskingSpanExporter(sink, capture_content=False)
        processor = CompactedContextProcessor()
        sink = NormalizingSpanExporter(sink, compacted_context=processor)
        self.provider = TracerProvider()
        self.provider.add_span_processor(self.started)
        self.provider.add_span_processor(processor)
        self.provider.add_span_processor(SimpleSpanProcessor(sink))
        _tracer.publish(self.provider)
        self.tracer = self.provider.get_tracer("app")

    def spans(self) -> list[ReadableSpan]:
        return list(self.exporter.get_finished_spans())

    def named(self, name: str) -> ReadableSpan:
        (span,) = [s for s in self.spans() if s.name == name]
        return span

    def llm(self, name: str = "llm") -> None:
        with self.tracer.start_as_current_span(name, attributes={OPENINFERENCE_SPAN_KIND: "LLM"}):
            pass


@pytest.fixture
def pipeline() -> Iterator[_Pipeline]:
    yield _Pipeline()


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _compaction(pipeline: _Pipeline) -> dict[str, Any]:
    return _attrs(pipeline.named("compaction"))


def _marked(pipeline: _Pipeline, name: str) -> bool:
    return _attrs(pipeline.named(name)).get(GEN_AI_CONVERSATION_COMPACTED) is True


# --- the span -----------------------------------------------------------------


def test_records_a_finished_chain_span_named_compaction(pipeline: _Pipeline) -> None:
    mark_compaction()
    span = pipeline.named("compaction")
    attrs = _attrs(span)
    assert attrs[OPENINFERENCE_SPAN_KIND] == "CHAIN"
    assert attrs[GEN_AI_OPERATION_NAME] == "compaction"
    assert attrs[RIUS_COMPACTION_EXECUTOR] == "client"
    assert attrs[RIUS_COMPACTION_KIND] == "summarize"
    assert span.end_time is not None


def test_is_a_child_of_the_current_span(pipeline: _Pipeline) -> None:
    with pipeline.tracer.start_as_current_span("agent") as parent:
        mark_compaction()
    child = pipeline.named("compaction")
    assert child.parent is not None
    assert child.parent.span_id == parent.get_span_context().span_id
    assert child.context.trace_id == parent.get_span_context().trace_id


def test_is_a_root_when_no_span_is_current(pipeline: _Pipeline) -> None:
    mark_compaction()
    assert pipeline.named("compaction").parent is None


def test_does_not_become_the_current_span(pipeline: _Pipeline) -> None:
    with pipeline.tracer.start_as_current_span("agent"):
        mark_compaction()
        pipeline.llm("after")
    assert pipeline.named("after").parent == pipeline.named("agent").context


@pytest.mark.parametrize("kind", ["summarize", "clear_tool_results", "clear_thinking", "truncate"])
def test_records_the_kind_it_is_given(pipeline: _Pipeline, kind: Any) -> None:
    mark_compaction(kind)
    assert _compaction(pipeline)[RIUS_COMPACTION_KIND] == kind


@pytest.mark.parametrize("trigger", ["auto", "manual"])
def test_records_the_trigger_it_is_given(pipeline: _Pipeline, trigger: Any) -> None:
    mark_compaction(trigger=trigger)
    assert _compaction(pipeline)[RIUS_COMPACTION_TRIGGER] == trigger


def test_an_argument_left_out_leaves_its_key_out(pipeline: _Pipeline) -> None:
    mark_compaction()
    attrs = _compaction(pipeline)
    for key in (
        RIUS_COMPACTION_TRIGGER,
        RIUS_COMPACTION_TOKENS_BEFORE,
        RIUS_COMPACTION_TOKENS_AFTER,
        RIUS_COMPACTION_TOKENS_DROPPED,
        RIUS_COMPACTION_MESSAGES_COMPACTED,
        RIUS_COMPACTION_MESSAGES_KEPT,
        RIUS_COMPACTION_SUMMARY_BYTES,
        RIUS_COMPACTION_SUMMARY_TEXT,
    ):
        assert key not in attrs


def test_token_counts_and_what_they_dropped(pipeline: _Pipeline) -> None:
    mark_compaction(tokens_before=9000, tokens_after=2500)
    attrs = _compaction(pipeline)
    assert attrs[RIUS_COMPACTION_TOKENS_BEFORE] == 9000
    assert attrs[RIUS_COMPACTION_TOKENS_AFTER] == 2500
    assert attrs[RIUS_COMPACTION_TOKENS_DROPPED] == 6500


def test_a_compaction_that_dropped_nothing_says_zero(pipeline: _Pipeline) -> None:
    mark_compaction(tokens_before=500, tokens_after=500)
    assert _compaction(pipeline)[RIUS_COMPACTION_TOKENS_DROPPED] == 0


def test_a_context_that_grew_has_no_dropped_count(pipeline: _Pipeline) -> None:
    mark_compaction(tokens_before=500, tokens_after=900)
    attrs = _compaction(pipeline)
    assert attrs[RIUS_COMPACTION_TOKENS_BEFORE] == 500
    assert attrs[RIUS_COMPACTION_TOKENS_AFTER] == 900
    assert RIUS_COMPACTION_TOKENS_DROPPED not in attrs


@pytest.mark.parametrize("known", ["tokens_before", "tokens_after"])
def test_one_count_alone_derives_no_dropped_count(pipeline: _Pipeline, known: str) -> None:
    mark_compaction(**{known: 700})
    attrs = _compaction(pipeline)
    assert RIUS_COMPACTION_TOKENS_DROPPED not in attrs


def test_message_counts(pipeline: _Pipeline) -> None:
    mark_compaction(messages_compacted=40, messages_kept=6)
    attrs = _compaction(pipeline)
    assert attrs[RIUS_COMPACTION_MESSAGES_COMPACTED] == 40
    assert attrs[RIUS_COMPACTION_MESSAGES_KEPT] == 6


def test_a_zero_count_is_kept_not_taken_for_absent(pipeline: _Pipeline) -> None:
    mark_compaction(messages_compacted=0, messages_kept=0, tokens_before=0, tokens_after=0)
    attrs = _compaction(pipeline)
    assert attrs[RIUS_COMPACTION_MESSAGES_COMPACTED] == 0
    assert attrs[RIUS_COMPACTION_MESSAGES_KEPT] == 0
    assert attrs[RIUS_COMPACTION_TOKENS_BEFORE] == 0


# --- the summary --------------------------------------------------------------


def test_the_summary_is_sized_in_utf8_bytes_and_carried_as_text(pipeline: _Pipeline) -> None:
    mark_compaction(summary="héllo ✓")
    attrs = _compaction(pipeline)
    assert attrs[RIUS_COMPACTION_SUMMARY_BYTES] == 10
    assert attrs[RIUS_COMPACTION_SUMMARY_TEXT] == "héllo ✓"


def test_an_empty_summary_is_still_a_summary(pipeline: _Pipeline) -> None:
    mark_compaction(summary="")
    assert _compaction(pipeline)[RIUS_COMPACTION_SUMMARY_BYTES] == 0


def test_the_summary_text_is_never_on_the_span_at_creation(pipeline: _Pipeline) -> None:
    mark_compaction(summary="secret plan")
    at_start = pipeline.started.at_start["compaction"]
    assert RIUS_COMPACTION_SUMMARY_TEXT not in at_start
    assert at_start[OPENINFERENCE_SPAN_KIND] == "CHAIN"


def test_capture_off_strips_the_text_and_keeps_the_size() -> None:
    pipeline = _Pipeline(capture_content=False)
    mark_compaction(summary="secret plan")
    attrs = _compaction(pipeline)
    assert RIUS_COMPACTION_SUMMARY_TEXT not in attrs
    assert attrs[RIUS_COMPACTION_SUMMARY_BYTES] == len("secret plan")


def test_a_long_summary_is_capped_like_any_attribute_but_sized_whole(pipeline: _Pipeline) -> None:
    summary = "x" * (MAX_ATTR_CHARS + 500)
    mark_compaction(summary=summary)
    attrs = _compaction(pipeline)
    assert attrs[RIUS_COMPACTION_SUMMARY_TEXT] == "x" * MAX_ATTR_CHARS + TRUNCATION_MARKER
    assert attrs[RIUS_COMPACTION_SUMMARY_BYTES] == len(summary)


# --- validation ---------------------------------------------------------------


def test_an_unknown_kind_is_a_value_error_and_records_nothing(pipeline: _Pipeline) -> None:
    with pytest.raises(ValueError, match="kind"):
        mark_compaction("shrink")  # type: ignore[arg-type]
    assert pipeline.spans() == []
    pipeline.llm()
    assert not _marked(pipeline, "llm")


def test_an_unknown_trigger_is_a_value_error_and_records_nothing(pipeline: _Pipeline) -> None:
    with pytest.raises(ValueError, match="trigger"):
        mark_compaction(trigger="sometimes")  # type: ignore[arg-type]
    assert pipeline.spans() == []
    pipeline.llm()
    assert not _marked(pipeline, "llm")


# --- the flag on the next model call -----------------------------------------


def test_the_next_llm_span_is_marked_compacted(pipeline: _Pipeline) -> None:
    mark_compaction()
    pipeline.llm()
    assert _marked(pipeline, "llm")


@pytest.mark.parametrize("operation", ["chat", "text_completion", "generate_content"])
def test_an_llm_span_known_only_by_its_operation_is_marked(
    pipeline: _Pipeline, operation: str
) -> None:
    mark_compaction()
    with pipeline.tracer.start_as_current_span(
        "call", attributes={GEN_AI_OPERATION_NAME: operation}
    ):
        pass
    assert _marked(pipeline, "call")


def test_a_model_call_started_while_the_compaction_span_ends_is_the_next_one(
    pipeline: _Pipeline,
) -> None:
    class _CallsOnEnd(SpanProcessor):
        def on_end(self, span: ReadableSpan) -> None:
            if span.name == "compaction":
                pipeline.llm("during")

    pipeline.provider.add_span_processor(_CallsOnEnd())
    mark_compaction()
    pipeline.llm("after")
    assert _marked(pipeline, "during")
    assert not _marked(pipeline, "after")


def test_a_span_whose_kind_is_set_after_it_started_is_the_next_model_call(
    pipeline: _Pipeline,
) -> None:
    # LangChain, LiteLLM and LlamaIndex start a span and learn it is a model
    # call only before it ends.
    mark_compaction()
    with pipeline.tracer.start_as_current_span("late") as span:
        span.set_attribute(OPENINFERENCE_SPAN_KIND, "LLM")
    pipeline.llm("after")
    assert _marked(pipeline, "late")
    assert not _marked(pipeline, "after")


def test_a_late_span_that_never_becomes_a_model_call_leaves_the_flag_for_the_next(
    pipeline: _Pipeline,
) -> None:
    mark_compaction()
    with pipeline.tracer.start_as_current_span("late") as span:
        span.set_attribute(OPENINFERENCE_SPAN_KIND, "CHAIN")
    pipeline.llm("after")
    assert not _marked(pipeline, "late")
    assert _marked(pipeline, "after")


def test_a_late_span_does_not_take_a_flag_a_known_call_already_took(
    pipeline: _Pipeline,
) -> None:
    mark_compaction()
    with pipeline.tracer.start_as_current_span("late") as span:
        pipeline.llm("known")
        span.set_attribute(OPENINFERENCE_SPAN_KIND, "LLM")
    assert _marked(pipeline, "known")
    assert not _marked(pipeline, "late")


def test_a_late_span_ending_without_an_exporter_pass_is_remembered_only_boundedly() -> None:
    processor = CompactedContextProcessor(max_tracked=2)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("app")
    _tracer.publish(provider)
    mark_compaction()
    spans = []
    for name in ("a", "b", "c"):
        spans.append(tracer.start_span(name))
    assert len(processor._open) == 2
    for span in spans:
        span.end()
    assert not processor._open


def test_a_pending_snapshot_does_not_spend_the_late_span_mark() -> None:
    processor = CompactedContextProcessor()
    provider = TracerProvider()
    provider.add_span_processor(processor)
    _tracer.publish(provider)
    mark_compaction()
    span = provider.get_tracer("app").start_span("late")
    span.set_attribute(OPENINFERENCE_SPAN_KIND, "LLM")
    span.end()
    snapshot = ReadableSpan(
        "late", context=span.get_span_context(), attributes={RIUS_SPAN_PENDING: True}
    )
    assert not processor.take(snapshot)
    assert processor.take(span)
    assert not processor.take(span)


def test_only_the_next_llm_span_is_marked(pipeline: _Pipeline) -> None:
    mark_compaction()
    pipeline.llm("first")
    pipeline.llm("second")
    assert _marked(pipeline, "first")
    assert not _marked(pipeline, "second")


def test_an_llm_span_with_no_compaction_before_it_gets_nothing(pipeline: _Pipeline) -> None:
    pipeline.llm()
    assert GEN_AI_CONVERSATION_COMPACTED not in _attrs(pipeline.named("llm"))


def test_a_span_that_is_not_a_model_call_neither_is_marked_nor_uses_the_flag(
    pipeline: _Pipeline,
) -> None:
    mark_compaction()
    with pipeline.tracer.start_as_current_span(
        "tool", attributes={OPENINFERENCE_SPAN_KIND: "TOOL"}
    ):
        pass
    pipeline.llm()
    assert GEN_AI_CONVERSATION_COMPACTED not in _attrs(pipeline.named("tool"))
    assert _marked(pipeline, "llm")


def test_a_compaction_span_does_not_use_the_flag_of_the_one_before(pipeline: _Pipeline) -> None:
    mark_compaction()
    mark_compaction()
    pipeline.llm()
    for span in pipeline.spans():
        if span.name == "compaction":
            assert GEN_AI_CONVERSATION_COMPACTED not in _attrs(span)
    assert _marked(pipeline, "llm")


def test_a_native_value_is_not_overwritten_and_uses_the_flag_up(pipeline: _Pipeline) -> None:
    mark_compaction()
    with pipeline.tracer.start_as_current_span(
        "native",
        attributes={OPENINFERENCE_SPAN_KIND: "LLM", GEN_AI_CONVERSATION_COMPACTED: False},
    ):
        pass
    pipeline.llm("later")
    assert _attrs(pipeline.named("native"))[GEN_AI_CONVERSATION_COMPACTED] is False
    assert not _marked(pipeline, "later")


def test_a_second_call_arms_the_flag_again(pipeline: _Pipeline) -> None:
    mark_compaction()
    pipeline.llm("first")
    mark_compaction()
    pipeline.llm("second")
    assert _marked(pipeline, "first")
    assert _marked(pipeline, "second")


def test_a_context_copied_before_the_call_is_not_marked(pipeline: _Pipeline) -> None:
    earlier = contextvars.copy_context()
    mark_compaction()
    earlier.run(pipeline.llm, "in_earlier")
    assert not _marked(pipeline, "in_earlier")


def test_a_copied_context_uses_the_flag_once_for_every_copy(pipeline: _Pipeline) -> None:
    mark_compaction()
    first, second = contextvars.copy_context(), contextvars.copy_context()
    first.run(pipeline.llm, "in_first")
    second.run(pipeline.llm, "in_second")
    pipeline.llm("in_main")
    assert _marked(pipeline, "in_first")
    assert not _marked(pipeline, "in_second")
    assert not _marked(pipeline, "in_main")


def test_threads_sharing_the_flag_mark_exactly_one_span(pipeline: _Pipeline) -> None:
    mark_compaction()
    workers = 16
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []

    def call(index: int) -> None:
        barrier.wait()
        try:
            contexts[index].run(pipeline.llm, f"llm-{index}")
        except BaseException as exc:  # surfaced below; a thread's error is otherwise lost
            errors.append(exc)

    # One copy per thread: a Context can be entered by one thread at a time,
    # and a second concurrent run() raises. Copies share the mark by reference.
    contexts = [contextvars.copy_context() for _ in range(workers)]
    threads = [threading.Thread(target=call, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sum(_marked(pipeline, f"llm-{i}") for i in range(workers)) == 1


# --- wiring -------------------------------------------------------------------


def test_a_client_flags_the_next_call_through_its_own_pipeline() -> None:
    sink = InMemorySpanExporter()
    client = init(span_exporter=sink, set_global=False, instruments=[], heartbeat=False)
    mark_compaction()
    with client.get_tracer().start_as_current_span(
        "llm", attributes={OPENINFERENCE_SPAN_KIND: "LLM"}
    ):
        pass
    client.flush()
    (span,) = [s for s in sink.get_finished_spans() if s.name == "llm"]
    assert _attrs(span)[GEN_AI_CONVERSATION_COMPACTED] is True
    client.shutdown()


_COMPACT_REPLY = json.dumps(
    {
        "id": "resp_c",
        "object": "response.compaction",
        "output": [{"id": "cmp_1", "type": "compaction", "encrypted_content": "opaque"}],
        "usage": {"input_tokens": 5000},
    }
)


def _compact_call(pipeline: _Pipeline, name: str, reply: str) -> None:
    with pipeline.tracer.start_as_current_span(
        name, attributes={OPENINFERENCE_SPAN_KIND: "LLM"}
    ) as span:
        span.set_attribute("output.value", reply)


def test_the_call_after_the_compact_endpoint_ran_on_the_compacted_context(
    pipeline: _Pipeline,
) -> None:
    _compact_call(pipeline, "compact", _COMPACT_REPLY)
    pipeline.llm("after")
    pipeline.llm("later")
    assert not _marked(pipeline, "compact")
    assert _marked(pipeline, "after")
    assert not _marked(pipeline, "later")


def test_a_reply_that_is_not_the_compact_endpoint_arms_nothing(pipeline: _Pipeline) -> None:
    _compact_call(pipeline, "chat", json.dumps({"id": "resp_1", "object": "response"}))
    pipeline.llm("after")
    assert not _marked(pipeline, "after")


def test_a_non_llm_span_carrying_the_compact_reply_arms_nothing(pipeline: _Pipeline) -> None:
    with pipeline.tracer.start_as_current_span(
        "tool", attributes={OPENINFERENCE_SPAN_KIND: "TOOL"}
    ) as span:
        span.set_attribute("output.value", _COMPACT_REPLY)
    pipeline.llm("after")
    assert not _marked(pipeline, "after")


def test_it_is_part_of_the_public_api() -> None:
    assert rius.mark_compaction is mark_compaction
    assert "mark_compaction" in rius.__all__


def test_the_operation_is_pinned_and_the_taxonomy_files_it_as_a_chain_step() -> None:
    assert OPERATION_COMPACTION == "compaction"
    assert kind_for_operation(OPERATION_COMPACTION) == SpanKind.CHAIN.value


def test_init_wires_the_late_span_mark_to_the_exporter() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    try:
        mark_compaction()
        with client.get_tracer().start_as_current_span("late") as span:
            span.set_attribute(OPENINFERENCE_SPAN_KIND, "LLM")
        client.flush()
    finally:
        client.shutdown()

    (late,) = [s for s in inner.get_finished_spans() if s.name == "late"]
    assert _attrs(late).get(GEN_AI_CONVERSATION_COMPACTED) is True
