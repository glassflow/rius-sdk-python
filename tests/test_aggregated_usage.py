"""A LiteLLM span whose OpenAI child recorded the same call keeps its usage aside."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.aggregated_usage import (
    LANGCHAIN_SCOPE,
    LITELLM_SCOPE,
    LLAMA_INDEX_SCOPE,
    WrappedCallTracker,
    move_aggregated_usage,
)
from rius.normalization import NormalizingSpanExporter
from rius.semconv import RIUS_SPAN_PENDING

OPENAI_SCOPE = "openinference.instrumentation.openai"

USAGE = {
    "llm.token_count.prompt": 42,
    "llm.token_count.completion": 9,
    "llm.token_count.total": 51,
}


class Pipeline:
    """A provider wired like init() wires it: tracker as a processor, and the
    same tracker in the exporter."""

    def __init__(self, tracker: WrappedCallTracker | None = None) -> None:
        self.tracker = tracker if tracker is not None else WrappedCallTracker()
        self.sink = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(self.tracker)
        self.provider.add_span_processor(
            SimpleSpanProcessor(NormalizingSpanExporter(self.sink, wrapped_calls=self.tracker))
        )

    def call(
        self,
        *,
        wrapper_scope: str = LITELLM_SCOPE,
        child: dict[str, Any] | None = None,
        wrapper: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """One model call: a wrapper span, optionally with a child that ends first."""
        tracer = self.provider.get_tracer(wrapper_scope)
        with tracer.start_as_current_span("completion") as parent:
            parent.set_attributes({"openinference.span.kind": "LLM", **(wrapper or USAGE)})
            if child is not None:
                child_tracer = self.provider.get_tracer(OPENAI_SCOPE)
                with child_tracer.start_as_current_span("ChatCompletion") as span:
                    span.set_attributes({"openinference.span.kind": "LLM", **child})
        return {s.name: dict(s.attributes or {}) for s in self.sink.get_finished_spans()}


def test_a_wrapper_whose_child_recorded_the_usage_keeps_it_aside() -> None:
    spans = Pipeline().call(child=USAGE)

    assert spans["completion"]["rius.aggregated_usage.input_tokens"] == 42
    assert spans["completion"]["rius.aggregated_usage.output_tokens"] == 9
    assert not [k for k in spans["completion"] if k.startswith("gen_ai.usage.")]
    # The child is the call itself, so it is the one the sums count.
    assert spans["ChatCompletion"]["gen_ai.usage.input_tokens"] == 42
    assert spans["ChatCompletion"]["gen_ai.usage.output_tokens"] == 9


def test_a_wrapper_with_no_child_keeps_its_usage() -> None:
    # Anthropic through LiteLLM: LiteLLM's own HTTP handler, no second span.
    spans = Pipeline().call(child=None)

    assert spans["completion"]["gen_ai.usage.input_tokens"] == 42
    assert not [k for k in spans["completion"] if k.startswith("rius.aggregated_usage.")]


def test_a_child_without_usage_leaves_the_wrapper_alone() -> None:
    # A failed call: the child records no usage, so the wrapper's is the record.
    spans = Pipeline().call(child={"llm.model_name": "gpt-4o-mini"})

    assert spans["completion"]["gen_ai.usage.input_tokens"] == 42


def test_a_child_with_canonical_usage_counts_too() -> None:
    spans = Pipeline().call(child={"gen_ai.usage.input_tokens": 42})

    assert spans["completion"]["rius.aggregated_usage.input_tokens"] == 42


@pytest.mark.parametrize("scope", [LITELLM_SCOPE, LANGCHAIN_SCOPE, LLAMA_INDEX_SCOPE])
def test_every_wrapper_scope_is_moved(scope: str) -> None:
    spans = Pipeline().call(wrapper_scope=scope, child=USAGE)

    assert spans["completion"]["rius.aggregated_usage.input_tokens"] == 42
    assert not [k for k in spans["completion"] if k.startswith("gen_ai.usage.")]


def test_only_wrapper_scopes_are_moved() -> None:
    # A span from any other instrumentor may be the only record of its usage.
    spans = Pipeline().call(wrapper_scope="some.other.instrumentation", child=USAGE)

    assert spans["completion"]["gen_ai.usage.input_tokens"] == 42


def test_the_child_and_the_wrapper_may_be_exported_in_different_batches() -> None:
    # SimpleSpanProcessor exports each span alone when it ends: the child goes
    # out first, in its own batch, and the mark still reaches the wrapper's.
    pipeline = Pipeline()
    pipeline.call(child=USAGE)

    batches = [s.name for s in pipeline.sink.get_finished_spans()]
    assert batches == ["ChatCompletion", "completion"]
    assert "rius.aggregated_usage.input_tokens" in (
        pipeline.sink.get_finished_spans()[1].attributes or {}
    )


def test_the_mark_is_dropped_once_the_wrapper_is_exported() -> None:
    pipeline = Pipeline()
    pipeline.call(child=USAGE)

    assert not pipeline.tracker._open
    assert not pipeline.tracker._marked


def test_the_books_are_bounded() -> None:
    tracker = WrappedCallTracker(max_tracked=2)
    provider = TracerProvider()
    provider.add_span_processor(tracker)
    tracer = provider.get_tracer(LITELLM_SCOPE)
    # Three wrappers that never end: the oldest is forgotten, not leaked.
    for name in ("a", "b", "c"):
        tracer.start_span(name)

    assert len(tracker._open) == 2


def test_a_pending_snapshot_does_not_take_the_mark() -> None:
    pipeline = Pipeline()
    tracer = pipeline.provider.get_tracer(LITELLM_SCOPE)
    with tracer.start_as_current_span("completion") as parent:
        parent.set_attributes(USAGE)
        with pipeline.provider.get_tracer(OPENAI_SCOPE).start_as_current_span(
            "ChatCompletion"
        ) as c:
            c.set_attributes(USAGE)
        snapshot = ReadableSpan(
            name="completion",
            context=parent.get_span_context(),
            attributes={RIUS_SPAN_PENDING: True},
        )
        assert pipeline.tracker.take(snapshot) is False

    final = next(s for s in pipeline.sink.get_finished_spans() if s.name == "completion")
    assert "rius.aggregated_usage.input_tokens" in (final.attributes or {})


def test_exporting_a_moved_span_again_changes_nothing() -> None:
    pipeline = Pipeline()
    pipeline.call(child=USAGE)
    first = list(pipeline.sink.get_finished_spans())

    again = InMemorySpanExporter()
    NormalizingSpanExporter(again, wrapped_calls=WrappedCallTracker()).export(first)

    assert [dict(s.attributes or {}) for s in again.get_finished_spans()] == [
        dict(s.attributes or {}) for s in first
    ]


def test_move_aggregated_usage_renames_only_the_usage_family() -> None:
    assert move_aggregated_usage({"gen_ai.request.model": "m", "gen_ai.usage.input_tokens": 1}) == {
        "gen_ai.request.model": "m",
        "rius.aggregated_usage.input_tokens": 1,
    }
    assert move_aggregated_usage({"gen_ai.request.model": "m"}) is None


@pytest.mark.parametrize("with_child", [True, False])
def test_init_wires_the_tracker(with_child: bool) -> None:
    sink = InMemorySpanExporter()
    client = init(api_key="test", span_exporter=sink, heartbeat=False, set_global=False)
    try:
        litellm = client.get_tracer(LITELLM_SCOPE)
        with litellm.start_as_current_span("completion") as parent:
            parent.set_attributes({"openinference.span.kind": "LLM", **USAGE})
            if with_child:
                openai = client.get_tracer(OPENAI_SCOPE)
                with openai.start_as_current_span("ChatCompletion") as child:
                    child.set_attributes({"openinference.span.kind": "LLM", **USAGE})
        client.flush()
    finally:
        client.shutdown()

    wrapper = next(s for s in sink.get_finished_spans() if s.name == "completion")
    attributes = dict(wrapper.attributes or {})
    if with_child:
        assert attributes["rius.aggregated_usage.input_tokens"] == 42
        assert "gen_ai.usage.input_tokens" not in attributes
    else:
        assert attributes["gen_ai.usage.input_tokens"] == 42


def test_a_wrapper_exported_at_shutdown_keeps_its_usage_aside() -> None:
    # No flush: the spans are still queued in the batch processor when the
    # provider shuts down, and the tracker is shut down before it.
    sink = InMemorySpanExporter()
    client = init(api_key="test", span_exporter=sink, heartbeat=False, set_global=False)
    langchain = client.get_tracer(LANGCHAIN_SCOPE)
    with langchain.start_as_current_span("ChatOpenAI") as parent:
        parent.set_attributes({"openinference.span.kind": "LLM", **USAGE})
        openai = client.get_tracer(OPENAI_SCOPE)
        with openai.start_as_current_span("ChatCompletion") as child:
            child.set_attributes({"openinference.span.kind": "LLM", **USAGE})
    client.shutdown()

    wrapper = next(s for s in sink.get_finished_spans() if s.name == "ChatOpenAI")
    attributes = dict(wrapper.attributes or {})
    assert attributes["rius.aggregated_usage.input_tokens"] == 42
    assert not any(key.startswith("gen_ai.usage.") for key in attributes)


def test_shutting_the_tracker_down_keeps_its_marks() -> None:
    # The batch processor exports its queue after the tracker has shut down.
    tracker = WrappedCallTracker()
    provider = TracerProvider()
    provider.add_span_processor(tracker)
    wrapper = provider.get_tracer(LITELLM_SCOPE)
    openai = provider.get_tracer(OPENAI_SCOPE)
    with (
        wrapper.start_as_current_span("completion") as parent,
        openai.start_as_current_span("ChatCompletion") as child,
    ):
        child.set_attributes(USAGE)
    tracker.shutdown()
    assert isinstance(parent, ReadableSpan)
    assert tracker.take(parent)
