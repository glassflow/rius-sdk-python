"""Partial (pending) spans: a content-free snapshot exported at span start.

Wire contract under test: same trace/span/parent ids, same name and start
timestamp as the final span; zero duration; the pending marker attribute;
identity/taxonomy attributes only, never content.
"""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import SpanProcessor as _SpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.semconv import RIUS_SPAN_PENDING


def _memory_client(**kwargs: object):
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        **kwargs,  # type: ignore[arg-type]
    )
    return client, exporter


def _split(spans):
    pending = [s for s in spans if s.attributes.get(RIUS_SPAN_PENDING)]
    final = [s for s in spans if not s.attributes.get(RIUS_SPAN_PENDING)]
    return pending, final


def test_flag_off_by_default_behavior_unchanged() -> None:
    client, exporter = _memory_client()
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert RIUS_SPAN_PENDING not in spans[0].attributes


def test_pending_snapshot_mirrors_identity_of_final_span() -> None:
    client, exporter = _memory_client(partial_spans=True)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    (pending,), (final,) = _split(exporter.get_finished_spans())

    assert pending.attributes[RIUS_SPAN_PENDING] is True
    # identical identity: the ClickHouse sort key must match for replacement
    assert pending.context.trace_id == final.context.trace_id
    assert pending.context.span_id == final.context.span_id
    assert pending.name == final.name
    assert pending.start_time == final.start_time
    # zero duration: OTLP cannot represent an unfinished span
    assert pending.end_time == pending.start_time
    assert final.end_time > final.start_time


def test_pending_preserves_parent_linkage() -> None:
    client, exporter = _memory_client(partial_spans=True)
    tracer = client.get_tracer()
    with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
        pass
    client.flush()
    pending, final = _split(exporter.get_finished_spans())
    pending_child = next(s for s in pending if s.name == "child")
    final_child = next(s for s in final if s.name == "child")
    assert pending_child.parent is not None
    assert pending_child.parent.span_id == final_child.parent.span_id


def test_pending_carries_identity_attributes_but_never_content() -> None:
    client, exporter = _memory_client(partial_spans=True)
    with client.get_tracer().start_as_current_span(
        "chat",
        attributes={
            "openinference.span.kind": "LLM",
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.request.temperature": 0.2,
            "input.value": "SECRET-CONTENT",
        },
    ):
        pass
    client.flush()
    (pending,), (final,) = _split(exporter.get_finished_spans())
    assert pending.attributes["openinference.span.kind"] == "LLM"
    assert pending.attributes["gen_ai.request.model"] == "gpt-4o"
    assert pending.attributes["gen_ai.request.temperature"] == 0.2
    assert "input.value" not in pending.attributes, "content must NEVER ride a pending span"
    assert final.attributes["input.value"] == "SECRET-CONTENT"


def test_sampled_out_spans_produce_no_pending() -> None:
    client, exporter = _memory_client(partial_spans=True, sample_rate=0.0)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans() == ()


def test_disabled_kills_pendings_too() -> None:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter, set_global=False, disabled=True, partial_spans=True, instruments=[]
    )
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()
    assert exporter.get_finished_spans() == ()


def test_env_var_enables_partial_spans(monkeypatch) -> None:
    from rius.config import resolve_config

    monkeypatch.setenv("RIUS_PARTIAL_SPANS", "true")
    assert resolve_config().partial_spans is True
    # explicit argument wins over the environment
    assert resolve_config(partial_spans=False).partial_spans is False


class _StartAttributeRecorder(_SpanProcessor):
    """Span processor recording each span's attributes as seen at on_start."""

    def __init__(self) -> None:
        self.seen: dict[str, dict] = {}

    def on_start(self, span, parent_context=None) -> None:  # noqa: ANN001
        self.seen[span.name] = dict(span.attributes or {})

    def on_end(self, span) -> None:  # noqa: ANN001
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def test_sdk_helpers_expose_identity_attributes_at_span_start() -> None:
    """Pending snapshots are built at on_start, so the SDK's own APIs must
    attach kind/model/provider at CREATION — set_attribute after the fact is
    invisible to the snapshot."""
    from opentelemetry import trace as otel_trace

    import rius

    recorder = _StartAttributeRecorder()
    otel_trace.get_tracer_provider().add_span_processor(recorder)  # type: ignore[attr-defined]

    with rius.start_as_current_span(
        "kindly", kind=rius.SpanKind.RETRIEVER, data_source_id="docs-index", top_k=3
    ):
        pass
    with rius.start_as_current_generation("genny", model="gpt-4o", provider="openai"):
        pass

    kindly = recorder.seen["kindly"]
    assert kindly["openinference.span.kind"] == "RETRIEVER"
    assert kindly["gen_ai.operation.name"] == "retrieval"
    # The data source composes the span name, so a pending retrieval that
    # never finishes is still attributable to the index it was searching.
    assert kindly["gen_ai.data_source.id"] == "docs-index"
    # How many documents were asked for is equally a property of the request.
    assert kindly["gen_ai.retrieval.top_k"] == 3
    # What came back cannot be, and must not appear on a start-time snapshot.
    assert "gen_ai.retrieval.documents" not in kindly
    genny = recorder.seen["genny"]
    assert genny["openinference.span.kind"] == "LLM"
    assert genny["gen_ai.operation.name"] == "chat"
    assert genny["gen_ai.request.model"] == "gpt-4o"
    assert genny["gen_ai.provider.name"] == "openai"


def test_observe_exposes_kind_at_span_start() -> None:
    """@observe used to set its kind AFTER start_span, so its pending
    snapshots had no openinference.span.kind and the live view could not
    classify them."""
    from opentelemetry import trace as otel_trace

    import rius

    recorder = _StartAttributeRecorder()
    otel_trace.get_tracer_provider().add_span_processor(recorder)  # type: ignore[attr-defined]

    @rius.observe(name="observed-tool", kind=rius.SpanKind.TOOL)
    def tool() -> None:
        pass

    @rius.observe(name="observed-gen")
    def gen():  # noqa: ANN202
        yield 1

    tool()
    list(gen())
    assert recorder.seen["observed-tool"]["openinference.span.kind"] == "TOOL"
    assert recorder.seen["observed-tool"]["gen_ai.operation.name"] == "execute_tool"
    assert recorder.seen["observed-gen"]["openinference.span.kind"] == "CHAIN"


def test_pending_snapshot_of_local_tool_span_carries_gen_ai_tool_name() -> None:
    """A still-running local tool must be identifiable by name in the live
    view, so the name has to be on the span at creation; read the EXPORTED
    snapshot because the pending allowlist runs between span and wire."""
    from rius import _tracer
    from rius.semconv import SpanKind
    from rius.spans import start_span

    client, exporter = _memory_client(partial_spans=True)
    # Route the SDK helpers to this scoped client (the lifecycle fixture
    # withdraws it again after the test).
    _tracer.publish(client._provider)
    span = start_span("weather", kind=SpanKind.TOOL, input={"city": "Berlin"})
    client.flush()  # the snapshot is exported while the span is still open
    pending, _ = _split(exporter.get_finished_spans())
    span.end()
    assert pending, "expected a pending snapshot"
    attrs = pending[0].attributes
    assert attrs["gen_ai.tool.name"] == "weather"
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert "input.value" not in attrs  # the allowlist still strips content


def test_creation_identity_keys_are_pending_allowlisted() -> None:
    """PENDING_IDENTITY_ATTRIBUTES is an allowlist, so an identity key added to
    a creation-attribute builder without an allowlist update is silently
    dropped from pending snapshots (this is how the MCP marker went missing
    once). Every key the SDK itself sets at span creation must pass it."""
    from rius.generation import _creation_attributes as generation_attributes
    from rius.instrumentation_mcp import _call_attributes
    from rius.semconv import (
        PENDING_IDENTITY_ATTRIBUTES,
        PENDING_IDENTITY_PREFIXES,
        SpanKind,
        kind_attributes,
    )
    from rius.spans import _creation as span_creation

    builders: dict[str, dict[str, str]] = {
        f"kind_attributes({kind.name})": kind_attributes(
            kind,
            "tool-name",
            data_source_id="docs",
            top_k=5,
            agent_name="planner",
            agent_id="ag_1",
            agent_version="7",
            executing_agent_name="executor",
            tool_call_id="call_1",
            tool_type="function",
        )
        for kind in SpanKind
    }
    builders["spans"] = span_creation("weather", SpanKind.TOOL, "u", tool_name="weather")[1]
    builders["spans(agent)"] = span_creation(
        "plan", SpanKind.AGENT, "u", agent_name="planner", agent_id="ag_1", agent_version="7"
    )[1]
    builders["generation"] = generation_attributes(
        model="m", provider="p", operation="chat", user_id="u", output_type="json"
    )
    builders["mcp"] = _call_attributes("search", protocol_version="2026-07-28")

    for builder, attributes in builders.items():
        for key in attributes:
            assert key in PENDING_IDENTITY_ATTRIBUTES or key.startswith(
                PENDING_IDENTITY_PREFIXES
            ), f"{builder} sets {key!r} at creation but it is not pending-allowlisted"


def test_pending_snapshot_of_an_agent_span_carries_the_agent_name() -> None:
    """The waterfall's live view must say which agent is running, not just
    that some agent is."""
    from rius import _tracer
    from rius.semconv import SpanKind
    from rius.spans import start_span

    client, exporter = _memory_client(partial_spans=True)
    _tracer.publish(client._provider)
    span = start_span("plan", kind=SpanKind.AGENT, agent_name="researcher", agent_id="ag_1")
    client.flush()
    pending, _ = _split(exporter.get_finished_spans())
    span.end()
    assert pending, "expected a pending snapshot"
    attrs = pending[0].attributes
    assert attrs["gen_ai.agent.name"] == "researcher"
    assert attrs["gen_ai.agent.id"] == "ag_1"
    assert attrs["gen_ai.operation.name"] == "invoke_agent"


def test_pending_and_final_agree_on_a_composed_span_name() -> None:
    """The snapshot carries the name the final span will have, which is why
    the composed name is built from the REQUEST model: a response model
    arriving later would make the two disagree, and the backend replaces the
    snapshot by identity, not by name."""
    from rius import _tracer, start_as_current_generation

    client, exporter = _memory_client(partial_spans=True)
    # Route the SDK helpers to this scoped client (the lifecycle fixture
    # withdraws it again after the test).
    _tracer.publish(client._provider)
    try:
        with start_as_current_generation(model="gpt-4o") as generation:
            generation.set_response_model("gpt-4o-2024-08-06")
    finally:
        client.flush()
    (pending,), (final,) = _split(exporter.get_finished_spans())
    assert pending.name == final.name == "chat gpt-4o"


def test_pending_and_final_agree_on_a_composed_tool_span_name() -> None:
    from rius import _tracer, start_as_current_span
    from rius.semconv import SpanKind

    client, exporter = _memory_client(partial_spans=True)
    _tracer.publish(client._provider)
    try:
        with start_as_current_span(kind=SpanKind.TOOL, tool_name="get_weather"):
            pass
    finally:
        client.flush()
    (pending,), (final,) = _split(exporter.get_finished_spans())
    assert pending.name == final.name == "execute_tool get_weather"


def test_pending_snapshot_of_a_tool_span_carries_the_executing_agent() -> None:
    """A still-running tool must be attributable to the agent that launched
    it: which agent is stuck is the live view's first question."""
    from rius import _tracer
    from rius.semconv import SpanKind
    from rius.spans import start_as_current_span, start_span

    client, exporter = _memory_client(partial_spans=True, agent_name="configured")
    _tracer.publish(client._provider)
    with start_as_current_span("plan", kind=SpanKind.AGENT, agent_name="researcher"):
        span = start_span("weather", kind=SpanKind.TOOL, tool_name="weather")
        client.flush()
        pending, _ = _split(exporter.get_finished_spans())
        span.end()
    tool_pending = [s for s in pending if s.name == "weather"]
    assert tool_pending, "expected a pending snapshot for the tool span"
    assert tool_pending[0].attributes["gen_ai.agent.name"] == "researcher"


def test_pending_snapshot_keeps_the_request_shaped_conformance_keys() -> None:
    """End-to-end counterpart to the structural allowlist check above: the
    requested output modality and the tool call this execution answers are
    both known at span start, so the live view already has them."""
    from rius import _tracer, start_as_current_generation, start_as_current_span
    from rius.semconv import SpanKind

    client, exporter = _memory_client(partial_spans=True)
    # Route the SDK helpers to this scoped client (the lifecycle fixture
    # withdraws it again after the test).
    _tracer.publish(client._provider)
    with start_as_current_generation(model="gpt-4o", output_type="json"):
        pass
    with start_as_current_span(
        kind=SpanKind.TOOL, tool_name="get_weather", tool_call_id="call_1", tool_type="function"
    ):
        pass
    client.flush()
    pending, _final = _split(exporter.get_finished_spans())
    generation, tool = pending
    assert generation.attributes["gen_ai.output.type"] == "json"
    assert tool.attributes["gen_ai.tool.call.id"] == "call_1"
    assert tool.attributes["gen_ai.tool.type"] == "function"


def test_pending_snapshot_of_an_agent_span_carries_the_agent_version() -> None:
    """Set at CREATION so the live view can tell which VERSION of an agent
    definition is the one currently stuck."""
    from rius import _tracer
    from rius.semconv import SpanKind
    from rius.spans import start_span

    client, exporter = _memory_client(partial_spans=True)
    _tracer.publish(client._provider)
    span = start_span("plan", kind=SpanKind.AGENT, agent_name="researcher", agent_version="7")
    client.flush()
    pending, _ = _split(exporter.get_finished_spans())
    span.end()
    assert pending, "expected a pending snapshot"
    assert pending[0].attributes["gen_ai.agent.version"] == "7"


def test_pending_snapshot_carries_both_request_namespaces() -> None:
    """Request parameters are chosen before the call runs, so the live view
    must already show them — under either namespace."""
    from rius import _tracer, start_generation

    client, exporter = _memory_client(partial_spans=True)
    _tracer.publish(client._provider)
    span = start_generation("chat", model_parameters={"temperature": 0.7, "my_custom_knob": 3})
    client.flush()
    pending, _ = _split(exporter.get_finished_spans())
    span.end()
    assert pending, "expected a pending snapshot"
    attrs = pending[0].attributes
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["rius.request.my_custom_knob"] == 3


def test_pending_snapshot_does_not_leak_tool_definitions_from_request_parameters() -> None:
    """rius.request.* rides pending snapshots by prefix, so a `tools` model
    parameter reaches one. Masking runs at export and covers the pending path
    too — the snapshot must not be the way tool definitions escape
    capture_content=False."""
    from rius import _tracer, start_generation

    client, exporter = _memory_client(partial_spans=True, capture_content=False)
    _tracer.publish(client._provider)
    span = start_generation(
        "chat",
        model_parameters={"tools": [{"description": "SECRET"}], "temperature": 0.7},
    )
    client.flush()
    pending, _ = _split(exporter.get_finished_spans())
    span.end()
    assert pending, "expected a pending snapshot"
    attrs = pending[0].attributes
    assert "rius.request.tools" not in attrs
    assert "SECRET" not in json.dumps(dict(attrs))
    assert attrs["gen_ai.request.temperature"] == 0.7


# --- The inverted guard ------------------------------------------------------
#
# test_creation_identity_keys_are_pending_allowlisted above walks the creation
# builders and asserts every key they set is allowlisted. That catches a
# missing allowlist entry. It cannot catch the opposite mistake: a key that IS
# allowlisted but is written after the span exists, so the snapshot built at
# on_start never sees it.
#
# That mistake was live for the whole life of partial spans. model_parameters
# was applied after start_span returned, so no request parameter ever reached a
# snapshot, while the prefix rule sat in the allowlist looking correct because
# nothing exercised it. The guard below is the invariant the allowlist actually
# encodes: allowlisted means "reaches the live view".

#: Allowlisted keys that legitimately cannot be known when the span opens.
#: Each one is an argued exemption, not an oversight, and the set existing is
#: the point: a new late write has to be defended in a diff instead of passing
#: unnoticed.
PENDING_LATE_EXEMPTIONS = frozenset(
    {
        # Set by record_first_token. Whether the response streamed is only
        # answerable once a first chunk has actually arrived, which is after
        # the span started by definition.
        "gen_ai.request.stream",
    }
)


def _fully_populated_generation() -> None:
    from rius.generation import start_generation

    generation = start_generation(
        model="gpt-4o",
        provider="openai",
        input=[{"role": "user", "content": "hello"}],
        model_parameters={
            "temperature": 0.7,
            "max_completion_tokens": 256,  # a recognised provider spelling
            "my_custom_knob": 3,  # lands in rius.request.*
        },
        operation="chat",
        reasoning_level="high",
        tools=[{"name": "get_weather"}],
        user_id="u-1",
        output_type="json",
    )
    # Everything a caller can add once the call is under way, so the final span
    # is as rich as it ever gets and the comparison has something to bite on.
    generation.record_first_token()
    generation.set_response_model("gpt-4o-2024-08-06")
    generation.set_response_id("resp_1")
    generation.set_usage(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=2,
        cache_write_input_tokens=1,
        reasoning_output_tokens=3,
    )
    generation.set_finish_reasons("stop")
    generation.set_output([{"role": "assistant", "content": "hi"}])
    generation.end()


def _fully_populated_tool() -> None:
    from rius.semconv import SpanKind
    from rius.spans import start_span

    observation = start_span(
        "lookup",
        kind=SpanKind.TOOL,
        input={"city": "Berlin"},
        user_id="u-1",
        tool_name="get_weather",
        tool_call_id="call_1",
        tool_type="function",
    )
    observation.set_output({"temp": 20})
    observation.end()


def _fully_populated_agent() -> None:
    from rius.semconv import SpanKind
    from rius.spans import start_span

    observation = start_span(
        kind=SpanKind.AGENT,
        input="research this",
        user_id="u-1",
        agent_name="researcher",
        agent_id="ag_1",
        agent_version="1.0.0",
    )
    observation.set_output("done")
    observation.end()


def _fully_populated_retriever() -> None:
    from rius.semconv import SpanKind
    from rius.spans import start_span

    observation = start_span(
        kind=SpanKind.RETRIEVER,
        input="weather berlin",
        user_id="u-1",
        data_source_id="product-kb",
        top_k=5,
    )
    observation.set_retrieved_documents([{"id": "doc-1", "score": 0.9}])
    observation.end()


@pytest.mark.parametrize(
    "build",
    [
        _fully_populated_generation,
        _fully_populated_tool,
        _fully_populated_agent,
        _fully_populated_retriever,
    ],
    ids=["generation", "tool", "agent", "retriever"],
)
def test_allowlisted_attributes_on_the_final_span_are_on_the_snapshot(build) -> None:  # noqa: ANN001
    """An allowlisted key written after the span opens never reaches the live
    view, and nothing today would notice. Open one span per helper with every
    optional argument populated, then assert the snapshot carries every
    allowlisted attribute the finished span ended up with."""
    from rius import _tracer
    from rius.semconv import (
        PENDING_IDENTITY_ATTRIBUTES,
        PENDING_IDENTITY_PREFIXES,
    )
    from rius.session import session

    client, exporter = _memory_client(partial_spans=True)
    _tracer.publish(client._provider)
    # A session scope so session.id is on the span too: it is allowlisted, and
    # a scope-derived attribute is exactly the kind that could be applied late.
    with session("sess-1"):
        build()
    client.flush()

    pending, final = _split(exporter.get_finished_spans())
    assert len(pending) == 1 and len(final) == 1
    snapshot, finished = pending[0], final[0]
    assert snapshot.context.span_id == finished.context.span_id

    missing = {
        key: value
        for key, value in finished.attributes.items()
        if (key in PENDING_IDENTITY_ATTRIBUTES or key.startswith(PENDING_IDENTITY_PREFIXES))
        and key not in PENDING_LATE_EXEMPTIONS
        and key not in snapshot.attributes
    }
    assert not missing, (
        f"allowlisted but written too late to reach the pending snapshot: {missing}. "
        "Either set it in the span's creation attributes, or add it to "
        "PENDING_LATE_EXEMPTIONS with the reason it cannot be known at span start."
    )
