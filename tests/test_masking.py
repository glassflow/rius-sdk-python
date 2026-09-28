import json
from typing import Any

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init


def test_mask_redacts_content_attributes() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("input.value", "secret")
        span.set_attribute("gen_ai.input.messages", '[{"role":"user","content":"hi"}]')
        span.set_attribute("gen_ai.request.model", "gpt-4o")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs["input.value"] == "***"
    assert attrs["gen_ai.input.messages"] == "***"
    assert attrs["gen_ai.request.model"] == "gpt-4o"  # non-content untouched


def test_capture_content_false_strips_content() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("input.value", "secret")
        span.set_attribute("gen_ai.request.model", "gpt-4o")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "input.value" not in attrs
    assert attrs["gen_ai.request.model"] == "gpt-4o"


def test_mask_covers_third_party_instrumentation_keys() -> None:
    # Export-stage masking is a single choke point: it must also redact content
    # emitted by bundled third-party instrumentation, not just our own keys.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("llm.input_messages", "third-party prompt")
        span.set_attribute("gen_ai.prompt", "another prompt")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs["llm.input_messages"] == "***"
    assert attrs["gen_ai.prompt"] == "***"


def test_mask_covers_flattened_third_party_content_keys() -> None:
    # OpenInference/OpenLLMetry instrumentors flatten message content into
    # indexed keys — masking must match those by prefix, not just exact keys.
    # The fields normalization reassembles arrive as gen_ai.input.messages;
    # a field it does not read (message.name) still arrives flattened.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("llm.input_messages.0.message.content", "secret")
        span.set_attribute("llm.input_messages.0.message.name", "secret")
        span.set_attribute("gen_ai.prompt.0.content", "secret")  # OpenLLMetry style
        span.set_attribute("llm.model_name", "gpt-4o")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs["gen_ai.input.messages"] == "***"
    assert "llm.input_messages.0.message.content" not in attrs
    assert attrs["llm.input_messages.0.message.name"] == "***"
    assert attrs["gen_ai.prompt.0.content"] == "***"
    assert attrs["llm.model_name"] == "gpt-4o"


def test_mask_returning_none_drops_attribute_not_leaks_original() -> None:
    # BoundedAttributes silently refuses invalid values — a mask returning None
    # must drop the attribute (fail closed), never leave the original in place.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: None)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("input.value", "secret")
        span.set_attribute("gen_ai.request.model", "gpt-4o")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "input.value" not in attrs
    assert attrs["gen_ai.request.model"] == "gpt-4o"


def test_mask_returning_non_primitive_is_serialized_not_leaked() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: {"redacted": True})
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("input.value", "secret")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs.get("input.value") == '{"redacted":true}'


def test_masking_does_not_mutate_spans_seen_by_other_processors() -> None:
    # ReadableSpan shares its attribute dict with every processor on the
    # provider — masking must export copies, never rewrite shared state.
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    inner = InMemorySpanExporter()
    other = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    client._provider.add_span_processor(SimpleSpanProcessor(other))
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("input.value", "secret")
    client.flush()

    assert inner.get_finished_spans()[0].attributes["input.value"] == "***"
    # the other processor's copy of history must be untouched
    assert other.get_finished_spans()[0].attributes["input.value"] == "secret"


def test_capture_content_false_covers_retriever_embedding_and_traceloop_keys() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("retrieval.documents.0.document.content", "secret doc")
        span.set_attribute("embedding.embeddings.0.embedding.text", "secret text")
        span.set_attribute("llm.prompt_template.variables", '{"user_q": "secret"}')
        span.set_attribute("traceloop.entity.input", "secret input")
        span.set_attribute("traceloop.entity.output", "secret output")
        span.set_attribute("retrieval.documents.0.document.id", "doc-1")  # metadata
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "retrieval.documents.0.document.content" not in attrs
    assert "embedding.embeddings.0.embedding.text" not in attrs
    assert "llm.prompt_template.variables" not in attrs
    assert "traceloop.entity.input" not in attrs
    assert "traceloop.entity.output" not in attrs
    assert attrs["retrieval.documents.0.document.id"] == "doc-1"


def test_capture_content_false_covers_unflattened_prompt_keys() -> None:
    # Instrumentations do not always flatten llm.prompts into indexed
    # llm.prompts.0 keys; the bare key must be covered too, or the privacy
    # switch fails open on exactly the attribute that carries every prompt.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("llm.prompts", '["secret prompt one", "secret prompt two"]')
        span.set_attribute("llm.prompt_template", "Answer as {persona}: {question}")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "llm.prompts" not in attrs
    assert "llm.prompt_template" not in attrs


def test_every_content_prefix_has_its_bare_key_covered() -> None:
    # The invariant whose silent violation caused the llm.prompts gap: a
    # family covered by prefix (indexed keys) must also cover the bare,
    # unflattened attribute the prefix is derived from.
    from rius.semconv import CONTENT_ATTRIBUTE_PREFIXES, CONTENT_ATTRIBUTES

    missing = {
        prefix.rstrip(".")
        for prefix in CONTENT_ATTRIBUTE_PREFIXES
        if prefix.rstrip(".") not in CONTENT_ATTRIBUTES
    }
    assert not missing, f"prefixes without their bare key in CONTENT_ATTRIBUTES: {missing}"


def test_capture_content_false_strips_content_in_event_attributes() -> None:
    # OTel GenAI has an event-based shape where message content arrives as
    # span events; the privacy boundary is the span, not one collection on it.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.add_event(
            "gen_ai.user.message",
            {"gen_ai.input.messages": "secret prompt", "gen_ai.request.model": "gpt-4o"},
        )
    client.flush()
    events = inner.get_finished_spans()[0].events
    assert len(events) == 1
    assert events[0].name == "gen_ai.user.message"
    assert "gen_ai.input.messages" not in events[0].attributes
    assert events[0].attributes["gen_ai.request.model"] == "gpt-4o"


def test_mask_redacts_content_in_event_attributes() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    with client.get_tracer().start_as_current_span("op") as span:
        span.add_event("gen_ai.choice", {"gen_ai.output.messages": "secret answer"})
    client.flush()
    events = inner.get_finished_spans()[0].events
    assert events[0].attributes["gen_ai.output.messages"] == "***"


def test_capture_content_false_strips_content_in_link_attributes() -> None:
    from opentelemetry.trace import Link

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    tracer = client.get_tracer()
    with tracer.start_as_current_span("linked-to") as target:
        target_context = target.get_span_context()
    link = Link(target_context, {"input.value": "secret", "link.kind": "followup"})
    with tracer.start_as_current_span("op", links=[link]):
        pass
    client.flush()
    exported = {s.name: s for s in inner.get_finished_spans()}
    links = exported["op"].links
    assert len(links) == 1
    assert "input.value" not in links[0].attributes
    assert links[0].attributes["link.kind"] == "followup"


def test_capture_content_false_strips_exception_payload_keeps_type() -> None:
    # Providers echo the rejected request into the error string, so the
    # exception event carries the same content the attribute strip removed.
    # Policy (matching the TypeScript SDK): drop message and stacktrace,
    # keep the event and exception.type so failures stay diagnosable.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        try:
            raise ValueError('400 invalid request: {"messages": "secret prompt"}')
        except ValueError as exc:
            span.record_exception(exc)
    client.flush()
    events = inner.get_finished_spans()[0].events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes["exception.type"] == "ValueError"
    assert "exception.message" not in events[0].attributes
    assert "exception.stacktrace" not in events[0].attributes


def test_mask_applies_to_exception_message_and_stacktrace() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    with client.get_tracer().start_as_current_span("op") as span:
        try:
            raise ValueError("secret prompt echo")
        except ValueError as exc:
            span.record_exception(exc)
    client.flush()
    events = inner.get_finished_spans()[0].events
    assert events[0].attributes["exception.type"] == "ValueError"
    assert events[0].attributes["exception.message"] == "***"
    assert events[0].attributes["exception.stacktrace"] == "***"


def test_events_untouched_when_content_capture_on_and_no_mask() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.add_event("gen_ai.user.message", {"gen_ai.input.messages": "kept"})
        try:
            raise ValueError("kept too")
        except ValueError as exc:
            span.record_exception(exc)
    client.flush()
    events = inner.get_finished_spans()[0].events
    assert events[0].attributes["gen_ai.input.messages"] == "kept"
    assert events[1].attributes["exception.message"] == "kept too"


def test_mask_accepting_key_receives_attribute_key() -> None:
    seen: list[str] = []

    def keyed_mask(value: object, *, key: str) -> str:
        seen.append(key)
        return "***"

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=keyed_mask)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("input.value", "secret")
    client.flush()
    assert inner.get_finished_spans()[0].attributes["input.value"] == "***"
    assert seen == ["input.value"]


def test_no_masking_by_default() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("input.value", "secret")
    client.flush()
    assert inner.get_finished_spans()[0].attributes["input.value"] == "secret"


def test_capture_content_false_strips_tool_definition_content() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("gen_ai.tool.definitions", '[{"name":"q","description":"secret"}]')
        span.set_attribute("gen_ai.tool.description", "runs SQL against the billing db")
        span.set_attribute("gen_ai.tool.name", "query_billing")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "gen_ai.tool.definitions" not in attrs
    assert "gen_ai.tool.description" not in attrs
    # identity, not content: the tool NAME survives, so traces stay navigable
    assert attrs["gen_ai.tool.name"] == "query_billing"


def test_mask_redacts_tool_definition_content() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("gen_ai.tool.definitions", '[{"name":"q"}]')
        span.set_attribute("gen_ai.tool.description", "does things")
        span.set_attribute("gen_ai.tool.name", "q")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs["gen_ai.tool.definitions"] == "***"
    assert attrs["gen_ai.tool.description"] == "***"
    assert attrs["gen_ai.tool.name"] == "q"


def test_capture_content_false_strips_tool_definitions() -> None:
    # The exact key the generation API's tools capture emits: definitions are
    # content (they embed prompt engineering, sometimes secrets in defaults).
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("chat") as span:
        span.set_attribute("gen_ai.tool.definitions", '[{"name": "secret_tool"}]')
        span.set_attribute("gen_ai.request.model", "gpt-4o")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "gen_ai.tool.definitions" not in attrs
    assert attrs["gen_ai.request.model"] == "gpt-4o"


def test_capture_content_false_strips_wrapper_tool_and_vercel_content_keys() -> None:
    # Pinned empirically against the bundled instrumentors (2026-09-14):
    # every OpenInference path emits llm.tools.{i}.tool.json_schema, and the
    # Vercel AI SDK path (TS) leaves raw ai.* keys carrying full messages,
    # response text and tool definitions. All of it is content.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("chat") as span:
        span.set_attribute("llm.tools.0.tool.json_schema", '{"name": "secret_tool"}')
        span.set_attribute("llm.tools", '[{"name": "secret_tool"}]')
        span.set_attribute("ai.prompt", '{"prompt": "secret"}')
        span.set_attribute("ai.prompt.messages", '[{"role": "user", "content": "secret"}]')
        span.set_attribute("ai.prompt.tools", '["secret schema"]')
        span.set_attribute("ai.response.text", "secret answer")
        span.set_attribute("ai.response.object", '{"secret": 1}')
        span.set_attribute("ai.toolCall.args", '{"city": "secret"}')
        span.set_attribute("ai.toolCall.result", '{"weather": "secret"}')
        # identity stays: names, ids, models
        span.set_attribute("ai.toolCall.name", "get_weather")
        span.set_attribute("ai.response.model", "gpt-test")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    for key in (
        "llm.tools.0.tool.json_schema",
        "llm.tools",
        "ai.prompt",
        "ai.prompt.messages",
        "ai.prompt.tools",
        "ai.response.text",
        "ai.response.object",
        "ai.toolCall.args",
        "ai.toolCall.result",
    ):
        assert key not in attrs, key
    assert attrs["ai.toolCall.name"] == "get_weather"
    assert attrs["ai.response.model"] == "gpt-test"


def test_invocation_parameters_dropped_whole_when_stripping() -> None:
    # The bag's membership is open and provider-defined: litellm and langchain
    # embed the request tools array in it, and nothing stops a provider adding
    # a system prompt or customer data beside it. A per-member blocklist is
    # always one provider behind, so the whole bag is content.
    #
    # That costs nothing the caller can name, because normalization runs
    # before masking and has already lifted every member the rule table knows
    # onto gen_ai.request.*. What is dropped here is exactly what nothing
    # classified.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("chat") as span:
        span.set_attribute(
            "llm.invocation_parameters",
            '{"model": "claude-sonnet-4", "temperature": 0.2,'
            ' "tools": [{"name": "refund", "description": "never above $500"}],'
            ' "system": "Our margin floor is 12%.",'
            ' "extra_body": {"customer_tier": "enterprise-gold"}}',
        )
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes

    assert "llm.invocation_parameters" not in attrs
    # Neither the tools array nor the unrecognised members survive anywhere.
    for secret in ("refund", "margin floor", "enterprise-gold"):
        assert secret not in json.dumps(dict(attrs)), secret
    # The recognised knobs did survive, promoted before masking saw the bag.
    assert attrs["gen_ai.request.model"] == "claude-sonnet-4"
    assert attrs["gen_ai.request.temperature"] == 0.2


def test_invocation_parameters_survive_when_content_is_captured() -> None:
    # Content capture on and no mask is the default, and content keys pass
    # through untouched there. Classifying the bag must not change that.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    with client.get_tracer().start_as_current_span("chat") as span:
        span.set_attribute("llm.invocation_parameters", '{"user": "u-1"}')
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs["llm.invocation_parameters"] == '{"user": "u-1"}'


def test_invocation_parameters_unparseable_dropped_when_stripping() -> None:
    # Once the bag is a content key this is no longer a special case, but it
    # is the input that used to take the partial-redaction path, and that path
    # deleted the key a second time. Kept as the regression guard.
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("chat") as span:
        span.set_attribute("llm.invocation_parameters", "not json {")
    client.flush()
    assert "llm.invocation_parameters" not in inner.get_finished_spans()[0].attributes


def test_invocation_parameters_reach_the_mask_with_their_key() -> None:
    # Before the bag was content it was invisible to a caller-supplied mask:
    # mask dispatch is keyed on the content allowlist, so a user who wanted a
    # member redacted had no hook at all. Now it arrives like any content key,
    # and a key-aware mask can act on it precisely.
    seen: dict[str, object] = {}

    def mask(value: object, *, key: str) -> object:
        seen[key] = value
        return "[masked]" if key == "llm.invocation_parameters" else value

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=mask)
    with client.get_tracer().start_as_current_span("chat") as span:
        span.set_attribute("llm.invocation_parameters", '{"user": "u-1"}')
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "llm.invocation_parameters" in seen
    assert attrs["llm.invocation_parameters"] == "[masked]"


# --- span status: provider errors echo the rejected request into the message ---


def _error_span_exported(**init_kwargs):
    from opentelemetry.trace import Status, StatusCode

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, **init_kwargs)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_status(Status(StatusCode.ERROR, "400: messages=[{'content':'SSN 123-45-6789'}]"))
    client.flush()
    return inner.get_finished_spans()[0]


def test_capture_content_false_drops_status_description_keeps_code() -> None:
    from opentelemetry.trace import StatusCode

    span = _error_span_exported(capture_content=False)
    assert span.status.status_code == StatusCode.ERROR
    assert not span.status.description


def test_mask_runs_over_status_description() -> None:
    from opentelemetry.trace import StatusCode

    span = _error_span_exported(mask=lambda _v: "***")
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "***"


def test_status_description_untouched_without_sanitization() -> None:
    span = _error_span_exported()
    assert "SSN 123-45-6789" in (span.status.description or "")


def test_status_sanitization_does_not_mutate_the_span_other_processors_see() -> None:
    from opentelemetry.trace import Status, StatusCode

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_status(Status(StatusCode.ERROR, "secret"))
        live = span
    client.flush()
    assert live.status.description == "secret"  # original untouched
    assert not inner.get_finished_spans()[0].status.description


# --- GenAI semconv tool-call / system-instruction keys, and OpenInference TOOL keys ---


def test_capture_content_false_strips_genai_tool_call_and_system_keys() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("gen_ai.system_instructions", "SECRET SYSTEM PROMPT")
        span.set_attribute("gen_ai.tool.call.arguments", '{"city":"Berlin"}')
        span.set_attribute("gen_ai.tool.call.result", '{"temp":21}')
        span.set_attribute("tool.description", "Looks up weather; internal URL http://x")
        span.set_attribute("tool.parameters", '{"type":"object"}')
        span.set_attribute("gen_ai.tool.name", "get_weather")
        span.set_attribute("gen_ai.tool.call.id", "call_1")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    for key in (
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "tool.description",
        "tool.parameters",
    ):
        assert key not in attrs, key
    assert attrs["gen_ai.tool.name"] == "get_weather"  # identity stays
    assert attrs["gen_ai.tool.call.id"] == "call_1"


# --- metadata.* mirrors: the OpenInference Vercel transform copies attributes
#     it does not translate under metadata.<original key> ---


def test_metadata_mirror_of_a_content_key_is_content() -> None:
    from rius.masking import _is_content_key

    assert _is_content_key("metadata.gen_ai.system_instructions")
    assert _is_content_key("metadata.gen_ai.input.messages")
    assert _is_content_key("metadata.llm.input_messages.0.message.content")
    assert not _is_content_key("metadata.gen_ai.tool.name")
    assert not _is_content_key("metadata.ai.model.id")


def test_capture_content_false_strips_metadata_mirrors() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("metadata.gen_ai.system_instructions", "SECRET")
        span.set_attribute("metadata.gen_ai.tool.name", "get_weather")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "metadata.gen_ai.system_instructions" not in attrs
    assert attrs["metadata.gen_ai.tool.name"] == "get_weather"


def test_capture_content_false_keeps_both_request_namespaces() -> None:
    """RIUS-917: request parameters are not content. A caller parameter in
    rius.request.* survives exactly as a spec one in gen_ai.request.* does."""
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("gen_ai.request.temperature", 0.7)
        span.set_attribute("rius.request.my_custom_knob", 3)
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["rius.request.my_custom_knob"] == 3


def test_capture_content_false_strips_tool_definitions_from_request_parameters() -> None:
    """A tools array is content wherever it lands. Three routes reach the same
    definitions — gen_ai.tool.definitions, the tools member inside
    llm.invocation_parameters, and a `tools` model parameter — and they must
    not disagree about whether capture_content=False protects them."""
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, capture_content=False)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("rius.request.tools", '[{"description": "SECRET"}]')
        span.set_attribute("rius.request.functions", '[{"description": "SECRET"}]')
        span.set_attribute("gen_ai.request.tools", '[{"description": "SECRET"}]')
        span.set_attribute("gen_ai.request.functions", '[{"description": "SECRET"}]')
        # The scalar knobs around them are identity and must survive.
        span.set_attribute("gen_ai.request.temperature", 0.7)
        span.set_attribute("rius.request.my_custom_knob", 3)
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert "rius.request.tools" not in attrs
    assert "rius.request.functions" not in attrs
    assert "gen_ai.request.tools" not in attrs
    assert "gen_ai.request.functions" not in attrs
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["rius.request.my_custom_knob"] == 3


def test_mask_redacts_tool_definitions_from_request_parameters() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("rius.request.tools", '[{"description": "SECRET"}]')
        span.set_attribute("gen_ai.request.temperature", 0.7)
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs["rius.request.tools"] == "***"
    assert attrs["gen_ai.request.temperature"] == 0.7


# --- content keys of other OTel-based SDKs (RIUS-1070): with the bridge on, a
#     Langfuse or Logfire span reaches this exporter, and so it does whenever
#     rius holds the global provider those SDKs then write to ---

THIRD_PARTY_CONTENT_KEYS = (
    # langfuse 3.15, langfuse/_client/attributes.py
    "langfuse.observation.input",
    "langfuse.observation.output",
    "langfuse.trace.input",
    "langfuse.trace.output",
    "langfuse.observation.metadata",
    "langfuse.observation.metadata.customer_note",
    "langfuse.trace.metadata",
    "langfuse.trace.metadata.customer_note",
    "langfuse.observation.status_message",
    "langfuse.experiment.metadata",
    "langfuse.experiment.item.metadata",
    "langfuse.experiment.item.expected_output",
    # traceloop-sdk 0.62.3 and its instrumentations
    "traceloop.entity.input",
    "traceloop.entity.output",
    "traceloop.prompt.template",
    "traceloop.prompt.template_variables",
    "traceloop.prompt.template_variables.question",
    "gen_ai.task.input",
    "gen_ai.task.output",
    "mcp.response.value",
    # logfire 5.1.1
    "request_data",
    "response_data",
    "events",
    "all_messages_events",
    "pydantic_ai.all_messages",
    "raw_input",
    "response",
    "input",
    "output",
    # mlflow-tracing 3.16.1, mlflow/tracing/constant.py
    "mlflow.spanInputs",
    "mlflow.spanOutputs",
    "mlflow.chat.tools",
    "mlflow.trace.intermediate_outputs",
    "mlflow.chunk.value",
    # openlit 1.45.0, openlit/semcov/__init__.py
    "gen_ai.retrieval.query.text",
    "gen_ai.content.reasoning",
    "gen_ai.content.revised_prompt",
    "gen_ai.tool.args",
    "gen_ai.response.tool_calls",
    "gen_ai.workflow.input",
    "gen_ai.workflow.output",
    "gen_ai.framework.pipeline.input_data",
    "gen_ai.framework.pipeline.output_data",
    "gen_ai.framework.error.message",
    "gen_ai.agent.context",
    "gen_ai.agent.instructions",
    "gen_ai.agent.goal",
    "gen_ai.agent.action.tool_input",
    "gen_ai.agent.final_result",
    "gen_ai.agent.next_goal",
    "gen_ai.agent.step_messages",
    "gen_ai.memory.search.query",
    "gen_ai.vectordb.search.query",
    "gen_ai.extraction.instruction",
    "mcp.tool.arguments",
    "mcp.tool.result",
    "mcp.result",
    "mcp.params",
    "mcp.sampling.messages",
    "mcp.fastmcp.prompt.arguments",
    "mcp.completion.argument.value",
    "mcp.completion.context.arguments",
    "mcp.completion.values",
    "mcp.error.message",
)

# Keys the same SDKs write next to their content that must survive content
# capture off: usage, identity and shape, several of them one suffix away
# from a content key.
THIRD_PARTY_KEYS_THAT_SURVIVE = (
    "input_tokens",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "input.mime_type",
    "output.mime_type",
    "llm.token_count.prompt",
    "llm.token_count.completion",
    "llm.token_count.total",
    "langfuse.observation.type",
    "langfuse.observation.level",
    "langfuse.observation.model.name",
    "langfuse.observation.usage_details",
    "langfuse.trace.name",
    "langfuse.trace.tags",
    "session.id",
    "user.id",
    "traceloop.entity.name",
    "traceloop.prompt.key",
    "mlflow.spanType",
    "mlflow.chat.tokenUsage",
    "gen_ai.agent.description",
    "mcp.method.name",
)


def _exported_attributes(values: dict[str, str], **init_kwargs: Any) -> dict[str, object]:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, **init_kwargs)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attributes(values)
    client.flush()
    return dict(inner.get_finished_spans()[0].attributes or {})


def test_capture_content_false_strips_third_party_sdk_content_keys() -> None:
    attrs = _exported_attributes(
        {key: "SECRET" for key in THIRD_PARTY_CONTENT_KEYS}, capture_content=False
    )
    leaked = sorted(key for key, value in attrs.items() if value == "SECRET")
    assert leaked == []


def test_third_party_usage_and_identity_keys_are_not_content() -> None:
    from rius.masking import _is_content_key

    assert [key for key in THIRD_PARTY_KEYS_THAT_SURVIVE if _is_content_key(key)] == []


def test_capture_content_false_keeps_third_party_usage_and_identity_keys() -> None:
    # Normalization renames llm.token_count.{prompt,completion} before masking
    # runs, so only the key-level test above can speak for those two.
    renamed = {"llm.token_count.prompt", "llm.token_count.completion"}
    values = {key: "kept" for key in THIRD_PARTY_KEYS_THAT_SURVIVE if key not in renamed}
    attrs = _exported_attributes(values, capture_content=False)
    assert {key: attrs.get(key) for key in values} == values


def test_third_party_sdk_content_keys_reach_the_mask_with_their_key() -> None:
    seen: list[str] = []

    def mask(value: object, *, key: str) -> str:
        seen.append(key)
        return "***"

    _exported_attributes({key: "SECRET" for key in THIRD_PARTY_CONTENT_KEYS}, mask=mask)
    assert sorted(seen) == sorted(THIRD_PARTY_CONTENT_KEYS)
