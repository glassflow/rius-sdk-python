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
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=lambda _v: "***")
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("llm.input_messages.0.message.content", "secret")
        span.set_attribute("gen_ai.prompt.0.content", "secret")  # OpenLLMetry style
        span.set_attribute("llm.model_name", "gpt-4o")
    client.flush()
    attrs = inner.get_finished_spans()[0].attributes
    assert attrs["llm.input_messages.0.message.content"] == "***"
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
    assert attrs.get("input.value") == '{"redacted": true}'


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
