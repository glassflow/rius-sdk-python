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
