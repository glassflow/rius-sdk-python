"""OpenInference EMBEDDING spans: the model mapped, the vectors dropped.

The raw attribute set below is what ``openinference-instrumentation-openai``
0.1.52 writes for ``OpenAI().embeddings.create(model=..., input=[...])``,
captured from the real instrumentor on a bare ``TracerProvider``. Without
normalization the span has no ``gen_ai.request.model`` (so the backend cannot
price it) and carries every vector, each a list of 1536 floats for
``text-embedding-3-small``, which ``output.value`` already holds as well.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.masking import MaskingSpanExporter
from rius.normalization import DEFAULT_TABLE, NormalizingSpanExporter, drop_embedding_vectors
from rius.semconv import GEN_AI_OPERATION_NAME, GEN_AI_REQUEST_MODEL, GEN_AI_USAGE_INPUT_TOKENS

_INPUT_VALUE = (
    '{"input": ["hello", "world"], "model": "text-embedding-3-small", "encoding_format": "base64"}'
)
_OUTPUT_VALUE = (
    '{"data":[{"embedding":[0.1,0.2,0.3],"index":0,"object":"embedding"},'
    '{"embedding":[0.4,0.5,0.6],"index":1,"object":"embedding"}],'
    '"model":"text-embedding-3-small","object":"list",'
    '"usage":{"prompt_tokens":8,"total_tokens":8}}'
)

#: In the order the instrumentor writes them.
_RAW_OPENAI_EMBEDDING: dict[str, Any] = {
    "llm.system": "openai",
    "input.value": _INPUT_VALUE,
    "input.mime_type": "application/json",
    "output.value": _OUTPUT_VALUE,
    "output.mime_type": "application/json",
    "embedding.invocation_parameters": (
        '{"model": "text-embedding-3-small", "encoding_format": "base64"}'
    ),
    "embedding.embeddings.0.embedding.text": "hello",
    "embedding.embeddings.1.embedding.text": "world",
    "llm.token_count.total": 8,
    "llm.token_count.prompt": 8,
    "embedding.model_name": "text-embedding-3-small",
    "embedding.embeddings.0.embedding.vector": (0.1, 0.2, 0.3),
    "embedding.embeddings.1.embedding.vector": (0.4, 0.5, 0.6),
    "openinference.span.kind": "EMBEDDING",
}


def _export(attributes: dict[str, Any], *, capture_content: bool = True) -> ReadableSpan:
    """One span with ``attributes`` through the normalizer (and masking) on a bare provider."""
    inner = InMemorySpanExporter()
    chain: Any = inner
    if not capture_content:
        chain = MaskingSpanExporter(chain, capture_content=False)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(NormalizingSpanExporter(chain)))
    with provider.get_tracer("t").start_as_current_span("CreateEmbeddings") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    (exported,) = inner.get_finished_spans()
    return exported


def test_openai_embedding_span_golden() -> None:
    exported = _export(_RAW_OPENAI_EMBEDDING)
    assert dict(exported.attributes or {}) == {
        "llm.system": "openai",
        "input.value": _INPUT_VALUE,
        "input.mime_type": "application/json",
        "output.value": _OUTPUT_VALUE,
        "output.mime_type": "application/json",
        "embedding.invocation_parameters": (
            '{"model": "text-embedding-3-small", "encoding_format": "base64"}'
        ),
        "embedding.embeddings.0.embedding.text": "hello",
        "embedding.embeddings.1.embedding.text": "world",
        "llm.token_count.total": 8,
        GEN_AI_USAGE_INPUT_TOKENS: 8,
        GEN_AI_REQUEST_MODEL: "text-embedding-3-small",
        "openinference.span.kind": "EMBEDDING",
        GEN_AI_OPERATION_NAME: "embeddings",
    }


def test_capture_off_strips_the_embedded_text_but_keeps_the_model() -> None:
    attributes = dict(_export(_RAW_OPENAI_EMBEDDING, capture_content=False).attributes or {})
    assert not [k for k in attributes if k.startswith("embedding.embeddings.")]
    assert "input.value" not in attributes
    assert "output.value" not in attributes
    assert attributes[GEN_AI_REQUEST_MODEL] == "text-embedding-3-small"
    assert attributes[GEN_AI_OPERATION_NAME] == "embeddings"
    assert attributes[GEN_AI_USAGE_INPUT_TOKENS] == 8


# --- the model rule ---------------------------------------------------------


def _normalize(attributes: dict[str, Any]) -> dict[str, Any]:
    out = DEFAULT_TABLE.normalize(attributes)
    return dict(attributes) if out is None else out


def test_embedding_model_name_is_mapped_and_deleted() -> None:
    assert _normalize({"embedding.model_name": "text-embedding-3-small"}) == {
        GEN_AI_REQUEST_MODEL: "text-embedding-3-small"
    }


def test_a_native_request_model_wins_and_the_source_goes() -> None:
    out = _normalize({"embedding.model_name": "m-source", GEN_AI_REQUEST_MODEL: "m-native"})
    assert out == {GEN_AI_REQUEST_MODEL: "m-native"}


@pytest.mark.parametrize("value", ["", 3])
def test_an_unusable_model_name_maps_to_nothing(value: Any) -> None:
    assert GEN_AI_REQUEST_MODEL not in _normalize({"embedding.model_name": value})


def test_the_embedding_rule_does_not_change_llm_model_precedence() -> None:
    # Placed after every existing model spelling: on a span that somehow
    # carries both, the dedicated LLM rule and the request bag keep the key.
    assert _normalize({"llm.request.model_name": "m-llm", "embedding.model_name": "m-emb"})[
        GEN_AI_REQUEST_MODEL
    ] == ("m-llm")
    assert _normalize(
        {"llm.invocation_parameters": '{"model": "m-bag"}', "embedding.model_name": "m-emb"}
    )[GEN_AI_REQUEST_MODEL] == ("m-bag")


def test_start_time_processor_names_the_model_on_the_live_span() -> None:
    from rius.normalization import NormalizingSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(NormalizingSpanProcessor())
    tracer = provider.get_tracer("t")
    span = tracer.start_span("e", attributes={"embedding.model_name": "text-embedding-3-small"})
    assert (span.attributes or {})[GEN_AI_REQUEST_MODEL] == "text-embedding-3-small"
    span.end()


# --- the vectors ------------------------------------------------------------


def test_every_vector_is_dropped_and_the_text_is_kept() -> None:
    out = drop_embedding_vectors(
        {
            "embedding.embeddings.0.embedding.vector": (0.1,),
            "embedding.embeddings.12.embedding.vector": (0.2,),
            "embedding.embeddings.0.embedding.text": "hello",
            "other": 1,
        }
    )
    assert out == {"embedding.embeddings.0.embedding.text": "hello", "other": 1}


def test_any_run_of_ascii_digits_is_an_index() -> None:
    # No upper bound: the rule is the digit pattern, as in the TypeScript SDK.
    key = f"embedding.embeddings.{2**64}.embedding.vector"
    assert drop_embedding_vectors({key: (0.1,)}) == {}


def test_nothing_to_drop_returns_none() -> None:
    assert drop_embedding_vectors({"embedding.embeddings.0.embedding.text": "x"}) is None
    assert drop_embedding_vectors({}) is None
    assert drop_embedding_vectors(None) is None


@pytest.mark.parametrize(
    "key",
    [
        "embedding.embeddings.x.embedding.vector",  # not an index
        # ASCII digits only, matched with the TypeScript SDK: no sign.
        "embedding.embeddings.+1.embedding.vector",
        "embedding.embeddings.-0.embedding.vector",
        "embedding.embeddings.\u0661.embedding.vector",  # an Arabic-Indic digit
        "embedding.embeddings.0.embedding.vector.extra",
        "embedding.embeddings.embedding.vector",
        "other.embeddings.0.embedding.vector",
        # Same lengths as the real prefix and suffix, so only the prefix and
        # suffix checks themselves can reject them.
        "embedding.embeddingz.0.embedding.vector",
        "embedding.embeddings.0.embedding.tokens",
    ],
)
def test_a_key_outside_the_family_is_left_alone(key: str) -> None:
    assert drop_embedding_vectors({key: (0.1,)}) is None


def test_an_llm_span_without_vectors_is_not_rebuilt() -> None:
    # The vector pass must not add a copy to spans it has nothing to do for.
    attributes = {"openinference.span.kind": "LLM", "gen_ai.operation.name": "chat"}
    assert drop_embedding_vectors(attributes) is None


# --- end to end: the real instrumentor through init() -----------------------


@pytest.fixture
def openai_instrumentor() -> Iterator[Any]:
    oi = pytest.importorskip("openinference.instrumentation.openai")
    instrumentor = oi.OpenAIInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    yield instrumentor
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


_EMBEDDING_REPLY = {
    "object": "list",
    "data": [
        {"object": "embedding", "index": i, "embedding": [0.01 * j for j in range(1536)]}
        for i in range(2)
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 8, "total_tokens": 8},
}


@pytest.mark.integration
@pytest.mark.parametrize("capture_content", [True, False])
def test_real_openai_embeddings_span_has_a_model_and_no_vectors(
    openai_instrumentor: Any, capture_content: bool
) -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    inner = InMemorySpanExporter()
    client = init(
        span_exporter=inner,
        set_global=False,
        instruments=["openai"],
        capture_content=capture_content,
    )
    oai = openai.OpenAI(
        api_key="test-key",
        base_url="http://mock.invalid/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=_EMBEDDING_REPLY))
        ),
    )
    oai.embeddings.create(model="text-embedding-3-small", input=["hello", "world"])
    client.flush()
    client.shutdown()

    (span,) = inner.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[GEN_AI_REQUEST_MODEL] == "text-embedding-3-small"
    assert attributes[GEN_AI_OPERATION_NAME] == "embeddings"
    assert attributes["openinference.span.kind"] == "EMBEDDING"
    assert attributes[GEN_AI_USAGE_INPUT_TOKENS] == 8
    assert "embedding.model_name" not in attributes
    assert not [k for k in attributes if k.endswith(".embedding.vector")]
    texts = sorted(k for k in attributes if k.endswith(".embedding.text"))
    if capture_content:
        assert texts == [
            "embedding.embeddings.0.embedding.text",
            "embedding.embeddings.1.embedding.text",
        ]
    else:
        assert texts == []
