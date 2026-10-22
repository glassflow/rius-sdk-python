"""Credential headers leave no span, whatever capture_content and mask say.

A header is not content: masking all of them would hide harmless ones, and
with capture on and no mask a credential would still ship. So a key naming a
credential header (ai.request.headers.<name>, http.request.header.<name>) is
dropped from every span, final and pending, by a step that is always on.
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanContext

from rius import init
from rius._credential_headers import CredentialHeaderDroppingExporter, is_credential_header_key
from rius.semconv import RIUS_SPAN_PENDING


@pytest.mark.parametrize(
    "key",
    [
        "ai.request.headers.authorization",
        # @ai-sdk/otel keeps the caller's spelling on some operations.
        "ai.request.headers.Authorization",
        "ai.request.headers.proxy-authorization",
        "ai.request.headers.cookie",
        "ai.request.headers.x-api-key",
        "ai.request.headers.api-key",
        "ai.request.headers.x-goog-api-key",
        "ai.request.headers.x-auth-token",
        "ai.request.headers.X-Client-Secret",
        # Gateways' own key headers: Portkey, LiteLLM, Kong, Azure APIM, Helicone.
        "ai.request.headers.x-portkey-api-key",
        "ai.request.headers.x-litellm-api-key",
        "ai.request.headers.x-apikey",
        "http.request.header.ocp_apim_subscription_key",
        "ai.request.headers.helicone-auth",
        "http.request.header.authorization",
        "http.response.header.set-cookie",
        # OpenTelemetry Python's HTTP instrumentations spell the header with "_".
        "http.request.header.x_api_key",
        "http.response.header.set_cookie",
        # The OpenInference transform's mirror of an untranslated key.
        "metadata.ai.request.headers.authorization",
        # The name is what follows the segment's last occurrence.
        "ai.request.headers.x.headers.authorization",
    ],
)
def test_a_credential_header_key_is_recognised(key: str) -> None:
    assert is_credential_header_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "ai.request.headers.content-type",
        "ai.request.headers.user-agent",
        "http.request.header.x-forwarded-for",
        # A credential name outside a header segment is someone else's key.
        "authorization",
        "gen_ai.request.model",
        "ai.request.headers",
        # Only a whole hyphen-separated word: "token" alone is not "-token".
        "ai.request.headers.token",
        "ai.request.headers.x-tokens",
        "ai.request.headers.x-ratelimit-remaining-tokens",
        # -api-key, not a bare -key.
        "ai.request.headers.idempotency-key",
    ],
)
def test_other_keys_are_not(key: str) -> None:
    assert not is_credential_header_key(key)


HEADERS: dict[str, Any] = {
    # What @ai-sdk/otel writes for headers={"Authorization": ..., "X-Trace": ...}.
    "ai.request.headers.Authorization": "Bearer sk-secret",
    "ai.request.headers.X-Trace": "kept",
    # What opentelemetry-util-http writes with header capture configured.
    "http.request.header.x_api_key": ("sk-secret",),
    "http.request.header.content_type": ("application/json",),
}
DROPPED = {"ai.request.headers.Authorization", "http.request.header.x_api_key"}
KEPT = {"ai.request.headers.X-Trace", "http.request.header.content_type"}


def _client(**kwargs: Any) -> tuple[Any, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        **kwargs,
    )
    return client, exporter


@pytest.mark.parametrize(
    "settings",
    [
        {"capture_content": False},
        {"capture_content": True},
        {"capture_content": True, "mask": lambda value: value},
    ],
    ids=["capture-off", "capture-on", "capture-on-identity-mask"],
)
def test_dropped_from_attributes_events_and_links(settings: dict[str, Any]) -> None:
    client, exporter = _client(**settings)
    linked = SpanContext(trace_id=1, span_id=1, is_remote=True)
    with client.get_tracer().start_as_current_span(
        "ai.generateText", attributes=HEADERS, links=[Link(linked, HEADERS)]
    ) as span:
        span.add_event("request", HEADERS)
    client.flush()

    (out,) = exporter.get_finished_spans()
    for bag in (out.attributes, out.events[0].attributes, out.links[0].attributes):
        assert bag is not None
        assert not DROPPED & set(bag)
        assert set(bag) >= KEPT


@pytest.mark.parametrize("capture_content", [False, True])
def test_dropped_from_pending_snapshots(capture_content: bool) -> None:
    # gen_ai.request.* rides a pending snapshot as identity, so a header key
    # under it would reach the snapshot through the prefix.
    client, exporter = _client(partial_spans=True, capture_content=capture_content)
    with client.get_tracer().start_as_current_span(
        "chat",
        attributes={
            "gen_ai.request.headers.authorization": "Bearer sk-secret",
            "gen_ai.request.model": "gpt-test",
        },
    ):
        pass
    client.flush()

    spans = exporter.get_finished_spans()
    pending = [s for s in spans if s.attributes.get(RIUS_SPAN_PENDING)]
    assert pending
    assert pending[0].attributes["gen_ai.request.model"] == "gpt-test"
    for out in spans:
        assert "gen_ai.request.headers.authorization" not in out.attributes


def test_the_exporter_leaves_the_original_span_alone() -> None:
    # A ReadableSpan shares its attributes with every processor on the
    # provider, so the drop works on a copy, as masking does.
    inner = InMemorySpanExporter()
    span = ReadableSpan(name="s", attributes=dict(HEADERS))
    CredentialHeaderDroppingExporter(inner).export([span])

    (out,) = inner.get_finished_spans()
    assert out is not span
    assert not DROPPED & set(out.attributes or {})
    assert set(span.attributes or {}) == set(HEADERS)


def test_rebuilt_events_and_links_keep_their_dropped_count() -> None:
    # An event or link that hit the attribute limit says so after the drop.
    def bounded() -> BoundedAttributes:
        attributes = BoundedAttributes(attributes=dict(HEADERS), immutable=True)
        attributes.dropped = 3
        return attributes

    inner = InMemorySpanExporter()
    linked = SpanContext(trace_id=1, span_id=1, is_remote=True)
    span = ReadableSpan(
        name="s",
        events=(Event("request", bounded(), 1),),
        links=(Link(linked, bounded()),),
    )
    CredentialHeaderDroppingExporter(inner).export([span])

    (out,) = inner.get_finished_spans()
    for item in (out.events[0], out.links[0]):
        assert not DROPPED & set(item.attributes or {})
        assert item.dropped_attributes == 3
