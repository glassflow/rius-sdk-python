"""Export-time rewrites must hold up on the oldest OpenTelemetry SDK we declare.

Before 1.25 an SDK ``Event`` has no ``dropped_attributes`` (``Link`` gains it
in 1.27), and the OTLP encoder reads ``attributes.dropped`` straight off every
event and link, so an event or link rebuilt with a plain dict crashes the
whole batch. These tests pin both contracts on the locked SDK; the floor CI
job runs the full suite on the old one.
"""

from __future__ import annotations

from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius._attributes import replacement_attributes

DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"


def _bounded(dropped: int) -> BoundedAttributes:
    attributes = BoundedAttributes(attributes={"k": "v"}, immutable=False)
    attributes.dropped = dropped
    return attributes


def test_the_dropped_count_is_read_off_an_item_without_dropped_attributes() -> None:
    # An Event before OTel 1.25 or a Link before 1.27: the count lives only
    # on its BoundedAttributes.
    legacy_event = SimpleNamespace(attributes=_bounded(3))
    carried = replacement_attributes(legacy_event, {"k": "x"})  # type: ignore[arg-type]
    assert isinstance(carried, BoundedAttributes)
    assert carried.dropped == 3


def test_an_item_without_any_dropped_count_keeps_the_plain_mapping() -> None:
    plain = {"k": "x"}
    assert replacement_attributes(SimpleNamespace(attributes={"k": "v"}), plain) is plain  # type: ignore[arg-type]


def _exported_with_event_and_link(**init_kwargs: object) -> object:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, **init_kwargs)  # type: ignore[arg-type]
    link_context = trace.SpanContext(
        trace_id=1, span_id=2, is_remote=True, trace_flags=trace.TraceFlags(1)
    )
    span = client.get_tracer().start_span(
        "s", links=[trace.Link(link_context, {"image": DATA_URL, "input.value": "s"})]
    )
    span.add_event("e", {"image": DATA_URL, "input.value": "s"})
    span.end()
    client.flush()
    [finished] = inner.get_finished_spans()
    return finished


def _assert_encodable(finished: object) -> None:
    for event in finished.events:  # type: ignore[attr-defined]
        assert isinstance(event.attributes, BoundedAttributes), event.attributes
    for link in finished.links:  # type: ignore[attr-defined]
        assert isinstance(link.attributes, BoundedAttributes), link.attributes


def test_media_stripped_events_and_links_stay_encodable() -> None:
    finished = _exported_with_event_and_link()
    assert finished.events[0].attributes["image"] != DATA_URL  # type: ignore[attr-defined]
    _assert_encodable(finished)


def test_masked_events_and_links_stay_encodable() -> None:
    # Content capture off: masking drops input.value from both, no media involved.
    finished = _exported_with_event_and_link(capture_content=False)
    assert "input.value" not in finished.events[0].attributes  # type: ignore[attr-defined]
    _assert_encodable(finished)


def test_events_and_links_masked_by_a_callable_stay_encodable() -> None:
    finished = _exported_with_event_and_link(mask=lambda value: "***" if value == "s" else value)
    assert finished.events[0].attributes["input.value"] == "***"  # type: ignore[attr-defined]
    _assert_encodable(finished)
