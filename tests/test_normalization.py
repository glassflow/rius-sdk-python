"""Normalization: third-party attribute keys mapped onto the canonical wire.

Two components under test, because ``ReadableSpan.attributes`` is a read-only
``mappingproxy``: a ``SpanProcessor`` that maps at ``on_start`` (where the
object is a live ``Span``) and a ``SpanExporter`` wrapper that maps at export
(rebuilding from a copy, never in place).
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.normalization import (
    NormalizationTable,
    NormalizingSpanExporter,
    NormalizingSpanProcessor,
    Rule,
    copy_value,
    json_member,
    to_int,
    total,
    wrap_in_list,
)
from rius.semconv import GEN_AI_INPUT_MESSAGES, GEN_AI_REQUEST_MODEL, RIUS_SPAN_PENDING


def _table(*rules: Rule) -> NormalizationTable:
    return NormalizationTable(rules)


def _normalize(table: NormalizationTable, attributes: dict[str, Any]) -> dict[str, Any]:
    out = table.normalize(attributes)
    return attributes if out is None else out


# --- the contract rules -----------------------------------------------------


def test_source_key_is_mapped_to_its_target() -> None:
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    assert _normalize(table, {"vendor.model": "gpt-4o"}) == {GEN_AI_REQUEST_MODEL: "gpt-4o"}


def test_native_canonical_key_is_never_overwritten() -> None:
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    out = _normalize(table, {"vendor.model": "wrong", GEN_AI_REQUEST_MODEL: "native"})
    assert out[GEN_AI_REQUEST_MODEL] == "native"


def test_mapped_source_key_is_deleted() -> None:
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    assert "vendor.model" not in _normalize(table, {"vendor.model": "gpt-4o"})


def test_source_key_is_deleted_even_when_the_native_key_wins() -> None:
    """It is a duplicate of a key we already have; keeping it double-counts."""
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    out = _normalize(table, {"vendor.model": "wrong", GEN_AI_REQUEST_MODEL: "native"})
    assert "vendor.model" not in out


def test_unmapped_key_is_left_untouched() -> None:
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    out = _normalize(table, {"vendor.model": "gpt-4o", "vendor.other": 7})
    assert out["vendor.other"] == 7


def test_fast_path_leaves_a_span_without_source_keys_alone() -> None:
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    attributes = {"gen_ai.operation.name": "chat", "session.id": "s"}
    assert table.applies(attributes) is False
    assert table.normalize(attributes) is None


# --- converters -------------------------------------------------------------


def test_to_int_parses_a_string_count() -> None:
    table = _table(Rule("vendor.tokens", "gen_ai.usage.input_tokens", to_int))
    assert _normalize(table, {"vendor.tokens": "12"})["gen_ai.usage.input_tokens"] == 12


def test_to_int_skips_an_unparseable_value() -> None:
    table = _table(Rule("vendor.tokens", "gen_ai.usage.input_tokens", to_int))
    out = _normalize(table, {"vendor.tokens": "not a number"})
    assert "gen_ai.usage.input_tokens" not in out
    assert "vendor.tokens" not in out


def test_wrap_in_list_promotes_a_scalar() -> None:
    table = _table(Rule("vendor.reason", "gen_ai.response.finish_reasons", wrap_in_list))
    out = _normalize(table, {"vendor.reason": "stop"})
    assert out["gen_ai.response.finish_reasons"] == ["stop"]


def test_json_member_promotes_a_member_of_a_json_payload() -> None:
    table = _table(Rule("vendor.params", GEN_AI_REQUEST_MODEL, json_member("model")))
    out = _normalize(table, {"vendor.params": '{"model": "gpt-4o", "temperature": 0}'})
    assert out[GEN_AI_REQUEST_MODEL] == "gpt-4o"


def test_json_member_skips_when_the_member_is_absent() -> None:
    table = _table(Rule("vendor.params", GEN_AI_REQUEST_MODEL, json_member("model")))
    out = _normalize(table, {"vendor.params": '{"temperature": 0}'})
    assert GEN_AI_REQUEST_MODEL not in out


def test_total_sums_several_source_keys() -> None:
    table = _table(Rule(("vendor.a", "vendor.b"), "gen_ai.usage.input_tokens", total))
    out = _normalize(table, {"vendor.a": 3, "vendor.b": 4})
    assert out["gen_ai.usage.input_tokens"] == 7
    assert "vendor.a" not in out and "vendor.b" not in out


def test_total_sums_the_sources_that_are_present() -> None:
    table = _table(Rule(("vendor.a", "vendor.b"), "gen_ai.usage.input_tokens", total))
    assert _normalize(table, {"vendor.a": 3})["gen_ai.usage.input_tokens"] == 3


def test_a_converter_that_raises_drops_only_its_own_rule() -> None:
    def boom(values: list[Any]) -> Any:
        raise RuntimeError("nope")

    table = _table(
        Rule("vendor.bad", "gen_ai.request.temperature", boom),
        Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value),
    )
    out = _normalize(table, {"vendor.bad": 1, "vendor.model": "gpt-4o"})
    assert "gen_ai.request.temperature" not in out
    assert out[GEN_AI_REQUEST_MODEL] == "gpt-4o"


# --- the exporter wrapper ---------------------------------------------------


def _exported(table: NormalizationTable, attributes: dict[str, Any]):
    inner = InMemorySpanExporter()
    exporter = NormalizingSpanExporter(inner, table=table)
    provider_exporter = InMemorySpanExporter()
    client = init(
        span_exporter=provider_exporter,
        set_global=False,
        service_name="test-svc",
        instruments=[],
    )
    try:
        with client.get_tracer().start_as_current_span("op", attributes=attributes):
            pass
        client.flush()
        (span,) = provider_exporter.get_finished_spans()
        exporter.export([span])
        return span, inner.get_finished_spans()[0]
    finally:
        client.shutdown()


def test_exporter_maps_at_export_time() -> None:
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    _, exported = _exported(table, {"vendor.model": "gpt-4o"})
    assert exported.attributes[GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert "vendor.model" not in exported.attributes


def test_exporter_does_not_mutate_the_span_in_place() -> None:
    """A ReadableSpan's attributes are shared by reference with every exporter."""
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    original, exported = _exported(table, {"vendor.model": "gpt-4o"})
    assert original.attributes["vendor.model"] == "gpt-4o"
    assert GEN_AI_REQUEST_MODEL not in original.attributes
    assert exported is not original


def test_exporter_returns_the_same_object_when_nothing_maps() -> None:
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    original, exported = _exported(table, {"other.key": "x"})
    assert exported is original


# --- end-to-end wiring ------------------------------------------------------


def _memory_client(**kwargs: Any):
    exporter = InMemorySpanExporter()
    client = init(
        span_exporter=exporter,
        set_global=False,
        service_name="test-svc",
        instruments=[],
        **kwargs,
    )
    return client, exporter


def _split(spans):
    pending = [s for s in spans if s.attributes.get(RIUS_SPAN_PENDING)]
    final = [s for s in spans if not s.attributes.get(RIUS_SPAN_PENDING)]
    return pending, final


def test_normalization_is_wired_into_init() -> None:
    client, exporter = _memory_client()
    try:
        with client.get_tracer().start_as_current_span("op", attributes={"llm.model_name": "m"}):
            pass
        client.flush()
        (span,) = exporter.get_finished_spans()
        assert span.attributes[GEN_AI_REQUEST_MODEL] == "m"
        assert "llm.model_name" not in span.attributes
    finally:
        client.shutdown()


def test_start_time_mapping_reaches_a_pending_snapshot() -> None:
    """The exporter alone is too late: pending snapshots are built at on_start."""
    client, exporter = _memory_client(partial_spans=True)
    try:
        with client.get_tracer().start_as_current_span("op", attributes={"llm.model_name": "m"}):
            pass
        client.flush()
        (pending,), _ = _split(exporter.get_finished_spans())
        assert pending.attributes[GEN_AI_REQUEST_MODEL] == "m"
    finally:
        client.shutdown()


def test_normalization_runs_before_masking() -> None:
    """Reversed, masking would strip llm.input_messages before it is mapped."""
    seen: list[str] = []

    def mask(value: Any, key: str) -> Any:
        seen.append(key)
        return "***"

    client, exporter = _memory_client(mask=mask)
    try:
        with client.get_tracer().start_as_current_span(
            "op", attributes={"llm.input_messages": '[{"role": "user"}]'}
        ):
            pass
        client.flush()
        (span,) = exporter.get_finished_spans()
    finally:
        client.shutdown()
    assert GEN_AI_INPUT_MESSAGES in seen
    assert "llm.input_messages" not in seen
    assert span.attributes[GEN_AI_INPUT_MESSAGES] == "***"


def test_mapped_content_key_is_stripped_with_capture_content_off() -> None:
    client, exporter = _memory_client(capture_content=False)
    try:
        with client.get_tracer().start_as_current_span(
            "op", attributes={"llm.input_messages": '[{"role": "user"}]'}
        ):
            pass
        client.flush()
        (span,) = exporter.get_finished_spans()
    finally:
        client.shutdown()
    assert GEN_AI_INPUT_MESSAGES not in span.attributes
    assert "llm.input_messages" not in span.attributes


def test_a_second_registered_exporter_sees_an_unnormalized_span() -> None:
    """In-place mutation would rewrite what every other exporter sees."""
    other = InMemorySpanExporter()
    client, exporter = _memory_client()
    try:
        client._provider.add_span_processor(SimpleSpanProcessor(other))
        with client.get_tracer().start_as_current_span("op") as span:
            span.set_attribute("llm.model_name", "m")
        client.flush()
        (seen,) = other.get_finished_spans()
        (normalized,) = exporter.get_finished_spans()
    finally:
        client.shutdown()
    assert seen.attributes["llm.model_name"] == "m"
    assert normalized.attributes[GEN_AI_REQUEST_MODEL] == "m"


# --- the processor ----------------------------------------------------------


def test_processor_maps_at_start_without_deleting_the_source() -> None:
    """A live Span can be added to but not deleted from; the exporter deletes."""
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, service_name="test-svc", instruments=[])
    try:
        client._provider.add_span_processor(NormalizingSpanProcessor(table=table))
        with client.get_tracer().start_as_current_span(
            "op", attributes={"vendor.model": "gpt-4o"}
        ) as span:
            assert span.attributes[GEN_AI_REQUEST_MODEL] == "gpt-4o"
            # a live Span has no delete; the source rides on until export
            assert span.attributes["vendor.model"] == "gpt-4o"
    finally:
        client.shutdown()


@pytest.mark.parametrize("attributes", [{}, {"unrelated": 1}])
def test_processor_fast_path(attributes: dict[str, Any]) -> None:
    table = _table(Rule("vendor.model", GEN_AI_REQUEST_MODEL, copy_value))
    assert table.applies(attributes) is False
