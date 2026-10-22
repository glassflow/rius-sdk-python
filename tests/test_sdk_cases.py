"""This SDK against rius-semconv's shared SDK cases.

``tests/fixtures/semconv/sdk_cases`` is rius-semconv's ``sdk_cases/``, synced
by scripts/sync-semconv-fixtures.sh. The TypeScript SDK runs the same cases, so
both SDKs are held to one expected result for each: how input.value and
output.value are written, how set_attribute encodes a value, how messages and
rius.context.sizes are recorded, what the masking exporter keeps, how RIUS_*
variables resolve and how the resource takes in OTEL_RESOURCE_ATTRIBUTES.
Their README describes the format, including the tagged values JSON cannot
hold (``{"$bytes": N}``, ``{"$number": "NaN"}``).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

import rius
from rius.config import resolve_config
from rius.masking import MaskingSpanExporter

CASES = Path(__file__).parent / "fixtures" / "semconv" / "sdk_cases"


def _cases(name: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads((CASES / f"{name}.json").read_text())["cases"]
    assert cases, f"no cases in {name}.json: is sdk_cases synced?"
    return cases


def _ids(name: str) -> list[str]:
    return [case["name"] for case in _cases(name)]


def _value(raw: Any) -> Any:
    """A case's value with its tagged forms decoded."""
    if isinstance(raw, dict):
        if set(raw) == {"$bytes"}:
            return bytes(raw["$bytes"])
        if set(raw) == {"$number"}:
            return float(raw["$number"])
        return {key: _value(item) for key, item in raw.items()}
    if isinstance(raw, list):
        return [_value(item) for item in raw]
    return raw


class _Spans:
    """A client exporting to memory, read back one span at a time.

    No instrumentation: patching a library here would leak into later tests.
    """

    def __init__(self) -> None:
        self.sink = InMemorySpanExporter()
        self.client = rius.init(
            span_exporter=self.sink, heartbeat=False, instruments=[], service_name="svc"
        )

    def attributes(self) -> dict[str, Any]:
        self.client.flush()
        (span,) = self.sink.get_finished_spans()
        return dict(span.attributes or {})


def _mixes_int_and_float(values: list[Any]) -> bool:
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return any(isinstance(v, int) for v in numbers) and any(isinstance(v, float) for v in numbers)


def _same(got: Any, want: Any, int_as_float: bool = False) -> bool:
    """``got`` equals ``want`` with its types: ``==`` alone takes True for 1
    and 1 for 1.0. An int may be written as a float only inside an expected
    array that mixes ints and floats, the one place the SDK makes it a float
    so OpenTelemetry Python before 1.25 keeps the array."""
    if isinstance(want, list):
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            return False
        loose = _mixes_int_and_float(want)
        return all(_same(g, w, loose) for g, w in zip(got, want, strict=True))
    if int_as_float and type(want) is int and type(got) is float:
        return got == want
    return type(got) is type(want) and got == want


@pytest.fixture
def exporter() -> Iterator[_Spans]:
    spans = _Spans()
    yield spans
    spans.client.shutdown()


@pytest.mark.parametrize("case", _cases("values"), ids=_ids("values"))
def test_input_and_output_value(case: dict[str, Any], exporter: _Spans) -> None:
    span = rius.start_span("s")
    span.set_input(_value(case["value"]))
    span.set_output(_value(case["value"]))
    span.end()
    attributes = exporter.attributes()
    assert attributes["input.value"] == case["expected"]
    assert attributes["output.value"] == case["expected"]


@pytest.mark.parametrize("case", _cases("attributes"), ids=_ids("attributes"))
def test_set_attribute(case: dict[str, Any], exporter: _Spans) -> None:
    span = rius.start_span("s")
    span.set_attribute("k", _value(case["value"]))
    span.end()
    written = exporter.attributes()["k"]
    assert _same(written, case["expected"]), (written, case["expected"])


@pytest.mark.parametrize("case", _cases("messages"), ids=_ids("messages"))
def test_messages(case: dict[str, Any], exporter: _Spans) -> None:
    generation = rius.start_generation(model="m")
    if case["direction"] == "input":
        generation.set_input(_value(case["value"]))
    else:
        generation.set_output(_value(case["value"]))
    generation.end()
    attributes = exporter.attributes()
    key = f"gen_ai.{case['direction']}.messages"
    assert json.loads(attributes[key]) == case["messages"]
    assert json.loads(attributes["rius.context.sizes"]) == case["context_sizes"]


def _mask(spec: dict[str, Any] | None) -> Any:
    if spec is None:
        return None
    if "returns" in spec:
        returned = _value(spec["returns"])
        return lambda value, key: returned
    if "format" in spec:
        template = spec["format"]
        return lambda value, key: template.format(key=key, value=value)
    if spec.get("throws"):

        def throws(value: Any, key: str) -> Any:
            raise RuntimeError("mask failed")

        return throws
    raise ValueError(f"unknown mask {spec}")


@pytest.mark.parametrize("case", _cases("masking"), ids=_ids("masking"))
def test_masking(case: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    inner = InMemorySpanExporter()
    masking = MaskingSpanExporter(
        inner, capture_content=case["capture_content"], mask=_mask(case["mask"])
    )
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(masking))
    given = case["span"]
    with caplog.at_level(logging.CRITICAL, logger="rius.masking"):
        span = provider.get_tracer("cases").start_span("s", attributes=given["attributes"])
        for event in given.get("events", []):
            span.add_event(event["name"], event["attributes"])
        if "status" in given:
            span.set_status(Status(StatusCode.ERROR, given["status"]))
        span.end()
    (exported,) = inner.get_finished_spans()
    expected = case["expected"]
    assert dict(exported.attributes or {}) == expected["attributes"]
    assert [
        {"name": event.name, "attributes": dict(event.attributes or {})}
        for event in exported.events
    ] == expected["events"]
    assert (exported.status.description or None) == expected["status"]
    if "status" in given:
        assert exported.status.status_code is StatusCode.ERROR


#: The Python field each env case names, as resolve_config() returns it.
_ENV_FIELDS = {
    "endpoint",
    "service_name",
    "agent_name",
    "service_version",
    "disabled",
    "capture_content",
    "heartbeat",
    "partial_spans",
    "bridge_foreign_provider",
    "sample_rate",
    "heartbeat_interval",
    "partial_spans_delay",
}


def _clear_rius_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("RIUS_") or name == "OTEL_RESOURCE_ATTRIBUTES":
            monkeypatch.delenv(name)


@pytest.mark.parametrize("case", _cases("env"), ids=_ids("env"))
def test_env(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clear_rius_env(monkeypatch)
    for name, raw in case["env"].items():
        monkeypatch.setenv(name, raw)
    with caplog.at_level(logging.WARNING, logger="rius.config"):
        config = resolve_config()
    assert set(case["expected"]) <= _ENV_FIELDS
    assert {field: getattr(config, field) for field in case["expected"]} == case["expected"]
    if "warns" in case:
        warned = [r for r in caplog.records if r.name == "rius.config"]
        assert bool(warned) == case["warns"], [r.getMessage() for r in warned]


@pytest.mark.parametrize("case", _cases("resource"), ids=_ids("resource"))
def test_resource(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_rius_env(monkeypatch)
    if case["otel_resource_attributes"] is not None:
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", case["otel_resource_attributes"])
    sink = InMemorySpanExporter()
    client = rius.init(span_exporter=sink, heartbeat=False, instruments=[], **case["options"])
    try:
        rius.start_span("s").end()
        client.flush()
        (span,) = sink.get_finished_spans()
        attributes = dict(span.resource.attributes)
    finally:
        client.shutdown()
    assert {key: attributes.get(key) for key in case["expected"]} == case["expected"]
    assert [key for key in case["absent"] if key in attributes] == []
