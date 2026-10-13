# Vendored from rius-semconv 2d189f8493c2f157c9be44c2dcc6356bf88d6a6d fixtures/_replay.py by
# scripts/sync-semconv-fixtures.sh. Do not edit: change it in rius-semconv.
"""Turn a stored recording back into OpenTelemetry ReadableSpans (Python).

The inverse of `_recording.scenario_json`: every span of a scenario becomes a
finished ReadableSpan carrying the recorded name, kind, status, attributes,
events, scope, resource, ids, parent link and times, so it can be fed to any
span exporter or processor exactly as if the instrumentor had just produced it.

Python consumers (the expected-result tool here, the Python SDK's conformance
test) use this module directly. Other languages decode the same OTLP/JSON with
their own OTLP types.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

_KINDS = {f"SPAN_KIND_{kind.name}": kind for kind in SpanKind}
_STATUS = {
    "STATUS_CODE_UNSET": StatusCode.UNSET,
    "STATUS_CODE_OK": StatusCode.OK,
    "STATUS_CODE_ERROR": StatusCode.ERROR,
}


def any_value(value: dict[str, Any]) -> Any:
    """An OTLP/JSON AnyValue as the Python value an attribute holds."""
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "arrayValue" in value:
        return tuple(any_value(v) for v in value["arrayValue"].get("values", []))
    raise ValueError(f"unsupported attribute value: {value}")


def attributes(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {item["key"]: any_value(item["value"]) for item in items}


def scenario_spans(scenario: dict[str, Any]) -> list[ReadableSpan]:
    """The finished spans of one recorded scenario, in the recorded order."""
    resource = Resource(attributes(scenario.get("resource", {}).get("attributes", [])))
    spans = []
    for entry in scenario["spans"]:
        d = entry["span"]
        trace_id = int(d["traceId"], 16)
        context = SpanContext(
            trace_id, int(d["spanId"], 16), is_remote=False, trace_flags=TraceFlags.SAMPLED
        )
        parent = None
        if d.get("parentSpanId"):
            parent = SpanContext(
                trace_id,
                int(d["parentSpanId"], 16),
                is_remote=False,
                trace_flags=TraceFlags.SAMPLED,
            )
        status = d.get("status", {})
        code = _STATUS[status.get("code", "STATUS_CODE_UNSET")]
        # OpenTelemetry keeps a description only on ERROR, and warns otherwise.
        description = status.get("message") if code is StatusCode.ERROR else None
        spans.append(
            ReadableSpan(
                name=d["name"],
                context=context,
                parent=parent,
                resource=resource,
                attributes=attributes(d.get("attributes", [])),
                events=[
                    Event(e["name"], attributes(e.get("attributes", [])), int(e["timeUnixNano"]))
                    for e in d.get("events", [])
                ],
                kind=_KINDS[d.get("kind", "SPAN_KIND_INTERNAL")],
                status=Status(code, description),
                start_time=int(d["startTimeUnixNano"]),
                end_time=int(d["endTimeUnixNano"]),
                instrumentation_scope=InstrumentationScope(
                    entry["scope"]["name"], entry["scope"].get("version") or None
                ),
            )
        )
    return spans
