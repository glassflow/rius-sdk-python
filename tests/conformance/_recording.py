# Vendored from rius-semconv 8436c523f40e67af0caf1c25bce5e8d24f1f3667 fixtures/_recording.py by
# scripts/sync-semconv-fixtures.sh. Do not edit: change it in rius-semconv.
"""Shared helpers for the recorders under fixtures/<family>/record.py.

A recorder drives a real instrumentor against a scripted provider in-process
(no network, no API keys), captures the finished spans with an in-memory
exporter, and hands each scenario's spans to `scenario_json`. That turns them
into the OTLP/JSON form (the protobuf JSON mapping of
opentelemetry.proto.trace.v1.Span) every consumer can parse with its own OTLP
types, made deterministic so a re-recording in the pinned environment is
byte-identical:

- ids: each distinct trace id becomes 1, 2, ... and each span id 1, 2, ...,
  numbered in start order, so parent links survive;
- times: every timestamp in the scenario (span starts and ends, event times)
  is replaced by its rank in the scenario's timeline, one millisecond apart
  from a fixed base, so ordering and nesting survive and the first-token
  event still lands after its span's start;
- an exception event's stack trace, which names machine paths, is elided;
- a status message is cut where a Python traceback starts, for the same
  reason (LangChain appends the formatted traceback to it);
- every UUID in a string value (frameworks write per-run ids into JSON
  payloads) becomes a placeholder numbered by first appearance.

This module is imported by recorders running in their own pinned virtualenv,
so it depends only on opentelemetry-sdk, opentelemetry-exporter-otlp-proto-common
and protobuf, which every such environment already holds.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan

#: 2023-11-14T22:13:20Z. Any fixed value works; this one is easy to spot.
BASE_TIME_NS = 1_700_000_000_000_000_000
TICK_NS = 1_000_000

STACKTRACE = "exception.stacktrace"
ELIDED = "<stacktrace elided>"
#: Python's traceback header. Some instrumentors (LangChain) append the whole
#: formatted traceback to the status message.
TRACEBACK = "Traceback (most recent call last):"


class ScriptedProvider:
    """A mock-transport handler: canned responses per URL, every request answered locally.

    Each scripted response answers exactly one request, in order, so an agent
    that calls the same endpoint twice (a tool call, then the final answer) is
    scripted with two responses. Takes the httpx module the client under test
    uses: openai >= 3 and anthropic >= 1 moved to the httpx2 fork, other
    clients still use httpx.
    """

    def __init__(self, httpx: Any) -> None:
        self.httpx = httpx
        self.responses: dict[str, list[Any]] = {}
        self.called: set[str] = set()

    def answer(self, url: str, *responses: Any) -> None:
        self.responses.setdefault(url, []).extend(responses)

    def reset(self) -> None:
        self.responses.clear()
        self.called.clear()

    def all_called(self) -> bool:
        """Every scripted response was used: none is left over."""
        return all(not queue for queue in self.responses.values())

    def transport(self) -> Any:
        return self.httpx.MockTransport(self)

    def intercept_all(self) -> None:
        """Answer every request any client in this process sends, from the script.

        For libraries that build their HTTP clients internally (LiteLLM keeps
        a module-level client per provider) and offer no clean injection point:
        passing `client=` would also end up in the span, serialized into the
        request-parameters bag. Replaces the real transport, so nothing can
        reach the network either.
        """
        self.httpx.HTTPTransport.handle_request = lambda _transport, request: self(request)

    def __call__(self, request: Any) -> Any:
        url = str(request.url).split("?")[0]
        if not self.responses.get(url):
            raise AssertionError(f"unscripted request: {url}")
        self.called.add(url)
        canned = self.responses[url].pop(0)
        return self.httpx.Response(
            canned.status_code, headers=canned.headers, content=canned.content
        )


def sse(events: Sequence[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    """A server-sent-events body. Each item is (event name or None, data)."""
    out = []
    for name, data in events:
        if name is not None:
            out.append(f"event: {name}\n")
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append(f"data: {payload}\n\n")
    return "".join(out).encode()


def spans_from_otlp(captured: Sequence[tuple[Any, Any, Any]]) -> list[ReadableSpan]:
    """Finished spans from OTLP protobuf captured off the wire.

    For recorders that cannot hold an in-memory exporter (the instrumented app
    runs as a child process and exports over OTLP/HTTP, as Laminar's recorder
    does): each captured (Resource, InstrumentationScope, Span) protobuf triple
    becomes a ReadableSpan with its own ids and times, through the same decoder
    every consumer uses (_replay.py), so `scenario_json` can make it canonical.
    All spans of a scenario must share one resource.
    """
    from _replay import scenario_spans

    resources = {r.SerializeToString(deterministic=True): r for r, _, _ in captured}
    if len(resources) > 1:
        raise ValueError(f"expected one resource, got {len(resources)}")
    resource = MessageToDict(next(iter(resources.values()))) if resources else {}
    spans = []
    for _, scope, span in captured:
        d = MessageToDict(span)
        # MessageToDict writes bytes as base64; the OTLP/JSON form is hex.
        d["traceId"] = span.trace_id.hex()
        d["spanId"] = span.span_id.hex()
        if span.parent_span_id:
            d["parentSpanId"] = span.parent_span_id.hex()
        else:
            d.pop("parentSpanId", None)
        spans.append({"scope": {"name": scope.name, "version": scope.version}, "span": d})
    return scenario_spans({"scenario": "", "resource": resource, "spans": spans})


def recorder_resource() -> Resource:
    """The resource every recorder's provider uses.

    service.instance.id is pinned: the OpenTelemetry SDK (1.45 and later) sets
    a random UUID otherwise, and the recording would differ on every run.
    """
    return Resource.create({"service.name": "rius-recorder", "service.instance.id": "recorder"})


def _hex(value: int, width: int) -> str:
    return format(value, f"0{width}x")


def _sid(span: ReadableSpan) -> int:
    return span.get_span_context().span_id


def _depths(spans: Sequence[ReadableSpan]) -> dict[int, int]:
    """Each span's depth below the scenario's roots (0 for a root)."""
    parent_of = {_sid(s): (s.parent.span_id if s.parent is not None else None) for s in spans}
    depth: dict[int, int] = {}

    def of(span_id: int) -> int:
        if span_id not in depth:
            parent = parent_of.get(span_id)
            depth[span_id] = 0 if parent not in parent_of else of(parent) + 1
        return depth[span_id]

    for span_id in parent_of:
        of(span_id)
    return depth


def scenario_json(
    name: str, spans: Sequence[ReadableSpan], *, canonicalize: bool = True
) -> dict[str, Any]:
    """One scenario: its resource and its spans, in start order.

    With `canonicalize` (recording), ids and times are renumbered as the module
    docstring describes. Without it (writing an expected result from replayed
    spans), the spans' own ids and times are kept as they are: they are already
    canonical, and renumbering would shift every later time whenever
    normalization drops an event.
    """
    if not spans:
        return {"scenario": name, "resource": {"attributes": []}, "spans": []}
    depth = _depths(spans)
    # Start order; on a tie a parent comes before its child, then the order the
    # spans finished in. Instrumentors that stamp times at millisecond
    # resolution (LangChain's run records) tie often, and a tie must not decide
    # the numbering by chance.
    ordered = sorted(
        enumerate(spans),
        key=lambda pair: (pair[1].start_time or 0, depth[_sid(pair[1])], pair[0]),
    )
    ordered_spans = [span for _, span in ordered]

    trace_ids: dict[int, str] = {}
    span_ids: dict[int, str] = {}
    for span in ordered_spans:
        context = span.get_span_context()
        if canonicalize:
            trace_ids.setdefault(context.trace_id, _hex(len(trace_ids) + 1, 32))
            span_ids.setdefault(context.span_id, _hex(len(span_ids) + 1, 16))
        else:
            trace_ids[context.trace_id] = _hex(context.trace_id, 32)
            span_ids[context.span_id] = _hex(context.span_id, 16)

    # Every start, event and end is its own point on the timeline, so a tie in
    # the real clock never merges two points. Ties are broken by structure: a
    # start before an event before an end; among starts the parent first,
    # among ends the child first; then span order. A span's start therefore
    # always precedes its own end, even when the clock saw no time pass.
    points: list[tuple[tuple, tuple]] = []
    for i, span in enumerate(ordered_spans):
        d = depth[_sid(span)]
        points.append(((span.start_time or 0, 0, d, i, 0), ("start", i)))
        for k, event in enumerate(span.events):
            points.append(((event.timestamp, 1, d, i, k), ("event", i, k)))
        points.append(((span.end_time or 0, 2, -d, i, 0), ("end", i)))
    if canonicalize:
        rank = {
            point: str(BASE_TIME_NS + n * TICK_NS)
            for n, (_, point) in enumerate(sorted(points, key=lambda p: p[0]))
        }
    else:
        rank = {point: str(key[0]) for key, point in points}

    request = encode_spans(ordered_spans)
    resource_spans = request.resource_spans
    resource = MessageToDict(resource_spans[0].resource) if resource_spans else {}
    resource["attributes"] = sorted(resource.get("attributes", []), key=lambda a: a["key"])

    encoded: dict[int, tuple[dict, dict]] = {}
    for rs in resource_spans:
        for ss in rs.scope_spans:
            scope = {"name": ss.scope.name, "version": ss.scope.version}
            for pb in ss.spans:
                span_id = int.from_bytes(pb.span_id, "big")
                encoded[span_id] = (scope, MessageToDict(pb))

    out = []
    for i, span in enumerate(ordered_spans):
        context = span.get_span_context()
        scope, d = encoded[context.span_id]
        d["traceId"] = trace_ids[context.trace_id]
        d["spanId"] = span_ids[context.span_id]
        parent = span.parent
        if parent is not None and parent.span_id in span_ids:
            d["parentSpanId"] = span_ids[parent.span_id]
        else:
            d.pop("parentSpanId", None)
        d.pop("flags", None)
        d["startTimeUnixNano"] = rank[("start", i)]
        d["endTimeUnixNano"] = rank[("end", i)]
        d["attributes"] = sorted(d.get("attributes", []), key=lambda a: a["key"])
        status = d.get("status", {})
        if TRACEBACK in status.get("message", ""):
            status["message"] = status["message"].split(TRACEBACK, 1)[0].rstrip()
        for k, event in enumerate(d.get("events", [])):
            event["timeUnixNano"] = rank[("event", i, k)]
            for attr in event.get("attributes", []):
                if attr["key"] == STACKTRACE:
                    attr["value"] = {"stringValue": ELIDED}
            event["attributes"] = sorted(event.get("attributes", []), key=lambda a: a["key"])
        out.append({"scope": scope, "span": d})
    scenario = {"scenario": name, "resource": resource, "spans": out}
    return _scrub_uuids(scenario) if canonicalize else scenario


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _scrub_uuids(scenario: dict[str, Any]) -> dict[str, Any]:
    """Replace every UUID in the scenario's string values with a placeholder.

    Frameworks mint ids per run and write them into attribute values, often
    inside a JSON string (LangChain's lc_run--<uuid>). Each distinct UUID
    becomes 00000000-0000-0000-0000-<n>, numbered by first appearance, so a
    repeated id stays recognisably the same id.
    """
    seen: dict[str, str] = {}

    def placeholder(match: re.Match[str]) -> str:
        found = match.group(0).lower()
        if found not in seen:
            seen[found] = f"00000000-0000-0000-0000-{len(seen) + 1:012d}"
        return seen[found]

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            return _UUID.sub(placeholder, value)
        return value

    return walk(scenario)


def write_recording(path: Path, recordings: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        json.dump({"recordings": recordings}, f, indent=2, ensure_ascii=False)
        f.write("\n")


def record_all(
    scenarios: dict[str, Any],
    exporter: Any,
    script: ScriptedProvider,
    packages: Sequence[str],
    argv: Sequence[str],
) -> None:
    """Run every scenario, write the recording, and report the environment.

    `argv` is `[<recording.json>, <environment.json>?]`. Each scenario starts
    from an empty exporter and an empty script, and must use every response it
    scripted, so a scenario that silently stopped calling the provider fails.
    """
    recordings = []
    for name, run in scenarios.items():
        exporter.clear()
        script.reset()
        run()
        leftover = {url: len(queue) for url, queue in script.responses.items() if queue}
        assert script.all_called(), f"{name}: scripted responses never requested: {leftover}"
        recordings.append(scenario_json(name, exporter.get_finished_spans()))
    write_recording(Path(argv[0]), recordings)
    env = json.dumps(environment(packages), indent=2)
    if len(argv) > 1:
        Path(argv[1]).write_text(env + "\n")
    else:
        print(env)


def environment(packages: Sequence[str]) -> dict[str, Any]:
    """The pins a recording was made with, for its manifest entry."""
    return {
        "python": sys.version.split()[0],
        "packages": {package: version(package) for package in packages},
    }
