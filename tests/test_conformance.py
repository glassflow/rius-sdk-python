"""Every shared recording, normalized by this SDK, must equal its expected result.

The recordings and their expected results come from rius-semconv (synced by
scripts/sync-semconv-fixtures.sh into tests/fixtures/semconv). Each recording is
real instrumentor output; its expected result is what the reference normalizer
produced from it. The Python SDK is that reference for every family it covers,
so for those this test is the guard that a normalizer change is made on purpose:
the expected result must be regenerated in rius-semconv in the same change, and
the TypeScript SDK and the sink then compare against the new one.

Families whose reference is another normalizer (the Vercel AI SDK, which only
the TypeScript SDK covers) are replayed too, so their drift stays visible. Every
difference must be reviewed in tests/conformance/known_differences.json; one that
is not listed fails, and so does a listed one that no longer occurs. A recorded
scenario with no expected result yet is a difference too (key ``scenario``): the
other normalizer has not produced it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from media_contract import MESSAGE_KEYS, canonical_errors, media_parts_in
from rius.aggregated_usage import WrappedCallTracker
from rius.normalization import NormalizingSpanExporter

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures" / "semconv" / "fixtures"
KNOWN = HERE / "conformance" / "known_differences.json"

sys.path.insert(0, str(HERE / "conformance"))

from _recording import scenario_json  # noqa: E402
from _replay import scenario_spans  # noqa: E402


@dataclass(frozen=True)
class Difference:
    family: str
    scenario: str
    span_id: str
    key: str
    detail: str


def _recordings() -> Iterator[tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]]:
    for manifest_path in sorted(FIXTURES.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for file, entry in sorted(manifest["recordings"].items()):
            expected_file = manifest_path.parent / entry["expected"]["file"]
            if not expected_file.name.endswith(".expected.json"):
                continue  # a sink-only family: its expected result is a stored row
            raw = json.loads((manifest_path.parent / file).read_text())["recordings"]
            expected = json.loads(expected_file.read_text())["recordings"]
            yield manifest["family"], entry["expected"]["source"], raw, expected


def _normalized(scenario: dict[str, Any]) -> dict[str, Any]:
    spans = scenario_spans(scenario)
    # The span processors that decide from end order, driven the way a live
    # provider drives them: every start, then every end in the order the spans
    # finished. A parent is still open when its child ends, as it was live.
    wrapped_calls = WrappedCallTracker()
    for span in sorted(spans, key=lambda s: s.start_time or 0):
        wrapped_calls.on_start(span)  # type: ignore[arg-type]
    for span in sorted(spans, key=lambda s: s.end_time or 0):
        wrapped_calls.on_end(span)
    sink = InMemorySpanExporter()
    NormalizingSpanExporter(sink, wrapped_calls=wrapped_calls).export(spans)
    return scenario_json(scenario["scenario"], sink.get_finished_spans(), canonicalize=False)


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    return {a["key"]: a["value"] for a in span.get("attributes", [])}


def _short(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _differences(family: str, got: dict[str, Any], want: dict[str, Any]) -> list[Difference]:
    out: list[Difference] = []
    name = want["scenario"]
    got_spans = {e["span"]["spanId"]: e["span"] for e in got["spans"]}
    for entry in want["spans"]:
        expected = entry["span"]
        span_id = expected["spanId"]
        actual = got_spans.get(span_id)
        if actual is None:
            out.append(Difference(family, name, span_id, "span", "missing"))
            continue
        a, b = _attributes(actual), _attributes(expected)
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                detail = f"{_short(b.get(key))} expected, {_short(a.get(key))} produced"
                out.append(Difference(family, name, span_id, key, detail))
        for field, key in (("events", "events"), ("status", "status"), ("name", "name")):
            if actual.get(field) != expected.get(field):
                out.append(Difference(family, name, span_id, key, "differs"))
        if actual.get("kind") != expected.get("kind"):
            out.append(Difference(family, name, span_id, "kind", "differs"))
    return out


def _known() -> list[dict[str, str]]:
    known: list[dict[str, str]] = json.loads(KNOWN.read_text())["differences"]
    for entry in known:
        assert entry.get("reason", "").strip(), f"known difference without a reason: {entry}"
    return known


def test_every_recording_normalizes_to_its_expected_result() -> None:
    known = _known()
    used: set[tuple[str, str]] = set()
    unexpected: list[str] = []
    spans = 0
    for family, _source, raw, expected in _recordings():
        by_name = {s["scenario"]: s for s in expected}
        for scenario in raw:
            want = by_name.get(scenario["scenario"])
            if want is None:
                diffs = [Difference(family, scenario["scenario"], "-", "scenario", "not expected")]
            else:
                spans += len(want["spans"])
                diffs = _differences(family, _normalized(scenario), want)
            for diff in diffs:
                matches = [
                    k for k in known if k["family"] == diff.family and k["key"] in ("*", diff.key)
                ]
                for k in matches:
                    used.add((k["family"], k["key"]))
                if not matches:
                    where = f"{diff.family}/{diff.scenario} span {diff.span_id}"
                    unexpected.append(f"{where}: {diff.key}: {diff.detail}")
    assert spans > 0, "no spans compared: is tests/fixtures/semconv synced?"
    stale = [f"{k['family']} {k['key']}" for k in known if (k["family"], k["key"]) not in used]
    assert not unexpected, "unreviewed differences:\n" + "\n".join(unexpected)
    assert not stale, "known differences that no longer occur:\n" + "\n".join(stale)


def test_every_normalized_media_part_is_canonical() -> None:
    """Whatever a recording holds, no media part this SDK writes is malformed:
    a file without its modality and file_id, say, fails here."""
    seen = 0
    failures: list[str] = []
    for family, _source, raw, _expected in _recordings():
        for scenario in raw:
            for entry in _normalized(scenario)["spans"]:
                attributes = _attributes(entry["span"])
                for key in MESSAGE_KEYS:
                    if "stringValue" not in attributes.get(key, {}):
                        continue
                    for part in media_parts_in(json.loads(attributes[key]["stringValue"])):
                        seen += 1
                        if errors := canonical_errors(part):
                            failures.append(f"{family}/{scenario['scenario']} {part}: {errors}")
    assert seen > 0, "no media part normalized: is tests/fixtures/semconv synced?"
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("family", sorted(p.parent.name for p in FIXTURES.glob("*/manifest.json")))
def test_replay_keeps_identity(family: str) -> None:
    """Ids, parents, times, names and kinds come from the recording: the
    normalizer never changes them, so any difference means the replay broke."""
    for fam, _source, raw, expected in _recordings():
        if fam != family:
            continue
        by_name = {s["scenario"]: s for s in expected}
        for scenario in raw:
            if scenario["scenario"] not in by_name:
                continue  # no expected result yet: reviewed as a known difference
            got = {e["span"]["spanId"]: e["span"] for e in _normalized(scenario)["spans"]}
            for entry in by_name[scenario["scenario"]]["spans"]:
                want = entry["span"]
                span = got[want["spanId"]]
                for field in ("traceId", "parentSpanId", "startTimeUnixNano", "endTimeUnixNano"):
                    assert span.get(field) == want.get(field), (family, scenario["scenario"], field)


def test_normalizing_an_expected_result_again_changes_nothing() -> None:
    """Normalized spans are normalized again in production: the sink normalizes
    what this SDK exported. So a second pass over this SDK's own output (the
    expected result, for every family it is the reference for) must be a no-op.
    """
    changed: list[str] = []
    spans = 0
    for family, source, _raw, expected in _recordings():
        if source != "python-sdk":
            continue
        for scenario in expected:
            spans += len(scenario["spans"])
            for diff in _differences(family, _normalized(scenario), scenario):
                where = f"{diff.family}/{diff.scenario} span {diff.span_id}"
                changed.append(f"{where}: {diff.key}: {diff.detail}")
    assert spans > 0, "no spans compared: is tests/fixtures/semconv synced?"
    assert not changed, "a second normalization changed:\n" + "\n".join(changed)
