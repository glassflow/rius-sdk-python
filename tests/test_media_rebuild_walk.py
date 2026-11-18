"""The media rebuild's walk over a messages value: its cost on large values,
and its output against a frozen copy of the walk it replaced.

The walk finds the part spans of a messages value without decoding the
strings it only steps over, so a value of many MB costs time and memory in
proportion to its size. It must give exactly what the decoding walk gave, for
every input: the rebuilt value is byte-identical across SDKs.
"""

from __future__ import annotations

import functools
import json
import random
import time
import timeit
import tracemalloc
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

from rius import _media
from rius._media import MESSAGE_MAX_DEPTH, _NotJson, _Open, _Parser, _part_edit, _role
from rius.normalization import NormalizingSpanExporter
from rius.semconv import GEN_AI_INPUT_MESSAGES

# --- frozen copy ---------------------------------------------------------------


def plain_key(scan: _Parser) -> str:
    scan.space()
    key = scan.string().value
    scan.space()
    scan.expect(":")
    scan.space()
    return str(key)


def plain_part_spans(text: str, parts_only: bool) -> list[tuple[int, int]]:
    scan = _Parser(text, 0, MESSAGE_MAX_DEPTH)
    stack: list[_Open] = []
    try:
        scan.space()
        while True:
            role = _role(stack, parts_only)
            start = scan.i
            ch = scan.peek()
            node: _Open | None = None
            if ch == "{" or ch == "[":
                node = _Open(ch == "{", role, start, spans=[])
                close = "}" if node.is_object else "]"
                scan.i += 1
                scan.enter()
                scan.space()
                if scan.peek() != close:
                    if node.is_object:
                        node.key = plain_key(scan)
                    stack.append(node)
                    continue
                scan.i += 1
                scan.depth -= 1
            else:
                scan.value()
            while True:
                if not stack:
                    scan.space()
                    if scan.i != len(text):
                        return []
                    return node.spans or [] if node is not None and node.role == "root" else []
                parent = stack[-1]
                if role == "part" and node is not None and node.is_object:
                    assert parent.spans is not None
                    parent.spans.append((start, scan.i))
                elif role == "parts":
                    parent.spans = node.spans if node is not None and not node.is_object else []
                elif role == "message" and node is not None and node.is_object:
                    assert parent.spans is not None and node.spans is not None
                    parent.spans.extend(node.spans)
                scan.space()
                if scan.peek() == ",":
                    scan.i += 1
                    scan.space()
                    if parent.is_object:
                        parent.key = plain_key(scan)
                    break
                scan.expect("}" if parent.is_object else "]")
                scan.depth -= 1
                node = stack.pop()
                role, start = node.role, node.start
    except _NotJson:
        return []


def plain_rebuild_native_parts(text: str, *, parts_only: bool) -> str:
    if '"blob"' not in text and '"uri"' not in text and '"file"' not in text and "\\" not in text:
        return text
    out, last = [], 0
    for start, end in plain_part_spans(text, parts_only):
        encoded = _part_edit(text[start:end])
        if encoded is not None:
            out += [text[last:start], encoded]
            last = end
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


# --- generated inputs ----------------------------------------------------------

_WS = ("", "", " ", "\n", "\t", "\r\n ")
_STRINGS = (
    "",
    "text",
    "ab\n",
    'q"uote',
    "back\\slash",
    "\x01ctl",
    "é",
    "😀",
    "\ud800",
    "\udfff",
    "😀",
    "/",
    "blob",
    "uri",
    "file",
    "parts",
    "image/png",
    "data:image/png;base64,iVBORw0KGgo=",
    "https://x.test/a.png",
)
_KEYS = (
    "type",
    "parts",
    "role",
    "content",
    "mime_type",
    "mimeType",
    "modality",
    "uri",
    "file_id",
    "byte_size",
    "base64",
    "url",
    "stripped_reason",
    "x",
)
_TYPES = ("text", "blob", "uri", "file", "image", "tool_call", "")


def _string(rng: random.Random) -> str:
    if rng.random() < 0.3:
        return "".join(rng.choice(_STRINGS) for _ in range(rng.randint(1, 4)))
    return rng.choice(_STRINGS)


def _part(rng: random.Random, depth: int) -> Any:
    kind = rng.choice(_TYPES)
    part: dict[str, Any] = {"type": kind}
    if kind == "blob":
        part["mime_type"] = rng.choice(("image/png", "audio/wav", "x", None, 3))
        part["content"] = rng.choice(("iVBORw0KGgo=", "", None, "data:image/png;base64,AAAA"))
        if rng.random() < 0.3:
            part["modality"] = rng.choice(("image", "audio", "video"))
    elif kind == "uri":
        part["uri"] = rng.choice(("https://x.test/a.png", "data:image/png;base64,AAAA", 7))
        part["mime_type"] = rng.choice(("image/png", "x"))
    elif kind == "file":
        if rng.random() < 0.5:
            part["file_id"] = "file-1"
        else:
            part[rng.choice(("base64", "url"))] = "AAAA"
            part["mime_type"] = "application/pdf"
    for _ in range(rng.randint(0, 3)):
        part[rng.choice(_KEYS)] = _leaf(rng, depth + 1)
    return part


def _leaf(rng: random.Random, depth: int) -> Any:
    r = rng.random()
    if r < 0.45 or depth > 4:
        return rng.choice((_string(rng), 0, -1.5e3, True, False, None))
    if r < 0.7:
        return [_leaf(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {rng.choice(_KEYS): _leaf(rng, depth + 1) for _ in range(rng.randint(0, 3))}


def _messages(rng: random.Random, parts_only: bool) -> Any:
    if parts_only:
        return [_part(rng, 0) if rng.random() < 0.85 else _leaf(rng, 1) for _ in range(3)]
    messages: list[Any] = []
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.1:
            messages.append(_leaf(rng, 1))
            continue
        message: dict[str, Any] = {"role": rng.choice(("user", "assistant"))}
        message["parts"] = [_part(rng, 2) for _ in range(rng.randint(0, 3))]
        if rng.random() < 0.15:
            message["x"] = _leaf(rng, 1)
        messages.append(message)
    return messages


def _spaced(text: str, rng: random.Random) -> str:
    """The JSON text with space put around its structural characters."""
    out: list[str] = []
    in_string = escaped = False
    for ch in text:
        out.append(ch)
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{}[],:" and rng.random() < 0.3:
            out.append(rng.choice(_WS))
    return "".join(out)


_ESCAPED_SPELLINGS = (
    ('"parts"', '"p\\u0061rts"'),
    ('"type"', '"t\\u0079pe"'),
    ('"blob"', '"bl\\u006fb"'),
    ('"uri"', '"\\u0075ri"'),
    ('"file"', '"fil\\u0065"'),
    ('"mime_type"', '"mime\\u005ftype"'),
    ('"role"', '"r\\u006fle"'),
)
_BREAKAGE = (
    ('"text"', '"te\\xt"'),
    ('"text"', '"te\x01xt"'),
    ('"text"', '"te\\u12"'),
    ('"text"', '"te\\u12G4"'),
    ('"text"', '"\\ud800\\u12"'),
    ('"text"', '"\\ud800\\uZZZZ"'),
    ("}", "}}"),
    (",", ",,"),
    (":", "::"),
    ("null", "nul"),
    ("[", "[["),
    ('"', ""),
)


def _text(rng: random.Random, parts_only: bool) -> str:
    value = _messages(rng, parts_only)
    text = json.dumps(value, ensure_ascii=rng.random() < 0.3)
    if rng.random() < 0.5:
        text = _spaced(text, rng)
    if rng.random() < 0.3:
        old, new = rng.choice(_ESCAPED_SPELLINGS)
        text = text.replace(old, new, rng.choice((1, -1)))
    if rng.random() < 0.15:
        old, new = rng.choice(_BREAKAGE)
        text = text.replace(old, new, 1)
    if rng.random() < 0.05 and text:
        text = text[: rng.randrange(len(text))]
    if rng.random() < 0.05:
        text = rng.choice(_WS) + text + rng.choice((" ", "x", "\n"))
    return text


@pytest.mark.parametrize("parts_only", [False, True])
def test_the_part_spans_match_the_decoding_walk(parts_only: bool) -> None:
    rng = random.Random(1888 + parts_only)
    found = 0
    for _ in range(6000):
        text = _text(rng, parts_only)
        spans = _media._part_spans(text, parts_only)
        assert spans == plain_part_spans(text, parts_only), text
        found += bool(spans)
    # The generator reaches values with parts, not only broken ones.
    assert found > 1000


@pytest.mark.parametrize("parts_only", [False, True])
def test_the_rebuild_matches_the_decoding_walk(parts_only: bool) -> None:
    rng = random.Random(8881 + parts_only)
    changed = 0
    for _ in range(6000):
        text = _text(rng, parts_only)
        rebuilt = _media.rebuild_native_parts(text, parts_only=parts_only)
        assert rebuilt == plain_rebuild_native_parts(text, parts_only=parts_only), text
        changed += rebuilt != text
    assert changed > 500


def test_the_walk_still_stops_at_its_depth() -> None:
    deep = "[" * MESSAGE_MAX_DEPTH + "]" * MESSAGE_MAX_DEPTH
    assert _media._part_spans(deep, False) == plain_part_spans(deep, False) == []
    too_deep = "[" + deep + "]"
    assert _media._part_spans(too_deep, False) == []


# --- cost on large values ------------------------------------------------------


@functools.cache
def _one_large_part() -> list[dict[str, Any]]:
    """30 MB of escape-heavy text in one part: each "\\n" is an escape in the
    JSON text. The decoding walk took several seconds and GBs of memory on it.
    Built on first use, so runs that deselect the cost tests do not pay for it."""
    return [{"role": "user", "parts": [{"type": "text", "content": "ab\n" * 10_000_000}]}]


@functools.cache
def _many_parts() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "ab\n" * 30} for _ in range(100_000)],
        }
    ]


def _export(value: Any) -> dict[str, Any]:
    context = SpanContext(trace_id=1, span_id=2, is_remote=False)
    span = ReadableSpan(name="chat", context=context, attributes={GEN_AI_INPUT_MESSAGES: value})
    sink = InMemorySpanExporter()
    NormalizingSpanExporter(sink).export([span])
    [out] = sink.get_finished_spans()
    return dict(out.attributes or {})


def _cost(value: Any) -> tuple[float, int]:
    """Seconds to export ``value`` (untraced), and the peak of the memory the
    export allocated (traced, so not timed)."""
    _export([{"role": "user", "parts": [{"type": "text", "content": "ab\n"}]}])  # warm imports
    started = time.perf_counter()
    _export(value)
    seconds = time.perf_counter() - started
    tracemalloc.start()
    try:
        _export(value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return seconds, peak


@pytest.mark.parametrize("form", ["json-string", "structured"])
def test_a_large_escaped_value_exports_in_linear_time_and_memory(form: str) -> None:
    value = _one_large_part()
    attribute = json.dumps(value) if form == "json-string" else value
    seconds, peak = _cost(attribute)
    # Bounds well clear of the linear walk (under a second, under three times
    # the text) and of the decoding one (several seconds, about sixty times
    # the text), so a slow CI runner does not flake.
    assert seconds < 3.0
    assert peak < 6 * len(json.dumps(value))


def test_the_walk_over_many_escaped_parts_is_faster_than_the_decoding_walk() -> None:
    # Against the frozen copy on the same machine, so a slow runner slows
    # both: the walk measured about five times faster. The best of several
    # runs of each, against a 2x bound, so one GC pause or preemption on a
    # shared runner does not flake it.
    text = json.dumps(_many_parts())

    def seconds(walk: Any) -> float:
        return min(timeit.repeat(lambda: walk(text, False), number=1, repeat=5))

    assert seconds(_media._part_spans) * 2 < seconds(plain_part_spans)
