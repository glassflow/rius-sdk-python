"""Value serialization for span attributes: JSON with a safe fallback and a bound.

The bound is 32 KB of serialized text, shared with the TypeScript SDK so a
payload lands the same size whichever SDK produced it. Media is stripped from
the text BEFORE it is cut (``_media.strip_raw_payload``): an image's base64 is
replaced in place by a short marker, so a request holding one stays whole JSON
instead of being cut inside the bytes, and no prefix of the bytes is kept.

The strip has to see the whole document, since a cut one does not parse and
only its data URLs could be found. So the work is bounded by a larger budget,
``MAX_STRIP_CHARS``, rather than by the cap: encoding stops there instead of
rendering an unbounded payload (the 2026-09 review measured rendering whole
and cutting at 200x the cost on large tool results), and the strip's own cost
on the caller's thread stays bounded. A payload past the budget is cut first:
the cut text no longer parses, so only its data URLs are found, and a keyed
image's base64 would survive. A conservative fallback replaces every run of
the base64 alphabet long enough to be media. That guard is this SDK's own: past
the budget the text is not byte-identical with the TypeScript SDK's.

A byte object (bytes, bytearray, memoryview) never leaves with its content:
the contract (``raw_payloads.byte_objects``) writes it as the replacement with
the unknown mime, sized in bytes, wherever it sits in the value.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from ._media import MAX_DEPTH, replacement, strip_raw_payload

MAX_ATTR_CHARS = 32 * 1024
TRUNCATION_MARKER = "…(truncated)"
#: How much of a value is rendered for the media strip to read it whole. Past
#: it the value is cut and stripped conservatively (see ``_cut_past_budget``).
MAX_STRIP_CHARS = 1024 * 1024
#: A run of the base64 alphabet (either variant, with padding): what the
#: fallback past the budget replaces once it is long enough to be media (see
#: ``_MIN_MEDIA_RUN``). A JSON text may write a character of the alphabet as an
#: escape (PHP writes "/" as "\/", others "+" as "\u002b"), so such an escape
#: is one character of the run. Each match is a whole run, its length checked
#: after, so every character is read once.
_BASE64_RUN = re.compile(
    r"(?:[A-Za-z0-9+/\-_=]|\\/"
    r"|\\u00(?:2[bdfBDF]|3[0-9dD]|4[1-9a-fA-F]|5[0-9aAfF]|6[1-9a-fA-F]|7[0-9aA]))+"
)
#: The shortest run the fallback takes for media rather than a word.
_MIN_MEDIA_RUN = 64

#: The byte objects: written as the replacement, never as their content.
BYTE_OBJECTS = (bytes, bytearray, memoryview)


#: What a failed read of JSON text returns, so a JSON ``null`` stays a value.
UNREADABLE: Any = object()


def loads_json(raw: str) -> Any:
    """``json.loads(raw)``, or ``UNREADABLE`` when the text cannot be read.

    ``json.loads`` raises ``ValueError`` on malformed text and
    ``RecursionError`` on text nested too deep for the decoder. How deep that
    is depends on the interpreter (the recursion limit up to 3.11, the C
    stack's limit from 3.12), so a deep text that does parse is read as any
    other.
    Every reader of a value some other code produced goes through here, so
    neither escapes into the caller's code or costs an export batch.
    """
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        return UNREADABLE


def loads_object(raw: str) -> dict[str, Any] | None:
    """``raw`` read as a JSON object; None when it is unreadable or not an object."""
    payload = loads_json(raw)
    return payload if isinstance(payload, dict) else None


def byte_object_size(value: bytes | bytearray | memoryview) -> int | None:
    """A byte object's size in bytes, not items; None for a released
    memoryview, whose size is unknown."""
    if isinstance(value, memoryview):
        try:
            return value.nbytes
        except ValueError:
            return None
    return len(value)


def byte_object_marker(value: bytes | bytearray | memoryview) -> str:
    """What a byte object is written as: the replacement, its mime unknown."""
    return replacement(None, byte_object_size(value))


#: A bytes literal in a repr, ``bytearray(...)`` around it or not, that does
#: not end a word. A repr is the encoder's fallback for any other object, and
#: writes a bytes field in it as its content.
_BYTES_REPR = re.compile(
    r"(?<![A-Za-z0-9_])(bytearray\()?b('(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\")(?(1)\))"
)


def _byte_literal_marker(match: re.Match[str]) -> str:
    try:
        size: int | None = len(ast.literal_eval("b" + match.group(2)))
    except (ValueError, SyntaxError):
        size = None
    return replacement(None, size)


def safe_repr(value: Any) -> str:
    """``repr(value)`` with every bytes literal in it written as the marker."""
    return _BYTES_REPR.sub(_byte_literal_marker, repr(value))


def _default(value: Any) -> str:
    """The encoder's fallback: a byte object is the marker, any other value
    its repr, with no byte object's content in it."""
    if isinstance(value, BYTE_OBJECTS):
        return byte_object_marker(value)
    return safe_repr(value)


# One JSON encoding for every producer of a JSON-valued attribute (this SDK,
# the TypeScript SDK and the sink): compact separators, raw UTF-8 rather than
# \uXXXX escapes, and no HTML escaping of < > & (Python never does that).
# It is what TypeScript's JSON.stringify writes, so the same value produces the
# same bytes whichever SDK recorded it.
_encoder = json.JSONEncoder(default=_default, separators=(",", ":"), ensure_ascii=False)

# ensure_ascii=False writes a lone surrogate raw, and a string holding one
# cannot be encoded as UTF-8: the OTLP exporter drops such an attribute. The
# escaping ensure_ascii used to do for it is kept here, which is also what
# JSON.stringify does with a lone surrogate. ``json.loads`` pairs the valid
# surrogates into one code point, so any left in a decoded ``str`` are unpaired.
LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _escape_surrogate(match: re.Match[str]) -> str:
    return f"\\u{ord(match.group()):04x}"


def truncate(text: str) -> str:
    """``text`` with its media stripped, cut at the attribute cap, the cut marked."""
    if len(text) > MAX_STRIP_CHARS:
        return _cut_past_budget(text)
    return _cut(_strip_media(text))


def strip_within_budget(text: str) -> str:
    """``text`` with its media stripped, the work bounded by the strip budget.

    Under the budget it is the contract's strip and nothing else, so a value
    another instrumentation recorded keeps its length. Past it, the value is
    cut and stripped as ``truncate`` does (see ``_cut_past_budget``).
    """
    if len(text) > MAX_STRIP_CHARS:
        return _cut_past_budget(text)
    return _strip_media(text)


def _strip_media(text: str) -> str:
    """The contract's strip, plus the base64-run fallback for a text nested
    deeper than its parser reads.

    The parser gives up past ``MAX_DEPTH`` arrays and objects, so such a text
    is not JSON to it and only its data URLs are found: a keyed image's base64
    would survive. The fallback is ``_cut_past_budget``'s, and the same
    SDK-local guard: only a text that deep is not byte-identical with the
    TypeScript SDK's.
    """
    stripped = strip_raw_payload(text)
    if _nests_past_parser_depth(stripped):
        return _BASE64_RUN.sub(_replace_media_run, stripped)
    return stripped


#: A JSON string, or one bracket outside any.
_STRING_OR_BRACKET = re.compile(r'"(?:[^"\\]|\\.)*"|[\[\]{}]', re.DOTALL)


def _nests_past_parser_depth(text: str) -> bool:
    """True when the parser may read past ``MAX_DEPTH`` somewhere in ``text``.

    An upper bound, not the parser's exact depth: the parser re-parses a
    JSON-looking string at the depth it sits at, so every opening bracket in a
    string (at any level of escaping) counts on top of that depth, and its
    closing brackets, which may close nothing, are not counted. A false
    positive only applies the run fallback.
    """
    if text.count("{") + text.count("[") <= MAX_DEPTH:
        return False  # the common case, without reading the text again
    depth = 0
    for match in _STRING_OR_BRACKET.finditer(text):
        token = match.group()
        if token[0] == '"':
            if depth + token.count("{") + token.count("[") > MAX_DEPTH:
                return True
        elif token in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                return True
        else:
            depth -= 1
    return False


def _cut_past_budget(text: str) -> str:
    """A text past the strip budget: cut to the cap, its data URLs stripped,
    then every remaining base64-alphabet run of 64 or more characters replaced.

    An SDK-local guard, not the contract's strip: the cut text no longer
    parses, so a keyed image's base64 is not found by its rule, and the
    fallback takes it, along with any long word of the same alphabet. The
    result is not byte-identical with the TypeScript SDK's past the budget.
    """
    stripped = strip_raw_payload(text[:MAX_ATTR_CHARS])
    return _BASE64_RUN.sub(_replace_media_run, stripped) + TRUNCATION_MARKER


def _replace_media_run(match: re.Match[str]) -> str:
    run = match.group()
    # A backslash starts an escape: "\/" writes one character, "\u00XX" one.
    length = len(run) - run.count("\\/") - 5 * run.count("\\u")
    return replacement(None, None) if length >= _MIN_MEDIA_RUN else run


def _cut(text: str) -> str:
    if len(text) > MAX_ATTR_CHARS:
        return text[:MAX_ATTR_CHARS] + TRUNCATION_MARKER
    return text


def text_value(value: Any) -> str:
    """A span's input or output as text: a string as-is, anything else serialized.

    A string is already the text a reader wants to see; JSON-encoding it would
    store ``"\\"Paris\\""`` and show the quotes. This is the rule the
    TypeScript SDK's ``toAttributeValue`` applies, so the same value lands the
    same way from either SDK. A lone surrogate becomes U+FFFD, the character
    a JavaScript string's UTF-8 encoding writes for it, since a raw one would
    get the attribute dropped by the OTLP exporter.
    """
    if isinstance(value, str):
        return truncate(LONE_SURROGATE.sub("\ufffd", value))
    return serialize(value)


def attribute_value(value: Any, text: Callable[[str], str] | None = None) -> Any:
    """A value set directly as an attribute, every byte object in it, at any
    depth of arrays and mappings, the bare marker (the contract's
    byte_objects.attribute_values); each string through ``text`` when given.
    The same object when nothing changes.

    An array holding a byte object beside an element that is then no string
    could not stay homogeneous: it is the array's JSON instead, one string,
    as any value this SDK serializes. Run before OpenTelemetry sees the
    value, which would hold a bytearray or memoryview as a tuple of ints.
    """
    if isinstance(value, str):
        return value if text is None else text(value)
    if isinstance(value, BYTE_OBJECTS):
        return byte_object_marker(value)
    if isinstance(value, (list, tuple)):
        items = [attribute_value(v, text) for v in value]
        if any(isinstance(v, BYTE_OBJECTS) for v in value) and not all(
            isinstance(v, str) for v in items
        ):
            return serialize(list(value))
        if all(new is old for new, old in zip(items, value, strict=True)):
            return value
        # A plain tuple or list, whatever the subclass: its constructor may
        # take other arguments, and OpenTelemetry holds either the same.
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, Mapping):
        members = {k: attribute_value(v, text) for k, v in value.items()}
        if all(members[k] is v for k, v in value.items()):
            return value
        return members
    return value


def serialize(value: Any) -> str:
    """Serialize a value to a bounded string suitable for a span attribute."""
    try:
        return _bounded_dumps(value)
    except Exception:
        # The default's repr can itself raise (broken __repr__, detached ORM
        # proxies); tracing must never crash the host over an unserializable
        # value. A key the encoder refuses (bytes, a tuple) lands here too.
        try:
            return truncate(safe_repr(value))
        except Exception:
            return f"<unserializable {type(value).__name__}>"


def _bounded_dumps(value: Any) -> str:
    # iterencode without _one_shot takes the pure-Python path, which yields
    # chunks lazily as it walks the value; the C encoder renders everything
    # first. A small payload costs a few extra microseconds this way; a large
    # one stops being walked at the strip budget instead of being rendered
    # whole.
    #
    # A chunk is escaped only when it fits the budget, so the escaping reads
    # at most the budget. One that does not fit is past it escaped or not (an
    # escape only lengthens it): the walk stops at the same chunk, and only
    # the prefix that is kept is escaped. Escaping is per character and
    # leaves no surrogate, so escaping the kept prefix of the mixed text is
    # the same prefix of the text escaped whole.
    chunks: list[str] = []
    size = 0
    for chunk in _encoder.iterencode(value, _one_shot=False):
        if size + len(chunk) <= MAX_STRIP_CHARS:
            chunk = LONE_SURROGATE.sub(_escape_surrogate, chunk)
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_STRIP_CHARS:
            kept = "".join(chunks)[:MAX_ATTR_CHARS]
            return _cut_past_budget(LONE_SURROGATE.sub(_escape_surrogate, kept))
    return truncate("".join(chunks))
