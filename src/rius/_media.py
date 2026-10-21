"""Media in a span: semconv media parts, and the strip of raw payloads.

Two halves of one rule, both from rius-semconv's ``registry/media_parts.yaml``:

* **Parts.** Image, audio and file content in a message becomes a semconv
  ``blob``, ``uri`` or ``file`` part (``media_parts`` for a provider's content
  block, ``native_part`` for a part that is already semconv-shaped). Inline
  bytes never survive: a blob carries its ``modality``, ``mime_type`` and
  ``byte_size`` and a ``stripped_reason``, and no ``content``.
* **Raw payloads.** ``strip_raw_payload`` replaces every media token in a
  string (a data URL, a keyed base64 string, a Buffer object) with
  ``[stripped <mime> <byte_size> bytes]`` IN PLACE: every other character of
  the string stays as it was, so the strip never parses and re-serializes the
  JSON. The TypeScript SDK and the sink write the same bytes for the same
  input; the golden examples in the registry pin it, and
  ``tests/test_media.py`` runs every one of them.

The strip is a port of ``generators/media_parts.py`` ``strip_raw_payload``, the
contract's reference implementation, with the rule data inlined (the test
checks it against the synced registry). Do not write a replacement with a JSON
encoder or size base64 with a strict decoder: both disagree across languages,
and the registry's rules are exhaustive.
"""

from __future__ import annotations

import bisect
import functools
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

# --- the contract's rule data (registry/media_parts.yaml) --------------------

MODALITIES = ("image", "video", "audio", "document")
NOT_CAPTURED = "not_captured"
REDACTED_UPSTREAM = "redacted_upstream"
STRIPPED_REASONS = (NOT_CAPTURED, REDACTED_UPSTREAM)

REPLACEMENT_FORMAT = "[stripped <mime> <byte_size> bytes]"
UNKNOWN = "unknown"
MIME_PATTERN = re.compile(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+")

#: Rule (b): a data URL, anywhere in a string. Group 1 the mime type, group 3
#: the base64 run.
DATA_URL_PATTERN = re.compile(
    r'[dD][aA][tT][aA]:([^;,:\t\n\f\r "]*)(;[^;,:\t\n\f\r "]*)*;base64,([A-Za-z0-9+/_-]*={0,2})'
)
#: Text every rule (b) token holds: the pattern's own literal.
DATA_URL_TEXT = ";base64,"
#: Rule (d): a Buffer object.
BUFFER_TYPE = "Buffer"
BUFFER_ARRAY_KEY = "data"
BUFFER_MIME_KEYS = ("media_type", "mediaType", "mime_type", "mimeType")
#: Rule (e): a keyed base64 string.
KEYED_KEYS = ("data", "b64_json", "result", "base64")
KEYED_SIBLING_KEYS = ("media_type", "mediaType", "mime_type", "mimeType", "output_format")
KEYED_PARENT_KEYS = ("input_audio", "audio", "inline_data", "inlineData")
KEYED_SOURCE_KEY = "source"
KEYED_SOURCE_TYPE = "base64"
KEYED_BLOB_TYPE = "blob"
KEYED_BLOB_KEY = "content"
#: (key, types): a key holding base64 whenever its object has one of the types.
KEYED_TYPED_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("base64", ("image", "audio", "video", "file", "text-plain")),
    ("input_audio", ("input_audio",)),
)
#: Keys holding base64 in an object under the source key, whatever its type.
KEYED_SOURCE_BYTES_KEYS = ("bytes",)
KEYED_MIME_KEYS = ("media_type", "mediaType", "mime_type", "mimeType")
#: (sibling key, mime template, parent keys it counts under or None for any).
KEYED_MIME_FORMATS: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("output_format", "image/<value>", None),
    ("format", "audio/<value>", ("input_audio", "audio")),
)

#: How deep a JSON candidate may nest, arrays and objects counted together
#: from the candidate's root, continued through nested documents.
MAX_DEPTH = 64

#: The whitespace a JSON candidate trims and base64 sizing removes.
ASCII_WHITESPACE = " \t\n\r"


# --- shared rules ------------------------------------------------------------


def is_mime(value: Any) -> bool:
    """A token/token MIME type: the only mime a replacement or a part holds."""
    return isinstance(value, str) and MIME_PATTERN.fullmatch(value) is not None


_BASE64_STANDARD = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_BASE64_STANDARD_EXTRA = frozenset("+/")
_BASE64_URL_SAFE_EXTRA = frozenset("-_")
_BASE64_ALL = _BASE64_STANDARD | _BASE64_STANDARD_EXTRA | _BASE64_URL_SAFE_EXTRA


_WHITESPACE_GONE = str.maketrans("", "", ASCII_WHITESPACE)


def base64_size(value: str) -> int | None:
    """The decoded size of a base64 string, or None when it is not valid base64.

    ASCII space, tab, newline and carriage return are removed first. Valid is
    one alphabet (standard or URL-safe, never mixed), at most two ``=`` at the
    end, and a length that is a multiple of 4, so unpadded base64 is not.
    """
    text = value.translate(_WHITESPACE_GONE)
    body = text.rstrip("=")
    padding = len(text) - len(body)
    chars = set(body)
    if padding > 2 or len(text) % 4 or not chars <= _BASE64_ALL:
        return None
    if chars & _BASE64_STANDARD_EXTRA and chars & _BASE64_URL_SAFE_EXTRA:
        return None
    return len(text) * 3 // 4 - padding


def replacement(mime: str | None, byte_size: int | None) -> str:
    """The text a media token in a raw payload is replaced by."""
    return REPLACEMENT_FORMAT.replace("<mime>", mime or UNKNOWN).replace(
        "<byte_size>", UNKNOWN if byte_size is None else str(byte_size)
    )


_REPLACEMENT_HEAD, _rest = REPLACEMENT_FORMAT.split("<mime>")
_REPLACEMENT_BETWEEN, _REPLACEMENT_TAIL = _rest.split("<byte_size>")


def is_replacement(value: str) -> bool:
    """Already a complete replacement: never stripped again, never context.

    The format with a mime that is unknown or token/token and a byte_size of
    ASCII digits or unknown; anything that only looks like one is a token.
    """
    head, between, tail = _REPLACEMENT_HEAD, _REPLACEMENT_BETWEEN, _REPLACEMENT_TAIL
    if len(value) < len(head) + len(between) + len(tail):
        return False
    if not (value.startswith(head) and value.endswith(tail)):
        return False
    mime, sep, size = value[len(head) : len(value) - len(tail)].rpartition(between)
    digits = bool(size) and all("0" <= ch <= "9" for ch in size)
    return bool(sep) and (mime == UNKNOWN or is_mime(mime)) and (digits or size == UNKNOWN)


# --- the strip ---------------------------------------------------------------
#
# Every span is on the original text, and only the tokens the rules find
# change. The structure follows the reference so the two can be read side by
# side: rules (d), then (e), then (b), on each value in document order.


#: Up to this many members a lookup scans them; past it they are indexed.
_SCAN_MEMBERS = 16


class Members:
    """An object's members in document order, duplicates kept, with a large
    object's values indexed by key on first lookup (a rule asks for a few keys
    several times)."""

    __slots__ = ("_by_key", "items", "names")

    def __init__(self, items: list[tuple[str, Any]]) -> None:
        self.items = items
        self.names = {name for name, _ in items}
        self._by_key: dict[str, list[Any]] | None = None

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(self.items)

    def values(self, key: str) -> Iterable[Any]:
        """The values of the members named ``key``, in document order."""
        if key not in self.names:
            return ()
        if len(self.items) <= _SCAN_MEMBERS:
            return (value for name, value in self.items if name == key)
        if self._by_key is None:
            by_key: dict[str, list[Any]] = {}
            for name, value in self.items:
                by_key.setdefault(name, []).append(value)
            self._by_key = by_key
        return self._by_key.get(key, [])


def _first(members: Members, key: str, ok: Callable[[Any], bool]) -> Any:
    """The value of the first member named ``key`` for which ``ok`` holds."""
    return next((v for v in members.values(key) if ok(v)), None)


def _sibling_string(value: Any) -> bool:
    """A value rule (e) reads as context: a non-empty string, not a replacement."""
    return isinstance(value, str) and bool(value) and not is_replacement(value)


#: A string rule (e) replaces: the same test, so stripping twice changes nothing.
_keyed_string = _sibling_string


def _mime_key(mime_keys: tuple[str, ...], members: Members) -> str | None:
    """The first mime key, in list order, holding a valid mime."""
    for key in mime_keys:
        mime = _first(members, key, lambda v: _sibling_string(v) and is_mime(v))
        if mime is not None:
            return str(mime)
    return None


def _buffer_data(members: Members) -> list[Any] | None:
    """Rule (d): a Buffer object's byte array, or None when it is not one."""
    if _first(members, "type", lambda v: v == BUFFER_TYPE) is None:
        return None
    data = _first(members, BUFFER_ARRAY_KEY, lambda v: isinstance(v, list) and bool(v))
    return data if isinstance(data, list) else None


def _keyed_positions(members: Members, parent_key: str | None) -> set[int]:
    """Rule (e): the positions of the members holding base64 media."""
    context = (
        any(k in KEYED_SIBLING_KEYS and _sibling_string(v) for k, v in members)
        or parent_key in KEYED_PARENT_KEYS
        or (
            parent_key == KEYED_SOURCE_KEY
            and _first(members, "type", lambda v: v == KEYED_SOURCE_TYPE) is not None
        )
    )
    keys = set(KEYED_KEYS) if context else set()
    if _first(members, "type", lambda v: v == KEYED_BLOB_TYPE) is not None:
        keys.add(KEYED_BLOB_KEY)
    for key, types in KEYED_TYPED_KEYS:
        if any(k == "type" and v in types for k, v in members):
            keys.add(key)
    if parent_key == KEYED_SOURCE_KEY:
        keys.update(KEYED_SOURCE_BYTES_KEYS)
    return {
        i
        for i, (k, v) in enumerate(members)
        if k in keys
        and (_keyed_string(v) or (isinstance(v, list) and any(_keyed_string(x) for x in v)))
    }


def _keyed_mime(members: Members, parent_key: str | None) -> str | None:
    """Rule (e)'s mime type: a mime key, else a mime format, else unknown."""
    mime = _mime_key(KEYED_MIME_KEYS, members)
    if mime:
        return mime
    for key, template, parents in KEYED_MIME_FORMATS:
        if parents is not None and parent_key not in parents:
            continue
        sibling = _first(members, key, _sibling_string)
        if sibling is None:
            continue
        mime = template.replace("<value>", sibling)
        if is_mime(mime):
            return mime
    return None


class _NotJson(ValueError):
    pass


@dataclass(slots=True)
class _Value:
    """A JSON value: its span on the text, its Python value, and its members
    (object), items (array) or the segments mapping each decoded character to
    the raw text (string)."""

    start: int
    end: int
    value: Any
    members: list[tuple[str, _Value]] | None = None
    items: list[_Value] | None = None
    segments: list[tuple[int, int, int]] | None = None

    def raw_span(self, index: int) -> tuple[int, int]:
        """Where a string's decoded character ``index`` sits in the raw text."""
        assert self.segments is not None
        at = bisect.bisect_right(self.segments, index, key=lambda s: s[0]) - 1
        decoded, raw, escape = self.segments[at]
        if escape:
            return raw, raw + escape
        return raw + index - decoded, raw + index - decoded + 1


_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_SHORT_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_HEX = frozenset("0123456789abcdefABCDEF")
_PLAIN_RUN = re.compile(r'[^"\\\x00-\x1f]*')
_SIMPLE_STRING = re.compile(r'"([^"\\\x00-\x1f]*)"')
_SPACE = re.compile(r"[ \t\n\r]*")
#: An escape-free member key and its colon, with the space around them.
_MEMBER_KEY = re.compile(r'[ \t\n\r]*"([^"\\\x00-\x1f]*)"[ \t\n\r]*:[ \t\n\r]*')
#: The separator after a member or an item, with the space before it.
_OBJECT_SEPARATOR = re.compile(r"[ \t\n\r]*([,}])")
_PLAIN_MEMBER_STRING = re.compile(r'"([^"\\\x00-\x1f]*)"[ \t\n\r]*([,}])')
_ARRAY_SEPARATOR = re.compile(r"[ \t\n\r]*([,\]])")
#: The value every parsed object holds: rules read an object's members, never
#: the object as a value (a rule comparing a member to a string, or testing it
#: for a list, sees an object as neither).
_OBJECT_VALUE: Any = object()
#: Shared leaves: a literal's value is all a rule reads of it, and a number's
#: not even that, so neither needs a node of its own.
_LITERALS = (
    ("true", _Value(0, 0, True)),
    ("false", _Value(0, 0, False)),
    ("null", _Value(0, 0, None)),
)
_NUMBER_LEAF = _Value(0, 0, 0)


class _Parser:
    """Strict RFC 8259, keeping every value's span and every duplicate member,
    at most MAX_DEPTH deep counting the ``depth`` arrays and objects around the
    text. The depth limit bounds the recursion, so no input raises out."""

    def __init__(self, text: str, depth: int, max_depth: int = MAX_DEPTH) -> None:
        self.text = text
        self.i = 0
        self.depth = depth
        self.max_depth = max_depth

    def enter(self) -> None:
        self.depth += 1
        if self.depth > self.max_depth:
            raise _NotJson("too deep")

    def document(self) -> _Value:
        self.space()
        value = self.value()
        self.space()
        if self.i != len(self.text):
            raise _NotJson("trailing text")
        return value

    def space(self) -> None:
        match = _SPACE.match(self.text, self.i)
        assert match is not None  # the pattern matches the empty string
        self.i = match.end()

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def expect(self, ch: str) -> None:
        if self.peek() != ch:
            raise _NotJson(f"expected {ch!r}")
        self.i += 1

    def value(self) -> _Value:
        ch = self.text[self.i : self.i + 1]
        if ch == "{":
            return self.object()
        if ch == "[":
            return self.array()
        if ch == '"':
            return self.string()
        for word, leaf in _LITERALS:
            if self.text.startswith(word, self.i):
                self.i += len(word)
                return leaf
        match = _NUMBER.match(self.text, self.i)
        if not match:
            raise _NotJson("no value")
        self.i = match.end()
        # A number's value and span are never read by a rule; only that it is one.
        return _NUMBER_LEAF

    def object(self) -> _Value:
        # The loops read a member's key and colon, and each separator, with one
        # match; anything those patterns miss takes the general path, which
        # fails as the plain grammar would.
        text, start = self.text, self.i
        self.expect("{")
        self.enter()
        members: list[tuple[str, _Value]] = []
        self.space()
        if self.peek() == "}":
            self.i += 1
        else:
            while True:
                key = _MEMBER_KEY.match(text, self.i)
                if key is not None:
                    name = key.group(1)
                    self.i = key.end()
                else:
                    self.space()
                    name = self.string().value
                    self.space()
                    self.expect(":")
                    self.space()
                plain = _PLAIN_MEMBER_STRING.match(text, self.i)
                if plain is not None:
                    # The commonest member: an escape-free string, then the separator.
                    value, close = plain.group(1, 2)
                    at = self.i
                    segments = [(0, at + 1, 0)] if value else []
                    members.append((name, _Value(at, plain.end(1) + 1, value, segments=segments)))
                    self.i = plain.end()
                    if close == "}":
                        break
                    continue
                members.append((name, self.value()))
                sep = _OBJECT_SEPARATOR.match(text, self.i)
                if sep is None:
                    raise _NotJson("expected ',' or '}'")
                self.i = sep.end()
                if sep.group(1) == "}":
                    break
        self.depth -= 1
        # No rule reads an object as a value, only its members.
        return _Value(start, self.i, _OBJECT_VALUE, members=members)

    def array(self) -> _Value:
        text, start = self.text, self.i
        self.expect("[")
        self.enter()
        items: list[_Value] = []
        self.space()
        if self.peek() == "]":
            self.i += 1
        else:
            while True:
                self.space()
                items.append(self.value())
                sep = _ARRAY_SEPARATOR.match(text, self.i)
                if sep is None:
                    raise _NotJson("expected ',' or ']'")
                self.i = sep.end()
                if sep.group(1) == "]":
                    break
        self.depth -= 1
        return _Value(start, self.i, [v.value for v in items], items=items)

    def string(self) -> _Value:
        start = self.i
        text = self.text
        simple = _SIMPLE_STRING.match(text, start)
        if simple is not None:
            # No escape: one run, or none for the empty string.
            self.i = simple.end()
            value = simple.group(1)
            return _Value(start, self.i, value, segments=[(0, start + 1, 0)] if value else [])
        self.expect('"')
        i, end = self.i, len(text)
        out: list[str] = []
        # Runs of plain characters, and escapes, each as (decoded index, raw
        # index, raw length of the escape or 0 for a run): what maps a decoded
        # index back to the raw text. Base64 is one run.
        segments: list[tuple[int, int, int]] = []
        decoded = 0
        while True:
            run = _PLAIN_RUN.match(text, i)
            assert run is not None  # the pattern matches the empty string
            if run.end() > i:
                segments.append((decoded, i, 0))
                out.append(run.group())
                decoded += run.end() - i
                i = run.end()
            if i >= end:
                raise _NotJson("unterminated string")
            ch, at = text[i], i
            if ch == '"':
                i += 1
                break
            if ch != "\\":
                raise _NotJson("control character in string")
            kind = text[i + 1 : i + 2]
            short = _SHORT_ESCAPES.get(kind)
            if short is not None:
                out.append(short)
                i += 2
            elif kind == "u":
                code = self.hex4(i + 2)
                i += 6
                if 0xD800 <= code < 0xDC00 and text.startswith("\\u", i):
                    low = self.hex4(i + 2)
                    if 0xDC00 <= low < 0xE000:
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        i += 6
                out.append(chr(code))
            else:
                raise _NotJson("bad escape")
            segments.append((decoded, at, i - at))
            decoded += 1
        self.i = i
        return _Value(start, i, "".join(out), segments=segments)

    def hex4(self, at: int) -> int:
        digits = self.text[at : at + 4]
        if len(digits) != 4 or not set(digits) <= _HEX:
            raise _NotJson("bad \\u escape")
        return int(digits, 16)


def _parse(text: str, depth: int) -> _Value | None:
    try:
        return _Parser(text, depth).document()
    except _NotJson:
        return None


def _is_json_candidate(value: str) -> bool:
    return value.lstrip(ASCII_WHITESPACE)[:1] in ("{", "[")


_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


_NEEDS_ESCAPE = re.compile('["\\\\\x00-\x1f\ud800-\udfff]')


def json_escape(text: str) -> str:
    """Text as written inside a JSON string: the minimal escapes JSON.stringify
    writes (lowercase ``\\u00xx`` for other controls and lone surrogates), and
    nothing else."""
    return _NEEDS_ESCAPE.sub(_escaped, text)


def _escaped(match: re.Match[str]) -> str:
    ch = match.group()
    return _ESCAPES.get(ch) or f"\\u{ord(ch):04x}"


#: An edit on a text: replace text[start:end] by the string.
_Edit = tuple[int, int, str]


@dataclass(frozen=True)
class MediaToken:
    """A media token the strip replaced: its rule, mime type and size (None
    when unknown)."""

    rule: str
    mime: str | None
    byte_size: int | None


@functools.lru_cache(maxsize=256)
def _literal(mime: str | None, byte_size: int | None) -> str:
    return '"' + json_escape(replacement(mime, byte_size)) + '"'


def _strip_text(text: str, depth: int, tokens: list[MediaToken]) -> list[_Edit]:
    """The edits on one string (an attribute, or a JSON string's unescaped
    value inside ``depth`` arrays and objects): the JSON walk when it is a
    candidate that parses, else rule (b) alone."""
    if is_replacement(text):
        return []
    if _is_json_candidate(text):
        document = _parse(text, depth)
        if document is not None:
            edits: list[_Edit] = []
            _walk(document, None, None, depth, tokens, edits)
            return edits
    edits = []
    for match in DATA_URL_PATTERN.finditer(text):
        mime = match.group(1) if is_mime(match.group(1)) else None
        size = base64_size(match.group(3))
        tokens.append(MediaToken("b", mime, size))
        edits.append((match.start(), match.end(), replacement(mime, size)))
    return edits


def _members(node: _Value) -> Members:
    assert node.members is not None
    return Members([(key, child.value) for key, child in node.members])


def _walk(
    node: _Value,
    parent_key: str | None,
    holder: _Value | None,
    depth: int,
    tokens: list[MediaToken],
    edits: list[_Edit],
) -> None:
    """Rules (d), then (e), then (b), on one value. ``holder`` is the nearest
    enclosing object (through arrays), for rule (d)'s mime type."""
    if node.members is not None:
        if not any(key in _RULE_KEYS for key, _ in node.members):
            # No member a rule (d) or (e) replaces: only the children can hold one.
            for key, child in node.members:
                _walk(child, key, node, depth + 1, tokens, edits)
            return
        members = _members(node)
        data = _buffer_data(members)
        if data is not None:
            mime = _mime_key(BUFFER_MIME_KEYS, _members(holder) if holder else Members([]))
            tokens.append(MediaToken("d", mime, len(data)))
            edits.append((node.start, node.end, _literal(mime, len(data))))
            return
        keyed = _keyed_positions(members, parent_key)
        mime = _keyed_mime(members, parent_key) if keyed else None
        for i, (key, child) in enumerate(node.members):
            if i not in keyed:
                _walk(child, key, node, depth + 1, tokens, edits)
            elif child.items is None:
                _keyed_edit(child, mime, tokens, edits)
            else:
                # A string array under the key: each string element is a token.
                for item in child.items:
                    if _keyed_string(item.value):
                        _keyed_edit(item, mime, tokens, edits)
                    else:
                        _walk(item, key, node, depth + 2, tokens, edits)
    elif node.items is not None:
        for child in node.items:
            _walk(child, parent_key, holder, depth + 1, tokens, edits)
    elif node.segments is not None:
        if not _may_hold_media(node.value):
            return  # the same test the whole text passed: no token can be in it
        # A JSON string: strip its unescaped value, then map each edit back to
        # the raw characters and write it escaped for this level.
        for start, end, text in _strip_text(node.value, depth, tokens):
            raw_start = node.raw_span(start)[0]
            raw_end = node.raw_span(end - 1)[1] if end > start else raw_start
            edits.append((raw_start, raw_end, json_escape(text)))


#: The member keys rules (d) and (e) replace: an object with none of them is
#: only walked into.
_RULE_KEYS = frozenset(
    (
        BUFFER_ARRAY_KEY,
        KEYED_BLOB_KEY,
        *KEYED_KEYS,
        *(key for key, _ in KEYED_TYPED_KEYS),
        *KEYED_SOURCE_BYTES_KEYS,
    )
)


def _keyed_edit(
    node: _Value, mime: str | None, tokens: list[MediaToken], edits: list[_Edit]
) -> None:
    """Rule (e) on one string literal: replaced whole, quotes included."""
    size = base64_size(node.value)
    tokens.append(MediaToken("e", mime, size))
    edits.append((node.start, node.end, _literal(mime, size)))


#: Rule (e) replaces a keyed member only beside a sibling mime key, under a
#: parent key, or in a base64 source, so one of these words is written out too.
_KEYED_CONTEXT_TEXT = (*KEYED_SIBLING_KEYS, *KEYED_PARENT_KEYS, KEYED_SOURCE_KEY)


def _may_hold_media(text: str) -> bool:
    """False only when no rule can find a token in ``text``.

    Every token needs its rule's words written out, unless a ``\\u`` escape
    hides them: rule (b)'s literal ``;base64,``, rule (d)'s ``Buffer`` and
    ``data``, and rule (e)'s keyed key with its context word, or ``blob`` and
    ``content`` for a blob part, a typed key with one of its types, or a source
    bytes key with ``source``. The other JSON escapes write none of these
    characters. Most attributes, and most strings inside them, hold none and
    skip the parse; the test holds for a JSON string's unescaped value too.
    """
    if "\\u" in text or DATA_URL_TEXT in text:
        return True
    if BUFFER_TYPE in text and BUFFER_ARRAY_KEY in text:
        return True
    if KEYED_BLOB_TYPE in text and KEYED_BLOB_KEY in text:
        return True
    for key, types in KEYED_TYPED_KEYS:
        if key in text and any(t in text for t in types):
            return True
    if KEYED_SOURCE_KEY in text and any(key in text for key in KEYED_SOURCE_BYTES_KEYS):
        return True
    return any(key in text for key in KEYED_KEYS) and any(
        word in text for word in _KEYED_CONTEXT_TEXT
    )


def _strip(text: str, tokens: list[MediaToken]) -> tuple[str, list[_Edit]]:
    if not _may_hold_media(text):
        return text, []
    edits = _strip_text(text, 0, tokens)
    if not edits:
        return text, edits
    out, last = [], 0
    for start, end, new in edits:
        out += [text[last:start], new]
        last = end
    out.append(text[last:])
    return "".join(out), edits


def strip_raw_payload(text: str) -> str:
    """``text`` with every media token replaced in place.

    Every other character is unchanged, so the result is byte-identical to the
    reference's for the same input. Never raises: the parser's depth limit
    bounds every recursion.
    """
    return _strip(text, [])[0]


def split_media(text: str) -> tuple[int, list[MediaToken]]:
    """A text's prose bytes and the media tokens in it, in document order.

    The prose is the stripped text less what the strip wrote in place of each
    token, in UTF-8 bytes: a data URL or base64 in a text part is media, not
    something the model read as text.
    """
    tokens: list[MediaToken] = []
    stripped, edits = _strip(text, tokens)
    written = sum(len(new.encode()) for _, _, new in edits)
    return len(stripped.encode()) - written, tokens


# --- parts -------------------------------------------------------------------
#
# What a provider's content block becomes, row by row of the mapping table.
# Each part is built in the contract's field order, an unknown field omitted
# (never null), and mime_type written only when it is token/token:
#
#   blob  {"type","modality","mime_type"?,"byte_size"?,"stripped_reason"}
#   uri   {"type","modality","mime_type"?,"uri"}
#   file  {"type","modality","mime_type"?,"file_id"}
#
# The TypeScript SDK (mediaParts.ts) builds the same parts from the same
# input, byte for byte; keep the two in step.

Part = dict[str, Any]


def modality_of(mime: Any) -> str | None:
    """The modality a token/token MIME type names: image/*, audio/* and
    video/* their own, any other (PDF, text/*, office formats) document. None
    when there is no valid MIME type, for the caller's part-kind fallback."""
    if not is_mime(mime):
        return None
    top = str(mime).split("/", 1)[0].lower()
    return top if top in ("image", "audio", "video") else "document"


def _with_mime(part: Part, mime: Any) -> Part:
    if is_mime(mime):
        part["mime_type"] = mime
    return part


#: The largest whole number JavaScript holds exactly (Number.MAX_SAFE_INTEGER).
MAX_SAFE_INTEGER = 2**53 - 1


def safe_size(value: Any) -> int | None:
    """A byte size as the TypeScript SDK and the sink read one: a whole,
    non-negative, exactly representable number (``8``, ``8.0`` and ``8e0``
    alike), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value <= MAX_SAFE_INTEGER or value != int(value):
        return None
    return int(value)


def blob_part(kind: str, mime: Any, byte_size: int | None, reason: str) -> Part:
    """A blob: inline bytes, never their content."""
    part = _with_mime({"type": "blob", "modality": modality_of(mime) or kind}, mime)
    if byte_size is not None:
        part["byte_size"] = byte_size
    part["stripped_reason"] = reason
    return part


def uri_part(kind: str, mime: Any, uri: str) -> Part:
    """A uri: media referenced by URL."""
    part = _with_mime({"type": "uri", "modality": modality_of(mime) or kind}, mime)
    part["uri"] = uri
    return part


def file_part(kind: str, mime: Any, file_id: str) -> Part:
    """A file: a provider file id."""
    part = _with_mime({"type": "file", "modality": modality_of(mime) or kind}, mime)
    part["file_id"] = file_id
    return part


def is_data_url(value: Any) -> bool:
    """A data URL, the scheme in any case."""
    return isinstance(value, str) and value[:5].lower() == "data:"


def data_url_blob(url: str, kind: str, mime_hint: Any = None) -> Part:
    """A data URL as a blob, never a uri: its own media type (else the hint),
    and the size of its base64 run. A data URL that is not base64 is a blob
    with no size; one with no payload at all (``data:None;url,None``, what
    OpenInference's Anthropic instrumentor writes for a url image) was
    emptied before we saw it, so it is redacted upstream."""
    match = DATA_URL_PATTERN.fullmatch(url)
    if match is not None:
        mime = match.group(1) if is_mime(match.group(1)) else mime_hint
        return blob_part(kind, mime, base64_size(match.group(3)), NOT_CAPTURED)
    header = re.split("[;,]", url[5:], maxsplit=1)[0]
    _, comma, payload = url.partition(",")
    mime = header if is_mime(header) else mime_hint
    if not comma or payload in ("", "None"):
        return blob_part(kind, mime, None, REDACTED_UPSTREAM)
    return blob_part(kind, mime, None, NOT_CAPTURED)


#: What OpenInference writes in place of an image over its size limit.
OPENINFERENCE_REDACTED = "__REDACTED__"


def url_part(url: str, kind: str, mime: Any = None) -> Part:
    """A URL as a part: a data URL is a blob, anything else a uri."""
    return data_url_blob(url, kind, mime) if is_data_url(url) else uri_part(kind, mime, url)


def base64_blob(data: str, kind: str, mime: Any) -> Part:
    """Inline base64 (or a data URL in its place) as a blob."""
    if is_data_url(data):
        return data_url_blob(data, kind, mime)
    return blob_part(kind, mime, base64_size(data), NOT_CAPTURED)


#: Filename extensions that name a modality, for file data with no MIME type.
#: Anything else is a document.
_EXTENSION_MODALITY = {
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "webp": "image",
    "wav": "audio",
    "mp3": "audio",
    "m4a": "audio",
    "ogg": "audio",
    "flac": "audio",
    "mp4": "video",
    "mov": "video",
    "webm": "video",
}


def _filename_modality(filename: Any) -> str:
    if not isinstance(filename, str):
        return "document"
    _, dot, extension = filename.rpartition(".")
    return _EXTENSION_MODALITY.get(extension.lower(), "document") if dot else "document"


def _string(value: Any) -> str | None:
    """A non-empty string, else None."""
    return value if isinstance(value, str) and value else None


def media_parts(item: Any) -> list[Part] | None:
    """A provider's media content item as semconv parts, or None when it is
    not a media item this module knows (it then passes through unchanged).

    Covers the pass-through rows of the mapping table: OpenAI chat and
    Responses, Anthropic, Gemini, Mistral, LangChain v1, Bedrock Converse,
    and native semconv media parts.
    """
    if not isinstance(item, dict):
        return None
    block_type = item.get("type")
    if isinstance(block_type, str) and block_type in _LANGCHAIN_KINDS:
        data = item.get("base64")
        if isinstance(data, str):
            return [base64_blob(data, _LANGCHAIN_KINDS[block_type], item.get("mime_type"))]
    if block_type in ("blob", "uri", "file"):
        if block_type == "file" and isinstance(item.get("file"), dict):
            return _openai_file(item["file"])
        native = native_part(item)
        return None if native is None else [native]
    if block_type == "image_url":
        image = item.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        return [url_part(url, "image")] if isinstance(url, str) else None
    if block_type == "input_audio":
        audio = item.get("input_audio")
        if isinstance(audio, str):
            # Mistral's audio chunk: the bare base64, with no format.
            return [base64_blob(audio, "audio", None)]
        if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
            return None
        fmt = _string(audio.get("format"))
        return [base64_blob(audio["data"], "audio", fmt and f"audio/{fmt}")]
    if block_type == "input_image":
        url = _string(item.get("image_url"))
        if url is not None:
            return [url_part(url, "image")]
        file_id = _string(item.get("file_id"))
        return None if file_id is None else [file_part("image", None, file_id)]
    if block_type == "input_file":
        return _openai_file(item)
    if block_type == "image_generation_call":
        if not isinstance(item.get("result"), str):
            return None
        fmt = _string(item.get("output_format"))
        return [base64_blob(item["result"], "image", fmt and f"image/{fmt}")]
    if block_type in ("image", "document"):
        return _anthropic_source(block_type, item.get("source"))
    return _bedrock_part(item) or _gemini_part(item)


#: A LangChain v1 standard content block's type, when it carries base64, and
#: the modality it names.
_LANGCHAIN_KINDS = {
    "image": "image",
    "audio": "audio",
    "video": "video",
    "file": "document",
    "text-plain": "document",
}


def _bedrock_part(item: Part) -> list[Part] | None:
    """A Bedrock Converse image, video or document block, which has no type,
    by its source's bytes: base64 (or, from boto3, a bytes object) a blob. An
    image's format names its mime type; a video or document format is not
    always a MIME subtype, so it names none."""
    for kind in ("image", "video", "document"):
        block = item.get(kind)
        if not isinstance(block, dict):
            continue
        source = block.get("source")
        data = source.get("bytes") if isinstance(source, dict) else None
        fmt = _string(block.get("format")) if kind == "image" else None
        mime = fmt and f"image/{fmt}"
        if isinstance(data, str):
            return [base64_blob(data, kind, mime)]
        if isinstance(data, (bytes, bytearray)):
            return [blob_part(kind, mime, len(data), NOT_CAPTURED)]
    return None


def _openai_file(file: Part) -> list[Part] | None:
    """Chat's ``file {file_id | file_data, filename}`` or the Responses
    ``input_file {file_id | file_data | file_url, filename}``. File data is a
    data URL (its MIME type names the modality) or bare base64, whose
    modality comes from the filename."""
    kind = _filename_modality(file.get("filename"))
    file_id = _string(file.get("file_id"))
    if file_id is not None:
        return [file_part(kind, None, file_id)]
    data = _string(file.get("file_data"))
    if data is not None:
        return [base64_blob(data, kind, None)]
    url = _string(file.get("file_url"))
    return None if url is None else [url_part(url, kind)]


def _anthropic_source(block_type: str, source: Any) -> list[Part] | None:
    """An Anthropic image or document by its source: base64 a blob, url a uri,
    file a file. A text or content document is text to the model."""
    if not isinstance(source, dict):
        return None
    kind = "image" if block_type == "image" else "document"
    source_type = source.get("type")
    mime = source.get("media_type")
    if source_type == "base64":
        data = source.get("data")
        return [base64_blob(data, kind, mime)] if isinstance(data, str) else None
    if source_type == "url":
        url = source.get("url")
        return [url_part(url, kind, mime)] if isinstance(url, str) else None
    if source_type == "file":
        file_id = source.get("file_id")
        return [file_part(kind, mime, file_id)] if isinstance(file_id, str) else None
    if block_type != "document":
        return None
    if source_type == "text":
        data = source.get("data")
        return [{"type": "text", "content": data}] if isinstance(data, str) else None
    if source_type == "content":
        content = source.get("content")
        if isinstance(content, str):
            return [{"type": "text", "content": content}]
        if isinstance(content, list):
            return [
                {"type": "text", "content": block["text"]}
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
    return None


def _gemini_part(item: Part) -> list[Part] | None:
    """A Gemini part, which has no type, in either spelling: ``inline_data``
    a blob, ``file_data`` a uri (a blob when its URI is a data URL)."""
    inline = item.get("inline_data", item.get("inlineData"))
    if isinstance(inline, dict) and isinstance(inline.get("data"), str):
        mime = inline.get("mime_type", inline.get("mimeType"))
        return [base64_blob(inline["data"], "document", mime)]
    file = item.get("file_data", item.get("fileData"))
    if isinstance(file, dict):
        uri = file.get("file_uri", file.get("fileUri"))
        if isinstance(uri, str):
            mime = file.get("mime_type", file.get("mimeType"))
            return [url_part(uri, _uri_modality(uri), mime)]
    return None


def _uri_modality(uri: str) -> str:
    """The modality a URI's filename extension names (its path, without the
    query or fragment); document when it names none we know."""
    path = uri.split("#", 1)[0].split("?", 1)[0]
    return _filename_modality(path.rsplit("/", 1)[-1])


def native_part(part: Part) -> Part | None:
    """A native semconv media part rebuilt into the canonical shape, or None
    when it cannot be (a uri with no URI string, a file with no id), to be
    kept as it is.

    Every readable part is rebuilt: members in the contract's order, any
    other member dropped, ``mime_type`` only when it is token/token.

    * A blob carrying content becomes a stripped blob: sized from it,
      ``not_captured``, its modality from its MIME type first (some producers
      label every file an image), else the one it has, else document.
    * Any other part keeps a modality it has, else takes one from its MIME
      type, else (a uri) from its filename extension, else document. A blob
      without content keeps a valid size and stripped_reason.
    * A uri holding a data URL is a blob: a data URL is never a uri.
    """
    written = part.get("modality")
    own = written if isinstance(written, str) and written in MODALITIES else None
    part_type = part.get("type")
    mime = part.get("mime_type")
    if part_type == "blob" and isinstance(part.get("content"), str):
        return base64_blob(part["content"], own or "document", mime)
    if part_type == "uri" and is_data_url(part.get("uri")):
        return data_url_blob(part["uri"], own or "document", mime)
    if part_type == "blob":
        size = part.get("byte_size")
        reason = part.get("stripped_reason")
        if reason not in STRIPPED_REASONS:
            reason = REDACTED_UPSTREAM if part.get("content") is None else NOT_CAPTURED
        rebuilt = blob_part("document", mime, safe_size(size), str(reason))
    elif part_type == "uri" and isinstance(part.get("uri"), str):
        rebuilt = uri_part(_uri_modality(part["uri"]), mime, part["uri"])
    elif part_type == "file" and isinstance(part.get("file_id"), str):
        rebuilt = file_part("document", mime, part["file_id"])
    else:
        return None
    if own is not None:
        rebuilt["modality"] = own  # a modality the part has is kept
    return rebuilt


def _compact(part: Part) -> str:
    """A rebuilt part as JSON.stringify writes it: compact, strings with the
    strip's minimal escapes. Its members are strings and whole numbers."""
    members = (
        f'"{json_escape(key)}":'
        + (f'"{json_escape(value)}"' if isinstance(value, str) else str(value))
        for key, value in part.items()
    )
    return "{" + ",".join(members) + "}"


#: How many arrays and objects deep a messages value may nest and still have
#: its parts rebuilt, counting the value's own array: far above any real
#: message, and the same bound as the TypeScript SDK and the sink.
MESSAGE_MAX_DEPTH = 1000


@dataclass
class _Open:
    """An array or object the part scan is inside: what it is in the
    messages value, the member being read (object), and the part spans found
    in it so far."""

    is_object: bool
    role: str
    start: int
    key: str | None = None
    spans: list[tuple[int, int]] | None = None


def _key(scan: _Parser) -> str:
    """An object member's key and its colon, up to its value."""
    scan.space()
    key = scan.string().value
    scan.space()
    scan.expect(":")
    scan.space()
    return str(key)


def _role(stack: list[_Open], parts_only: bool) -> str:
    """What the next value is, from where it sits: the value itself, a
    message, a message's parts, a part, or anything else."""
    if not stack:
        return "root"
    parent = stack[-1]
    if parent.role == "root" and not parent.is_object:
        return "part" if parts_only else "message"
    if parent.role == "message" and parent.is_object and parent.key == "parts":
        return "parts"
    if parent.role == "parts" and not parent.is_object:
        return "part"
    return "other"


def _part_spans(text: str, parts_only: bool) -> list[tuple[int, int]]:
    """The spans of the objects a messages value holds as parts, in order:
    each message's ``parts`` items (the last ``parts`` member wins), or with
    ``parts_only`` the value's own items.

    A strict RFC 8259 walk with an explicit stack, so no nesting exhausts the
    interpreter's: a value that does not parse, or nests past
    MESSAGE_MAX_DEPTH, has none.
    """
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
                        node.key = _key(scan)
                    stack.append(node)
                    continue
                scan.i += 1
                scan.depth -= 1
            else:
                scan.value()  # a string, number or literal: never nests
            # The value from start to scan.i is done: hand it to its parents.
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
                        parent.key = _key(scan)
                    break
                scan.expect("}" if parent.is_object else "]")
                scan.depth -= 1
                node = stack.pop()
                role, start = node.role, node.start
    except _NotJson:
        return []


def _part_edit(raw: str) -> str | None:
    """One native media part's rebuilt encoding, when its bytes change.

    The part is read with ``json.loads``, as a standard decoder reads it (the
    last of a duplicated member wins). One nested too deep to load is not
    rebuilt."""
    try:
        part = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    rebuilt = native_part(part) if isinstance(part, dict) else None
    if rebuilt is None:
        return None
    encoded = _compact(rebuilt)
    return encoded if encoded != raw else None


def rebuild_native_parts(text: str, *, parts_only: bool) -> str:
    """A messages value with every native media part rebuilt, in place.

    Only a rebuilt part's own bytes are rewritten, compact; every other byte
    of the value stays as it came, and a value with nothing to rebuild, or
    that does not parse, is returned unchanged. ``parts_only`` is a value
    that is itself a list of parts (system instructions); otherwise each
    message's ``parts`` are rebuilt.
    """
    # Every media part names its type, unless an escape spells it.
    if '"blob"' not in text and '"uri"' not in text and '"file"' not in text and "\\" not in text:
        return text
    out, last = [], 0
    for start, end in _part_spans(text, parts_only):
        encoded = _part_edit(text[start:end])
        if encoded is not None:
            out += [text[last:start], encoded]
            last = end
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)
