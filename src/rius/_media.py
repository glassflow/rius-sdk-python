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
import re
from collections.abc import Callable
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
#: Rule (d): a Buffer object.
BUFFER_TYPE = "Buffer"
BUFFER_ARRAY_KEY = "data"
BUFFER_MIME_KEYS = ("media_type", "mediaType", "mime_type", "mimeType")
#: Rule (e): a keyed base64 string.
KEYED_KEYS = ("data", "b64_json", "result")
KEYED_SIBLING_KEYS = ("media_type", "mediaType", "mime_type", "mimeType", "output_format")
KEYED_PARENT_KEYS = ("input_audio", "audio", "inline_data", "inlineData")
KEYED_SOURCE_KEY = "source"
KEYED_SOURCE_TYPE = "base64"
KEYED_BLOB_TYPE = "blob"
KEYED_BLOB_KEY = "content"
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

#: An object's members in document order, duplicates kept.
Members = list[tuple[str, Any]]


def _first(members: Members, key: str, ok: Callable[[Any], bool]) -> Any:
    """The value of the first member named ``key`` for which ``ok`` holds."""
    return next((v for k, v in members if k == key and ok(v)), None)


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


@dataclass
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


class _Parser:
    """Strict RFC 8259, keeping every value's span and every duplicate member,
    at most MAX_DEPTH deep counting the ``depth`` arrays and objects around the
    text. The depth limit bounds the recursion, so no input raises out."""

    def __init__(self, text: str, depth: int) -> None:
        self.text = text
        self.i = 0
        self.depth = depth

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _NotJson("too deep")

    def document(self) -> _Value:
        self.space()
        value = self.value()
        self.space()
        if self.i != len(self.text):
            raise _NotJson("trailing text")
        return value

    def space(self) -> None:
        text, i = self.text, self.i
        while i < len(text) and text[i] in ASCII_WHITESPACE:
            i += 1
        self.i = i

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def expect(self, ch: str) -> None:
        if self.peek() != ch:
            raise _NotJson(f"expected {ch!r}")
        self.i += 1

    def value(self) -> _Value:
        ch = self.peek()
        if ch == "{":
            return self.object()
        if ch == "[":
            return self.array()
        if ch == '"':
            return self.string()
        for word, val in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(word, self.i):
                start, self.i = self.i, self.i + len(word)
                return _Value(start, self.i, val)
        match = _NUMBER.match(self.text, self.i)
        if not match:
            raise _NotJson("no value")
        self.i = match.end()
        # A number's value is never read by a rule; only that it is one.
        return _Value(match.start(), match.end(), 0)

    def object(self) -> _Value:
        start = self.i
        self.expect("{")
        self.enter()
        members: list[tuple[str, _Value]] = []
        self.space()
        if self.peek() == "}":
            self.i += 1
        else:
            while True:
                self.space()
                key = self.string()
                self.space()
                self.expect(":")
                self.space()
                members.append((key.value, self.value()))
                self.space()
                if self.peek() == ",":
                    self.i += 1
                    continue
                self.expect("}")
                break
        self.depth -= 1
        return _Value(start, self.i, {k: v.value for k, v in members}, members=members)

    def array(self) -> _Value:
        start = self.i
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
                self.space()
                if self.peek() == ",":
                    self.i += 1
                    continue
                self.expect("]")
                break
        self.depth -= 1
        return _Value(start, self.i, [v.value for v in items], items=items)

    def string(self) -> _Value:
        start = self.i
        self.expect('"')
        text = self.text
        out: list[str] = []
        # Runs of plain characters, and escapes, each as (decoded index, raw
        # index, raw length of the escape or 0 for a run): what maps a decoded
        # index back to the raw text. Base64 is one run.
        segments: list[tuple[int, int, int]] = []
        decoded = 0
        while True:
            run = _PLAIN_RUN.match(text, self.i)
            assert run is not None  # the pattern matches the empty string
            if run.end() > self.i:
                segments.append((decoded, self.i, 0))
                out.append(run.group())
                decoded += run.end() - self.i
                self.i = run.end()
            if self.i >= len(text):
                raise _NotJson("unterminated string")
            ch, at = text[self.i], self.i
            if ch == '"':
                self.i += 1
                break
            if ch != "\\":
                raise _NotJson("control character in string")
            kind = text[self.i + 1 : self.i + 2]
            if kind in _SHORT_ESCAPES:
                out.append(_SHORT_ESCAPES[kind])
                self.i += 2
            elif kind == "u":
                code = self.hex4(self.i + 2)
                self.i += 6
                if 0xD800 <= code < 0xDC00 and text.startswith("\\u", self.i):
                    low = self.hex4(self.i + 2)
                    if 0xDC00 <= low < 0xE000:
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        self.i += 6
                out.append(chr(code))
            else:
                raise _NotJson("bad escape")
            segments.append((decoded, at, self.i - at))
            decoded += 1
        return _Value(start, self.i, "".join(out), segments=segments)

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


def json_escape(text: str) -> str:
    """Text as written inside a JSON string: the minimal escapes JSON.stringify
    writes (lowercase ``\\u00xx`` for other controls and lone surrogates), and
    nothing else."""
    out = []
    for ch in text:
        code = ord(ch)
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif code < 0x20 or 0xD800 <= code < 0xE000:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


#: An edit on a text: replace text[start:end] by the string.
_Edit = tuple[int, int, str]


def _literal(mime: str | None, byte_size: int | None) -> str:
    return '"' + json_escape(replacement(mime, byte_size)) + '"'


def _strip_text(text: str, depth: int) -> list[_Edit]:
    """The edits on one string (an attribute, or a JSON string's unescaped
    value inside ``depth`` arrays and objects): the JSON walk when it is a
    candidate that parses, else rule (b) alone."""
    if is_replacement(text):
        return []
    if _is_json_candidate(text):
        document = _parse(text, depth)
        if document is not None:
            edits: list[_Edit] = []
            _walk(document, None, None, depth, edits)
            return edits
    edits = []
    for match in DATA_URL_PATTERN.finditer(text):
        mime = match.group(1) if is_mime(match.group(1)) else None
        edits.append((match.start(), match.end(), replacement(mime, base64_size(match.group(3)))))
    return edits


def _walk(
    node: _Value,
    parent_key: str | None,
    holder: Members | None,
    depth: int,
    edits: list[_Edit],
) -> None:
    """Rules (d), then (e), then (b), on one value. ``holder`` is the nearest
    enclosing object (through arrays), for rule (d)'s mime type."""
    if node.members is not None:
        members = [(key, child.value) for key, child in node.members]
        data = _buffer_data(members)
        if data is not None:
            mime = _mime_key(BUFFER_MIME_KEYS, holder or [])
            edits.append((node.start, node.end, _literal(mime, len(data))))
            return
        keyed = _keyed_positions(members, parent_key)
        mime = _keyed_mime(members, parent_key) if keyed else None
        for i, (key, child) in enumerate(node.members):
            if i not in keyed:
                _walk(child, key, members, depth + 1, edits)
            elif child.items is None:
                _keyed_edit(child, mime, edits)
            else:
                # A string array under the key: each string element is a token.
                for item in child.items:
                    if _keyed_string(item.value):
                        _keyed_edit(item, mime, edits)
                    else:
                        _walk(item, key, members, depth + 2, edits)
    elif node.items is not None:
        for child in node.items:
            _walk(child, parent_key, holder, depth + 1, edits)
    elif node.segments is not None:
        # A JSON string: strip its unescaped value, then map each edit back to
        # the raw characters and write it escaped for this level.
        for start, end, text in _strip_text(node.value, depth):
            raw_start = node.raw_span(start)[0]
            raw_end = node.raw_span(end - 1)[1] if end > start else raw_start
            edits.append((raw_start, raw_end, json_escape(text)))


def _keyed_edit(node: _Value, mime: str | None, edits: list[_Edit]) -> None:
    """Rule (e) on one string literal: replaced whole, quotes included."""
    edits.append((node.start, node.end, _literal(mime, base64_size(node.value))))


#: Text every token holds, written out, when no ``\u`` escape could hide it:
#: rule (b)'s literal ``;base64,``, rule (d)'s ``Buffer`` and rule (e)'s keys
#: (``blob`` for the content of a blob part). The other JSON escapes write
#: none of these characters, so a string holding none of them, and no
#: ``\u``, has no token. Most attributes are such strings, and skip the parse.
_TOKEN_TEXT = (";base64,", "Buffer", "data", "b64_json", "result", "blob")


def _may_hold_media(text: str) -> bool:
    return "\\u" in text or any(needle in text for needle in _TOKEN_TEXT)


def strip_raw_payload(text: str) -> str:
    """``text`` with every media token replaced in place.

    Every other character is unchanged, so the result is byte-identical to the
    reference's for the same input, and stripping twice changes nothing. Never
    raises: the parser's depth limit bounds every recursion.
    """
    if not _may_hold_media(text):
        return text
    edits = _strip_text(text, 0)
    if not edits:
        return text
    out, last = [], 0
    for start, end, new in edits:
        out += [text[last:start], new]
        last = end
    out.append(text[last:])
    return "".join(out)


# --- parts -------------------------------------------------------------------
#
# What a provider's content block becomes, row by row of the mapping table.
# Each part is built in the field order of the contract's part shapes:
#
#   blob  {"type","modality","mime_type"?,"byte_size"?,"stripped_reason"}
#   uri   {"type","modality","mime_type"?,"uri"}
#   file  {"type","modality","mime_type"?,"file_id"}

Part = dict[str, Any]


def modality(mime: str | None, kind: str) -> str:
    """The modality: from the MIME type when it is known, else the part kind.

    image/*, audio/* and video/* are their own; any other known type (PDF,
    text/*, office formats) is a document. A non-image MIME type is never an
    image, whatever the provider called the part.
    """
    if mime is None:
        return kind
    top = mime.split("/", 1)[0].lower()
    return top if top in ("image", "audio", "video") else "document"


def _mime_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and is_mime(value) else None


def blob_part(mime: str | None, byte_size: int | None, reason: str, kind: str) -> Part:
    part: Part = {"type": "blob", "modality": modality(mime, kind)}
    if mime is not None:
        part["mime_type"] = mime
    if byte_size is not None:
        part["byte_size"] = byte_size
    part["stripped_reason"] = reason
    return part


def _reference_part(part_type: str, field: str, value: str, mime: str | None, kind: str) -> Part:
    part: Part = {"type": part_type, "modality": modality(mime, kind)}
    if mime is not None:
        part["mime_type"] = mime
    part[field] = value
    return part


def uri_part(uri: str, mime: str | None, kind: str) -> Part:
    return _reference_part("uri", "uri", uri, mime, kind)


def file_part(file_id: str, mime: str | None, kind: str) -> Part:
    return _reference_part("file", "file_id", file_id, mime, kind)


def base64_blob(data: Any, mime: str | None, kind: str) -> Part:
    """Inline base64 bytes: sized when the bytes are valid base64."""
    if not isinstance(data, str) or not data:
        return blob_part(mime, None, REDACTED_UPSTREAM, kind)
    return blob_part(mime, base64_size(data), NOT_CAPTURED, kind)


def is_data_url(value: Any) -> bool:
    return isinstance(value, str) and value[:5].lower() == "data:"


def data_url_blob(url: str, kind: str, mime: str | None = None) -> Part:
    """A data URL as a blob: never a uri.

    The URL's own media type and the size of its base64 run. A data URL with
    no base64 bytes to measure (OpenInference's ``data:None;url,None``, an
    empty run) is a blob the instrumentor already emptied: redacted_upstream,
    with no size. ``mime`` is the provider's own type, used when the URL names
    none.
    """
    match = DATA_URL_PATTERN.match(url)
    if match is None:
        head = url[5:].split(",", 1)[0].split(";", 1)[0]
        return blob_part(_mime_or_none(head) or mime, None, REDACTED_UPSTREAM, kind)
    url_mime = _mime_or_none(match.group(1)) or mime
    if not match.group(3):
        return blob_part(url_mime, None, REDACTED_UPSTREAM, kind)
    return blob_part(url_mime, base64_size(match.group(3)), NOT_CAPTURED, kind)


#: What OpenInference writes in place of a data URL over its size limit.
OPENINFERENCE_REDACTED = "__REDACTED__"


def url_part(url: Any, kind: str, mime: str | None = None) -> Part:
    """A URL a provider references media by: a data URL is a blob, any other
    a uri. No URL at all, or OpenInference's redaction marker, is media the
    instrumentor dropped."""
    if not isinstance(url, str) or not url or url == OPENINFERENCE_REDACTED:
        return blob_part(mime, None, REDACTED_UPSTREAM, kind)
    if is_data_url(url):
        return data_url_blob(url, kind, mime)
    return uri_part(url, mime, kind)


#: Filename extensions that name a non-document modality, for an OpenAI file
#: whose data URL gives no MIME type. Anything else is a document.
_EXTENSION_MODALITY = {
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "webp": "image",
    "mp3": "audio",
    "wav": "audio",
    "m4a": "audio",
    "flac": "audio",
    "ogg": "audio",
    "mp4": "video",
    "mov": "video",
}


def _filename_kind(filename: Any) -> str:
    if not isinstance(filename, str) or "." not in filename:
        return "document"
    return _EXTENSION_MODALITY.get(filename.rsplit(".", 1)[1].lower(), "document")


def _file_data(data: Any, filename: Any) -> Part:
    """OpenAI file data: a data URL, or bare base64."""
    kind = _filename_kind(filename)
    if is_data_url(data):
        return data_url_blob(data, kind)
    return base64_blob(data, None, kind)


def _anthropic_source(block: Part, kind: str, text: Callable[[Any], list[Part]]) -> list[Part]:
    source = block.get("source")
    if not isinstance(source, dict):
        return [blob_part(None, None, REDACTED_UPSTREAM, kind)]
    mime = _mime_or_none(source.get("media_type"))
    source_type = source.get("type")
    if source_type == "base64":
        return [base64_blob(source.get("data"), mime, kind)]
    if source_type == "url" and isinstance(source.get("url"), str):
        return [url_part(source["url"], kind, mime)]
    if source_type == "file" and isinstance(source.get("file_id"), str):
        return [file_part(source["file_id"], mime, kind)]
    if kind == "document" and source_type in ("text", "content"):
        # Text to the model, so text parts.
        if source_type == "text":
            return text(source.get("data"))
        content = source.get("content")
        if isinstance(content, str):
            return text(content)
        if isinstance(content, list):
            return [part for item in content for part in text(item)]
    return [blob_part(mime, None, REDACTED_UPSTREAM, kind)]


def media_parts(item: Any, text: Callable[[Any], list[Part]]) -> list[Part] | None:
    """The semconv parts a provider content block becomes, or None when it is
    not media (the caller then keeps it as it would any other block).

    ``text`` turns what is text to the model into text parts: a string, or a
    content block (an Anthropic content-source document holds them). Covers
    every passthrough row of the contract's mapping table, plus the native
    semconv media parts (see ``native_part``).
    """
    if not isinstance(item, dict):
        return None
    block_type = item.get("type")
    # OpenAI chat
    if block_type == "image_url":
        image = item.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        return [url_part(url, "image")]
    if block_type == "input_audio":
        audio = item.get("input_audio")
        if not isinstance(audio, dict):
            return [blob_part(None, None, REDACTED_UPSTREAM, "audio")]
        return [base64_blob(audio.get("data"), _format_mime("audio", audio), "audio")]
    if block_type == "file" and isinstance(item.get("file"), dict):
        file = item["file"]
        if isinstance(file.get("file_id"), str):
            return [file_part(file["file_id"], None, _filename_kind(file.get("filename")))]
        return [_file_data(file.get("file_data"), file.get("filename"))]
    # OpenAI Responses
    if block_type == "input_image":
        if isinstance(item.get("file_id"), str) and not item.get("image_url"):
            return [file_part(item["file_id"], None, "image")]
        return [url_part(item.get("image_url"), "image")]
    if block_type == "input_file":
        kind = _filename_kind(item.get("filename"))
        if isinstance(item.get("file_id"), str):
            return [file_part(item["file_id"], None, kind)]
        if item.get("file_data") is not None:
            return [_file_data(item.get("file_data"), item.get("filename"))]
        return [url_part(item.get("file_url"), kind)]
    if block_type == "image_generation_call":
        mime = _format_mime("image", item, "output_format")
        return [base64_blob(item.get("result"), mime, "image")]
    # Anthropic messages
    if block_type == "image":
        return _anthropic_source(item, "image", text)
    if block_type == "document":
        return _anthropic_source(item, "document", text)
    return native_media_parts(item)


def native_media_parts(item: Any) -> list[Part] | None:
    """The parts a part of an already-shaped (``{"role", "parts"}``) message
    becomes, or None when it is not media: a semconv media part made
    conformant, or a Gemini ``inline_data`` / ``file_data`` part. Provider
    content blocks are not read here: a parts message speaks semconv (or
    Gemini, whose contents are parts too)."""
    if not isinstance(item, dict):
        return None
    block_type = item.get("type")
    if block_type in ("blob", "uri", "file"):
        native = native_part(item)
        return [item if native is None else native]
    if block_type is not None:
        return None
    # Gemini generate_content: untyped parts, in either spelling.
    inline = item.get("inline_data", item.get("inlineData"))
    if isinstance(inline, dict):
        mime = _mime_or_none(inline.get("mime_type", inline.get("mimeType")))
        return [base64_blob(inline.get("data"), mime, "document")]
    file_data = item.get("file_data", item.get("fileData"))
    if isinstance(file_data, dict):
        mime = _mime_or_none(file_data.get("mime_type", file_data.get("mimeType")))
        uri = file_data.get("file_uri", file_data.get("fileUri"))
        return [url_part(uri, "document", mime)]
    return None


def _format_mime(top: str, holder: Part, key: str = "format") -> str | None:
    """``<top>/<format>`` from a format field, when that is a valid mime."""
    fmt = holder.get(key)
    return _mime_or_none(f"{top}/{fmt}") if isinstance(fmt, str) and fmt else None


def native_part(part: Part) -> Part | None:
    """A semconv media part made conformant, or None when it already is one.

    * A blob loses its content: sized from it, ``not_captured``. A blob that
      is already stripped keeps its size and reason, so normalizing twice
      changes nothing.
    * The modality comes from the MIME type when that is known (some
      producers label every file an image), else stays as written. A part
      with neither gets none: there is nothing to tell it from.
    * A uri holding a data URL is a blob. A uri or file part is otherwise
      kept as written, its modality filled (or corrected) from the MIME type.
    """
    part_type = part.get("type")
    mime = _mime_or_none(part.get("mime_type"))
    written = part.get("modality")
    kind = written if isinstance(written, str) and written in MODALITIES else "document"
    if part_type == "blob":
        if "content" in part:
            fixed = base64_blob(part["content"], mime, kind)
        else:
            size = part.get("byte_size")
            reason = part.get("stripped_reason")
            fixed = blob_part(
                mime,
                size if isinstance(size, int) and not isinstance(size, bool) else None,
                reason if reason in STRIPPED_REASONS else REDACTED_UPSTREAM,
                kind,
            )
        if mime is None and kind != written:
            del fixed["modality"]
    elif part_type == "uri" and is_data_url(part.get("uri")):
        fixed = data_url_blob(part["uri"], kind, mime)
    elif part_type in ("uri", "file") and mime is not None:
        fixed = dict(part)
        fixed["modality"] = modality(mime, kind)
    else:
        return None
    return None if fixed == part and list(fixed) == list(part) else fixed


def has_media_marker(text: str) -> bool:
    """Whether a messages value could hold a part ``native_media_parts``
    rewrites: every such part names its type, or its Gemini field, in the
    JSON. A cheap test before parsing one."""
    return any(marker in text for marker in _MEDIA_MARKERS)


_MEDIA_MARKERS = ('"blob"', '"uri"', '"file"', "inline_data", "inlineData", "file_data", "fileData")
