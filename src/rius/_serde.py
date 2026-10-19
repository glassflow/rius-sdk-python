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
"""

from __future__ import annotations

import json
import re
from typing import Any

from ._media import replacement, strip_raw_payload

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

# One JSON encoding for every producer of a JSON-valued attribute (this SDK,
# the TypeScript SDK and the sink): compact separators, raw UTF-8 rather than
# \uXXXX escapes, and no HTML escaping of < > & (Python never does that).
# It is what TypeScript's JSON.stringify writes, so the same value produces the
# same bytes whichever SDK recorded it.
_encoder = json.JSONEncoder(default=repr, separators=(",", ":"), ensure_ascii=False)

# ensure_ascii=False writes a lone surrogate raw, and a string holding one
# cannot be encoded as UTF-8: the OTLP exporter drops such an attribute. The
# escaping ensure_ascii used to do for it is kept here, which is also what
# JSON.stringify does with a lone surrogate.
_SURROGATE = re.compile("[\ud800-\udfff]")


def _escape_surrogate(match: re.Match[str]) -> str:
    return f"\\u{ord(match.group()):04x}"


def truncate(text: str) -> str:
    """``text`` with its media stripped, cut at the attribute cap, the cut marked."""
    if len(text) > MAX_STRIP_CHARS:
        return _cut_past_budget(text)
    return _cut(strip_raw_payload(text))


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
        return truncate(_SURROGATE.sub("\ufffd", value))
    return serialize(value)


def serialize(value: Any) -> str:
    """Serialize a value to a bounded string suitable for a span attribute."""
    try:
        return _bounded_dumps(value)
    except Exception:
        # default=repr can itself raise (broken __repr__, detached ORM proxies);
        # tracing must never crash the host over an unserializable value.
        try:
            return truncate(repr(value))
        except Exception:
            return f"<unserializable {type(value).__name__}>"


def _bounded_dumps(value: Any) -> str:
    # iterencode without _one_shot takes the pure-Python path, which yields
    # chunks lazily as it walks the value; the C encoder renders everything
    # first. A small payload costs a few extra microseconds this way; a large
    # one stops being walked at the strip budget instead of being rendered
    # whole.
    chunks: list[str] = []
    size = 0
    for chunk in _encoder.iterencode(value, _one_shot=False):
        chunk = _SURROGATE.sub(_escape_surrogate, chunk)
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_STRIP_CHARS:
            return _cut_past_budget("".join(chunks))
    return truncate("".join(chunks))
