"""The fast base64 sizing and the bounded encoding, against frozen copies of
the straightforward versions they replaced.

Both are rewritten for speed on multi-MB media, and both must give exactly what
the plain versions gave, for every input: the strip's output is byte-identical
across SDKs, so a size or a cut that moved by one character would break it.
Each test runs the frozen copy and the live code over thousands of generated
inputs (wrapped, URL-safe, mixed-alphabet, padded and invalid base64, lone
surrogates, values on both sides of the budgets) and compares the results.
"""

from __future__ import annotations

import base64
import json
import random
from typing import Any

import pytest

from rius import _media, _serde
from rius._media import DATA_URL_PATTERN, base64_size, data_url_size, split_media
from rius._serde import _bounded_dumps

# --- frozen copies -----------------------------------------------------------

_STANDARD = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_STANDARD_EXTRA = frozenset("+/")
_URL_SAFE_EXTRA = frozenset("-_")
_ALL = _STANDARD | _STANDARD_EXTRA | _URL_SAFE_EXTRA
_WHITESPACE_GONE = str.maketrans("", "", " \t\n\r")


def plain_base64_size(value: str) -> int | None:
    text = value.translate(_WHITESPACE_GONE)
    body = text.rstrip("=")
    padding = len(text) - len(body)
    chars = set(body)
    if padding > 2 or len(text) % 4 or not chars <= _ALL:
        return None
    if chars & _STANDARD_EXTRA and chars & _URL_SAFE_EXTRA:
        return None
    return len(text) * 3 // 4 - padding


def plain_bounded_dumps(value: Any) -> str:
    chunks: list[str] = []
    size = 0
    for chunk in _serde._encoder.iterencode(value, _one_shot=False):
        chunk = _serde._SURROGATE.sub(_serde._escape_surrogate, chunk)
        chunks.append(chunk)
        size += len(chunk)
        if size > _serde.MAX_STRIP_CHARS:
            return _serde._cut_past_budget("".join(chunks))
    return _serde.truncate("".join(chunks))


# --- generated inputs ----------------------------------------------------------

_WS = (" ", "\n", "\t", "\r", "\r\n")
_ODD = ("é", "\ud800", "\udfff", "!", "=", "*", "\x00", "😀", "\\", '"')
_MIMES = ("image/png", "audio/wav", "", "x", "IMAGE/JPEG", "a/b;c", "é/x")


def _base64(rng: random.Random) -> str:
    raw = rng.randbytes(rng.randint(0, 40))
    encode = base64.urlsafe_b64encode if rng.random() < 0.4 else base64.b64encode
    text = encode(raw).decode()
    r = rng.random()
    if r < 0.15:
        text = text.rstrip("=")
    elif r < 0.22:
        text += "=" * rng.randint(1, 3)
    if rng.random() < 0.2 and text:
        step, sep = rng.choice((3, 4, 64, 76)), rng.choice(_WS)
        text = sep.join(text[i : i + step] for i in range(0, len(text), step))
        if rng.random() < 0.3:
            text = rng.choice(_WS) + text + rng.choice(_WS)
    if rng.random() < 0.15 and text:
        i = rng.randrange(len(text))
        text = text[:i] + rng.choice("+/-_") + text[i + 1 :]
    if rng.random() < 0.12:
        i = rng.randrange(len(text) + 1)
        text = text[:i] + rng.choice(_ODD) + text[i:]
    if rng.random() < 0.05:
        text = rng.choice(("", "=", "==", "===", " ", "\n\n", "A", "AB=", "A===", "-_+/"))
    return text


def _data_url(rng: random.Random) -> str:
    scheme = rng.choice(("data:", "DATA:", "Data:"))
    params = rng.choice(("", ";charset=utf-8", ";x;y"))
    return f"{scheme}{rng.choice(_MIMES)}{params};base64,{_base64(rng)}"


def _keyed_json(rng: random.Random) -> str:
    data = _base64(rng)
    doc: Any = rng.choice(
        (
            {"source": {"type": "base64", "media_type": rng.choice(_MIMES), "data": data}},
            {"inline_data": {"mime_type": "image/png", "data": data}},
            {"input_audio": {"format": "wav", "data": data}},
            {"b64_json": data, "output_format": "png"},
            {"type": "image", "base64": data, "mime_type": "image/png"},
            {"type": "blob", "content": data, "modality": "image"},
            {"type": "Buffer", "data": [rng.randint(0, 255) for _ in range(rng.randint(0, 8))]},
            {"type": "image_url", "image_url": {"url": _data_url(rng)}},
            {"type": "image", "image": rng.choice((data, _data_url(rng)))},
            {"text": "see " + _data_url(rng)},
        )
    )
    text = json.dumps(doc, ensure_ascii=rng.random() < 0.3)
    if rng.random() < 0.15:
        text = text.replace("/", "\\/")
    if rng.random() < 0.05:
        text = text[: rng.randrange(len(text) + 1)]
    return text


def _payload(rng: random.Random) -> str:
    if rng.random() < 0.35:
        return _keyed_json(rng)
    pieces = [rng.choice(("see ", "", "\ud800 ", "é ", "[stripped image/png 3 bytes] "))]
    for _ in range(rng.randint(0, 3)):
        pieces += [rng.choice((_data_url, _keyed_json, _base64))(rng), rng.choice((" ", "\n", ""))]
    return "".join(pieces)


def _surrogate_text(rng: random.Random, n: int) -> str:
    chars = list("x" * n)
    cuts = [_serde.MAX_ATTR_CHARS + d for d in range(-6, 7)]
    spots = [rng.randrange(n) for _ in range(max(1, n // 50))] if n else []
    for i in spots + [c for c in cuts if 0 <= c < n and rng.random() < 0.3]:
        chars[i] = rng.choice(("\ud800", "\udbff", "\udc00", "\udfff", "é", "😀"))
    return "".join(chars)


def _value(rng: random.Random) -> Any:
    budget = _serde.MAX_STRIP_CHARS
    near = budget + rng.randint(-40, 40)
    return rng.choice(
        (
            _surrogate_text(rng, rng.randint(0, 64)),
            _surrogate_text(rng, near),
            [_surrogate_text(rng, rng.randint(0, budget // 3)) for _ in range(rng.randint(1, 6))],
            {"a": _surrogate_text(rng, near), "b": [1, 2.5, None, True]},
            {_surrogate_text(rng, 8): _surrogate_text(rng, rng.randint(0, budget))},
            [_payload(rng) for _ in range(rng.randint(1, 4))],
            [b"\x00\x01", bytearray(b"ab"), memoryview(b"xyz")],
            # A key the encoder refuses, past the budget or before it.
            ["x" * (near - 10), {(1, 2): "past the budget"}],
            [_surrogate_text(rng, near - 5), {(1, 2): "near the budget"}],
            [{(1, 2): "early"}],
            ["\ud800" * rng.randint(budget // 6 - 5, budget // 6 + 5), "tail"],
        )
    )


def _outcome(fn: Any, value: Any) -> Any:
    """What ``fn`` returns, or the type of what it raises (a lone surrogate
    cannot be encoded as UTF-8, so a split of a text holding one raises)."""
    try:
        return fn(value)
    except Exception as exc:
        return type(exc)


@pytest.fixture
def small_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Budgets small enough that generated values cross them often."""
    monkeypatch.setattr(_serde, "MAX_STRIP_CHARS", 600)
    monkeypatch.setattr(_serde, "MAX_ATTR_CHARS", 64)


# --- base64 sizing -------------------------------------------------------------


def test_base64_size_matches_the_plain_version() -> None:
    rng = random.Random(1)
    for _ in range(5000):
        value = _base64(rng)
        assert base64_size(value) == plain_base64_size(value), repr(value)


def test_a_data_url_size_matches_the_plain_version_of_its_run() -> None:
    rng = random.Random(2)
    for _ in range(5000):
        for match in DATA_URL_PATTERN.finditer("data:x/y;base64," + _base64(rng)):
            run = match.group(3)
            assert data_url_size(run) == plain_base64_size(run), repr(run)


def test_the_strip_and_the_split_match_the_plain_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    # The strip's only change is its sizing: with the frozen copy patched back
    # in for both size functions, the strip is the one it replaced.
    rng = random.Random(3)
    payloads = [_payload(rng) for _ in range(2000)]
    live = [(_media.strip_raw_payload(p), _outcome(split_media, p)) for p in payloads]
    monkeypatch.setattr(_media, "base64_size", plain_base64_size)
    monkeypatch.setattr(_media, "data_url_size", plain_base64_size)
    assert live == [(_media.strip_raw_payload(p), _outcome(split_media, p)) for p in payloads]


# --- the bounded encoding ------------------------------------------------------


@pytest.mark.usefixtures("small_budgets")
def test_bounded_dumps_matches_the_plain_version_across_the_budget() -> None:
    rng = random.Random(5)
    for _ in range(3000):
        value = _value(rng)
        assert _outcome(_bounded_dumps, value) == _outcome(plain_bounded_dumps, value)


def test_bounded_dumps_matches_the_plain_version_at_the_real_budget() -> None:
    rng = random.Random(6)
    for _ in range(15):
        value = _value(rng)
        assert _outcome(_bounded_dumps, value) == _outcome(plain_bounded_dumps, value)
