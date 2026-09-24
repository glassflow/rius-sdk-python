"""Serialization must never crash the host, whatever the value does."""

from __future__ import annotations

from rius._serde import MAX_ATTR_CHARS, serialize


class _BrokenRepr:
    def __repr__(self) -> str:
        raise RuntimeError("detached ORM session")


def test_broken_repr_does_not_raise() -> None:
    text = serialize(_BrokenRepr())
    assert "unserializable" in text
    assert "_BrokenRepr" in text


def test_broken_repr_inside_container_does_not_raise() -> None:
    text = serialize({"obj": _BrokenRepr()})
    assert "unserializable" in text


def test_truncation_bounds_output() -> None:
    text = serialize("x" * (MAX_ATTR_CHARS * 2))
    assert len(text) <= MAX_ATTR_CHARS + len("…(truncated)")


# --- the 32 KB cap, shared with the TypeScript SDK ---


def test_cap_is_32_kib() -> None:
    assert MAX_ATTR_CHARS == 32 * 1024


def test_under_cap_output_is_plain_json() -> None:
    import json

    value = {"messages": [{"role": "user", "content": "x" * 100}] * 10}
    assert serialize(value) == json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def test_large_payload_is_truncated_with_marker() -> None:
    text = serialize(["y" * 1000] * 1000)  # ~1 MB when serialized
    assert text.endswith("…(truncated)")
    assert len(text) <= MAX_ATTR_CHARS + len("…(truncated)")


def test_truncate_bounds_raw_text() -> None:
    from rius._serde import truncate

    assert truncate("short") == "short"
    long = "z" * (MAX_ATTR_CHARS + 5)
    assert truncate(long) == "z" * MAX_ATTR_CHARS + "…(truncated)"


def test_bounded_encoding_does_not_walk_the_whole_payload() -> None:
    """The cap must bound the WORK, not just the output: a generator that
    would take forever to exhaust is cut off once the cap is reached."""

    def endless():  # noqa: ANN202
        i = 0
        while True:
            yield i
            i += 1

    class Lazy(list):  # json sees a list; iteration is unbounded
        def __iter__(self):  # noqa: ANN204
            return endless()

        def __len__(self) -> int:
            return 1

    text = serialize(Lazy([0]))
    assert text.endswith("…(truncated)")
    assert len(text) <= MAX_ATTR_CHARS + len("…(truncated)")


# --- one JSON encoding for every producer: compact, raw UTF-8, no HTML escaping ---


def test_output_uses_compact_separators() -> None:
    assert serialize({"a": [1, 2], "b": {"c": None}}) == '{"a":[1,2],"b":{"c":null}}'


def test_non_ascii_is_written_raw_not_escaped() -> None:
    assert serialize({"text": "héllo 日本 🙂"}) == '{"text":"héllo 日本 🙂"}'


def test_html_characters_are_not_escaped() -> None:
    assert serialize({"html": "<b>a & b</b>"}) == '{"html":"<b>a & b</b>"}'


def test_control_characters_are_still_escaped() -> None:
    assert serialize('line\nbreak\t"q"') == '"line\\nbreak\\t\\"q\\""'


def test_default_repr_fallback_is_kept() -> None:
    class Thing:
        def __repr__(self) -> str:
            return "<Thing é>"

    assert serialize({"t": Thing()}) == '{"t":"<Thing é>"}'


def test_non_ascii_truncation_counts_characters_not_escapes() -> None:
    """With raw UTF-8 each non-ASCII character is one character of the cap,
    not the six of a \\uXXXX escape, so the cut lands at exactly the cap."""
    text = serialize("é" * (MAX_ATTR_CHARS * 2))
    assert text.endswith("…(truncated)")
    body = text[: -len("…(truncated)")]
    assert len(body) == MAX_ATTR_CHARS
    assert body == '"' + "é" * (MAX_ATTR_CHARS - 1)
    assert "\\u" not in body


def test_non_ascii_under_cap_is_not_truncated() -> None:
    # 20 000 characters: over the cap as \\uXXXX escapes (120 000), under it raw.
    value = "日" * 20_000
    assert serialize(value) == '"' + value + '"'


def test_lone_surrogates_are_escaped_so_the_attribute_stays_valid_utf8() -> None:
    """A lone surrogate (from surrogateescape-decoded bytes, say) cannot be
    encoded as UTF-8, and the OTLP exporter drops an attribute it cannot
    encode. JSON.stringify escapes lone surrogates as \\uXXXX; so does this,
    in keys as well as values."""
    text = serialize({"k\udc80": "a\ud800b"})
    assert text == '{"k\\udc80":"a\\ud800b"}'
    text.encode("utf-8")  # must not raise


def test_escaped_surrogates_count_toward_the_cap_as_written() -> None:
    text = serialize("\udc80" * MAX_ATTR_CHARS)
    body = text[: -len("…(truncated)")]
    assert len(body) == MAX_ATTR_CHARS
    body.encode("utf-8")
