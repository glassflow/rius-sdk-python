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
    assert serialize(value) == json.dumps(value)


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
