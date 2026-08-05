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
