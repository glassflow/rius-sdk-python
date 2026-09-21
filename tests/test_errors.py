"""``error.type`` spelling shared by every span that records an exception."""

from __future__ import annotations

from rius._errors import error_type


class _Custom(RuntimeError):
    pass


def test_builtin_exceptions_stay_bare() -> None:
    assert error_type(ValueError("x")) == "ValueError"
    assert error_type(KeyError("k")) == "KeyError"


def test_user_exceptions_are_module_qualified() -> None:
    assert error_type(_Custom()) == f"{_Custom.__module__}.{_Custom.__qualname__}"


def test_nested_classes_keep_their_qualname() -> None:
    class Inner(Exception):
        pass

    assert error_type(Inner()).endswith(".Inner")
    assert Inner.__module__ in error_type(Inner())
