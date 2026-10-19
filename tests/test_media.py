"""The raw-payload strip, against the media contract's golden examples.

``tests/fixtures/semconv/media_parts.yaml`` is rius-semconv's
``registry/media_parts.yaml``, synced with the recordings by
scripts/sync-semconv-fixtures.sh. Its golden examples are the reference
implementation's tests and every normalizer's: the strip must turn each input
into exactly its output, byte for byte, and stripping that output again must
change nothing. The rule data the strip inlines is checked against the same
file, so a contract change that this SDK has not followed fails here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rius import _media
from rius._media import base64_size, is_replacement, strip_raw_payload

CONTRACT = Path(__file__).parent / "fixtures" / "semconv" / "media_parts.yaml"


def _contract() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(CONTRACT.read_text())
    return data


RAW = _contract()["raw_payloads"]
GOLDEN = RAW["golden"]


@pytest.mark.parametrize("golden", GOLDEN, ids=[g["name"] for g in GOLDEN])
def test_every_golden_example_strips_to_its_exact_output(golden: dict[str, Any]) -> None:
    assert strip_raw_payload(golden["input"]) == golden["output"]


@pytest.mark.parametrize("golden", GOLDEN, ids=[g["name"] for g in GOLDEN])
def test_stripping_a_golden_output_again_changes_nothing(golden: dict[str, Any]) -> None:
    assert strip_raw_payload(golden["output"]) == golden["output"]


def test_the_golden_examples_are_there() -> None:
    # A sync that lost them would make the two tests above vacuous.
    assert len(GOLDEN) >= 30


def test_the_in_text_example_strips_to_its_output() -> None:
    in_text = RAW["replacement"]["in_text"]
    assert strip_raw_payload(in_text["before"]) == in_text["after"]


def test_the_inlined_rule_data_is_the_contracts() -> None:
    contract = _contract()
    rules = {rule["id"]: rule for rule in RAW["detection"]}
    replacement = RAW["replacement"]
    assert list(_media.MODALITIES) == contract["modalities"]
    assert set(_media.STRIPPED_REASONS) == set(contract["stripped_reasons"])
    assert replacement["format"] == _media.REPLACEMENT_FORMAT
    assert replacement["unknown"] == _media.UNKNOWN
    assert _media.MIME_PATTERN.pattern == replacement["mime_pattern"]
    assert _media.DATA_URL_PATTERN.pattern == rules["b"]["pattern"]
    assert rules["d"]["type"] == _media.BUFFER_TYPE
    assert rules["d"]["array_key"] == _media.BUFFER_ARRAY_KEY
    assert list(_media.BUFFER_MIME_KEYS) == rules["d"]["mime_keys"]
    e = rules["e"]
    assert list(_media.KEYED_KEYS) == e["keys"]
    assert list(_media.KEYED_SIBLING_KEYS) == e["sibling_keys"]
    assert list(_media.KEYED_PARENT_KEYS) == e["parent_keys"]
    assert e["source_key"] == _media.KEYED_SOURCE_KEY
    assert e["source_type"] == _media.KEYED_SOURCE_TYPE
    assert e["blob_type"] == _media.KEYED_BLOB_TYPE
    assert e["blob_key"] == _media.KEYED_BLOB_KEY
    assert list(_media.KEYED_MIME_KEYS) == e["mime_keys"]
    assert [
        (key, mime, list(parents) if parents is not None else None)
        for key, mime, parents in _media.KEYED_MIME_FORMATS
    ] == [(f["key"], f["mime"], f.get("parent_keys")) for f in e["mime_formats"]]


@pytest.mark.parametrize("example", RAW["replacement"]["examples"])
def test_the_replacement_examples_are_complete_replacements(example: str) -> None:
    assert is_replacement(example)
    assert strip_raw_payload(example) == example


def test_no_depth_makes_the_strip_raise() -> None:
    # Far past the 64-level limit and Python's recursion limit: the candidate
    # does not parse, so only the data URL is replaced.
    deep = "[" * 5000 + '"data:image/png;base64,QUFB"' + "]" * 5000
    assert strip_raw_payload(deep) == "[" * 5000 + '"[stripped image/png 3 bytes]"' + "]" * 5000


def test_a_string_with_no_media_comes_back_as_the_same_object() -> None:
    text = '{"messages":[{"role":"user","content":"hi"}]}'
    assert strip_raw_payload(text) is text


@pytest.mark.parametrize(
    ("value", "size"),
    [
        ("QUFB", 3),
        ("QUE=", 2),
        ("QQ==", 1),
        ("Q UFB", 3),
        ("QUFB\n", 3),
        ("QUF", None),
        ("-_-_", 3),
        ("+_+_", None),
        ("QQ===", None),
        ("", 0),
    ],
)
def test_base64_size_follows_the_contract(value: str, size: int | None) -> None:
    assert base64_size(value) == size
