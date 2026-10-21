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

import json
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
def test_stripping_a_golden_output_again_gives_its_second_output(golden: dict[str, Any]) -> None:
    # Unchanged, but for the one input class the contract names: a candidate
    # that parses only once pass 1 has run, whose second output it lists.
    second = golden.get("second_output", golden["output"])
    assert strip_raw_payload(golden["output"]) == second
    assert strip_raw_payload(second) == second


def test_a_golden_example_holds_a_second_pass() -> None:
    assert any("second_output" in g for g in GOLDEN)


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
    assert [
        {
            "key": t.key,
            "types": list(t.types),
            **({"parent_keys": list(t.parent_keys)} if t.parent_keys is not None else {}),
            **({"holder_types": list(t.holder_types)} if t.holder_types is not None else {}),
            **({"keeps_urls": True} if t.keeps_urls else {}),
            **({"holder_mime": True} if t.holder_mime else {}),
        }
        for t in _media.KEYED_TYPED_KEYS
    ] == e["typed_keys"]
    assert _media.URL_SCHEME.pattern == e["url_scheme"]
    assert [
        {"key": k, "value": v, "keys": list(keys)} for k, v, keys in _media.KEYED_BASE64_MARKERS
    ] == e["base64_markers"]
    assert list(_media.KEYED_SOURCE_BYTES_KEYS) == e["source_bytes_keys"]
    assert list(_media.KEYED_MIME_KEYS) == e["mime_keys"]
    assert [
        (key, mime, list(parents) if parents is not None else None)
        for key, mime, parents in _media.KEYED_MIME_FORMATS
    ] == [(f["key"], f["mime"], f.get("parent_keys")) for f in e["mime_formats"]]


@pytest.mark.parametrize("example", RAW["replacement"]["examples"])
def test_the_replacement_examples_are_complete_replacements(example: str) -> None:
    assert is_replacement(example)
    assert strip_raw_payload(example) == example


#: data: strings rule (b) cannot replace whole, each holding the payload
#: "SECRETPAYLOAD": a wrapped run, percent-encoding, a broken run, an
#: uppercase scheme, parameters and no base64 marker at all.
_BROKEN_DATA_URLS = (
    "data:image/png;base64,U0VDUkVU\nUEFZTE9BRA==SECRETPAYLOAD",
    "data:image/png,SECRETPAYLOAD%00%01",
    "DATA:image/png;base64,U0VD*SECRETPAYLOAD",
    "data:;base64,SECRETPAYLOAD!",
    "data:image/png;name=a b;base64,SECRETPAYLOAD",
    "data:text/plain,SECRETPAYLOAD",
    "data:SECRETPAYLOAD",
)


@pytest.mark.parametrize("url", _BROKEN_DATA_URLS)
@pytest.mark.parametrize(
    "shape",
    [
        '{{"type":"image","image":{value}}}',
        '{{"type":"image","image":{value},"mediaType":"image/png"}}',
        '{{"type":"image","image":{{"type":"data","data":{value}}}}}',
        '{{"type":"file","data":{{"type":"data","data":{value}}},"mediaType":"image/png"}}',
    ],
)
def test_no_payload_survives_a_url_key(url: str, shape: str) -> None:
    text = shape.format(value=json.dumps(url))
    stripped = strip_raw_payload(text)
    assert "SECRETPAYLOAD" not in stripped
    assert strip_raw_payload(stripped) == stripped


def test_a_whole_base64_data_url_under_a_url_key_keeps_its_own_mime() -> None:
    text = '{"type":"image","image":"data:image/jpeg;base64,QUFB","mediaType":"image/png"}'
    assert strip_raw_payload(text) == (
        '{"type":"image","image":"[stripped image/jpeg 3 bytes]","mediaType":"image/png"}'
    )


def test_no_depth_makes_the_strip_raise() -> None:
    # Far past the 64-level limit and Python's recursion limit: the candidate
    # does not parse, so only the data URL is replaced.
    deep = "[" * 5000 + '"data:image/png;base64,QUFB"' + "]" * 5000
    assert strip_raw_payload(deep) == "[" * 5000 + '"[stripped image/png 3 bytes]"' + "]" * 5000


def test_a_string_with_no_media_comes_back_as_the_same_object() -> None:
    text = '{"messages":[{"role":"user","content":"hi"}]}'
    assert strip_raw_payload(text) is text


# The pre-check skips the parse only when no rule's words are written out:
# each case here needs one word of the check, and the reference strips it.
@pytest.mark.parametrize(
    ("text", "stripped"),
    [
        # A key hidden by a \u escape: only the "\u" in the text says to parse.
        (
            '{"mime_type":"image/png","d\\u0061ta":"QUJD"}',
            '{"mime_type":"image/png","d\\u0061ta":"[stripped image/png 3 bytes]"}',
        ),
        # A base64 source: "source" is the only context word written out.
        (
            '{"source":{"type":"base64","data":"QUJD"}}',
            '{"source":{"type":"base64","data":"[stripped unknown 3 bytes]"}}',
        ),
    ],
)
def test_the_pre_check_lets_through_every_media_the_rules_find(text: str, stripped: str) -> None:
    assert strip_raw_payload(text) == stripped


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
