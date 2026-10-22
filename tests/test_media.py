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
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from rius import _media
from rius._media import base64_size, is_replacement, split_media, strip_raw_payload

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
    assert _media.ESCAPED_DATA_URL_PATTERN.pattern == rules["b"]["escaped_pattern"]
    assert _media.PERCENT_SCHEME.pattern == rules["b"]["percent_scheme"]
    assert _media.PERCENT_DATA_URL_PATTERN.pattern == rules["b"]["percent_pattern"]
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
    ("text", "expected"),
    [
        ("plain prose", False),
        # Only the data URL word, only the \u escape, and both together.
        ("see data:image/png;base64,QUFB", True),
        ('{"d\\u0061ta":"QUJD"}', True),
        ("data:image/png;base64,QUFB and \\u0041", True),
        # A rule's words with no \u and no data URL still pass the check.
        ('{"type":"Buffer","data":[1,2]}', True),
        ('{"source":{"data":"QUJD"}}', True),
        ('{"hello":"world"}', False),
        # A percent-encoded data URL holds no ;base64, but always base64%2.
        ("see data%3Aimage%2Fpng%3Bbase64%2CQUFB", True),
        ("see data%3Aimage%2Fpng%3Bbase64%2cQUFB", True),
    ],
)
def test_the_pre_check_answers_the_same_whichever_test_runs_first(
    text: str, expected: bool
) -> None:
    assert _media._may_hold_media(text) is expected


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


# --- the final line break, escaped wraps and percent-encoded data URLs -------


def _tokens(text: str, escaped: bool = False) -> list[tuple[str, str | None, int | None]]:
    """Rule (b)'s tokens in a text: each one's text, mime type and size."""
    return [
        (match.group(0), *_media._match_token(match))
        for match in _media._data_url_matches(text, escaped=escaped)
    ]


@pytest.mark.parametrize(
    ("text", "run"),
    [
        ("data:a/b;base64,QUFB\\nQkJD", "QUFB\\nQkJD"),
        ("data:a/b;base64,QUFB\\r\\nQkJD", "QUFB\\r\\nQkJD"),
        ("data:a/b;base64,QUFB\\n> QkJD=", "QUFB\\n> QkJD="),
        ("data:a/b;base64,QUFB\\n\\nQkJD", "QUFB"),
        ("data:a/b;base64,QUFB\\rQkJD", "QUFB"),
        ("data:a/b;base64,QUFB\\tQkJD", "QUFB"),
        ("data:a/b;base64,QUE=\\nQkJD", "QUE="),
        ("data:a/b;base64,QUFB\\n", "QUFB"),
        ("data:a/b;base64,QUFB\nQkJD\\nQ0ND", "QUFB\nQkJD\\nQ0ND"),
        ("data:a/b;base64,QUFB\\nQkJD\nQ0ND", "QUFB\\nQkJD"),
        ("data:a/b;base64," + "A" * 77 + "\\nQkJD", "A" * 77),
        ("data:a/b;base64,QUFB\\ndata:a/b;base64,QkJD", "QUFB"),
    ],
)
def test_an_escaped_run_goes_on_over_an_escaped_line_break(text: str, run: str) -> None:
    assert _tokens(text, escaped=True)[0][0] == "data:a/b;base64," + run


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ('{"u":"data:a/b;base64,QUFB\\nQkJD', 6),
        ('{"u":"data:a/b;base64,QUFB\\r\\n> QkJD', 6),
        ('{"u":"data:a/b;base64,QUFB\\nQkJD"}', 6),
    ],
)
def test_only_json_that_does_not_parse_takes_an_escaped_line_break(text: str, size: int) -> None:
    [token] = split_media(text)[1]
    assert token.byte_size == size
    # The same text outside a candidate keeps to the raw pattern.
    [token] = split_media("x " + text)[1]
    assert token.byte_size == 3


@pytest.mark.parametrize(
    ("text", "mime", "size"),
    [
        ("data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D", "image/png", 8),
        ("DATA%3aimage%2fpng%3bbase64%2civBORw0KGgo%3d", "image/png", 8),
        ("data%3Aimage/png%3Bbase64%2C%2B%2B%2B%2B////AA%3D%3D", "image/png", 7),
        ("data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CQUFB", "image/svg+xml", 3),
        ("data%3Aimage%2Fpng%3Bname%3Da.png%3Bbase64%2CQUFB", "image/png", 3),
        ("data%3A%3Bbase64%2CQUFB", None, 3),
        ("data%3Aimage%3Bbase64%2CQUFB", None, 3),
        ("data%3Aa%2Fb%2Fc%3Bbase64%2CQUFB", None, 3),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%3D%3D%3D", "a/b", None),
        ("data%3Aa%2Fb%3Bbase64%2C", "a/b", 0),
        # The line breaks a payload was wrapped at before it was encoded.
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%0AQkJD", "a/b", 6),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%0d%0aQkJD", "a/b", 6),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%0BQkJD", "a/b", None),
    ],
)
def test_a_percent_encoded_data_url_is_one_token(text: str, mime: str | None, size: int) -> None:
    end = len(text) - 3 if text.endswith("%3D%3D%3D") else len(text)
    assert _tokens(text) == [(text[:end], mime, size)]


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("x data%3Aa%2Fb%3Bbase64%2CQUFB%0AQkJD", "data%3Aa%2Fb%3Bbase64%2CQUFB%0AQkJD"),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%0d%0aQkJD%0A ", "data%3Aa%2Fb%3Bbase64%2CQUFB%0d%0aQkJD%0A"),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%3D%0AQkJD", "data%3Aa%2Fb%3Bbase64%2CQUFB%3D"),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%20QkJD", "data%3Aa%2Fb%3Bbase64%2CQUFB"),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%0CQkJD", "data%3Aa%2Fb%3Bbase64%2CQUFB"),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB%2C", "data%3Aa%2Fb%3Bbase64%2CQUFB"),
        ("data%3Aa%2Fb%3Bbase64%2CQUFB&w=1", "data%3Aa%2Fb%3Bbase64%2CQUFB"),
        (
            "data%3Aa%2Fb%3Bbase64%2CQUFBdata%3Aa%2Fb%3Bbase64%2CQkJD",
            "data%3Aa%2Fb%3Bbase64%2CQUFB",
        ),
        ("data%3Aa%2Fb%3Bbase64%2CQUFBdata:a/b;base64,QkJD", "data%3Aa%2Fb%3Bbase64%2CQUFB"),
        ("data%3Aa%2Cb%3Bbase64%2CQUFB", None),
        ("data%3Aa b%3Bbase64%2CQUFB", None),
        ("data%253Aa%3Bbase64%2CQUFB", None),
        ("data%3Aa%3Bbase64%2cQUFB", "data%3Aa%3Bbase64%2cQUFB"),
        ("data%3Aa%3BBASE64%2CQUFB", None),
        # The uniform cut: a plain header holding data%3A is no token.
        ("data:xdata%3A;base64,QUFB", None),
    ],
)
def test_a_percent_encoded_data_url_ends_where_its_run_does(text: str, token: str | None) -> None:
    assert [t[0] for t in _tokens(text)][:1] == ([token] if token else [])


@pytest.mark.parametrize(
    ("tail", "kept"),
    [
        ("", True),
        ("\n", True),
        ("\r\n", True),
        ("\n\n", False),
        ("\r", False),
        (" \n", False),
        ("\n ", False),
        ("\n\r\n", False),
    ],
)
def test_a_url_key_takes_one_final_line_break(tail: str, kept: bool) -> None:
    value = "data:image/png;base64,iVBORw0KGgo=" + tail
    assert (_media.whole_data_url(value, line_break=True) is not None) == kept
    assert _media._kept_url(value) == kept
    assert (_media.whole_data_url(value) is not None) == (tail == "")


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        ("data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D", True),
        ("data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D\n", True),
        ("data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D ", False),
        ("data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D%0A", True),
        ("data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D%0d%0A", True),
        ("data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D%0A%0A", False),
        ("data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D%0A\n", False),
        ("data:image/png;base64,iVBORw0KGgo=%0A", False),
        ("https://x/?u=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D", True),
        (" data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgo%3D", False),
    ],
)
def test_a_url_key_keeps_a_whole_percent_encoded_data_url_for_rule_b(
    value: str, kept: bool
) -> None:
    assert _media._kept_url(value) == kept


@pytest.mark.parametrize(
    ("value", "stripped"),
    [
        (
            '"data:image/png;base64,iVBORw0KGgo=\\n"',
            '"[stripped image/png 8 bytes]"',
        ),
        (
            '"data:image/png;base64,iVBO\\r\\nRw0KGgo=\\r\\n"',
            '"[stripped image/png 8 bytes]"',
        ),
        (
            '["data:image/png;base64,iVBORw0KGgo=\\n"]',
            '["[stripped image/png 8 bytes]"]',
        ),
        (
            '"data%3aimage%2fpng%3bbase64%2civBORw0KGgo%3d%0A"',
            '"[stripped image/png 8 bytes]"',
        ),
        # Two breaks: no whole match, so a rule (e) token.
        (
            '"data:image/png;base64,iVBORw0KGgo=\\n\\n"',
            '"[stripped unknown unknown bytes]"',
        ),
    ],
)
def test_a_url_key_value_and_its_final_line_break_are_one_token(value: str, stripped: str) -> None:
    text = '{"type":"image","image":' + value + "}"
    out = strip_raw_payload(text)
    assert out == '{"type":"image","image":' + stripped + "}"
    assert strip_raw_payload(out) == out


#: Inputs one step from a token, each of which a backtracking matcher could
#: take in quadratic time: the strip must stay linear on all of them.
NEAR_MATCHES = {
    # JSON that does not parse, so the escaped pattern is tried.
    "escaped_blank_continuations": '{"a":"data:a/b;base64,A' + "\\n " * 20_000 + "!",
    "escaped_one_character_lines": '{"a":"data:a/b;base64,' + "A\\n" * 20_000 + "!",
    "escaped_lines_at_the_cap": '{"a":"data:a/b;base64,' + ("A" * 76 + "\\r\\n") * 500 + "!",
    "escaped_backslashes": '{"a":"data:a/b;base64,A' + "\\" * 40_000,
    "percent_schemes": "data%3A" * 6_000,
    "percent_headers_with_no_comma": "data%3A" + "%3Bbase64" * 4_000,
    "percent_tokens": "data%3Aa%3Bbase64%2C" * 2_000,
    "percent_escaped_run": "data%3Aa%2Fb%3Bbase64%2C" + "%2B" * 13_000 + "!",
    "percent_line_breaks": "data%3Aa%2Fb%3Bbase64%2C" + "A%0D%0A" * 6_000 + "!",
    "percent_wrapped_tokens": "data%3Aa%2Fb%3Bbase64%2CQUFB%0A" * 1_500,
}
#: The near matches that hold no token; the others are all tokens.
NO_TOKENS = {"percent_schemes", "percent_headers_with_no_comma"}


@pytest.mark.parametrize("name", NEAR_MATCHES)
def test_near_matches_strip_in_linear_time(name: str) -> None:
    text = NEAR_MATCHES[name]
    start = time.perf_counter()
    stripped = strip_raw_payload(text)
    assert time.perf_counter() - start < 0.5
    assert (stripped == text) == (name in NO_TOKENS)
    assert "base64" not in stripped or name in NO_TOKENS
