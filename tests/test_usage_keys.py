"""The normalization table follows the usage-key registry.

``tests/fixtures/semconv/usage_keys.yaml`` is rius-semconv's
``registry/usage_keys.yaml``, synced by scripts/sync-semconv-fixtures.sh. Every
row it says the Python SDK maps must be a rule of ``DEFAULT_TABLE`` with that
target, and no row it says is ignored may be one. The breakdowns the exporter
drops at 0 are exactly the rows it marks ``omit_zero``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rius.normalization import DEFAULT_TABLE, USAGE_BREAKDOWN_SUFFIXES, Rule
from rius.semconv import GEN_AI_USAGE_PREFIX

REGISTRY = Path(__file__).parent / "fixtures" / "semconv" / "usage_keys.yaml"


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = yaml.safe_load(REGISTRY.read_text())["keys"]
    return rows


def _targets() -> dict[str, str]:
    return {
        source: rule.target
        for rule in DEFAULT_TABLE.rules
        if isinstance(rule, Rule)
        for source in rule.sources
    }


def test_every_row_the_python_sdk_maps_is_a_rule_with_its_target() -> None:
    mapped = {
        row["key"]: row["target"]
        for row in _rows()
        if "key" in row and "python" in (row.get("mapped_by") or [])
    }
    assert mapped, "no row names the Python SDK: is usage_keys.yaml synced?"
    targets = _targets()
    assert {key: targets.get(key) for key in mapped} == mapped


def test_no_ignored_row_is_a_rule_source() -> None:
    ignored = {row["key"] for row in _rows() if row.get("decision") == "ignored" and "key" in row}
    assert not ignored & set(_targets())


def test_the_zero_dropped_breakdowns_are_the_rows_marked_omit_zero() -> None:
    omitted = {row["key"] for row in _rows() if row.get("omit_zero") is True}
    assert omitted, "no row is marked omit_zero: is usage_keys.yaml synced?"
    assert {GEN_AI_USAGE_PREFIX + suffix for suffix in USAGE_BREAKDOWN_SUFFIXES} == omitted
