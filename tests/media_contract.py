"""The media contract's canonical part shape, read from the vendored
``tests/fixtures/semconv/media_parts.yaml`` (rius-semconv's
``registry/media_parts.yaml``).

A normalized media part is canonical when its members are exactly the
contract's, in its order, none null, with a modality and the member its type
cannot do without: a blob's stripped_reason, a uri's uri, a file's file_id. A
bare ``{"type": "file"}`` is not.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

CONTRACT_FILE = Path(__file__).parent / "fixtures" / "semconv" / "media_parts.yaml"


def contract() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(CONTRACT_FILE.read_text())
    return data


MEMBERS: dict[str, list[str]] = contract()["part_shapes"]["members"]
MODALITIES: list[str] = contract()["modalities"]
STRIPPED_REASONS: list[str] = list(contract()["stripped_reasons"])
REQUIRED = {"blob": "stripped_reason", "uri": "uri", "file": "file_id"}

MESSAGE_KEYS = ("gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions")


def canonical_errors(part: dict[str, Any]) -> list[str]:
    """What keeps one media part (type blob, uri or file) from being canonical."""
    members = MEMBERS[part["type"]]
    errors = []
    if list(part) != [m for m in members if m in part]:
        errors.append(f"members {list(part)} are not in the order {members}")
    errors += [f"no {m}" for m in ("modality", REQUIRED[part["type"]]) if m not in part]
    errors += [f"{k} is null" for k, v in part.items() if v is None]
    if "modality" in part and part["modality"] not in MODALITIES:
        errors.append(f"modality {part['modality']!r}")
    if part["type"] == "blob" and part.get("stripped_reason") not in STRIPPED_REASONS:
        errors.append(f"stripped_reason {part.get('stripped_reason')!r}")
    return errors


def media_parts_in(messages: Any) -> Iterator[dict[str, Any]]:
    """Every media part of a message-key value: each message's parts, or the
    system instructions' parts."""
    for item in messages if isinstance(messages, list) else []:
        if not isinstance(item, dict):
            continue
        parts = item.get("parts") if isinstance(item.get("parts"), list) else [item]
        for part in parts:
            if isinstance(part, dict) and part.get("type") in MEMBERS:
                yield part
