#!/usr/bin/env python3
"""Check every gen_ai.* attribute we emit against the upstream registry.

The GenAI semantic conventions moved out of the versioned, published
open-telemetry/semantic-conventions repo into semantic-conventions-genai,
which cuts NO releases: renames and removals exist only on its main branch.
The published docs site and the old repo's registry lag it or list the old
names as tombstones, so reading the "official" docs is how we shipped
gen_ai.usage.cache_creation.input_tokens two weeks after upstream renamed it
to cache_write. Main is the only watchable surface; this script watches it.

Compares the module-level ``NAME = "gen_ai...."`` constants in
src/rius/semconv.py (the same shape gen-semconv-fixture extraction relies on
in the TypeScript repo) against the attribute names in the upstream registry
markdown. An attribute of ours that upstream no longer lists means a rename,
removal, or re-scope: a WIRE change and a judgement call, so the outcome is a
failing exit for a human, never an auto-fix.

Exits 0 when every emitted attribute is known upstream, 2 on drift, 3 when
the upstream registry cannot be fetched or parses to implausibly few
attributes (a fetch or format break must not masquerade as "no drift").
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

REGISTRY_URL = (
    "https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai"
    "/main/docs/registry/attributes/gen-ai.md"
)
SEMCONV = Path(__file__).resolve().parent.parent / "src" / "rius" / "semconv.py"

# Deliberately not upstream attributes; each entry needs a reason.
RIUS_ONLY = {
    # Rius event name for streaming TTFT, not a registry attribute; the
    # matching upstream concept is the gen_ai.server.time_to_first_token
    # metric (see the comment at its definition).
    "gen_ai.first_token",
}

# If the registry ever parses below this, assume the page format changed and
# fail loudly instead of reporting every attribute as drifted.
MIN_PLAUSIBLE_UPSTREAM = 30


def upstream_attributes() -> set[str]:
    with urllib.request.urlopen(REGISTRY_URL, timeout=30) as resp:
        text = resp.read().decode()
    return set(re.findall(r"`(gen_ai\.[a-z_.0-9]+)`", text))


def emitted_attributes() -> dict[str, str]:
    consts = re.findall(r'^([A-Z][A-Z_0-9]*) = "([^"]+)"$', SEMCONV.read_text(), re.M)
    return {
        name: value
        for name, value in consts
        if value.startswith("gen_ai.")
        # A trailing dot is a key-family prefix (e.g. gen_ai.request.),
        # not an attribute; its members are checked individually where we
        # emit them under known names.
        and not value.endswith(".")
        and value not in RIUS_ONLY
    }


def main() -> int:
    try:
        upstream = upstream_attributes()
    except OSError as err:
        print(f"::error::could not fetch the upstream registry: {err}")
        return 3
    if len(upstream) < MIN_PLAUSIBLE_UPSTREAM:
        print(
            f"::error::parsed only {len(upstream)} attributes from the upstream "
            "registry; the page format likely changed. Fix the parser before "
            "trusting this check again."
        )
        return 3

    drifted = {name: value for name, value in emitted_attributes().items() if value not in upstream}
    if not drifted:
        print(f"all emitted gen_ai.* attributes are in the upstream registry ({REGISTRY_URL})")
        return 0

    print(
        "::error::attributes we emit are no longer in the upstream GenAI "
        "registry — renamed, removed, or re-scoped upstream:"
    )
    for name, value in sorted(drifted.items()):
        print(f"::error::  {name} = {value}")
    print(
        "Resolve by porting the upstream change (check "
        "https://github.com/open-telemetry/semantic-conventions-genai/pulls for "
        "the rename), or add the attribute to RIUS_ONLY with a reason."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
