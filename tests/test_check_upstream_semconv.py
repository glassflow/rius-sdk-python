"""scripts/check_upstream_semconv.py checks only the gen_ai.* keys we emit.

The vendored module also names the source spellings the normalizers read from
other producers (legacy and producer-specific gen_ai.* keys no upstream
registry defines). They are never emitted, so upstream drift on them is not
ours to port.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_upstream_semconv.py"
_spec = importlib.util.spec_from_file_location("check_upstream_semconv", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

SAMPLE = """# --- Attribute keys ---
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"

# --- Normalizer source spellings ---
# Keys other producers write that the normalizers read.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_USAGE_DETAILS_THINKING_TOKENS = "gen_ai.usage.details.thinking_tokens"

# --- Coding-agent plugin ---
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
"""


def test_source_spellings_are_not_checked_as_emitted() -> None:
    emitted = check.emitted_attributes(SAMPLE)
    assert emitted == {
        "GEN_AI_REQUEST_MODEL": "gen_ai.request.model",
        "GEN_AI_AGENT_NAME": "gen_ai.agent.name",
    }


def test_without_the_section_heading_every_constant_is_checked() -> None:
    # Fails loud: if the heading ever changes, nothing is silently skipped.
    text = SAMPLE.replace("# --- Normalizer source spellings ---", "# --- Renamed ---")
    assert "GEN_AI_SYSTEM" in check.emitted_attributes(text)


def test_the_vendored_module_emits_no_source_spelling() -> None:
    emitted = set(check.emitted_attributes().values())
    assert "gen_ai.request.model" in emitted
    assert "gen_ai.system_instructions" in emitted
    assert "gen_ai.system" not in emitted
    assert "gen_ai.usage.details.reasoning_tokens" not in emitted
