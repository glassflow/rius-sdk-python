"""semconv.py re-exports the vendored rius-semconv module rather than redefining it.

CI's semconv-vendored workflow checks the module is byte-identical to the
pinned rius-semconv release; these tests check the hand-written module serves
it unchanged, so a constant cannot be shadowed by a stale hand-typed copy.
"""

import ast
import re
from pathlib import Path

import rius._semconv_generated as generated
import rius.semconv as semconv


def test_every_generated_name_is_served_by_semconv_unchanged() -> None:
    names = [name for name in vars(generated) if not name.startswith("__")]
    names = [name for name in names if name not in {"annotations", "Enum"}]
    assert names
    for name in names:
        assert getattr(semconv, name) is getattr(generated, name), name


def test_semconv_defines_no_constant_of_its_own() -> None:
    source = Path(semconv.__file__).read_text()
    assert not re.findall(r"^_?[A-Z][A-Z_0-9]*\s*[:=]", source, re.M)


def test_the_vendored_module_names_its_rius_semconv_release() -> None:
    first = Path(generated.__file__).read_text().splitlines()[0]
    assert re.fullmatch(
        r"# Vendored from rius-semconv v\d+\.\d+\.\d+ out/python/_semconv_generated\.py by", first
    )


# Registry keys the SDK once spelled by hand. Each is now imported from
# semconv, so a hand-typed copy cannot drift from the registry.
REGISTRY_KEYS = (
    "SERVICE_NAME",
    "EXCEPTION_MESSAGE",
    "EXCEPTION_STACKTRACE",
    "GEN_AI_SYSTEM_INSTRUCTIONS",
    "LLM_SYSTEM",
    "LLM_PROVIDER",
    "GEN_AI_SYSTEM",
    "LLM_REQUEST_MODEL_NAME",
    "LLM_RESPONSE_MODEL_NAME",
    "TOOL_NAME",
    "LLM_FINISH_REASON",
    "EMBEDDING_MODEL_NAME",
    "METADATA",
    "OUTPUT_MIME_TYPE",
    "GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK",
    "LLM_TOKEN_COUNT_PREFIX",
    "LLM_TOKEN_COUNT_PROMPT",
    "LLM_TOKEN_COUNT_COMPLETION",
    "LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ",
    "LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE",
    "LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO",
    "LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING",
    "LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO",
    "GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS",
    "GEN_AI_USAGE_INPUT_TOKEN_DETAILS_AUDIO_TOKENS",
    "GEN_AI_USAGE_DETAILS_REASONING_TOKENS",
    "GEN_AI_USAGE_DETAILS_THINKING_TOKENS",
    "GEN_AI_USAGE_DETAILS_THOUGHTS_TOKENS",
    "GEN_AI_USAGE_DETAILS_IMAGE_PROMPT_TOKENS",
    "GEN_AI_USAGE_DETAILS_IMAGE_CACHE_TOKENS",
    "GEN_AI_USAGE_DETAILS_IMAGE_CANDIDATES_TOKENS",
    "GEN_AI_USAGE_DETAILS_INPUT_AUDIO_TOKENS",
    "GEN_AI_USAGE_DETAILS_CACHE_AUDIO_READ_TOKENS",
    "GEN_AI_USAGE_DETAILS_OUTPUT_AUDIO_TOKENS",
    "GEN_AI_USAGE_DETAILS_AUDIO_PROMPT_TOKENS",
    "GEN_AI_USAGE_DETAILS_AUDIO_CACHE_TOKENS",
    "GEN_AI_USAGE_DETAILS_AUDIO_CANDIDATES_TOKENS",
    "GEN_AI_USAGE_DETAILS_AUDIO_RESPONSE_TOKENS",
)


def test_registry_keys_are_imported_not_hand_typed() -> None:
    values = {getattr(semconv, name): name for name in REGISTRY_KEYS}
    # Every name must resolve to a key, or the scan below would pass vacuously.
    for value, name in values.items():
        assert isinstance(value, str), name
        assert value, name
    assert len(values) == len(REGISTRY_KEYS)
    package = Path(semconv.__file__).parent
    found = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "_semconv_generated.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and node.value in values:
                found.append(f"{path.name}:{node.lineno} {node.value!r} ({values[node.value]})")
    assert not found, "\n".join(found)
