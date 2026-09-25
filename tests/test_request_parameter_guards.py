"""One parameter, one shape: the native model_parameters path and the
llm.invocation_parameters rule must agree on every canonical request key.

Both paths route a canonical ``gen_ai.request.*`` key through the same
guard, and both resolve two spellings of one parameter by the same
precedence. These tests pin that the two paths are wired to the same table,
not merely that each behaves on its own.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rius.generation import _request_attributes
from rius.normalization import (
    INVOCATION_PARAMETER_MEMBERS,
    REQUEST_PARAMETER_GUARDS,
    openinference_invocation_parameters,
)
from rius.semconv import GEN_AI_REQUEST_PARAMETERS, GEN_AI_REQUEST_PREFIX


def test_every_canonical_request_key_has_a_guard() -> None:
    assert set(GEN_AI_REQUEST_PARAMETERS.values()) == set(REQUEST_PARAMETER_GUARDS)


def test_the_invocation_parameters_rule_uses_the_same_guards() -> None:
    for member, target, convert in INVOCATION_PARAMETER_MEMBERS:
        assert convert is REQUEST_PARAMETER_GUARDS[target], member


def test_the_native_precedence_agrees_with_the_invocation_parameters_rule() -> None:
    """Where both tables list two spellings of one key, both put them in the
    same order, so a request carrying both keeps the same one either way."""
    native = list(GEN_AI_REQUEST_PARAMETERS)
    members = [(member, target) for member, target, _ in INVOCATION_PARAMETER_MEMBERS]
    for i, (first, target) in enumerate(members):
        for second, other in members[i + 1 :]:
            if other == target:
                assert native.index(first) < native.index(second), (first, second)


def _canonical(attributes: Any) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in attributes.items()
        if key.startswith(GEN_AI_REQUEST_PREFIX)
    }


@pytest.mark.parametrize(
    "bag",
    [
        {"stop": "END"},
        {"stop": ["a", "b"]},
        {"stop": [["a"]]},
        {"temperature": "0.2"},
        {"temperature": 1},
        {"temperature": True},
        {"max_tokens": {"limit": 5}},
        {"max_tokens": 256.0},
        {"seed": "7"},
        {"n": 2},
        {"model": ""},
        {"model": "gpt-4o"},
        {"stream": "yes"},
        {"stream": False},
        {"max_tokens": 100, "max_completion_tokens": 200},
        {"max_completion_tokens": 200, "max_tokens": 100},
        {"max_tokens": "100", "max_completion_tokens": 200},
        {"stop": "A", "stop_sequences": ["B"]},
        {"stop_sequences": ["B"], "stop": "A"},
        {"stop": 3, "stop_sequences": ["B"]},
    ],
)
def test_both_paths_put_the_same_canonical_keys_on_the_wire(bag: dict[str, Any]) -> None:
    native = _canonical(_request_attributes(bag))
    normalized = _canonical(openinference_invocation_parameters(json.dumps(bag)))
    assert native == normalized


def test_a_non_finite_member_stays_in_the_invocation_parameters_bag() -> None:
    """Python's json module reads Infinity and NaN; neither is promoted."""
    produced = openinference_invocation_parameters(
        '{"max_tokens": Infinity, "temperature": NaN, "seed": 1}'
    )
    assert produced["gen_ai.request.seed"] == 1
    assert "gen_ai.request.max_tokens" not in produced
    assert "gen_ai.request.temperature" not in produced


def test_an_int_beyond_a_doubles_range_is_not_a_crash() -> None:
    """float() of it raises; the guard must SKIP rather than fail the call."""
    produced = _request_attributes({"temperature": 10**400, "max_tokens": 10**400})
    assert "gen_ai.request.temperature" not in produced
    assert "rius.request.temperature" in produced
    assert produced["gen_ai.request.max_tokens"] == 10**400
