"""Registry smoke test: every entry must resolve against its real package.

``enable_instrumentations()`` swallows ImportError by design, so a typo'd
module or class name in the registry would ship silently — the extra installs
fine and then does nothing. This test runs in the CI ``extras-smoke`` job with
every extra installed (gated by GLASSFLOW_ALL_EXTRAS so regular runs, which
only carry the openai instrumentor, don't skip-hide a typo as "missing").
"""

from __future__ import annotations

import importlib
import os

import pytest

from rius.instrumentation import REGISTRY, InstrumentorSpec

pytestmark = pytest.mark.skipif(
    not os.getenv("GLASSFLOW_ALL_EXTRAS"),
    reason="requires an --all-extras environment (CI extras-smoke job)",
)


@pytest.mark.parametrize("spec", REGISTRY, ids=lambda spec: spec.name)
def test_registry_entry_resolves(spec: InstrumentorSpec) -> None:
    module = importlib.import_module(spec.module)
    instrumentor = getattr(module, spec.class_name)()
    assert hasattr(instrumentor, "instrument")
    assert hasattr(instrumentor, "uninstrument")
    assert hasattr(instrumentor, "is_instrumented_by_opentelemetry")
