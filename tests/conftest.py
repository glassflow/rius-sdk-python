"""Shared test fixtures.

Installs a global OpenTelemetry provider backed by an in-memory exporter once
(the global provider can only be set once per process), and clears captured
spans between tests. `@observe` and other helpers that use the global tracer are
exercised against this.
"""

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_EXPORTER = InMemorySpanExporter()


def _ensure_global_provider() -> None:
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
        trace.set_tracer_provider(provider)


@pytest.fixture(autouse=True)
def exported_spans() -> InMemorySpanExporter:
    _ensure_global_provider()
    _EXPORTER.clear()
    return _EXPORTER


@pytest.fixture(autouse=True)
def _reset_rius_lifecycle() -> "Iterator[None]":
    """Clear module-level init()/instrumentation state between tests."""
    yield
    from rius import client as client_module
    from rius import instrumentation as instrumentation_module

    client_module._current_client = None
    instrumentation_module._ENABLED.clear()
