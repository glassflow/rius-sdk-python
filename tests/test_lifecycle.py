"""init() lifecycle semantics.

Re-init is warn+skip: a second global ``init()`` keeps the existing
configuration and returns the existing client (OTel's global provider is
write-once, so silently building a second pipeline would split traces).
``shutdown()`` releases the slot. Scoped clients (``set_global=False``) are
independent and never auto-enable process-global instrumentors.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init


def test_second_global_init_warns_and_returns_existing_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = init(span_exporter=InMemorySpanExporter(), set_global=True)
    with caplog.at_level("WARNING", logger="rius.client"):
        second = init(span_exporter=InMemorySpanExporter(), set_global=True, service_name="other")
    assert second is first
    assert any("already" in record.message for record in caplog.records)


def test_shutdown_releases_the_slot_for_reinit() -> None:
    first = init(span_exporter=InMemorySpanExporter(), set_global=True)
    first.shutdown()
    second = init(span_exporter=InMemorySpanExporter(), set_global=True)
    assert second is not first


def test_scoped_init_does_not_trip_or_claim_the_guard() -> None:
    scoped = init(span_exporter=InMemorySpanExporter(), set_global=False)
    global_client = init(span_exporter=InMemorySpanExporter(), set_global=True)
    assert global_client is not scoped
    # and a scoped init after a global one is still independent
    another_scoped = init(span_exporter=InMemorySpanExporter(), set_global=False)
    assert another_scoped is not global_client


def test_disabled_init_does_not_claim_the_global_provider_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "rius.client.trace.set_tracer_provider", lambda provider: calls.append(provider)
    )
    init(disabled=True)
    assert calls == []


def test_enabled_global_init_sets_the_global_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "rius.client.trace.set_tracer_provider", lambda provider: calls.append(provider)
    )
    client = init(span_exporter=InMemorySpanExporter(), set_global=True)
    assert calls == [client._provider]
