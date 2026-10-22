"""init() lifecycle semantics.

Re-init is warn+skip: a second global ``init()`` keeps the existing
configuration and returns the existing client (OTel's global provider is
write-once, so silently building a second pipeline would split traces).
``shutdown()`` releases the slot. Scoped clients (``set_global=False``) are
independent and never auto-enable process-global instrumentors.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
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


# --- re-init must carry the SDK's own helpers along, not just instrumentors ---


def _helper_spans(exporter: InMemorySpanExporter, client) -> set[str]:
    from rius import observe, start_as_current_span, start_generation, start_span

    @observe(name="decorated")
    def decorated() -> None:
        pass

    start_span("manual").end()
    with start_as_current_span("scoped"):
        pass
    start_generation("gen", model="m").end()
    decorated()
    client.flush()
    return {s.name for s in exporter.get_finished_spans()}


def test_helpers_follow_the_client_across_a_reinit() -> None:
    first_exporter = InMemorySpanExporter()
    first = init(span_exporter=first_exporter, set_global=True, instruments=[])
    assert _helper_spans(first_exporter, first) == {"manual", "scoped", "gen", "decorated"}
    first.shutdown()

    second_exporter = InMemorySpanExporter()
    second = init(span_exporter=second_exporter, set_global=True, instruments=[])
    assert _helper_spans(second_exporter, second) == {"manual", "scoped", "gen", "decorated"}
    assert first_exporter.get_finished_spans() and len(first_exporter.get_finished_spans()) == 4


def test_helpers_use_the_global_provider_when_no_client_is_active(
    exported_spans: InMemorySpanExporter,
) -> None:
    # Without any rius client the helpers must still produce spans through
    # whatever provider the process has (conftest installs one), not crash.
    from rius import start_span

    start_span("no-client").end()
    assert [s.name for s in exported_spans.get_finished_spans()] == ["no-client"]


# --- re-init: the write-once OTel global still holds the first, dead provider ---

_REINIT_GLOBAL_TRACER = """
import logging, sys
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import rius

records = []
class Grab(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())
logging.getLogger("rius").addHandler(Grab())
logging.getLogger("rius").setLevel(logging.DEBUG)

first_exporter = InMemorySpanExporter()
first = rius.init(span_exporter=first_exporter, instruments=[])
assert trace.get_tracer_provider() is first._provider
first.shutdown()

second_exporter = InMemorySpanExporter()
second = rius.init(span_exporter=second_exporter, instruments=[])
trace.get_tracer("third.party").start_span("third-party").end()
second.flush()
second_names = [s.name for s in second_exporter.get_finished_spans()]
first_names = [s.name for s in first_exporter.get_finished_spans()]
assert second_names == ["third-party"], (second_names, first_names)
assert not any("another SDK" in r or "could not register" in r for r in records), records
second.shutdown()

# A third cycle bridges again, and the second client's bridge stays released.
third_exporter = InMemorySpanExporter()
third = rius.init(span_exporter=third_exporter, instruments=[])
trace.get_tracer("third.party").start_span("again").end()
third.flush()
assert [s.name for s in third_exporter.get_finished_spans()] == ["again"]
assert [s.name for s in second_exporter.get_finished_spans()] == ["third-party"]
third.shutdown()
"""


def test_reinit_reaches_global_tracer_spans_without_the_bridge_opt_in() -> None:
    # The OTel global is write-once, so after shutdown()+init() it still holds
    # the first client's (dead) provider. Spans third-party code starts through
    # it must reach the new client, as the init() docstring promises, without
    # the foreign-provider opt-in: that provider is ours, not another SDK's.
    # A subprocess, because this suite's conftest claims the global first.
    result = subprocess.run(
        [sys.executable, "-c", _REINIT_GLOBAL_TRACER],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "RIUS_HEARTBEAT": "false", "RIUS_BRIDGE_FOREIGN_PROVIDER": "false"},
    )
    assert result.returncode == 0, result.stderr


_REINIT_DEAD_PIPELINE = """
from opentelemetry import trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import rius

# Every processor a shut-down client still feeds. OpenTelemetry cannot remove
# a processor from a provider, so the first client's pipeline stays on the
# global provider; on OTel 1.24 its batch processor then logs "Already
# shutdown, dropping span." for every third-party span.
shut = []  # the processors themselves: an id() could be reused
fed_after_shutdown = []
real_shutdown, real_on_end = BatchSpanProcessor.shutdown, BatchSpanProcessor.on_end

def shutdown(self):
    shut.append(self)
    real_shutdown(self)

def on_end(self, span):
    if any(self is s for s in shut):
        fed_after_shutdown.append(span.name)
    real_on_end(self, span)

BatchSpanProcessor.shutdown = shutdown
BatchSpanProcessor.on_end = on_end

first = rius.init(span_exporter=InMemorySpanExporter(), instruments=[])
first.shutdown()
second_exporter = InMemorySpanExporter()
second = rius.init(span_exporter=second_exporter, instruments=[])
trace.get_tracer("third.party").start_span("third-party").end()
second.flush()
assert [s.name for s in second_exporter.get_finished_spans()] == ["third-party"]
assert fed_after_shutdown == [], fed_after_shutdown
second.shutdown()
"""


def test_a_shut_down_client_stops_receiving_global_tracer_spans() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _REINIT_DEAD_PIPELINE],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "RIUS_HEARTBEAT": "false", "RIUS_BRIDGE_FOREIGN_PROVIDER": "false"},
    )
    assert result.returncode == 0, result.stderr


class _UncachedTracerProvider(TracerProvider):
    """An SDK provider as it was before OTel 1.42: a new tracer per call."""

    def get_tracer(self, *args: Any, **kwargs: Any) -> trace.Tracer:
        tracer = super().get_tracer(*args, **kwargs)
        return copy.copy(tracer)


def test_withdraw_clears_the_tracer_even_when_the_provider_does_not_cache_tracers() -> None:
    from rius import _tracer

    provider = _UncachedTracerProvider()
    assert provider.get_tracer("x") is not provider.get_tracer("x")
    _tracer.publish(provider)
    assert _tracer._active is not None
    _tracer.withdraw(provider)
    assert _tracer._active is None


def test_withdraw_by_a_stale_provider_keeps_the_newer_tracer() -> None:
    from rius import _tracer

    stale, current = _UncachedTracerProvider(), _UncachedTracerProvider()
    _tracer.publish(current)
    published = _tracer.sdk_tracer()
    _tracer.withdraw(stale)
    assert _tracer.sdk_tracer() is published


def test_shutdown_returns_the_helpers_to_the_global_provider(
    exported_spans: InMemorySpanExporter,
) -> None:
    from rius import start_span

    client = init(span_exporter=InMemorySpanExporter(), set_global=True, instruments=[])
    client.shutdown()
    start_span("after-shutdown").end()
    assert [s.name for s in exported_spans.get_finished_spans()] == ["after-shutdown"]
