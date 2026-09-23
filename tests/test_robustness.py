"""Edge cases from the 2026-09-14 review that crashed, spun or silently dropped."""

from __future__ import annotations

import gc
import math
import threading

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init, observe, start_generation
from rius.config import DEFAULT_HEARTBEAT_INTERVAL, resolve_config

# --- NaN passes a min/max clamp, then spins the heartbeat or crashes init() ---


def test_nan_heartbeat_interval_falls_back_to_the_default() -> None:
    assert (
        resolve_config(heartbeat_interval=math.nan).heartbeat_interval == DEFAULT_HEARTBEAT_INTERVAL
    )


def test_nan_sample_rate_falls_back_to_the_default() -> None:
    assert resolve_config(sample_rate=math.nan).sample_rate == 1.0


def test_nan_partial_spans_delay_falls_back_to_the_default() -> None:
    assert resolve_config(partial_spans_delay=math.nan).partial_spans_delay == 0.0


def test_nan_from_the_environment_is_rejected_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_SAMPLE_RATE", "nan")
    monkeypatch.setenv("RIUS_HEARTBEAT_INTERVAL", "NaN")
    config = resolve_config()
    assert config.sample_rate == 1.0
    assert config.heartbeat_interval == DEFAULT_HEARTBEAT_INTERVAL
    init(span_exporter=InMemorySpanExporter(), set_global=False, sample_rate=math.nan)  # no raise


# --- model_parameters: dicts and None were dropped by OTel with only a warning ---


def test_model_parameters_serialize_non_primitives_and_skip_none(
    exported_spans: InMemorySpanExporter,
) -> None:
    start_generation(
        "g",
        model_parameters={
            "response_format": {"type": "json_object"},
            "temperature": None,
            "top_p": 0.5,
            "stop": ["a", "b"],
        },
    ).end()
    attrs = exported_spans.get_finished_spans()[0].attributes
    # response_format is not a spec-defined request attribute: our namespace.
    assert attrs["rius.request.response_format"] == '{"type": "json_object"}'
    assert "gen_ai.request.temperature" not in attrs
    assert attrs["gen_ai.request.top_p"] == 0.5
    # OpenAI's `stop` is a recognised spelling of gen_ai.request.stop_sequences.
    assert attrs["gen_ai.request.stop_sequences"] == ("a", "b")


# --- a kill-switched process must not throw at a call site that works when enabled ---


def test_register_workspace_is_a_noop_when_disabled() -> None:
    client = init(disabled=True, workspaces={}, set_global=False)
    client.register_workspace("acme", "key")  # no raise


# --- @observe on generators must be a transparent proxy ---


def test_observed_generator_forwards_send() -> None:
    @observe(name="echo")
    def echo():  # noqa: ANN202
        received = yield "ready"
        yield received

    g = echo()
    assert next(g) == "ready"
    assert g.send("hello") == "hello"


def test_observed_generator_forwards_throw() -> None:
    @observe(name="catcher")
    def catcher():  # noqa: ANN202
        try:
            yield 1
        except KeyError:
            yield "caught"

    g = catcher()
    next(g)
    assert g.throw(KeyError("k")) == "caught"


def test_observed_generator_close_closes_the_inner_generator() -> None:
    closed: list[bool] = []

    @observe(name="closer")
    def closer():  # noqa: ANN202
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    g = closer()
    next(g)
    g.close()
    assert closed == [True]  # not deferred to garbage collection


def test_observed_async_generator_aclose_closes_the_inner_generator() -> None:
    import asyncio

    closed: list[bool] = []

    @observe(name="acloser")
    async def acloser():  # noqa: ANN202
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    async def scenario() -> None:
        g = acloser()
        assert await g.__anext__() == 1
        await g.aclose()
        gc.collect()

    asyncio.run(scenario())
    assert closed == [True]


# --- a debounced pending must not be enqueued after its own final span ---


def test_pending_emission_holds_the_scheduler_lock() -> None:
    """pop_due used to release the lock before emitting, so a span ending in
    that window cancelled nothing and its final row was followed by a
    pending=true row. Emitting under the lock serialises cancel and emit."""
    from rius.pending import PendingScheduler

    observed: list[bool] = []

    def emit(_snapshot) -> None:  # noqa: ANN001
        # From ANOTHER thread, the lock must be unavailable while we emit.
        result: list[bool] = []

        def probe() -> None:
            result.append(scheduler._cond.acquire(blocking=False))
            if result[-1]:
                scheduler._cond.release()

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        observed.append(result[0])

    scheduler = PendingScheduler(emit=emit, delay=0.0, clock=lambda: 100.0, start_thread=False)
    scheduler.schedule(("t", "s"), object())  # type: ignore[arg-type]
    scheduler.pop_due()
    assert observed == [False]
