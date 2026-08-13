"""Agent-lifetime heartbeat sender.

Implements the heartbeat spec: payload v1, emission semantics (init-to-exit
lifetime, immediate first ping, stopped ping on graceful shutdown, silent
failure), and the config surface (on by default, interval clamped [5, 300],
agent_name defaults to service_name).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.config import resolve_config
from rius.heartbeat import HeartbeatSender, OpenRootSpanTracker

# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_heartbeat_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # The conftest guard forces heartbeat off suite-wide (network safety);
    # clear it here to observe the real default.
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    monkeypatch.delenv("GLASSFLOW_HEARTBEAT", raising=False)
    assert resolve_config().heartbeat is True


def test_heartbeat_opt_out_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HEARTBEAT", "false")
    assert resolve_config().heartbeat is False


def test_heartbeat_enabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HEARTBEAT", "true")
    assert resolve_config().heartbeat is True


def test_heartbeat_argument_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSFLOW_HEARTBEAT", "true")
    assert resolve_config(heartbeat=False).heartbeat is False


def test_heartbeat_interval_default() -> None:
    assert resolve_config().heartbeat_interval == 15.0


def test_heartbeat_interval_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSFLOW_HEARTBEAT_INTERVAL", "30")
    assert resolve_config().heartbeat_interval == 30.0


def test_heartbeat_interval_clamped_low_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert resolve_config(heartbeat_interval=1.0).heartbeat_interval == 5.0
    assert any("heartbeat_interval" in r.message for r in caplog.records)


def test_heartbeat_interval_clamped_high() -> None:
    assert resolve_config(heartbeat_interval=1000.0).heartbeat_interval == 300.0


def test_agent_name_defaults_to_service_name() -> None:
    config = resolve_config(service_name="checkout-agent")
    assert config.agent_name == "checkout-agent"


def test_agent_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSFLOW_AGENT_NAME", "env-agent")
    assert resolve_config(service_name="svc").agent_name == "env-agent"
    assert resolve_config(service_name="svc", agent_name="arg-agent").agent_name == "arg-agent"


def test_heartbeat_endpoint_property() -> None:
    config = resolve_config(endpoint="https://ingest.example.com/")
    assert config.heartbeat_endpoint == "https://ingest.example.com/v1/heartbeat"


# ---------------------------------------------------------------------------
# Open-root-span tracker
# ---------------------------------------------------------------------------


def _tracked_client() -> tuple[Any, OpenRootSpanTracker]:
    tracker = OpenRootSpanTracker()
    client = init(
        set_global=False,
        service_name="t",
        span_exporter=InMemorySpanExporter(),
    )
    client._provider.add_span_processor(tracker)  # noqa: SLF001 — test wiring
    return client, tracker


def test_tracker_counts_open_root_spans() -> None:
    client, tracker = _tracked_client()
    assert tracker.open_trace_ids() == []
    with client.get_tracer().start_as_current_span("root") as root:
        expected = format(root.get_span_context().trace_id, "032x")
        assert tracker.open_trace_ids() == [expected]
        # a child span of the same trace is NOT a new open root
        with client.get_tracer().start_as_current_span("child"):
            assert tracker.open_trace_ids() == [expected]
    assert tracker.open_trace_ids() == []


def test_tracker_handles_concurrent_roots() -> None:
    client, tracker = _tracked_client()
    tracer = client.get_tracer()
    a = tracer.start_span("a")
    b = tracer.start_span("b")
    assert len(tracker.open_trace_ids()) == 2
    a.end()
    assert len(tracker.open_trace_ids()) == 1
    b.end()
    assert tracker.open_trace_ids() == []


# ---------------------------------------------------------------------------
# Payload (spec v1)
# ---------------------------------------------------------------------------


def _sender(
    sent: list[dict[str, Any]],
    tracker: OpenRootSpanTracker | None = None,
    interval: float = 3600.0,
) -> HeartbeatSender:
    return HeartbeatSender(
        url="https://ingest.example.com/v1/heartbeat",
        headers={},
        interval=interval,
        agent_name="checkout-agent",
        tracker=tracker or OpenRootSpanTracker(),
        transport=sent.append,
    )


def test_payload_matches_spec_v1() -> None:
    sent: list[dict[str, Any]] = []
    sender = _sender(sent)
    sender._send_ping()  # noqa: SLF001 — payload unit test
    payload = sent[0]
    assert payload["v"] == 1
    uuid.UUID(payload["instance_id"])  # valid UUID
    assert payload["agent_name"] == "checkout-agent"
    # RFC3339 UTC with sub-second precision
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", payload["sent_at"])
    assert payload["sdk_language"] == "python"
    assert payload["sdk_version"]
    assert payload["open_traces"] == []
    assert payload["open_trace_count"] == 0
    assert "stopped" not in payload


def test_payload_open_traces_capped_at_32() -> None:
    sent: list[dict[str, Any]] = []
    tracker = OpenRootSpanTracker()
    for i in range(40):
        tracker._on_root_start(format(i + 1, "032x"))  # noqa: SLF001
    sender = _sender(sent, tracker=tracker)
    sender._send_ping()  # noqa: SLF001
    payload = sent[0]
    assert len(payload["open_traces"]) == 32
    assert payload["open_trace_count"] == 40


def test_instance_id_constant_across_pings() -> None:
    sent: list[dict[str, Any]] = []
    sender = _sender(sent)
    sender._send_ping()  # noqa: SLF001
    sender._send_ping()  # noqa: SLF001
    assert sent[0]["instance_id"] == sent[1]["instance_id"]


# ---------------------------------------------------------------------------
# Lifecycle: immediate first ping, stopped ping, idempotent stop
# ---------------------------------------------------------------------------


def test_start_sends_immediate_ping() -> None:
    sent: list[dict[str, Any]] = []
    first_ping = threading.Event()

    def transport(payload: dict[str, Any]) -> None:
        sent.append(payload)
        first_ping.set()

    sender = HeartbeatSender(
        url="u",
        headers={},
        interval=3600.0,
        agent_name="a",
        tracker=OpenRootSpanTracker(),
        transport=transport,
    )
    sender.start()
    try:
        assert first_ping.wait(timeout=5.0), "no ping arrived after start()"
        assert "stopped" not in sent[0]
    finally:
        sender.stop()


def test_stop_sends_stopped_ping_exactly_once() -> None:
    sent: list[dict[str, Any]] = []
    sender = _sender(sent)
    sender.start()
    sender.stop()
    sender.stop()  # idempotent
    stopped_pings = [p for p in sent if p.get("stopped") is True]
    assert len(stopped_pings) == 1
    assert sent[-1] is stopped_pings[0]


def test_client_shutdown_stops_heartbeat() -> None:
    sent: list[dict[str, Any]] = []
    client = init(
        set_global=False,
        service_name="hb-svc",
        heartbeat=True,
        heartbeat_transport=sent.append,
        span_exporter=InMemorySpanExporter(),
    )
    client.shutdown()
    assert any(p.get("stopped") is True for p in sent)
    assert sent[0]["agent_name"] == "hb-svc"  # agent_name defaulted to service_name


def test_open_traces_flow_into_payloads() -> None:
    sent: list[dict[str, Any]] = []
    client = init(
        set_global=False,
        service_name="hb-svc",
        heartbeat=True,
        heartbeat_transport=sent.append,
        span_exporter=InMemorySpanExporter(),
    )
    try:
        span = client.get_tracer().start_span("root")
        trace_id = format(span.get_span_context().trace_id, "032x")
        client._heartbeat._send_ping()  # noqa: SLF001 — deterministic mid-run ping
        span.end()
        client._heartbeat._send_ping()  # noqa: SLF001
        with_open = [p for p in sent if trace_id in p.get("open_traces", [])]
        assert with_open, "open root trace never appeared in a payload"
        assert sent[-1]["open_traces"] == []
    finally:
        client.shutdown()


def test_heartbeat_on_by_default_starts_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    monkeypatch.delenv("GLASSFLOW_HEARTBEAT", raising=False)
    sent: list[dict[str, Any]] = []
    client = init(
        set_global=False,
        service_name="svc",
        span_exporter=InMemorySpanExporter(),
        heartbeat_transport=sent.append,
    )
    assert client._heartbeat is not None  # noqa: SLF001
    client.shutdown()


def test_heartbeat_opt_out_starts_no_thread() -> None:
    client = init(
        set_global=False,
        service_name="svc",
        heartbeat=False,
        span_exporter=InMemorySpanExporter(),
    )
    assert client._heartbeat is None  # noqa: SLF001
    client.shutdown()


def test_disabled_kill_switch_disables_heartbeat() -> None:
    sent: list[dict[str, Any]] = []
    client = init(
        set_global=False,
        service_name="svc",
        disabled=True,
        heartbeat=True,
        heartbeat_transport=sent.append,
    )
    client.shutdown()
    assert sent == []


# ---------------------------------------------------------------------------
# Failure behavior: never raises, warns once
# ---------------------------------------------------------------------------


def test_transport_failure_never_raises_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(_: dict[str, Any]) -> None:
        raise ConnectionError("endpoint down")

    sender = HeartbeatSender(
        url="u",
        headers={},
        interval=3600.0,
        agent_name="a",
        tracker=OpenRootSpanTracker(),
        transport=broken,
    )
    with caplog.at_level(logging.DEBUG, logger="rius.heartbeat"):
        sender._send_ping()  # noqa: SLF001
        sender._send_ping()  # noqa: SLF001
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Real HTTP transport (no mocks): local server, auth header, 204
# ---------------------------------------------------------------------------


def test_http_transport_end_to_end() -> None:
    received: list[tuple[dict[str, Any], str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server API
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((json.loads(body), self.headers.get("Authorization")))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args: Any) -> None:  # silence test output
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/heartbeat"
        sender = HeartbeatSender(
            url=url,
            headers={"Authorization": "Bearer test-key"},
            interval=3600.0,
            agent_name="e2e-agent",
            tracker=OpenRootSpanTracker(),
        )
        sender._send_ping()  # noqa: SLF001
        assert len(received) == 1
        payload, auth = received[0]
        assert payload["v"] == 1
        assert payload["agent_name"] == "e2e-agent"
        assert auth == "Bearer test-key"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# atexit: a real subprocess that exits cleanly sends the stopped ping
# ---------------------------------------------------------------------------


def test_atexit_sends_stopped_ping_from_real_process() -> None:
    import os
    import subprocess
    import sys

    heartbeats: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server API
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/v1/heartbeat":
                heartbeats.append(json.loads(body))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = {
            **os.environ,
            "RIUS_ENDPOINT": f"http://127.0.0.1:{server.server_port}",
            "RIUS_HEARTBEAT": "1",  # overrides the conftest guard in os.environ
            "RIUS_SERVICE_NAME": "atexit-agent",
        }
        # init() then exit normally WITHOUT calling shutdown(): the atexit
        # hook alone must produce the stopped ping.
        result = subprocess.run(
            [sys.executable, "-c", "import rius; rius.init(instruments=[])"],
            env=env,
            timeout=30,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert heartbeats, "no heartbeat arrived from the subprocess"
        assert "stopped" not in heartbeats[0]
        assert heartbeats[-1].get("stopped") is True
        assert all(p["agent_name"] == "atexit-agent" for p in heartbeats)
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Shutdown latency & resource hygiene (security/robustness review)
# ---------------------------------------------------------------------------


def test_stop_with_dead_slow_endpoint_returns_quickly() -> None:
    """A dead endpoint must not hold the user's process exit hostage."""
    import socket
    import time

    # A socket that accepts but never responds — the worst-case endpoint.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/v1/heartbeat"
        sender = HeartbeatSender(
            url=url,
            headers={},
            interval=3600.0,
            agent_name="a",
            tracker=OpenRootSpanTracker(),
            ping_timeout=0.3,
            final_ping_timeout=0.3,
        )
        sender.start()
        time.sleep(0.05)  # let the first (hanging) ping get in flight
        started = time.monotonic()
        sender.stop()
        elapsed = time.monotonic() - started
        # bound: in-flight ping timeout + final ping timeout + slack
        assert elapsed < 2.0, f"stop() took {elapsed:.2f}s against a dead endpoint"
    finally:
        listener.close()


def test_final_ping_timeout_defaults_shorter_than_ping_timeout() -> None:
    sender = HeartbeatSender(
        url="u",
        headers={},
        interval=3600.0,
        agent_name="a",
        tracker=OpenRootSpanTracker(),
    )
    assert sender._final_ping_timeout < sender._ping_timeout  # noqa: SLF001
    assert sender._final_ping_timeout == 1.0  # noqa: SLF001


def test_stopped_senders_leave_the_fork_registry() -> None:
    """init/shutdown cycles must not accumulate fork-handler references."""
    from rius.heartbeat import _active_senders

    before = len(_active_senders)
    sent: list[dict[str, Any]] = []
    sender = _sender(sent)
    sender.start()
    assert len(_active_senders) == before + 1
    sender.stop()
    assert len(_active_senders) == before
