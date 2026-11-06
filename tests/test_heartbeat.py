"""Agent-lifetime heartbeat sender.

Implements the heartbeat spec: payload v1, emission semantics (init-to-exit
lifetime, immediate first ping, stopped ping on graceful shutdown, silent
failure), and the config surface (on by default, interval clamped [5, 300],
agent_name defaults to service_name).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init
from rius.config import resolve_config
from rius.heartbeat import (
    HeartbeatSender,
    OpenRootSpanTracker,
    derive_instance_id,
    mint_instance_identity,
)

# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_heartbeat_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # The conftest guard forces heartbeat off suite-wide (network safety);
    # clear it here to observe the real default.
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    assert resolve_config().heartbeat is True


def test_heartbeat_opt_out_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HEARTBEAT", "false")
    assert resolve_config().heartbeat is False


def test_heartbeat_enabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HEARTBEAT", "true")
    assert resolve_config().heartbeat is True


def test_heartbeat_argument_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HEARTBEAT", "true")
    assert resolve_config(heartbeat=False).heartbeat is False


def test_heartbeat_interval_default() -> None:
    assert resolve_config().heartbeat_interval == 15.0


def test_heartbeat_interval_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HEARTBEAT_INTERVAL", "30")
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
    monkeypatch.setenv("RIUS_AGENT_NAME", "env-agent")
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
        instance_id=str(uuid.uuid4()),
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
        instance_id=str(uuid.uuid4()),
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
        instance_id=str(uuid.uuid4()),
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
            instance_id=str(uuid.uuid4()),
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


def _serve_heartbeats() -> tuple[HTTPServer, threading.Thread, list[tuple[dict[str, Any], Any]]]:
    received: list[tuple[dict[str, Any], Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server API
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((json.loads(body), self.headers))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args: Any) -> None:  # silence test output
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, received


def _decode_secret(secret: str) -> bytes:
    return base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4))


def test_instance_id_derivation_matches_the_receiver() -> None:
    # Same vector as argus-core heartbeat/secret_test.go and the TS SDK.
    assert derive_instance_id(bytes(range(32))) == "630dcd29-66c4-8366-9112-5448bbb25b4f"


def test_minted_identity_is_a_secret_and_the_id_it_derives() -> None:
    instance_id, secret = mint_instance_identity()
    raw = _decode_secret(secret)
    assert len(raw) == 32 and "=" not in secret
    assert instance_id == derive_instance_id(raw)
    assert uuid.UUID(instance_id).version == 8
    assert mint_instance_identity() != (instance_id, secret)


def test_init_uses_the_derived_id_for_spans_and_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    client = init(
        set_global=False,
        service_name="svc",
        api_key="k",
        endpoint="http://127.0.0.1:1",
        span_exporter=InMemorySpanExporter(),
    )
    try:
        sender = client._heartbeat  # noqa: SLF001
        assert sender is not None
        resource_id = client._provider.resource.attributes["service.instance.id"]  # noqa: SLF001
        assert sender.instance_id == resource_id
        assert sender.instance_id == derive_instance_id(_decode_secret(sender._secret or ""))  # noqa: SLF001
    finally:
        client.shutdown()


def test_a_custom_transport_pings_as_a_legacy_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    # It is handed the payload only, so it could never send the secret, and a
    # derived id without its secret is refused on every ping.
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    client = init(
        set_global=False,
        service_name="svc",
        span_exporter=InMemorySpanExporter(),
        heartbeat_transport=[].append,
    )
    try:
        sender = client._heartbeat  # noqa: SLF001
        assert sender is not None
        assert sender._secret is None  # noqa: SLF001
        assert uuid.UUID(sender.instance_id).version == 4
        resource_id = client._provider.resource.attributes["service.instance.id"]  # noqa: SLF001
        assert sender.instance_id == resource_id
    finally:
        client.shutdown()


def test_fork_reset_keeps_a_legacy_sender_legacy() -> None:
    sender = HeartbeatSender(
        url="http://unused",
        headers={},
        interval=3600.0,
        agent_name="a",
        instance_id=str(uuid.uuid4()),
        tracker=OpenRootSpanTracker(),
        transport=lambda payload: None,
    )
    before = sender.instance_id
    try:
        sender._reset_in_child()  # noqa: SLF001
        assert sender.instance_id != before
        assert sender._secret is None  # noqa: SLF001
        assert uuid.UUID(sender.instance_id).version == 4
    finally:
        sender.stop()


def test_every_ping_including_the_final_one_carries_the_secret() -> None:
    server, thread, received = _serve_heartbeats()
    instance_id, secret = mint_instance_identity()
    try:
        sender = HeartbeatSender(
            url=f"http://127.0.0.1:{server.server_port}/v1/heartbeat",
            headers={"Authorization": "Bearer test-key"},
            interval=3600.0,
            agent_name="e2e-agent",
            instance_id=instance_id,
            instance_secret=secret,
            tracker=OpenRootSpanTracker(),
        )
        sender.start()
        sender.stop()
        assert [payload.get("stopped", False) for payload, _ in received] == [False, True]
        assert [headers.get("Rius-Instance-Secret") for _, headers in received] == [secret, secret]
        for payload, _ in received:
            assert payload["instance_id"] == instance_id
            assert secret not in json.dumps(payload)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_a_sender_without_a_secret_sends_no_secret_header() -> None:
    server, thread, received = _serve_heartbeats()
    try:
        sender = HeartbeatSender(
            url=f"http://127.0.0.1:{server.server_port}/v1/heartbeat",
            headers={},
            interval=3600.0,
            agent_name="e2e-agent",
            instance_id=str(uuid.uuid4()),
            tracker=OpenRootSpanTracker(),
        )
        sender._send_ping()  # noqa: SLF001
        assert received[0][1].get("Rius-Instance-Secret") is None
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_a_caller_header_cannot_replace_the_secret() -> None:
    server, thread, received = _serve_heartbeats()
    instance_id, secret = mint_instance_identity()
    try:
        for name in ("Rius-Instance-Secret", "rius-instance-secret", "RIUS-INSTANCE-SECRET"):
            sender = HeartbeatSender(
                url=f"http://127.0.0.1:{server.server_port}/v1/heartbeat",
                headers={"Authorization": "Bearer test-key", name: "caller-value"},
                interval=3600.0,
                agent_name="e2e-agent",
                instance_id=instance_id,
                instance_secret=secret,
                tracker=OpenRootSpanTracker(),
            )
            sender._send_ping()  # noqa: SLF001
            assert received[-1][1].get_all("Rius-Instance-Secret") == [secret], name
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_fork_reset_mints_a_new_secret_and_its_id() -> None:
    instance_id, secret = mint_instance_identity()
    sender = HeartbeatSender(
        url="https://ingest.example.com/v1/heartbeat",
        headers={},
        interval=3600.0,
        agent_name="a",
        instance_id=instance_id,
        instance_secret=secret,
        tracker=OpenRootSpanTracker(),
        transport=[].append,
    )
    sender._reset_in_child()  # noqa: SLF001
    try:
        assert sender._secret not in (None, secret)  # noqa: SLF001
        assert sender.instance_id == derive_instance_id(_decode_secret(sender._secret or ""))  # noqa: SLF001
    finally:
        sender.stop()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_real_fork_gives_the_child_its_own_derived_identity() -> None:
    instance_id, secret = mint_instance_identity()
    sender = HeartbeatSender(
        url="https://ingest.example.com/v1/heartbeat",
        headers={},
        interval=3600.0,
        agent_name="a",
        instance_id=instance_id,
        instance_secret=secret,
        tracker=OpenRootSpanTracker(),
        transport=[].append,
    )
    sender.start()
    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork with live threads
        pid = os.fork()
    if pid == 0:  # pragma: no cover — runs in the child
        os.close(read_fd)
        child_secret = sender._secret or ""  # noqa: SLF001
        ok = (
            sender.instance_id != instance_id
            and child_secret != secret
            and sender.instance_id == derive_instance_id(_decode_secret(child_secret))
        )
        os.write(write_fd, b"1" if ok else b"0")
        os._exit(0)
    os.close(write_fd)
    try:
        assert os.read(read_fd, 1) == b"1"
        os.waitpid(pid, 0)
        assert sender.instance_id == instance_id  # the parent keeps its own
    finally:
        os.close(read_fd)
        sender.stop()


def test_a_refused_secret_warns_specifically_and_never_logs_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Refuse(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server API
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(403)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Refuse)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    instance_id, secret = mint_instance_identity()
    try:
        sender = HeartbeatSender(
            url=f"http://127.0.0.1:{server.server_port}/v1/heartbeat",
            headers={},
            interval=3600.0,
            agent_name="a",
            instance_id=instance_id,
            instance_secret=secret,
            tracker=OpenRootSpanTracker(),
        )
        with caplog.at_level(logging.DEBUG, logger="rius"):
            sender._send_ping()  # noqa: SLF001
            sender._send_ping()  # noqa: SLF001
        refused = [r for r in caplog.records if "instance secret" in r.getMessage()]
        assert len(refused) == 1 and refused[0].levelno == logging.WARNING
        assert all(secret not in r.getMessage() for r in caplog.records)
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
            # The default transport only pings with credentials.
            "RIUS_API_KEY": "k",
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
            instance_id=str(uuid.uuid4()),
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
        instance_id=str(uuid.uuid4()),
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


# ---------------------------------------------------------------------------
# No credentials, no heartbeat
# ---------------------------------------------------------------------------
# The managed endpoint rejects an unauthenticated ping, so the default
# transport would only fail every interval. An injected transport knows its
# own destination and still runs.


def test_no_api_key_starts_no_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    monkeypatch.delenv("RIUS_API_KEY", raising=False)
    client = init(
        set_global=False,
        service_name="svc",
        endpoint="http://127.0.0.1:1",
        span_exporter=InMemorySpanExporter(),
    )
    try:
        assert client._heartbeat is None  # noqa: SLF001
    finally:
        client.shutdown()


def test_a_heartbeat_skipped_for_lack_of_credentials_logs_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    monkeypatch.delenv("RIUS_API_KEY", raising=False)
    with caplog.at_level(logging.DEBUG, logger="rius.client"):
        client = init(
            set_global=False,
            service_name="svc",
            endpoint="http://127.0.0.1:1",
            span_exporter=InMemorySpanExporter(),
        )
    client.shutdown()
    assert any(
        r.levelno == logging.DEBUG and "heartbeat" in r.getMessage().lower() for r in caplog.records
    )


def test_a_disabled_heartbeat_logs_no_credentials_message(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("RIUS_API_KEY", raising=False)
    with caplog.at_level(logging.DEBUG, logger="rius.client"):
        client = init(
            set_global=False,
            service_name="svc",
            heartbeat=False,
            endpoint="http://127.0.0.1:1",
            span_exporter=InMemorySpanExporter(),
        )
    client.shutdown()
    assert not any("credentials" in r.getMessage() for r in caplog.records)


def test_an_api_key_starts_the_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    client = init(
        set_global=False,
        service_name="svc",
        api_key="k",
        endpoint="http://127.0.0.1:1",
        span_exporter=InMemorySpanExporter(),
    )
    try:
        assert client._heartbeat is not None  # noqa: SLF001
    finally:
        client.shutdown()


def test_an_explicit_authorization_header_starts_the_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RIUS_HEARTBEAT", raising=False)
    monkeypatch.delenv("RIUS_API_KEY", raising=False)
    client = init(
        set_global=False,
        service_name="svc",
        headers={"authorization": "Bearer custom"},
        endpoint="http://127.0.0.1:1",
        span_exporter=InMemorySpanExporter(),
    )
    try:
        assert client._heartbeat is not None  # noqa: SLF001
    finally:
        client.shutdown()


def test_an_injected_transport_heartbeats_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RIUS_API_KEY", raising=False)
    sent: list[dict[str, Any]] = []
    client = init(
        set_global=False,
        service_name="svc",
        heartbeat=True,
        heartbeat_transport=sent.append,
        span_exporter=InMemorySpanExporter(),
    )
    client.shutdown()
    assert sent
