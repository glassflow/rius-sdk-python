"""Hints received on the heartbeat response.

Implements the "Hints" section of the heartbeat contract: the opt-in
``accepts`` capability, the capped response read, the SDK-side rules
(de-duplication, own-trace filter, per-trace budget, bounded queue) and the
``acked`` request field. Pings go to a real local HTTP server.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import deque
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rius
from rius import Hint, drain_hints, init
from rius.config import resolve_config
from rius.heartbeat import HeartbeatSender, OpenRootSpanTracker, mint_instance_identity
from rius.hints import HintInbox

TRACE_A = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACE_B = "0af7651916cd43dd8448eb211c80319c"


def _hint_id(n: int) -> str:
    return f"h_{n:026d}"


def _wire_hint(n: int, trace_id: str | None = None, **overrides: Any) -> dict[str, Any]:
    hint: dict[str, Any] = {
        "id": _hint_id(n),
        "code": "CF-01",
        "template_version": 1,
        "text": f"hint {n}",
        "params": {"tool": "search_docs", "calls": 5, "broken": True},
        "span_ids": ["00f067aa0ba902b7"],
        "created_at": "2026-10-28T10:00:00.000Z",
        "expires_at": "2026-10-28T10:10:00.000Z",
    }
    if trace_id is not None:
        hint["trace_id"] = trace_id
    hint.update(overrides)
    return hint


def _body(*hints: dict[str, Any], **top: Any) -> bytes:
    return json.dumps({"format": "hints", "hints": list(hints), **top}).encode()


# ---------------------------------------------------------------------------
# A real heartbeat endpoint with scripted responses
# ---------------------------------------------------------------------------


class _Endpoint:
    """Local heartbeat receiver: records each request, answers from a script."""

    def __init__(self) -> None:
        self.requests: list[tuple[dict[str, Any], int]] = []
        self.replies: deque[tuple[int, bytes]] = deque()
        self.arrived = threading.Condition()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server API
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                # The reply is taken before the request is recorded, and the
                # request recorded before answering: a test that saw a request
                # arrive never races it for a reply, and a ping has returned
                # only once it is recorded.
                status, body = endpoint.replies.popleft() if endpoint.replies else (204, b"")
                with endpoint.arrived:
                    endpoint.requests.append((json.loads(raw), len(raw)))
                    endpoint.arrived.notify_all()
                self.send_response(status)
                if body:
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # silence test output
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    @property
    def url(self) -> str:
        return self.base_url + "/v1/heartbeat"

    def reply(self, status: int, body: bytes = b"") -> None:
        self.replies.append((status, body))

    def payloads(self) -> list[dict[str, Any]]:
        return [payload for payload, _ in self.requests]

    def wait_for(self, count: int, timeout: float = 5.0) -> None:
        with self.arrived:
            assert self.arrived.wait_for(lambda: len(self.requests) >= count, timeout)

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)


@pytest.fixture
def endpoint() -> Iterator[_Endpoint]:
    served = _Endpoint()
    yield served
    served.close()


def _sender(
    endpoint: _Endpoint,
    *,
    inbox: HintInbox | None = None,
    tracker: OpenRootSpanTracker | None = None,
    agent_name: str = "agent",
) -> HeartbeatSender:
    instance_id, secret = mint_instance_identity()
    return HeartbeatSender(
        url=endpoint.url,
        headers={"Authorization": "Bearer k"},
        interval=3600.0,
        agent_name=agent_name,
        instance_id=instance_id,
        instance_secret=secret,
        tracker=tracker or OpenRootSpanTracker(),
        hints=inbox,
    )


def _open(*trace_ids: str) -> OpenRootSpanTracker:
    tracker = OpenRootSpanTracker()
    for trace_id in trace_ids:
        tracker._on_root_start(trace_id)  # noqa: SLF001 — test wiring
    return tracker


def _inbox(tracker: OpenRootSpanTracker, **kwargs: Any) -> HintInbox:
    return HintInbox(open_traces=tracker.open_trace_ids, **kwargs)


def _ids(hints: list[Hint]) -> list[str]:
    return [hint.id for hint in hints]


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_hints_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIUS_HINTS", raising=False)
    assert resolve_config().hints is False


def test_hints_on_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HINTS", "true")
    assert resolve_config().hints is True


def test_hints_argument_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HINTS", "true")
    assert resolve_config(hints=False).hints is False


def test_hints_per_trace_defaults_to_three(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIUS_HINTS_PER_TRACE", raising=False)
    assert resolve_config().hints_per_trace == 3


def test_hints_per_trace_from_env_and_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIUS_HINTS_PER_TRACE", "5")
    assert resolve_config().hints_per_trace == 5
    assert resolve_config(hints_per_trace=1).hints_per_trace == 1


def test_hints_per_trace_rejects_junk_and_negatives(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RIUS_HINTS_PER_TRACE", "many")
    with caplog.at_level(logging.WARNING):
        assert resolve_config().hints_per_trace == 3
        assert resolve_config(hints_per_trace=-2).hints_per_trace == 0
    assert any("RIUS_HINTS_PER_TRACE" in r.getMessage() for r in caplog.records)
    assert any("hints_per_trace" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Capability: accepts / acked only when hints are on
# ---------------------------------------------------------------------------


def test_hints_off_sends_no_accepts_and_ignores_a_body(endpoint: _Endpoint) -> None:
    endpoint.reply(200, _body(_wire_hint(1)))
    sender = _sender(endpoint)
    sender._send_ping()  # noqa: SLF001
    payload = endpoint.payloads()[0]
    assert "accepts" not in payload
    assert "acked" not in payload


def test_hints_on_sends_accepts_and_an_empty_acked(endpoint: _Endpoint) -> None:
    sender = _sender(endpoint, inbox=_inbox(_open()))
    sender._send_ping()  # noqa: SLF001
    payload = endpoint.payloads()[0]
    assert payload["accepts"] == ["hints"]
    assert payload["acked"] == []


def test_a_sender_without_a_secret_never_asks_for_hints(endpoint: _Endpoint) -> None:
    # The receiver delivers only to authenticated pings, so a legacy sender
    # asking for hints would only ever get nothing back.
    sender = HeartbeatSender(
        url=endpoint.url,
        headers={},
        interval=3600.0,
        agent_name="a",
        instance_id=str(uuid.uuid4()),
        tracker=OpenRootSpanTracker(),
        hints=_inbox(_open()),
    )
    sender._send_ping()  # noqa: SLF001
    assert "accepts" not in endpoint.payloads()[0]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_a_hint_is_parsed_into_a_frozen_hint(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open(TRACE_A))
    endpoint.reply(200, _body(_wire_hint(1, TRACE_A)))
    _sender(endpoint, inbox=inbox)._send_ping()  # noqa: SLF001
    [hint] = inbox.drain(TRACE_A)
    assert hint == Hint(
        id=_hint_id(1),
        code="CF-01",
        template_version=1,
        text="hint 1",
        params={"tool": "search_docs", "calls": 5, "broken": True},
        trace_id=TRACE_A,
        span_ids=("00f067aa0ba902b7",),
        created_at="2026-10-28T10:00:00.000Z",
        expires_at="2026-10-28T10:10:00.000Z",
    )
    with pytest.raises(AttributeError):
        hint.text = "changed"  # type: ignore[misc]


def test_unknown_fields_are_ignored(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    endpoint.reply(200, _body(_wire_hint(1, signature="sig", later={"x": 1}), extra=True))
    _sender(endpoint, inbox=inbox)._send_ping()  # noqa: SLF001
    assert _ids(inbox.drain(None)) == [_hint_id(1)]


def test_an_instance_wide_hint_has_no_trace_id(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    endpoint.reply(200, _body(_wire_hint(1)))
    _sender(endpoint, inbox=inbox)._send_ping()  # noqa: SLF001
    [hint] = inbox.drain(None)
    assert hint.trace_id is None


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"\xff\xfe",
        b"[]",
        b'{"hints": []}',
        json.dumps({"format": "hints.v2", "hints": [_wire_hint(1)]}).encode(),
        json.dumps({"format": "hints", "hints": {"id": "h_1"}}).encode(),
        json.dumps({"format": "hints"}).encode(),
    ],
    ids=["empty", "not-json", "not-utf8", "list", "no-format", "unasked-format", "dict", "none"],
)
def test_a_body_of_the_wrong_kind_means_no_hints(
    endpoint: _Endpoint, body: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    inbox = _inbox(_open())
    endpoint.reply(200, body)
    endpoint.reply(204)
    sender = _sender(endpoint, inbox=inbox)
    with caplog.at_level(logging.WARNING, logger="rius"):
        sender._send_ping()  # noqa: SLF001
        sender._send_ping()  # noqa: SLF001
    assert inbox.drain(None) == []
    assert endpoint.payloads()[1]["acked"] == []
    assert not caplog.records  # a liveness ping that got a 2xx is not a delivery failure


def test_a_malformed_hint_is_dropped_and_acked_but_its_neighbours_are_kept(
    endpoint: _Endpoint,
) -> None:
    inbox = _inbox(_open())
    endpoint.reply(
        200,
        _body(
            _wire_hint(1, template_version="one"),
            _wire_hint(2, trace_id="NOT-A-TRACE"),
            {"no": "id"},
            "a string",
            _wire_hint(3),
        ),
    )
    sender = _sender(endpoint, inbox=inbox)
    sender._send_ping()  # noqa: SLF001
    sender._send_ping()  # noqa: SLF001
    assert _ids(inbox.drain(None)) == [_hint_id(3)]
    assert endpoint.payloads()[1]["acked"] == [_hint_id(1), _hint_id(2), _hint_id(3)]


def test_a_body_over_64_kib_means_no_hints(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    body = _body(_wire_hint(1), pad="x" * (64 * 1024))
    assert len(body) > 64 * 1024
    endpoint.reply(200, body)
    _sender(endpoint, inbox=inbox)._send_ping()  # noqa: SLF001
    assert inbox.drain(None) == []


def test_a_body_of_exactly_64_kib_is_read(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    body = _body(_wire_hint(1))
    body = body[:-1] + b" " * (64 * 1024 - len(body)) + b"}"
    assert len(body) == 64 * 1024
    endpoint.reply(200, body)
    _sender(endpoint, inbox=inbox)._send_ping()  # noqa: SLF001
    assert _ids(inbox.drain(None)) == [_hint_id(1)]


def test_a_stopped_pings_body_is_ignored(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    endpoint.reply(200, _body(_wire_hint(1)))
    _sender(endpoint, inbox=inbox)._send_ping(stopped=True)  # noqa: SLF001
    assert inbox.drain(None) == []


# ---------------------------------------------------------------------------
# Rules: de-duplication, own traces, budget, queue
# ---------------------------------------------------------------------------


def test_a_redelivered_hint_is_queued_once_and_acked_again(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(_wire_hint(1)))
    endpoint.reply(200, _body(_wire_hint(1)))
    sender._send_ping()  # noqa: SLF001
    sender._send_ping()  # noqa: SLF001  — carries the first ack, gets the hint again
    sender._send_ping()  # noqa: SLF001
    assert _ids(inbox.drain(None)) == [_hint_id(1)]
    assert [p["acked"] for p in endpoint.payloads()] == [[], [_hint_id(1)], [_hint_id(1)]]


def test_dedupe_remembers_the_last_1024_ids() -> None:
    inbox = _inbox(_open(), max_queued=2000)
    for start in range(0, 1025, 100):  # one body holds ~250 under the 64 KiB cap
        inbox.accept(_body(*(_wire_hint(n) for n in range(start, min(start + 100, 1025)))))
    assert len(inbox.drain(None)) == 1025
    # Of 1,025 ids, the oldest (0) has been forgotten and the next (1) is kept.
    inbox.accept(_body(_wire_hint(1)))
    assert inbox.drain(None) == []
    inbox.accept(_body(_wire_hint(0)))
    assert _ids(inbox.drain(None)) == [_hint_id(0)]


def test_a_hint_for_a_trace_that_is_not_open_here_is_dropped_and_acked(
    endpoint: _Endpoint,
) -> None:
    inbox = _inbox(_open(TRACE_A))
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(_wire_hint(1, TRACE_B), _wire_hint(2, TRACE_A)))
    sender._send_ping()  # noqa: SLF001
    sender._send_ping()  # noqa: SLF001
    assert _ids(inbox.drain(TRACE_B)) == []
    assert _ids(inbox.drain(TRACE_A)) == [_hint_id(2)]
    assert endpoint.payloads()[1]["acked"] == [_hint_id(1), _hint_id(2)]


def test_trace_ids_are_compared_as_lowercase_hex(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open(TRACE_A))
    endpoint.reply(200, _body(_wire_hint(1, TRACE_A.upper())))
    _sender(endpoint, inbox=inbox)._send_ping()  # noqa: SLF001
    [hint] = inbox.drain(TRACE_A.upper())
    assert hint.trace_id == TRACE_A


def test_expires_at_is_never_compared_with_the_local_clock(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    long_gone = _wire_hint(
        1, created_at="2001-01-01T00:00:00.000Z", expires_at="2001-01-01T00:10:00.000Z"
    )
    endpoint.reply(200, _body(long_gone))
    _sender(endpoint, inbox=inbox)._send_ping()  # noqa: SLF001
    assert _ids(inbox.drain(None)) == [_hint_id(1)]


def test_at_most_three_hints_per_trace_by_default(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open(TRACE_A, TRACE_B))
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(*(_wire_hint(n, TRACE_A) for n in range(1, 5))))
    endpoint.reply(200, _body(_wire_hint(5, TRACE_A), _wire_hint(6, TRACE_B), _wire_hint(7)))
    sender._send_ping()  # noqa: SLF001
    sender._send_ping()  # noqa: SLF001
    sender._send_ping()  # noqa: SLF001
    # The budget counts accepted hints, not queued ones: draining frees nothing.
    assert _ids(inbox.drain(TRACE_A)) == [_hint_id(n) for n in (1, 2, 3, 7)]
    assert _ids(inbox.drain(TRACE_B)) == [_hint_id(6)]
    acked = [i for p in endpoint.payloads() for i in p["acked"]]
    assert sorted(acked) == [_hint_id(n) for n in range(1, 8)]


def test_the_per_trace_budget_is_configurable() -> None:
    inbox = _inbox(_open(TRACE_A), per_trace=1)
    inbox.accept(_body(_wire_hint(1, TRACE_A), _wire_hint(2, TRACE_A)))
    assert _ids(inbox.drain(TRACE_A)) == [_hint_id(1)]


def test_a_full_queue_drops_the_oldest_with_a_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    inbox = _inbox(_open(), max_queued=2)
    with caplog.at_level(logging.DEBUG, logger="rius.hints"):
        inbox.accept(_body(_wire_hint(1), _wire_hint(2), _wire_hint(3)))
    assert _ids(inbox.drain(None)) == [_hint_id(2), _hint_id(3)]
    assert any(_hint_id(1) in r.getMessage() for r in caplog.records)
    assert inbox.acks_to_send() == [_hint_id(1), _hint_id(2), _hint_id(3)]


def test_a_hint_that_fails_verification_is_dropped_and_acked() -> None:
    inbox = _inbox(_open(), verify=lambda raw: raw["id"] != _hint_id(1))
    inbox.accept(_body(_wire_hint(1), _wire_hint(2)))
    assert _ids(inbox.drain(None)) == [_hint_id(2)]
    assert inbox.acks_to_send() == [_hint_id(1), _hint_id(2)]


# ---------------------------------------------------------------------------
# on_hint callback
# ---------------------------------------------------------------------------


def test_on_hint_is_called_once_per_newly_accepted_hint(endpoint: _Endpoint) -> None:
    seen: list[tuple[str, str]] = []
    inbox = _inbox(
        _open(), on_hint=lambda hint: seen.append((hint.id, threading.current_thread().name))
    )
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(_wire_hint(1), _wire_hint(2, TRACE_B)))
    endpoint.reply(200, _body(_wire_hint(1)))
    sender.start()
    endpoint.wait_for(1)
    sender._send_ping()  # noqa: SLF001
    sender.stop()
    assert seen == [(_hint_id(1), "rius-heartbeat")]


def test_an_on_hint_exception_is_logged_and_never_breaks_the_ping(
    endpoint: _Endpoint, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(hint: Hint) -> None:
        raise RuntimeError("callback bug")

    inbox = _inbox(_open(), on_hint=explode)
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(_wire_hint(1), _wire_hint(2)))
    with caplog.at_level(logging.DEBUG, logger="rius"):
        sender._send_ping()  # noqa: SLF001
        sender._send_ping()  # noqa: SLF001
    assert _ids(inbox.drain(None)) == [_hint_id(1), _hint_id(2)]
    assert endpoint.payloads()[1]["acked"] == [_hint_id(1), _hint_id(2)]
    assert any("on_hint" in r.getMessage() for r in caplog.records)
    assert not any("heartbeat delivery failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Acks
# ---------------------------------------------------------------------------


def test_an_ack_is_carried_until_a_ping_that_carried_it_gets_a_2xx(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(_wire_hint(1)))
    endpoint.reply(503)
    endpoint.reply(500)
    endpoint.reply(204)
    for _ in range(5):
        sender._send_ping()  # noqa: SLF001
    assert [p["acked"] for p in endpoint.payloads()] == [
        [],
        [_hint_id(1)],
        [_hint_id(1)],
        [_hint_id(1)],
        [],
    ]


def test_acked_holds_at_most_16_ids_oldest_first(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open(), max_queued=64)
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(*(_wire_hint(n) for n in range(20))))
    for _ in range(3):
        sender._send_ping()  # noqa: SLF001
    acked = [p["acked"] for p in endpoint.payloads()]
    assert acked[1] == [_hint_id(n) for n in range(16)]
    assert acked[2] == [_hint_id(n) for n in range(16, 20)]


def test_acks_are_trimmed_to_keep_the_request_within_3_5_kib(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open(), max_queued=64)
    # A long agent name leaves room for only a few ids under 3,584 bytes.
    sender = _sender(endpoint, inbox=inbox, agent_name="a" * 3100)
    endpoint.reply(200, _body(*(_wire_hint(n) for n in range(16))))
    for _ in range(6):
        sender._send_ping()  # noqa: SLF001
    sizes = [size for _, size in endpoint.requests]
    acked = [p["acked"] for p in endpoint.payloads()]
    assert all(size <= 3584 for size in sizes)
    assert 0 < len(acked[1]) < 16
    assert acked[1] == [_hint_id(n) for n in range(len(acked[1]))]
    sent = [i for ids in acked for i in ids]
    assert sent == [_hint_id(n) for n in range(16)]  # every id goes, in order, exactly once


def test_an_oversized_payload_is_still_sent_with_no_acks(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    sender = _sender(endpoint, inbox=inbox, agent_name="a" * 5000)
    endpoint.reply(200, _body(_wire_hint(1)))
    sender._send_ping()  # noqa: SLF001
    sender._send_ping()  # noqa: SLF001
    assert [p["acked"] for p in endpoint.payloads()] == [[], []]
    assert inbox.acks_to_send() == [_hint_id(1)]


def test_after_a_413_the_next_ping_sends_acked_empty(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(_wire_hint(1)))
    endpoint.reply(413)
    for _ in range(4):
        sender._send_ping()  # noqa: SLF001
    assert [p["acked"] for p in endpoint.payloads()] == [[], [_hint_id(1)], [], [_hint_id(1)]]


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


def test_a_forked_child_forgets_the_parents_hints_acks_and_ids(endpoint: _Endpoint) -> None:
    inbox = _inbox(_open())
    sender = _sender(endpoint, inbox=inbox)
    endpoint.reply(200, _body(_wire_hint(1), _wire_hint(2)))
    sender._send_ping()  # noqa: SLF001
    sender._reset_in_child()  # noqa: SLF001 — what the fork hook runs in the child
    try:
        assert inbox.drain(None) == []
        assert inbox.acks_to_send() == []
        inbox.accept(_body(_wire_hint(1)))  # the parent's id is new to the child
        assert _ids(inbox.drain(None)) == [_hint_id(1)]
    finally:
        sender.stop()


# ---------------------------------------------------------------------------
# init() wiring and the public drain
# ---------------------------------------------------------------------------


def _hint_client(endpoint: _Endpoint, **kwargs: Any) -> rius.GlassflowClient:
    return init(
        service_name="svc",
        api_key="k",
        endpoint=endpoint.base_url,
        heartbeat=True,
        heartbeat_interval=300.0,
        span_exporter=InMemorySpanExporter(),
        **kwargs,
    )


def test_a_custom_transport_keeps_hints_off_and_says_why_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sent: list[dict[str, Any]] = []
    with caplog.at_level(logging.DEBUG, logger="rius"):
        client = init(
            set_global=False,
            service_name="svc",
            heartbeat=True,
            hints=True,
            heartbeat_transport=sent.append,
            span_exporter=InMemorySpanExporter(),
        )
    try:
        assert client.drain_hints() == []
    finally:
        client.shutdown()
    assert sent and all("accepts" not in payload for payload in sent)
    messages = [r.getMessage() for r in caplog.records if "hints" in r.getMessage()]
    assert len(messages) == 1
    assert "heartbeat_transport" in messages[0]


def test_hints_off_by_default_in_init(endpoint: _Endpoint, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIUS_HINTS", raising=False)
    client = _hint_client(endpoint, set_global=False)
    try:
        endpoint.wait_for(1)
        assert "accepts" not in endpoint.payloads()[0]
        assert client.drain_hints() == []
    finally:
        client.shutdown()


def test_init_on_hint_and_drain_by_the_current_trace(endpoint: _Endpoint) -> None:
    received: list[Hint] = []
    client = _hint_client(endpoint, set_global=False, hints=True, on_hint=received.append)
    try:
        endpoint.wait_for(1)
        assert endpoint.payloads()[0]["accepts"] == ["hints"]
        tracer = client.get_tracer()
        other = tracer.start_span("other run")
        other_trace = format(other.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("run") as run:
            this_trace = format(run.get_span_context().trace_id, "032x")
            endpoint.reply(
                200,
                _body(_wire_hint(1, this_trace), _wire_hint(2, other_trace), _wire_hint(3)),
            )
            assert client._heartbeat is not None  # noqa: SLF001
            client._heartbeat._send_ping()  # noqa: SLF001
            assert _ids(received) == [_hint_id(1), _hint_id(2), _hint_id(3)]
            with tracer.start_as_current_span("model call"):
                assert _ids(client.drain_hints()) == [_hint_id(1), _hint_id(3)]
            assert client.drain_hints() == []
        assert client.drain_hints() == []  # no current trace: instance-wide only
        assert _ids(client.drain_hints(trace_id=other_trace)) == [_hint_id(2)]
        other.end()
    finally:
        client.shutdown()


def test_the_module_drain_uses_the_global_client(endpoint: _Endpoint) -> None:
    assert drain_hints() == []  # no client yet
    client = _hint_client(endpoint, hints=True, instruments=[])
    try:
        endpoint.wait_for(1)
        endpoint.reply(200, _body(_wire_hint(1)))
        assert client._heartbeat is not None  # noqa: SLF001
        client._heartbeat._send_ping()  # noqa: SLF001
        assert _ids(drain_hints()) == [_hint_id(1)]
    finally:
        client.shutdown()
    assert drain_hints() == []


def test_init_passes_the_per_trace_budget_through(endpoint: _Endpoint) -> None:
    client = _hint_client(endpoint, set_global=False, hints=True, hints_per_trace=1)
    try:
        endpoint.wait_for(1)
        with client.get_tracer().start_as_current_span("run") as run:
            this_trace = format(run.get_span_context().trace_id, "032x")
            endpoint.reply(200, _body(_wire_hint(1, this_trace), _wire_hint(2, this_trace)))
            assert client._heartbeat is not None  # noqa: SLF001
            client._heartbeat._send_ping()  # noqa: SLF001
            assert _ids(client.drain_hints()) == [_hint_id(1)]
    finally:
        client.shutdown()


def test_a_second_client_keeps_its_own_hints(endpoint: _Endpoint) -> None:
    first = _hint_client(endpoint, set_global=False, hints=True)
    second = _hint_client(endpoint, set_global=False, hints=True)
    try:
        endpoint.wait_for(2)
        endpoint.reply(200, _body(_wire_hint(1)))
        assert first._heartbeat is not None  # noqa: SLF001
        first._heartbeat._send_ping()  # noqa: SLF001
        assert second.drain_hints() == []
        assert _ids(first.drain_hints()) == [_hint_id(1)]
    finally:
        first.shutdown()
        second.shutdown()


def test_hint_names_are_public() -> None:
    assert "Hint" in rius.__all__
    assert "drain_hints" in rius.__all__
