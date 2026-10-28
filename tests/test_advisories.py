"""Advisories delivered on heartbeat responses (spike)."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from rius import drain_advisories
from rius.advisories import accept
from rius.heartbeat import HeartbeatSender, OpenRootSpanTracker


@pytest.fixture(autouse=True)
def _empty_queue() -> Iterator[None]:
    drain_advisories()
    yield
    drain_advisories()


def _id() -> str:
    return str(uuid.uuid4())


def test_accept_queues_advisories_and_drain_clears_them() -> None:
    first, second = _id(), _id()
    accept({"advisories": [{"id": first, "text": "a"}, {"id": second, "text": "b", "code": "X"}]})
    drained = drain_advisories()
    assert [a["id"] for a in drained] == [first, second]
    assert drained[1]["code"] == "X"
    assert drain_advisories() == []


def test_accept_drops_a_redelivered_advisory() -> None:
    advisory_id = _id()
    accept({"advisories": [{"id": advisory_id, "text": "a"}]})
    drain_advisories()
    accept({"advisories": [{"id": advisory_id, "text": "a"}]})
    assert drain_advisories() == []


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {"advisories": "nope"},
        {"advisories": [{"text": "no id"}]},
        {"advisories": [{"id": "x", "text": ""}]},
        {"advisories": [{"id": 1, "text": "id not a string"}]},
        {"advisories": ["not a dict"]},
    ],
)
def test_accept_ignores_malformed_bodies(body: Any) -> None:
    accept(body)
    assert drain_advisories() == []


def _sender(
    sent: list[dict[str, Any]], on_response: Any = None, reply: Any = None
) -> HeartbeatSender:
    def transport(payload: dict[str, Any]) -> Any:
        sent.append(payload)
        return reply

    return HeartbeatSender(
        url="https://ingest.example.com/v1/heartbeat",
        headers={},
        interval=3600.0,
        agent_name="checkout-agent",
        instance_id=_id(),
        tracker=OpenRootSpanTracker(),
        transport=transport,
        on_response=on_response,
    )


def test_payload_announces_advisories_only_when_a_handler_is_wired() -> None:
    sent: list[dict[str, Any]] = []
    _sender(sent)._send_ping()  # noqa: SLF001
    _sender(sent, on_response=lambda _r: None)._send_ping()  # noqa: SLF001
    assert "accepts" not in sent[0]
    assert sent[1]["accepts"] == ["advisories"]


def test_a_failing_response_handler_never_raises() -> None:
    def boom(_response: Any) -> None:
        raise RuntimeError("boom")

    _sender([], on_response=boom, reply={"advisories": []})._send_ping()  # noqa: SLF001


@pytest.mark.parametrize(("status", "body"), [(204, b""), (200, b"not json")])
def test_http_transport_hands_nothing_on_for_an_empty_or_unparsable_body(
    status: int, body: bytes
) -> None:
    assert _post_to_server(status, body) == []


def test_http_transport_hands_a_json_body_to_the_handler() -> None:
    advisory_id = _id()
    body = json.dumps({"advisories": [{"id": advisory_id, "text": "try lookup_index"}]})
    responses = _post_to_server(200, body.encode())
    assert responses == [{"advisories": [{"id": advisory_id, "text": "try lookup_index"}]}]


def _post_to_server(status: int, body: bytes) -> list[Any]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server API
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            if body:
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    responses: list[Any] = []
    try:
        sender = HeartbeatSender(
            url=f"http://127.0.0.1:{server.server_port}/v1/heartbeat",
            headers={},
            interval=3600.0,
            agent_name="e2e-agent",
            instance_id=_id(),
            tracker=OpenRootSpanTracker(),
            on_response=responses.append,
        )
        sender._send_ping()  # noqa: SLF001
    finally:
        server.shutdown()
        server.server_close()
    return responses
