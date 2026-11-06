"""Local stand-in for the Rius receiver that answers heartbeats with hints.

Run it, point the agent's RIUS_ENDPOINT at it, then queue a hint by hand:

    uv run python spike/hints/fake_receiver.py [--port 8787] [--interval 5]
    curl -X POST localhost:8787/admin/hint -d '{"text": "..."}'

It follows the receiver's side of the heartbeat contract ("Hints" in
argus-core's docs/heartbeat-contract.md), so the SDK's real hint client runs
against it unchanged:

- Hints are returned only to an authenticated ping that sends
  ``accepts: ["hints"]``: its ``Rius-Instance-Secret`` must derive its
  ``instance_id``. A derived (v8) id without the secret gets 403; a legacy v4
  id without one is accepted and never gets hints.
- A hint stays pending until the SDK acks it, so it can be returned more than
  once (at-least-once), and is dropped once it expires.
- A run-scoped hint is returned while its run is in ``open_traces``. When the
  list is complete and the hint is older than one ping interval, a run missing
  from it has finished, and the hint is dropped undelivered. "One ping
  interval" is ``--interval`` here (default 5 s, the agent's); the real
  receiver uses ARGUS_HEARTBEAT_PING_INTERVAL (default 15 s), so drops come
  later on a real stack.
- A ``stopped`` ping removes the instance's hints. One response carries at
  most 8 hints and 16 KiB of them; the rest wait for the next ping.

``/admin/hint`` takes ``text`` and optionally ``code``, ``template_version``,
``params``, ``span_ids``, ``trace_id`` and ``instance_id``. It defaults to the
last instance that pinged with authentication and to that instance's first
open run, and only queues run-scoped hints, as the hint writer does.
OTLP traces are accepted and dropped.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SECRET_HEADER = "Rius-Instance-Secret"
MIN_SECRET_BYTES, MAX_SECRET_BYTES = 32, 64
OPEN_TRACES_CAP = 32
# The most hints, and bytes of them, one response carries; the rest wait.
MAX_HINTS_PER_RESPONSE, MAX_RESPONSE_BYTES = 8, 16 * 1024
LIFETIME = timedelta(minutes=10)
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

lock = threading.Lock()
pending: dict[str, list[dict[str, Any]]] = {}
open_runs: dict[str, tuple[list[str], int]] = {}
last_instance: str | None = None
interval = timedelta(seconds=5)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def derive_instance_id(secret: bytes) -> str:
    """The instance id a secret derives: SHA-256(secret)[:16] as a UUIDv8."""
    b = bytearray(hashlib.sha256(secret).digest()[:16])
    b[6] = (b[6] & 0x0F) | 0x80
    b[8] = (b[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(b)))


def is_derived(instance_id: str) -> bool:
    try:
        return uuid.UUID(instance_id).version == 8
    except ValueError:
        return False


def authenticate(instance_id: str, header: str | None) -> int | bool:
    """True if authenticated, False for a legacy ping, else the status to refuse it with.

    As the receiver's CheckInstanceSecret: a derived (v8) id without its
    secret, or a secret that does not derive the id exactly, is 403; a header
    that is not unpadded base64url of 32 to 64 bytes is 400. Any other id
    without a secret is legacy.
    """
    header = (header or "").strip()
    if not header:
        return 403 if is_derived(instance_id) else False
    if not re.fullmatch(r"[A-Za-z0-9_-]+", header):
        return 400
    try:
        secret = base64.urlsafe_b64decode(header + "=" * (-len(header) % 4))
    except ValueError:
        return 400
    if not MIN_SECRET_BYTES <= len(secret) <= MAX_SECRET_BYTES:
        return 400
    return True if derive_instance_id(secret) == instance_id else 403


def new_hint_id(now: datetime) -> str:
    """h_ followed by a ULID: 48 bits of milliseconds, 80 random bits."""
    n = (int(now.timestamp() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    return "h_" + "".join(CROCKFORD[(n >> shift) & 31] for shift in range(125, -1, -5))


def stamp(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def due(instance: str, ping: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    """Apply the ping's acks and expiry, then the hints to return on it."""
    if ping.get("stopped"):
        dropped = pending.pop(instance, [])
        if dropped:
            log(f"  stopped: removed {len(dropped)} hint(s)")
        return []
    acked = set(ping.get("acked") or [])
    listed = [t.lower() for t in ping.get("open_traces") or []]
    traces = set(listed)
    count = ping.get("open_trace_count", 0)
    complete = count <= OPEN_TRACES_CAP and len(listed) >= count
    keep: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    size = 0
    for hint in pending.get(instance, []):
        if hint["id"] in acked:
            log(f"  acked {hint['id']}")
            continue
        if datetime.fromisoformat(hint["expires_at"].replace("Z", "+00:00")) <= now:
            log(f"  expired {hint['id']}")
            continue
        run = hint.get("trace_id")
        if run is None or run in traces or not complete:
            size += len(json.dumps(hint))
            if len(out) < MAX_HINTS_PER_RESPONSE and size <= MAX_RESPONSE_BYTES:
                out.append(hint)
        elif now - datetime.fromisoformat(hint["created_at"].replace("Z", "+00:00")) > interval:
            log(f"  dropped {hint['id']}: run {run[:8]} has finished")
            continue
        keep.append(hint)
    pending[instance] = keep
    return out


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path not in ("/v1/heartbeat", "/admin/hint"):  # /v1/traces and the rest
            self._reply(200)
            return
        try:
            item = json.loads(body)
        except ValueError:
            self._reply(400, {"error": "body is not JSON"})
            return
        if not isinstance(item, dict):
            self._reply(400, {"error": "body is not a JSON object"})
        elif self.path == "/v1/heartbeat":
            self._heartbeat(item)
        else:
            self._queue(item)

    def _heartbeat(self, ping: dict[str, Any]) -> None:
        global last_instance
        instance = ping.get("instance_id")
        if not isinstance(instance, str) or not instance:
            self._reply(400, {"error": "no instance_id"})
            return
        authenticated = authenticate(instance, self.headers.get(SECRET_HEADER))
        if not isinstance(authenticated, bool):
            log(f"heartbeat {instance[:8]} refused with {authenticated}: bad instance secret")
            self._reply(authenticated)
            return
        now = datetime.now(timezone.utc)
        with lock:
            hints: list[dict[str, Any]] = []
            if authenticated and "hints" in (ping.get("accepts") or []):
                last_instance = instance
                open_runs[instance] = (
                    [t.lower() for t in ping.get("open_traces") or []],
                    ping.get("open_trace_count", 0),
                )
                hints = due(instance, ping, now)
        log(
            f"heartbeat {instance[:8]} open_traces={ping.get('open_trace_count', 0)} "
            f"authenticated={authenticated} -> {len(hints)} hint(s)"
        )
        if hints:
            self._reply(200, {"format": "hints", "hints": hints})
        else:
            self._reply(204)

    def _queue(self, item: dict[str, Any]) -> None:
        if not isinstance(item.get("text"), str) or not item["text"]:
            self._reply(400, {"error": "text is required"})
            return
        now = datetime.now(timezone.utc)
        with lock:
            instance = item.get("instance_id") or last_instance
            if instance is None:
                self._reply(409, {"error": "no instance has pinged with hints on yet"})
                return
            runs = open_runs.get(instance, ([], 0))[0]
            trace_id = (item.get("trace_id") or (runs[0] if runs else "")).lower() or None
            if trace_id is None:
                self._reply(409, {"error": "the instance has no open run"})
                return
            hint = {
                "id": new_hint_id(now),
                "code": item.get("code", "CF-TEST"),
                "template_version": item.get("template_version", 1),
                "text": item["text"],
                "params": item.get("params", {}),
                "trace_id": trace_id,
                "span_ids": item.get("span_ids", []),
                "created_at": stamp(now),
                "expires_at": stamp(now + LIFETIME),
            }
            pending.setdefault(instance, []).append(hint)
        log(f"queued {hint['id']} for {instance[:8]} run {trace_id[:8]}")
        self._reply(202, {"instance_id": instance, **hint})

    def _reply(self, status: int, payload: Any = None) -> None:
        data = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status)
        if data:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--interval", type=float, default=5, help="the SDK's ping interval, s")
    opts = parser.parse_args()
    interval = timedelta(seconds=opts.interval)
    log(f"fake receiver on http://127.0.0.1:{opts.port}")
    ThreadingHTTPServer(("127.0.0.1", opts.port), Handler).serve_forever()
