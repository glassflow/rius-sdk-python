"""Agent-lifetime heartbeat sender (payload v1).

The heartbeat answers one question traces cannot: is this agent process
alive right now? Spans export only when they finish, so an idle or crashed
agent is indistinguishable from a healthy quiet one. Heartbeats are a
process-lifetime signal, fully independent of trace traffic: a daemon
thread pings ``POST /v1/heartbeat`` from ``init()`` until process exit.

Contract highlights (the spec is normative; this module implements it):

- First ping immediately at start (the agent appears without waiting an
  interval), then every ``interval`` seconds.
- Graceful shutdown (``client.shutdown()`` / ``atexit``) sends a final
  ``stopped: true`` ping. No signal handlers are installed, because a
  library must not own process signals; an unhandled SIGTERM/SIGKILL means
  no stopped ping, and the backend's stale→gone path covers exactly that.
- Never raises into user code. Pings have a short timeout, are never
  retried or queued (liveness is only true fresh; a late heartbeat is
  misinformation), and delivery problems warn once per process.
- ``fork()``: the child re-arms with a NEW ``instance_id``, so one identity
  never speaks for two processes.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import urllib.request
import uuid
import weakref
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from . import __version__

logger = logging.getLogger(__name__)

PAYLOAD_VERSION = 1
OPEN_TRACES_CAP = 32
_PING_TIMEOUT_S = 3.0
# The final stopped ping runs inside atexit: it must never hold the user's
# process exit hostage, so it gets a tighter budget than regular pings (a
# missed stopped ping just reports as gone instead of stopped — acceptable).
_FINAL_PING_TIMEOUT_S = 1.0

# os.register_at_fork callbacks can NEVER be unregistered, so per-sender
# registration would leak a callback + sender reference for every init()
# in a long-lived app. One module-level handler + a weak registry instead:
# stopped/collected senders simply vanish from the set.
_active_senders: weakref.WeakSet[HeartbeatSender] = weakref.WeakSet()
_fork_hook_installed = False
_fork_hook_lock = threading.Lock()


def _reset_active_senders_in_child() -> None:  # pragma: no cover — fork-only
    for sender in list(_active_senders):
        sender._reset_in_child()  # noqa: SLF001 — module-internal


def _install_fork_hook() -> None:
    global _fork_hook_installed
    with _fork_hook_lock:
        if _fork_hook_installed or not hasattr(os, "register_at_fork"):
            return
        os.register_at_fork(after_in_child=_reset_active_senders_in_child)
        _fork_hook_installed = True


class OpenRootSpanTracker(SpanProcessor):
    """Tracks trace ids of currently-open root spans.

    A root span is one started with no parent context; children of the same
    trace never touch the set. This is what lets the backend derive
    ``running`` vs ``ready`` from the heartbeat payload alone.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # trace_id (32-hex) -> count of open root spans in that trace (a
        # trace id normally has one root, but be safe about duplicates).
        self._open: dict[str, int] = {}

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        if span.parent is not None:
            return
        context = span.get_span_context()
        if context is None:
            return
        self._on_root_start(format(context.trace_id, "032x"))

    def on_end(self, span: ReadableSpan) -> None:
        if span.parent is not None:
            return
        context = span.get_span_context()
        if context is None:
            return
        trace_id = format(context.trace_id, "032x")
        with self._lock:
            count = self._open.get(trace_id, 0) - 1
            if count <= 0:
                self._open.pop(trace_id, None)
            else:
                self._open[trace_id] = count

    def _on_root_start(self, trace_id: str) -> None:
        with self._lock:
            self._open[trace_id] = self._open.get(trace_id, 0) + 1

    def open_trace_ids(self) -> list[str]:
        """Trace ids of currently-open root spans (insertion order)."""
        with self._lock:
            return list(self._open)

    def shutdown(self) -> None:  # pragma: no cover — SpanProcessor API
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # pragma: no cover
        return True


def _http_transport(url: str, headers: dict[str, str]) -> Callable[[dict[str, Any], float], None]:
    """Default transport: a plain POST with a per-call timeout, no retries.

    TLS certificate verification is urllib's default and is deliberately not
    configurable here: a liveness signal must not become a reason to accept
    unverified endpoints.
    """

    def send(payload: dict[str, Any], timeout: float) -> None:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout):
            pass  # 2xx is success; the body is ignored by contract

    return send


class HeartbeatSender:
    """Daemon thread pinging the heartbeat endpoint for the process lifetime."""

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        interval: float,
        agent_name: str,
        tracker: OpenRootSpanTracker,
        transport: Callable[[dict[str, Any]], None] | None = None,
        ping_timeout: float = _PING_TIMEOUT_S,
        final_ping_timeout: float = _FINAL_PING_TIMEOUT_S,
    ) -> None:
        self._interval = interval
        self._agent_name = agent_name
        self._tracker = tracker
        self._ping_timeout = ping_timeout
        self._final_ping_timeout = final_ping_timeout
        if transport is not None:
            # Injected transports (tests) take the payload only.
            self._send: Callable[[dict[str, Any], float], None] = lambda p, _t: transport(p)
        else:
            self._send = _http_transport(url, headers)
        self._instance_id = str(uuid.uuid4())
        self._stop_event = threading.Event()
        self._stopped = False
        self._lock = threading.Lock()
        self._delivery_warned = False
        self._thread: threading.Thread | None = None

    @property
    def instance_id(self) -> str:
        """Identity of one process lifetime; fresh per process (and per fork)."""
        return self._instance_id

    def start(self) -> None:
        """Start the daemon thread; first ping goes out immediately."""
        self._thread = threading.Thread(target=self._run, name="glassflow-heartbeat", daemon=True)
        self._thread.start()
        atexit.register(self.stop)
        # A forked child must never reuse the parent's identity; the shared
        # module-level fork hook re-arms every live sender with a fresh
        # instance_id (same pattern the OTel exporter uses).
        _active_senders.add(self)
        _install_fork_hook()

    def stop(self) -> None:
        """Stop the thread and send the final ``stopped`` ping. Idempotent.

        Exit-latency budget: an in-flight ping can hold the join for at most
        ``ping_timeout``; the final ping gets ``final_ping_timeout``. A dead
        endpoint therefore delays process exit by a bounded few seconds, not
        by the interval.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        _active_senders.discard(self)
        atexit.unregister(self.stop)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._ping_timeout + 1.0)
        self._send_ping(stopped=True)

    def _run(self) -> None:
        self._send_ping()
        while not self._stop_event.wait(self._interval):
            self._send_ping()

    def _reset_in_child(self) -> None:  # pragma: no cover — fork-only path
        if self._stopped:
            return
        self._instance_id = str(uuid.uuid4())
        self._stop_event = threading.Event()
        self.start()

    def __del__(self) -> None:  # pragma: no cover — GC-timing dependent
        atexit.unregister(self.stop)

    def _build_payload(self, *, stopped: bool = False) -> dict[str, Any]:
        open_ids = self._tracker.open_trace_ids()
        payload: dict[str, Any] = {
            "v": PAYLOAD_VERSION,
            "instance_id": self._instance_id,
            "agent_name": self._agent_name,
            # RFC3339 UTC with sub-second precision, Z suffix
            "sent_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "sdk_language": "python",
            "sdk_version": __version__,
            "open_traces": open_ids[:OPEN_TRACES_CAP],
            "open_trace_count": len(open_ids),
        }
        if stopped:
            # Present-and-true only on the final ping; false is never sent.
            payload["stopped"] = True
        return payload

    def _send_ping(self, *, stopped: bool = False) -> None:
        timeout = self._final_ping_timeout if stopped else self._ping_timeout
        try:
            self._send(self._build_payload(stopped=stopped), timeout)
        except Exception as exc:  # noqa: BLE001 — never raises into user code
            if not self._delivery_warned:
                self._delivery_warned = True
                logger.warning("heartbeat delivery failed (%s); further failures log at DEBUG", exc)
            else:
                logger.debug("heartbeat delivery failed: %s", exc)
