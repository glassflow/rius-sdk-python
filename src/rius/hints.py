"""Hints received on the heartbeat response.

A hint is a short note from Rius about a problem it detected in one of this
instance's open runs. It is advice, not a command: the SDK queues it, and the
agent's code drains it at a safe point (typically before its next model call)
and decides what to do with it.

The heartbeat contract's "Hints" section is normative; this module implements
the SDK side of it:

- The response body is "no hints" unless it is ``{"format": <a format we
  listed in accepts>, "hints": [...]}``. Unknown fields are ignored.
- Every hint the inbox processes is acked, whether it was queued or dropped
  (duplicate, failed verification, not one of this process's open traces,
  over the per-trace budget, malformed). An ack means "stop sending this".
- ``expires_at`` is never compared with the local clock: expiry is the
  receiver's job, and a skewed client clock would drop every hint silently.
- Nothing here raises into the heartbeat ping or into user code.

All state lives on one :class:`HintInbox` per client, so a scoped client or a
second ``init()`` keeps its own hints.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from opentelemetry import trace

from .config import DEFAULT_HINTS_PER_TRACE

logger = logging.getLogger(__name__)

# The capability strings this SDK sends in ``accepts``; a response in any
# other ``format`` is ignored.
ACCEPTED_FORMATS: tuple[str, ...] = ("hints",)
# The SDK reads at most this much of a response body; a longer one is "no hints".
MAX_RESPONSE_BYTES = 64 * 1024
# ``acked`` holds at most this many ids, and the encoded request at most this
# many bytes; ids that do not fit are carried to the next ping.
MAX_ACKED_PER_PING = 16
MAX_REQUEST_BYTES = 3584

_SEEN_IDS = 1024
_DEFAULT_MAX_QUEUED = 32
# Acks waiting for a 2xx. Bounded so a receiver that never answers cannot grow
# it; an id lost here only means the hint comes back, is recognised as a
# duplicate and is acked again.
_MAX_PENDING_ACKS = 1024
_TRACE_ID = re.compile(r"[0-9a-f]{32}")

ParamValue = str | int | float | bool


@dataclass(frozen=True)
class Hint:
    """One hint, as delivered on the heartbeat response.

    ``created_at`` and ``expires_at`` are the writer's RFC3339 UTC stamps,
    kept as strings: the SDK never compares them with its own clock.
    ``trace_id`` is the root trace of the run the hint is about (32 lowercase
    hex), or ``None`` for a hint about the instance as a whole.
    """

    id: str
    code: str
    template_version: int
    text: str
    # Read-only; left out of the hash so a hint stays hashable.
    params: Mapping[str, ParamValue] = field(hash=False)
    trace_id: str | None
    span_ids: tuple[str, ...]
    created_at: str
    expires_at: str


def verify_hint(raw: Mapping[str, Any]) -> bool:
    """Whether a received hint may be accepted. A hint that fails is dropped and acked.

    Accepts everything for now: hints are not signed yet. Signature checking
    goes here once the receiver delivers signed hints.
    """
    return True


def current_trace_id() -> str | None:
    """The trace id of the current span context (32 lowercase hex), if any."""
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")


def parse_response(body: bytes | None) -> list[Any]:
    """The raw ``hints`` list of a response body, or ``[]`` for "no hints"."""
    if not body or len(body) > MAX_RESPONSE_BYTES:
        return []
    try:
        decoded = json.loads(body)
    except (ValueError, RecursionError):
        return []
    if not isinstance(decoded, dict) or decoded.get("format") not in ACCEPTED_FORMATS:
        return []
    hints = decoded.get("hints")
    return hints if isinstance(hints, list) else []


def _is_param(value: object) -> bool:
    return isinstance(value, (str, int, float, bool))


def _to_hint(raw: Mapping[str, Any]) -> Hint | None:
    """A :class:`Hint` from one wire hint, or ``None`` when it is malformed."""
    code, version, text = raw.get("code"), raw.get("template_version"), raw.get("text")
    created_at, expires_at = raw.get("created_at"), raw.get("expires_at")
    if not (
        isinstance(code, str)
        and isinstance(version, int)
        and not isinstance(version, bool)
        and isinstance(text, str)
        and isinstance(created_at, str)
        and isinstance(expires_at, str)
    ):
        return None
    params = raw.get("params", {})
    if not isinstance(params, dict) or not all(
        isinstance(key, str) and _is_param(value) for key, value in params.items()
    ):
        return None
    span_ids = raw.get("span_ids", [])
    if not isinstance(span_ids, list) or not all(isinstance(s, str) for s in span_ids):
        return None
    trace_id = raw.get("trace_id")
    if trace_id is not None:
        if not isinstance(trace_id, str) or not _TRACE_ID.fullmatch(trace_id.lower()):
            return None
        trace_id = trace_id.lower()
    return Hint(
        id=raw["id"],
        code=code,
        template_version=version,
        text=text,
        params=MappingProxyType(dict(params)),
        trace_id=trace_id,
        span_ids=tuple(span_ids),
        created_at=created_at,
        expires_at=expires_at,
    )


class HintInbox:
    """One client's hints: the queue, the ids it has seen, and the acks to send.

    The heartbeat thread feeds it each response body (:meth:`accept`) and asks
    it for the ids to ack; the agent drains it from any thread.
    """

    def __init__(
        self,
        open_traces: Callable[[], Sequence[str]],
        *,
        per_trace: int = DEFAULT_HINTS_PER_TRACE,
        max_queued: int = _DEFAULT_MAX_QUEUED,
        on_hint: Callable[[Hint], object] | None = None,
        verify: Callable[[Mapping[str, Any]], bool] = verify_hint,
    ) -> None:
        self._open_traces = open_traces
        self._per_trace = per_trace
        self._max_queued = max_queued
        self._on_hint = on_hint
        self._verify = verify
        self._on_hint_warned = False
        self._lock = threading.Lock()
        self._queue: deque[Hint] = deque()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._accepted_per_trace: dict[str, int] = {}
        self._pending_acks: OrderedDict[str, None] = OrderedDict()
        self._skip_acks_once = False

    # -- heartbeat thread ---------------------------------------------------

    def accept(self, body: bytes | None) -> None:
        """Process the hints in one response body. Never raises."""
        try:
            accepted = self._accept(parse_response(body))
        except Exception as exc:  # noqa: BLE001 — the hint path never breaks a ping
            logger.debug("rius hints: could not process a heartbeat response: %s", exc)
            return
        for hint in accepted:
            self._notify(hint)

    def _accept(self, raw_hints: list[Any]) -> list[Hint]:
        open_traces = {t.lower() for t in self._open_traces()}
        accepted: list[Hint] = []
        with self._lock:
            # The budget is per run: forget the counts of runs that have ended.
            for trace_id in [t for t in self._accepted_per_trace if t not in open_traces]:
                del self._accepted_per_trace[trace_id]
            for raw in raw_hints:
                if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
                    continue  # nothing to ack it by
                hint_id: str = raw["id"]
                self._ack(hint_id)
                if hint_id in self._seen:
                    continue  # a redelivery: acking it again is all it needs
                self._remember(hint_id)
                hint = self._admit(raw, open_traces)
                if hint is not None:
                    accepted.append(hint)
        return accepted

    def _admit(self, raw: dict[str, Any], open_traces: set[str]) -> Hint | None:
        """Queue one new hint if every rule lets it through (lock held)."""
        hint = _to_hint(raw)
        if hint is None:
            logger.debug("rius hints: dropped malformed hint %s", raw["id"])
            return None
        if not self._verify(raw):
            logger.debug("rius hints: dropped hint %s, verification failed", hint.id)
            return None
        if hint.trace_id is not None:
            if hint.trace_id not in open_traces:
                logger.debug("rius hints: dropped hint %s, its trace is not open here", hint.id)
                return None
            accepted = self._accepted_per_trace.get(hint.trace_id, 0)
            if accepted >= self._per_trace:
                logger.debug("rius hints: dropped hint %s, its run's budget is spent", hint.id)
                return None
            self._accepted_per_trace[hint.trace_id] = accepted + 1
        self._queue.append(hint)
        while len(self._queue) > self._max_queued:
            dropped = self._queue.popleft()
            logger.debug("rius hints: queue full, dropped the oldest hint %s", dropped.id)
        return hint

    def _remember(self, hint_id: str) -> None:
        self._seen[hint_id] = None
        while len(self._seen) > _SEEN_IDS:
            self._seen.popitem(last=False)

    def _ack(self, hint_id: str) -> None:
        self._pending_acks[hint_id] = None
        while len(self._pending_acks) > _MAX_PENDING_ACKS:
            self._pending_acks.popitem(last=False)

    def _notify(self, hint: Hint) -> None:
        if self._on_hint is None:
            return
        try:
            self._on_hint(hint)
        except Exception as exc:  # noqa: BLE001 — a callback bug never breaks the ping
            if self._on_hint_warned:
                logger.debug("rius on_hint raised for hint %s: %r", hint.id, exc)
            else:
                self._on_hint_warned = True
                logger.warning(
                    "rius on_hint raised for hint %s (%r); the hint is still queued. "
                    "Further on_hint errors log at DEBUG",
                    hint.id,
                    exc,
                )

    def acks_to_send(self) -> list[str]:
        """The ids for the next ping's ``acked``: at most 16, oldest first.

        Empty once after a ``413`` (:meth:`request_too_large`).
        """
        with self._lock:
            if self._skip_acks_once:
                self._skip_acks_once = False
                return []
            return list(self._pending_acks)[:MAX_ACKED_PER_PING]

    def acks_delivered(self, ids: Sequence[str]) -> None:
        """A ping that carried ``ids`` got a 2xx: stop sending them."""
        with self._lock:
            for hint_id in ids:
                self._pending_acks.pop(hint_id, None)

    def request_too_large(self) -> None:
        """A ping was refused with 413: send the next one with ``acked`` empty."""
        with self._lock:
            self._skip_acks_once = True

    def reset(self) -> None:
        """Forget everything; a forked child's hints are not its parent's.

        The lock is replaced, not taken: another parent thread may have held it
        at the fork, and it would never be released in the child.
        """
        self._lock = threading.Lock()
        self._queue.clear()
        self._seen.clear()
        self._accepted_per_trace.clear()
        self._pending_acks.clear()
        self._skip_acks_once = False

    # -- agent side ---------------------------------------------------------

    def drain(self, trace_id: str | None) -> list[Hint]:
        """Remove and return the hints for ``trace_id`` plus the instance-wide ones.

        ``None`` returns the instance-wide hints only. Oldest first.
        """
        wanted = None if trace_id is None else trace_id.lower()
        drained: list[Hint] = []
        kept: deque[Hint] = deque()
        with self._lock:
            for hint in self._queue:
                (drained if hint.trace_id in (None, wanted) else kept).append(hint)
            self._queue = kept
        # This is where a hint is handed to the agent: the place to record its
        # delivery on the current span once the semantic conventions name it.
        return drained
