"""Advisories delivered on heartbeat responses (spike).

Rius can answer a heartbeat with ``{"advisories": [...]}``: short notes about
a problem it detected in this agent's runs. The heartbeat thread queues them
here; the agent drains the queue at a safe point (before its next model call)
and decides how to show them to the model.

Each advisory is a dict with at least ``id`` and ``text``. Other fields
(``code``, ``trace_id``, ...) pass through untouched. Advisories are
de-duplicated by ``id`` so a redelivery never reaches the model twice.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from typing import Any

logger = logging.getLogger(__name__)

_MAX_QUEUED = 32
_MAX_SEEN_IDS = 1024
_MAX_TEXT_LEN = 2000

_lock = threading.Lock()
_queue: deque[dict[str, Any]] = deque(maxlen=_MAX_QUEUED)
_seen: OrderedDict[str, None] = OrderedDict()


def accept(response: Any) -> None:
    """Queue the advisories found in a heartbeat response body."""
    if not isinstance(response, dict):
        return
    items = response.get("advisories")
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        advisory_id, text = item.get("id"), item.get("text")
        if not isinstance(advisory_id, str) or not isinstance(text, str) or not text:
            continue
        with _lock:
            if advisory_id in _seen:
                continue
            _seen[advisory_id] = None
            while len(_seen) > _MAX_SEEN_IDS:
                _seen.popitem(last=False)
            _queue.append({**item, "text": text[:_MAX_TEXT_LEN]})
        logger.info("rius advisory received: id=%s code=%s", advisory_id, item.get("code"))


def drain_advisories() -> list[dict[str, Any]]:
    """Return and clear the advisories received since the last drain."""
    with _lock:
        drained = list(_queue)
        _queue.clear()
    return drained
