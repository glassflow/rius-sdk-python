"""Intentional skips: saying a run had nothing to do, on purpose.

A run that finished without calling a model or a tool looks the same to the
backend whether it meant to (an empty queue, a guard that decided no work was
needed) or its work went missing. Rius's empty-run detection (CF-05) flags the
second; :func:`mark_run_skipped` tells it about the first, so the run is an
informational finding instead of a warning and stays out of the agent's
baseline.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.trace import Span

from .semconv import OPENINFERENCE_SPAN_KIND, SpanKind

logger = logging.getLogger(__name__)

#: The agent-root attribute that marks an intentional skip. Planned in the
#: rius-semconv registry (``RIUS_RUN_SKIPPED_REASON``); it moves to the
#: generated constants once a registry release ships the row.
RUN_SKIPPED_REASON = "rius.run.skipped_reason"

#: The longest reason sent. The backend cuts what it shows to 200 characters
#: too, and strips control characters.
MAX_SKIPPED_REASON_CHARS = 200


def _clean(reason: str) -> str:
    if not isinstance(reason, str):
        raise TypeError(f"reason must be a str, got {type(reason).__name__}")
    cleaned = " ".join(reason.split())
    if not cleaned:
        raise ValueError("reason must not be empty: an empty reason is no skip marker")
    return cleaned[:MAX_SKIPPED_REASON_CHARS]


def mark_skipped(span: Span, reason: str) -> bool:
    """Mark ``span``, an agent span, as a run skipped on purpose.

    Returns True when the attribute was set, False (with a debug log) when the
    span is not recording or is not an AGENT span: the marker means something
    only on the run's agent root, and setting it elsewhere would be silently
    ignored. Raises ``ValueError`` for an empty or whitespace-only reason and
    ``TypeError`` for a reason that is not a string.
    """
    cleaned = _clean(reason)
    if not span.is_recording():
        logger.debug("mark_run_skipped: the span is not recording; nothing marked")
        return False
    attributes = getattr(span, "attributes", None) or {}
    if attributes.get(OPENINFERENCE_SPAN_KIND) != SpanKind.AGENT.value:
        logger.debug("mark_run_skipped: the span is not an AGENT span; nothing marked")
        return False
    span.set_attribute(RUN_SKIPPED_REASON, cleaned)
    return True


def mark_run_skipped(reason: str) -> bool:
    """Mark the current agent run as intentionally skipped, with a short reason.

    Call it inside the run's agent span, before it ends: in a function
    decorated ``@observe(kind=SpanKind.AGENT)`` or in
    ``start_as_current_span(..., kind=SpanKind.AGENT)``, when the run decides
    it has nothing to do::

        @rius.observe(kind=rius.SpanKind.AGENT)
        def triage() -> None:
            tickets = fetch_new_tickets_from_cache()
            if not tickets:
                rius.mark_run_skipped("no new tickets")
                return
            ...

    The backend reads the marker on the run's root agent span only: mark the
    outermost agent of the trace. A marked run that still calls a model or a
    tool is judged on that work, not on the marker.

    ``reason`` is shown to people as quoted text: keep it short and
    operational ("no new tickets", "input already processed"). Whitespace is
    collapsed and it is cut to 200 characters. It is metadata, not content, so
    it is sent even with ``capture_content=False`` and is not masked: never put
    user input, message text or secrets in it.

    Returns True when the current span was marked, False (with a debug log)
    when the current span is not a recording AGENT span. Raises ``ValueError``
    for an empty reason.
    """
    return mark_skipped(trace.get_current_span(), reason)
