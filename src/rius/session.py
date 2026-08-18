"""Sessions: group traces of one conversation or agent run under a caller id.

A trace is a causal unit and stays short; a session is a correlation unit the
application assigns, open ended and spanning turns. The caller mints the id
(``session()`` for a scope, ``init(session_id=...)`` for a process-wide
default) and ``SessionSpanProcessor`` stamps it as the OpenInference
``session.id`` attribute on every span started in scope.

Stamping happens in ``on_start``, for two reasons:

* The sink derives its ``SessionId`` column per span, falling back to the
  trace id when the attribute is missing, so a session id set only on the
  root would scatter child spans into per-trace pseudo-sessions.
* Pending snapshots are built at span start from the identity allowlist;
  an attribute set later never reaches them.

The id rides OTel context, not a bare module global, so ``session()`` scopes
nest, unwind with the block even on error, and follow async tasks the same
way the active span does. Threads inherit whatever the closure captured,
exactly like the rest of OTel context.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import Span, SpanProcessor

from .semconv import SESSION_ID

_SESSION_KEY = otel_context.create_key("rius-session-id")


@contextmanager
def session(session_id: str | None = None) -> Iterator[str]:
    """Scope every span started in the block to one session id.

    Pass the application's own id (a conversation id, a job id) to correlate
    with it; call with no argument to mint a fresh UUID for the scope. Either
    way the block's id is yielded, so it can be logged or handed to the next
    turn. Nested scopes override outer ones, and an active scope overrides
    the ``init(session_id=...)`` default.

    There is deliberately no process-wide auto-generation: without an id the
    backend groups each trace as its own session, which stays true in a
    server handling many users, while an auto-minted global id would merge
    every user into one. A generated id is only ever scoped to an explicit
    block, where "this is one session" is the caller's own claim.

    Example:

    ```python
    with rius.session(conversation_id):
        handle_turn(message)  # every span of the turn carries session.id
    ```
    """
    resolved = session_id if session_id is not None else str(uuid.uuid4())
    token = otel_context.attach(otel_context.set_value(_SESSION_KEY, resolved))
    try:
        yield resolved
    finally:
        otel_context.detach(token)


class SessionSpanProcessor(SpanProcessor):
    """Stamps ``session.id`` on every span at start.

    The active ``session()`` scope wins; otherwise ``default_session_id``
    (from ``init(session_id=...)`` / ``RIUS_SESSION_ID``) applies; with
    neither, the attribute is not set and the sink groups the span by its
    trace id.
    """

    def __init__(self, default_session_id: str | None = None) -> None:
        self._default = default_session_id

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        value = otel_context.get_value(_SESSION_KEY, context=parent_context)
        session_id = value if isinstance(value, str) else self._default
        if session_id is not None:
            span.set_attribute(SESSION_ID, session_id)

    def on_end(self, span: Any) -> None:  # pragma: no cover - nothing to do
        pass

    def shutdown(self) -> None:  # pragma: no cover - nothing to hold
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True
