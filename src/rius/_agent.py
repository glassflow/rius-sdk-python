"""The configured agent name, readable from the span helpers.

An AGENT span should say which agent it invokes. The caller usually names it,
but a process that runs a single agent already told us its name at ``init()``,
and the conventions make the key Conditionally Required rather than optional,
so the helpers fall back to the configured name instead of leaving the span
silent. The value is also on the Resource; a span carrying it too is not
redundant, because resource attributes do not survive every collector pipeline
and a consumer reading one span in isolation still needs the answer.

The span helpers cannot import ``client`` (``__init__`` imports both, and the
dependency runs the other way), so ``init()`` publishes the name here and
``shutdown()`` withdraws it, exactly as ``_tracer`` does for the tracer. Keyed
on the provider for the same reason: a stale client's ``shutdown()`` must not
pull the name out from under a newer ``init()``.
"""

from __future__ import annotations

import threading
from typing import Any

from .config import DEFAULT_SERVICE_NAME
from .semconv import SpanKind

_lock = threading.Lock()
_active: tuple[Any, str] | None = None


def publish(provider: Any, name: str) -> None:
    """Make ``name`` the fallback agent name for spans this process opens."""
    global _active
    with _lock:
        _active = (provider, name)


def withdraw(provider: Any) -> None:
    """Drop the published name, if ``provider`` is the one that published it."""
    global _active
    with _lock:
        if _active is not None and _active[0] is provider:
            _active = None


def configured_agent_name() -> str | None:
    """The agent name from ``init()``, or ``None`` when no client is active."""
    active = _active  # single attribute read; no lock needed on the read path
    return active[1] if active is not None else None


def resolve_agent_name(agent_name: str | None, kind: SpanKind) -> str | None:
    """Explicit name, else the configured one; never the span or function name.

    A tool name falls back to the span name because the two were historically
    the same string. An agent name does not: a function name is not an agent's
    identity, so the fallback is the name this process was configured with,
    which is a real answer, and nothing after that.

    Except when that configured name is the placeholder. Nothing was named,
    and ``DEFAULT_SERVICE_NAME`` is what BOTH the service name and the agent
    name resolve to in that case, so emitting it would claim an identity the
    caller never gave and would name every such span ``invoke_agent
    unknown_service``. A bare ``invoke_agent`` is the honest answer, and it is
    what the conventions prescribe when the name is not available.

    A caller who literally names their agent ``"unknown_service"`` is treated
    as not having named one. That is the placeholder's meaning everywhere else
    in the pipeline, and the alternative is threading a "was this defaulted"
    flag through the client for a case nobody has.
    """
    if kind is not SpanKind.AGENT:
        return None
    if agent_name is not None:
        return agent_name
    configured = configured_agent_name()
    return None if configured == DEFAULT_SERVICE_NAME else configured
