"""Agent identity for spans: who is invoked, and who is executing.

``gen_ai.agent.name`` carries TWO different facts, told apart by
``gen_ai.operation.name``:

* on an ``invoke_agent`` span it names the agent BEING INVOKED — the caller's
  choice, resolved by :func:`resolve_agent_name`;
* on an ``execute_tool`` span it names the agent EXECUTING the tool — nobody
  passes it, it is whichever agent's scope the call happens inside, resolved
  by :func:`executing_agent_name`.

Both are Conditionally Required by the conventions. They must never be
conflated: merging them would label a tool span with the agent it *is*, and
an agent span with whoever called it.

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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context

from .config import DEFAULT_SERVICE_NAME
from .semconv import GEN_AI_AGENT_NAME, SpanKind

_lock = threading.Lock()
_active: tuple[Any, str] | None = None

# The ENCLOSING agent, i.e. the one currently executing. Rides OTel context
# rather than a module global for the reasons session.py spells out: scopes
# nest, unwind with their block even on error, and follow async tasks the way
# the active span does. Deliberately NOT derived from the parent span: a tool
# is not always a direct child of its agent span (a chain, a retriever or an
# auto-instrumented span can sit in between), and a parent span's attributes
# are not readable from a child in the OTel API anyway.
_EXECUTING_AGENT_KEY = otel_context.create_key("rius-executing-agent-name")


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
    return _named_agent()


def named_agent(name: str | None) -> str | None:
    """``name``, unless it is the unnamed placeholder, in which case ``None``.

    The suppression rule itself, stated once and applied at BOTH scopes: by
    the span helpers below, and by the resource's ``rius.main_agent.name``,
    which is the same resolved value. A process that named nothing must claim
    no agent identity anywhere.
    """
    return None if name == DEFAULT_SERVICE_NAME else name


def _named_agent() -> str | None:
    """The configured agent name, unless it is the unnamed placeholder.

    Shared by both readings of ``gen_ai.agent.name`` so the suppression rule
    is stated once: a process that named nothing emits nothing.
    """
    return named_agent(configured_agent_name())


def invoked_agent_name(kind: SpanKind, attributes: Mapping[str, str | int]) -> str | None:
    """The agent an AGENT span INVOKES, read back off its own creation attributes.

    Read back rather than re-resolved so the scope a span opens can only ever
    be the name that span actually carries, the way ``compose_span_name``
    takes its target from the same map. On any other kind there is no invoked
    agent, and on a TOOL span the same key means something else entirely.
    """
    if kind is not SpanKind.AGENT:
        return None
    value = attributes.get(GEN_AI_AGENT_NAME)
    return value if isinstance(value, str) else None


def executing_agent_name() -> str | None:
    """The agent EXECUTING right here: the enclosing scope, else the configured one.

    This is what ``gen_ai.agent.name`` means on an ``execute_tool`` span, and
    it is resolved, never passed: the innermost enclosing AGENT scope wins,
    because a supervisor's tool call and its sub-agent's tool call were made
    by different agents. With no scope at all, a single-agent process still
    knows the answer, so the configured name applies under the same
    placeholder suppression the invoked name uses.
    """
    scoped = otel_context.get_value(_EXECUTING_AGENT_KEY)
    if isinstance(scoped, str):
        return scoped
    return _named_agent()


def context_with_executing_agent(
    name: str | None, context: otel_context.Context | None = None
) -> otel_context.Context:
    """``context`` (or the current one) with ``name`` as the executing agent.

    A no-op when ``name`` is ``None``, so an unnamed agent span leaves an
    outer scope standing instead of blanking it.
    """
    current = context if context is not None else otel_context.get_current()
    if name is None:
        return current
    return otel_context.set_value(_EXECUTING_AGENT_KEY, name, current)


@contextmanager
def executing_agent_scope(name: str | None) -> Iterator[None]:
    """Make ``name`` the executing agent for the block; a no-op when ``None``.

    Only the helpers that ACTIVATE context open one — ``start_as_current_span``
    and ``@observe``. ``start_span`` does not activate anything, so it cannot,
    exactly as it cannot scope ``user_id``.
    """
    if name is None:
        yield
        return
    token = otel_context.attach(context_with_executing_agent(name))
    try:
        yield
    finally:
        otel_context.detach(token)
