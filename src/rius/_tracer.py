"""The tracer the SDK's own helpers open spans with.

``start_span``, ``observe``, the generation helpers and the MCP wrapper all
need a tracer, and they used to ask the OpenTelemetry global for one on every
call. That has two costs. The global provider is write-once, so after
``shutdown()`` + ``init()`` the helpers kept handing spans to the dead first
provider while instrumentors, which are re-bound explicitly, followed the new
one. And ``get_tracer()`` is a few microseconds per span, a real share of a
span's cost.

``init()`` therefore publishes the active client's tracer here and
``shutdown()`` withdraws it. Helpers call ``sdk_tracer()``, which returns the
published tracer when a global client is active and otherwise falls back to
the OpenTelemetry global, so code that never called ``init()`` (or only built
scoped clients) behaves exactly as before.
"""

from __future__ import annotations

import threading

from opentelemetry import trace

from . import __version__
from .semconv import TRACER_NAME

_lock = threading.Lock()
# The provider rides along with its tracer so withdraw() can compare
# providers: OTel SDKs before 1.42 hand out a fresh tracer per get_tracer()
# call, so asking the provider again would never match what was published.
_active: tuple[trace.TracerProvider, trace.Tracer] | None = None


def publish(provider: trace.TracerProvider) -> None:
    """Make ``provider``'s SDK tracer the one the helpers use."""
    global _active
    tracer = provider.get_tracer(TRACER_NAME, __version__)
    with _lock:
        _active = (provider, tracer)


def withdraw(provider: trace.TracerProvider) -> None:
    """Stop using ``provider``'s tracer, if it is the published one.

    Keyed on the provider so a stale client's ``shutdown()`` cannot pull the
    rug from under a newer ``init()``.
    """
    global _active
    with _lock:
        if _active is not None and _active[0] is provider:
            _active = None


def sdk_tracer() -> trace.Tracer:
    """The tracer for SDK-created spans: the active client's, else the global."""
    active = _active  # single attribute read; no lock needed on the hot path
    if active is not None:
        return active[1]
    return trace.get_tracer(TRACER_NAME, __version__)
