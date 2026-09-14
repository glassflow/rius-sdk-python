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
_active: trace.Tracer | None = None


def publish(provider: trace.TracerProvider) -> None:
    """Make ``provider``'s SDK tracer the one the helpers use."""
    global _active
    tracer = provider.get_tracer(TRACER_NAME, __version__)
    with _lock:
        _active = tracer


def withdraw(provider: trace.TracerProvider) -> None:
    """Stop using ``provider``'s tracer, if it is the published one.

    Keyed on the provider so a stale client's ``shutdown()`` cannot pull the
    rug from under a newer ``init()``.
    """
    global _active
    with _lock:
        if _active is not None and _active is _tracer_of(provider):
            _active = None


def _tracer_of(provider: trace.TracerProvider) -> trace.Tracer:
    # SDK providers cache tracers per (name, version), so this returns the
    # same object publish() stored; the API's proxy provider does not, but a
    # proxy is never what init() publishes.
    return provider.get_tracer(TRACER_NAME, __version__)


def sdk_tracer() -> trace.Tracer:
    """The tracer for SDK-created spans: the active client's, else the global."""
    tracer = _active  # single attribute read; no lock needed on the hot path
    if tracer is not None:
        return tracer
    return trace.get_tracer(TRACER_NAME, __version__)
