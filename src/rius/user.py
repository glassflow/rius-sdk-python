"""Users: attribute every span of a request to the end user it served.

The caller supplies the id (``user()`` for a scope) and ``UserSpanProcessor``
stamps it as the ``user.id`` attribute on every span started in scope. That
key is the one OpenInference defines, Langfuse reads natively, and the
OpenTelemetry registry lists; the sink also accepts OTel's ``enduser.id`` from
third-party instrumentors, but this SDK emits one name for one fact.

Stamping happens in ``on_start``, as for sessions: the sink derives its
``UserId`` column per span, and pending snapshots are built at span start from
the identity allowlist, so an attribute set later would reach neither.

Two deliberate differences from ``session()``:

* No process-wide default and no environment variable. A user is a property
  of a request, and a global default would attribute every request a process
  ever handles to one person.
* Nothing is minted when the caller passes nothing. A session without an id
  is still a session; a span without a user is simply anonymous, and the
  backend treats an empty ``UserId`` as exactly that.

The id rides OTel context, so scopes nest, unwind with the block even on
error, and follow async tasks the same way the active span does.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import Span, SpanProcessor

from .semconv import USER_ID

_USER_KEY = otel_context.create_key("rius-user-id")


@contextmanager
def user(user_id: str) -> Iterator[str]:
    """Scope every span started in the block to one end user.

    Pass the application's own identifier for the person the request serves.
    Prefer an opaque id over an email: it is stored on every span and shown in
    the console. When only sensitive identifiers exist, hash them first (OTel's
    ``user.hash`` / ``enduser.pseudo.id`` guidance). Nested scopes override
    outer ones. The id is yielded so it can be logged alongside the work.

    Example:

    ```python
    with rius.user(request.user_id):
        handle(request)  # every span of the request carries user.id
    ```
    """
    token = otel_context.attach(otel_context.set_value(_USER_KEY, user_id))
    try:
        yield user_id
    finally:
        otel_context.detach(token)


class UserSpanProcessor(SpanProcessor):
    """Stamps ``user.id`` on every span started inside a ``user()`` scope.

    Outside any scope the attribute is not set; there is no default to fall
    back to, by design (see the module docstring).
    """

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        value = otel_context.get_value(_USER_KEY, context=parent_context)
        if isinstance(value, str):
            span.set_attribute(USER_ID, value)

    def on_end(self, span: Any) -> None:  # pragma: no cover - nothing to do
        pass

    def shutdown(self) -> None:  # pragma: no cover - nothing to hold
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True
