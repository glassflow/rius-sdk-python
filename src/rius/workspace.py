"""Workspaces: route spans from one process to per-customer destinations.

One client, one provider, one batch pipeline; the *destination* is a
context-scoped property. ``workspace(alias)`` sets an OTel context key
(exactly like ``session()``), ``WorkspaceSpanProcessor`` stamps it as a
transient attribute at span start, and ``RoutingSpanExporter`` partitions
each export batch by that attribute, strips it, and forwards every partition
to the exporter registered for its alias. Spans started outside any scope go
to the default destination.

Because the alias rides OTel context, everything started in scope routes
together: ``observe`` wrappers, generations, sessions, and spans created by
auto-instrumentation, which is the property a second client can never give
(instrumentors are process-global and bound to one provider).

Two rules the design enforces or warns about:

- The routing attribute never reaches the wire. The destination's API key is
  what tells the backend which workspace a span belongs to; the alias is
  process-local configuration, so the exporter strips it from copies (the
  same copy-on-write discipline as ``MaskingSpanExporter``).
- One trace, one workspace. The backend derives the workspace from the API
  key per request, so a trace split across scopes would come apart. Starting
  a span under a different alias than its parent's logs a warning; switch
  workspaces at request boundaries, not inside a trace.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .semconv import WORKSPACE_ROUTE

logger = logging.getLogger(__name__)

_WORKSPACE_KEY = otel_context.create_key("rius-workspace-alias")

ExporterFactory = Callable[[str], SpanExporter]


@contextmanager
def workspace(alias: str) -> Iterator[str]:
    """Scope every span started in the block to one workspace destination.

    ``alias`` names a workspace registered via ``init(workspaces={...})`` or
    ``register_workspace()``; the block's spans are exported with that
    workspace's API key. Scopes nest and unwind with the block, and follow
    async tasks the way all OTel context does, but a trace must stay inside
    one workspace: enter the scope at a request boundary, before the root
    span starts.

    Example:

    ```python
    with rius.workspace("acme"):
        handle(request)  # every span of the request lands in acme's workspace
    ```
    """
    if not alias:
        raise ValueError("workspace alias must be a non-empty string")
    token = otel_context.attach(otel_context.set_value(_WORKSPACE_KEY, alias))
    try:
        yield alias
    finally:
        otel_context.detach(token)


class WorkspaceSpanProcessor(SpanProcessor):
    """Stamps the active workspace alias on every span at start.

    Stamping happens in ``on_start`` for the same reason sessions stamp
    there: pending snapshots are built from start-time attributes, and a
    snapshot must route to the same workspace its final span will.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        alias = otel_context.get_value(_WORKSPACE_KEY, context=parent_context)
        if alias is None:
            return
        parent = trace.get_current_span(parent_context)
        parent_alias = None
        if isinstance(parent, ReadableSpan) and parent.attributes:
            parent_alias = parent.attributes.get(WORKSPACE_ROUTE)
        if parent_alias is not None and parent_alias != alias:
            logger.warning(
                "span %r starts under workspace %r but its parent is stamped %r; "
                "a trace cannot straddle two workspaces (the backend derives the "
                "workspace from the API key). Switch workspaces at request "
                "boundaries, before the root span starts.",
                span.name,
                alias,
                parent_alias,
            )
        span.set_attribute(WORKSPACE_ROUTE, str(alias))

    def on_end(self, span: ReadableSpan) -> None:  # pragma: no cover - no-op
        pass

    def shutdown(self) -> None:  # pragma: no cover - no-op
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # pragma: no cover
        return True


class RoutingSpanExporter(SpanExporter):
    """Partition each batch by the routing attribute and fan out.

    Wraps the default exporter plus one lazily-created exporter per
    registered workspace key. Sits innermost in the export chain, so masking
    and export-health wrap the whole fan-out. Spans with no stamp go to the
    default destination; a stamp whose alias is not registered also goes to
    the default (the vendor's own workspace) with a warning, which keeps the
    data rather than dropping it and cannot leak one customer's spans to
    another.
    """

    def __init__(
        self,
        default_exporter: SpanExporter,
        exporter_factory: ExporterFactory,
        routes: dict[str, str] | None = None,
    ) -> None:
        self._default = default_exporter
        self._factory = exporter_factory
        self._routes: dict[str, str] = dict(routes or {})
        self._exporters: dict[str, SpanExporter] = {}
        self._warned_aliases: set[str] = set()
        self._lock = threading.Lock()

    def register(self, alias: str, api_key: str) -> None:
        """Add or replace a route. Replacing supports key rotation."""
        if not alias or not api_key:
            raise ValueError("workspace alias and api_key must be non-empty strings")
        with self._lock:
            self._routes[alias] = api_key
            self._warned_aliases.discard(alias)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        groups: dict[str | None, list[ReadableSpan]] = {}
        for span in spans:
            alias = None
            if span.attributes and WORKSPACE_ROUTE in span.attributes:
                raw = span.attributes[WORKSPACE_ROUTE]
                alias = raw if isinstance(raw, str) else str(raw)
                span = self._stripped(span)
            groups.setdefault(alias, []).append(span)

        result = SpanExportResult.SUCCESS
        for alias, group in groups.items():
            exporter = self._resolve(alias)
            if exporter.export(group) is not SpanExportResult.SUCCESS:
                result = SpanExportResult.FAILURE
        return result

    def _resolve(self, alias: str | None) -> SpanExporter:
        if alias is None:
            return self._default
        with self._lock:
            api_key = self._routes.get(alias)
            if api_key is None:
                if alias not in self._warned_aliases:
                    self._warned_aliases.add(alias)
                    logger.warning(
                        "no workspace registered for alias %r; its spans go to the "
                        "default destination. Register it with "
                        "rius.register_workspace(%r, api_key) or in "
                        "init(workspaces={...}).",
                        alias,
                        alias,
                    )
                return self._default
            exporter = self._exporters.get(api_key)
            if exporter is None:
                exporter = self._factory(api_key)
                self._exporters[api_key] = exporter
            return exporter

    @staticmethod
    def _stripped(span: ReadableSpan) -> ReadableSpan:
        # Copy-on-write, same discipline as MaskingSpanExporter: the attribute
        # dict is shared with every other processor on the provider.
        new_attributes = dict(span.attributes or {})
        del new_attributes[WORKSPACE_ROUTE]
        stripped = copy.copy(span)
        stripped._attributes = new_attributes
        return stripped

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        ok = self._default.force_flush(timeout_millis)
        with self._lock:
            exporters = list(self._exporters.values())
        for exporter in exporters:
            ok = exporter.force_flush(timeout_millis) and ok
        return ok

    def shutdown(self) -> None:
        self._default.shutdown()
        with self._lock:
            exporters = list(self._exporters.values())
        for exporter in exporters:
            exporter.shutdown()
