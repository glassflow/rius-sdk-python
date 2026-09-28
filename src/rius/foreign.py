"""Another SDK's tracer provider in the same process (RIUS-1070).

The OpenTelemetry global provider is write-once. When another SDK (Langfuse
v3, for one) claims it before ``rius.init()``, rius keeps a private provider
and its instrumentors bind to that, but span CONTEXT is process-wide: an LLM
call made inside a span of the other provider takes that span as parent, and
rius exports a child whose parent it never receives.

Three pieces address this:

* ``bridge`` attaches rius's span-processor pipeline to the other provider, so
  its spans (the parents) reach rius too. When rius wins the race it IS the
  global provider and exports every span in the process, so init order must not
  change what gets exported.
* ``ResourceAdoptingExporter`` gives those bridged spans rius's resource
  identity at export: the sink derives agent name and instance id from resource
  attributes, and the other provider's resource carries neither.
* ``ForeignParentDetector`` flags a span whose parent is local
  (``is_remote=False``) yet never started through rius's pipeline. That parent
  can only belong to another in-process provider; remote parents are ordinary
  distributed tracing and are never flagged.
"""

from __future__ import annotations

import copy
import logging
import threading
import weakref
from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .semconv import RIUS_PARENT_FOREIGN

logger = logging.getLogger(__name__)


class ForeignParentDetector(SpanProcessor):
    """Stamp ``rius.parent.foreign`` on spans whose local parent rius never saw.

    "Seen" is a weak set of the span objects this pipeline started, so memory is
    bounded by live spans. It needs no removal at ``on_end``: a child names its
    local parent through a context holding the parent span object, which keeps
    it alive for exactly as long as the lookup matters. (A span context
    re-wrapped by hand in ``NonRecordingSpan`` is not that object, and is
    flagged.) With the
    bridge on, the other provider's spans pass through ``on_start`` as well, so
    they count as seen and their children are correctly left unflagged.
    """

    def __init__(self) -> None:
        self._seen: weakref.WeakSet[Span] = weakref.WeakSet()
        self._lock = threading.Lock()
        self._count = 0

    @property
    def count(self) -> int:
        """Spans flagged since this detector was built."""
        return self._count

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        parent = span.parent
        with self._lock:
            foreign = (
                parent is not None
                and parent.is_valid
                and not parent.is_remote
                and trace.get_current_span(parent_context) not in self._seen
            )
            self._seen.add(span)
            if foreign:
                self._count += 1
            first = foreign and self._count == 1
        if foreign:
            span.set_attribute(RIUS_PARENT_FOREIGN, True)
            if first:
                logger.warning(
                    "span %r has a parent from another OpenTelemetry provider in this "
                    "process that rius does not receive, so its trace arrives without "
                    "that parent (flagged %s). This happens when another SDK set the "
                    "global provider first and bridge_foreign_provider is off.",
                    span.name,
                    RIUS_PARENT_FOREIGN,
                )

    def on_end(self, span: ReadableSpan) -> None:  # pragma: no cover - nothing to do
        pass

    def shutdown(self) -> None:  # pragma: no cover - nothing to hold
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True


class _Forwarder(SpanProcessor):
    """The one processor rius ever adds to another provider.

    Its lifecycle belongs to rius, not to the provider it rides: that provider's
    ``shutdown``/``force_flush`` do nothing here, and ``release`` makes it inert,
    since OTel offers no way to remove a processor.
    """

    def __init__(self) -> None:
        self._target: SpanProcessor | None = None

    def attach(self, target: SpanProcessor) -> None:
        self._target = target

    def release(self, target: SpanProcessor) -> None:
        if self._target is target:
            self._target = None

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        target = self._target
        if target is not None:
            target.on_start(span, parent_context=parent_context)

    def _on_ending(self, span: Span) -> None:
        target = self._target
        if target is not None:
            target._on_ending(span)  # noqa: SLF001 - the SpanProcessor hook itself

    def on_end(self, span: ReadableSpan) -> None:
        target = self._target
        if target is not None:
            target.on_end(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# One forwarder per provider, re-targeted on every init(), so a shutdown()+init()
# cycle never stacks a second one on a provider that cannot drop the first.
_forwarders: weakref.WeakKeyDictionary[TracerProvider, _Forwarder] = weakref.WeakKeyDictionary()
_forwarders_lock = threading.Lock()


class Bridge:
    """Handle on rius's pipeline attached to another provider."""

    def __init__(self, forwarder: _Forwarder, pipeline: SpanProcessor) -> None:
        self._forwarder = forwarder
        self._pipeline = pipeline

    def release(self) -> None:
        """Stop feeding the other provider's spans into this pipeline. Idempotent."""
        self._forwarder.release(self._pipeline)


def bridge(provider: TracerProvider, pipeline: SpanProcessor) -> Bridge:
    """Feed every span ``provider`` starts and ends into ``pipeline``."""
    with _forwarders_lock:
        forwarder = _forwarders.get(provider)
        if forwarder is None:
            forwarder = _Forwarder()
            provider.add_span_processor(forwarder)
            _forwarders[provider] = forwarder
        forwarder.attach(pipeline)
    return Bridge(forwarder, pipeline)


class ResourceAdoptingExporter(SpanExporter):
    """Export spans of a bridged provider under rius's resource identity.

    The other provider's resource wins nothing it shares with ``resource``
    (``service.name``, ``service.instance.id``, the agent identity keys) and
    keeps whatever else it carries. Spans of rius's own provider pass through
    untouched.
    """

    def __init__(self, inner: SpanExporter, resource: Resource) -> None:
        self._inner = inner
        self._resource = resource

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._adopted(span) for span in spans])

    def _adopted(self, span: ReadableSpan) -> ReadableSpan:
        if span.resource is self._resource:
            return span
        adopted = copy.copy(span)
        # Not Resource.merge: on a schema_url conflict it keeps the OLD resource.
        adopted._resource = Resource(  # noqa: SLF001 - as NormalizingSpanExporter does
            {**span.resource.attributes, **self._resource.attributes},
            self._resource.schema_url or span.resource.schema_url,
        )
        return adopted

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()
