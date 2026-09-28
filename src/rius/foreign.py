"""Another SDK's tracer provider in the same process (RIUS-1070).

The OpenTelemetry global provider is write-once. When another SDK (Langfuse
v3, for one) claims it before ``rius.init()``, rius keeps a private provider
and its instrumentors bind to that, but span CONTEXT is process-wide: an LLM
call made inside a span of the other provider takes that span as parent, and
rius exports a child whose parent it never receives.

Four pieces address this:

* ``bridge`` (opt-in: ``bridge_foreign_provider=True``) attaches rius's
  span-processor pipeline to the other provider, so its spans (the parents)
  reach rius too, as they would had rius won the race and become the global.
  Off by default: exporting another SDK's spans needs the caller's consent.
* ``ShadowPipeline`` keeps rius's hands off those spans. A live span has one
  attribute dict and the other provider exports it too, so whatever rius
  stamped on it (session, user, route, canonical keys) would reach the other
  vendor. rius's pipeline works on a private twin instead, and exports a copy
  of the finished span carrying what it added to the twin. It also applies
  rius's sampling decision, which ``BridgeAwareSampler`` then extends to rius's
  own spans started under a bridged parent.
* ``ResourceAdoptingExporter`` gives those bridged spans rius's resource
  identity at export: the sink derives agent name and instance id from resource
  attributes, and the other provider's resource carries neither.
* ``ForeignParentDetector`` flags a span whose parent is local
  (``is_remote=False``) yet never started through rius's pipeline. That parent
  can only belong to another in-process provider; remote parents are ordinary
  distributed tracing and are never flagged. It runs with the bridge off too,
  which is how rius tells the caller the bridge exists.
"""

from __future__ import annotations

import copy
import logging
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    Decision,
    Sampler,
    SamplingResult,
)
from opentelemetry.trace import Link, SpanContext, SpanKind
from opentelemetry.trace.span import TraceState
from opentelemetry.util.types import Attributes

from ._attributes import replacement_attributes
from .semconv import RIUS_PARENT_FOREIGN

logger = logging.getLogger(__name__)


class ForeignParentDetector(SpanProcessor):
    """Stamp ``rius.parent.foreign`` on spans whose local parent rius never saw.

    "Seen" is a weak set of the span objects this pipeline started, so memory is
    bounded by live spans. It needs no removal at ``on_end``: a child names its
    local parent through a context holding the parent span object, which keeps
    it alive for exactly as long as the lookup matters. (A span context
    re-wrapped by hand in ``NonRecordingSpan`` is not that object, and is
    flagged.) With the bridge on, the other provider's spans pass through
    ``on_start`` as their twins, so they count as seen and their children are
    correctly left unflagged.
    """

    def __init__(self, global_provider: str | None = None) -> None:
        self._global_provider = global_provider
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
                and twin_of(trace.get_current_span(parent_context)) not in self._seen
            )
            self._seen.add(span)
            if foreign:
                self._count += 1
            first = foreign and self._count == 1
        if foreign:
            span.set_attribute(RIUS_PARENT_FOREIGN, True)
            if first:
                self._warn(span.name)

    def _warn(self, name: str) -> None:
        owner = (
            f"another OpenTelemetry SDK ({self._global_provider})"
            if self._global_provider
            else "another OpenTelemetry provider in this process"
        )
        logger.warning(
            "%s owns the global tracer provider; rius spans have parents it never "
            "receives (first: %r, flagged %s). Set RIUS_BRIDGE_FOREIGN_PROVIDER=true or "
            "init(bridge_foreign_provider=True) to send them to Rius.",
            owner,
            name,
            RIUS_PARENT_FOREIGN,
        )

    def on_end(self, span: ReadableSpan) -> None:  # pragma: no cover - nothing to do
        pass

    def shutdown(self) -> None:  # pragma: no cover - nothing to hold
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True


@dataclass(frozen=True)
class _Shadow:
    twin: Span
    seed: Mapping[str, Any]


# The bridged spans rius sampled in (with their shadow) and those it dropped.
# Weak keys: an entry lives exactly as long as the other SDK's span. on_end
# gets a snapshot of the span, not the span, hence the index by span context.
_shadows: weakref.WeakKeyDictionary[Span, _Shadow] = weakref.WeakKeyDictionary()
_shadows_by_context: weakref.WeakValueDictionary[tuple[int, int], _Shadow] = (
    weakref.WeakValueDictionary()
)
_dropped: weakref.WeakSet[Span] = weakref.WeakSet()
_ABSENT = object()


def _context_key(context: SpanContext) -> tuple[int, int]:
    return (context.trace_id, context.span_id)


def _shadow_of(span: object) -> _Shadow | None:
    return _shadows.get(span) if isinstance(span, Span) else None


def twin_of(span: object) -> object:
    """The twin rius's pipeline saw in place of a bridged span; any other span as is."""
    shadow = _shadow_of(span)
    return span if shadow is None else shadow.twin


def _rius_decision(span: object) -> bool | None:
    """Whether rius sampled a bridged span in, or None when it is not one."""
    if span in _dropped:
        return False
    return True if _shadow_of(span) is not None else None


class BridgeAwareSampler(Sampler):
    """rius's sampler, following rius's own decision on a bridged parent.

    ``ParentBased`` follows the parent's sampled flag, and a bridged parent's
    flag is the other SDK's decision (usually always-on), which would ship
    every trace it touches whatever ``sample_rate`` says.
    """

    def __init__(self, inner: Sampler) -> None:
        self._inner = inner

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        kept = _rius_decision(trace.get_current_span(parent_context))
        sampler = self._inner if kept is None else ALWAYS_ON if kept else ALWAYS_OFF
        return sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"BridgeAware{{{self._inner.get_description()}}}"


class _Twin(Span):
    """A span no provider owns: rius's pipeline stamps it in place of a bridged span."""


def _twin(span: Span, parent_context: Context | None) -> Span:
    twin = _Twin(
        span.name,
        span.get_span_context(),
        parent=span.parent,
        resource=span.resource,
        attributes=span.attributes,
        links=span.links,
        kind=span.kind,
        instrumentation_scope=span.instrumentation_scope,
    )
    twin.start(start_time=span.start_time, parent_context=parent_context)
    return twin


def _stamped(span: ReadableSpan, shadow: _Shadow) -> ReadableSpan:
    """The finished span plus what rius's pipeline wrote on its twin at start.

    A key the other SDK rewrote after start keeps that value, as a later
    ``set_attribute`` wins over a start-time stamp on rius's own spans.
    """
    attributes = dict(span.attributes or {})
    for key, value in (shadow.twin.attributes or {}).items():
        started = shadow.seed.get(key, _ABSENT)
        if value != started and attributes.get(key, _ABSENT) == started:
            attributes[key] = value
    stamped = copy.copy(span)
    stamped._attributes = replacement_attributes(span, attributes)  # noqa: SLF001
    return stamped


class ShadowPipeline(SpanProcessor):
    """rius's pipeline as another provider feeds it, never writing to that provider's spans.

    Spans rius's sampler drops never reach the pipeline, and neither does a
    span that started before the bridge was attached: rius never saw it start,
    so its children are flagged ``rius.parent.foreign`` instead.
    """

    def __init__(self, pipeline: SpanProcessor, sampler: Sampler) -> None:
        self._pipeline = pipeline
        self._sampler = sampler

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        result = self._sampler.should_sample(
            parent_context,
            span.get_span_context().trace_id,
            span.name,
            span.kind,
            span.attributes,
            span.links,
        )
        if result.decision is not Decision.RECORD_AND_SAMPLE:
            _dropped.add(span)
            return
        twin = _twin(span, parent_context)
        shadow = _Shadow(twin, dict(span.attributes or {}))
        _shadows[span] = shadow
        _shadows_by_context[_context_key(twin.get_span_context())] = shadow
        self._pipeline.on_start(twin, parent_context=parent_context)

    def _on_ending(self, span: Span) -> None:
        shadow = _shadow_of(span)
        if shadow is not None:
            self._pipeline._on_ending(shadow.twin)  # noqa: SLF001 - the SpanProcessor hook

    def on_end(self, span: ReadableSpan) -> None:
        shadow = (
            None if span.context is None else _shadows_by_context.get(_context_key(span.context))
        )
        if shadow is not None:
            self._pipeline.on_end(_stamped(span, shadow))

    def shutdown(self) -> None:  # pragma: no cover - the rider never shuts its target down
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


def bridge(provider: TracerProvider, pipeline: SpanProcessor, sampler: Sampler) -> Bridge:
    """Feed the spans ``provider`` starts and ends that ``sampler`` keeps into ``pipeline``."""
    shadow = ShadowPipeline(pipeline, sampler)
    with _forwarders_lock:
        forwarder = _forwarders.get(provider)
        if forwarder is None:
            forwarder = _Forwarder()
            provider.add_span_processor(forwarder)
            _forwarders[provider] = forwarder
        forwarder.attach(shadow)
    return Bridge(forwarder, shadow)


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
