"""Normalization: third-party attribute keys mapped onto the canonical wire.

Instrumentations disagree about names for the same fact (``llm.model_name``,
``gen_ai.request.model``, ...). Rather than teach the backend every dialect,
the SDK maps the dialects it sees onto the conventions we already emit, so
everything downstream — masking, the pending allowlist, the sink — only has
to recognise one spelling. The mapping is **data** (a
:class:`NormalizationTable` of :class:`Rule`\\ s), not per-instrumentor code,
so the sink can mirror the same table.

Two components, because the same mapping has to happen at two moments and
the object is a different type at each:

* :class:`NormalizingSpanProcessor` runs at ``on_start``, where the object is
  a live ``Span`` and ``set_attribute`` works. This is what lets a pending
  snapshot — built at start — see canonical keys. A live span has no delete,
  so start-time mapping is **add-only**; the source key rides along and the
  exporter removes it.
* :class:`NormalizingSpanExporter` runs at export. It cannot mutate: the
  ``ReadableSpan.attributes`` handed to ``on_end``/``export`` is a read-only
  ``mappingproxy`` (assignment raises ``TypeError``), and even reaching past
  it would be wrong — a ``ReadableSpan`` shares its attribute dict by
  reference with every processor on the provider, so an in-place edit
  rewrites what other exporters see and races with their iteration. So it
  rebuilds each span from a copy, exactly like ``masking.py``.

Contract, in both components:

* A canonical key already present is NEVER overwritten. Native wins: an
  instrumentation that already speaks the convention is authoritative.
* A source key that a rule covers is DELETED once the rule has run — even
  when the native key won, because it is then a duplicate and keeping it
  double-counts downstream.
* An unmapped key is left untouched. Normalization adds spellings; it is not
  an allowlist.

Always on, no opt-out. A fast path skips the whole pass when a span carries
no key under any source namespace at all, which is every span we emit
ourselves; on an empty table it short-circuits on the first check.

**The shipped table is deliberately empty**, and a rule is not a casual
addition. Normalization is wired into ``init()`` unconditionally, so anything
in ``DEFAULT_TABLE`` is live in every process on the next release — and a rule
DELETES its source key, so a wrong mapping is unrecoverable: the original is
gone and a wrongly-shaped value sits under a canonical key that the sink and
console read as conventional. Half-migrating a concept is worse than not
migrating it. A rule therefore belongs here only once the mapping is known to
be both correct and total for that source; the per-instrumentation tables are
their own tickets, and they are purely additive to this file. Rules used to
exercise the machinery live in the tests, injected through ``table=``.

Ordering: the normalizing exporter must run BEFORE the masking exporter
(i.e. it wraps it), so masking only has to recognise canonical content keys.
Reversed, masking would strip ``llm.input_messages`` before it could be
mapped and the canonical key would arrive empty.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

#: A converter's way of saying "no value"; the target key is then not set.
SKIP: Any = object()

#: Takes the values of the rule's present source keys, in rule order.
Converter = Callable[[list[Any]], Any]


def copy_value(values: list[Any]) -> Any:
    """The source value, unchanged."""
    return values[0]


def to_int(values: list[Any]) -> Any:
    """The source value as an int; SKIP when it does not parse."""
    try:
        return int(values[0])
    except (TypeError, ValueError):
        return SKIP


def wrap_in_list(values: list[Any]) -> Any:
    """The source scalar as a one-element list (e.g. a lone finish reason)."""
    value = values[0]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def json_member(member: str) -> Converter:
    """Promote one member out of a JSON-object source value."""

    def convert(values: list[Any]) -> Any:
        raw = values[0]
        if not isinstance(raw, str):
            return SKIP
        try:
            payload = json.loads(raw)
        except ValueError:
            return SKIP
        if not isinstance(payload, dict) or member not in payload:
            return SKIP
        return payload[member]

    return convert


def total(values: list[Any]) -> Any:
    """The sum of the present source values; SKIP if none are numeric."""
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(numbers) if numbers else SKIP


class Rule:
    """One mapping: source key(s) → target key, through a converter."""

    __slots__ = ("sources", "target", "convert")

    def __init__(
        self,
        sources: str | Sequence[str],
        target: str,
        convert: Converter = copy_value,
    ) -> None:
        self.sources: tuple[str, ...] = (sources,) if isinstance(sources, str) else tuple(sources)
        self.target = target
        self.convert = convert


def _namespace(key: str) -> str:
    """The fast-path prefix a source key belongs to (``llm.model_name`` → ``llm.``)."""
    head, dot, _ = key.partition(".")
    return head + dot if dot else key


class NormalizationTable:
    """A set of :class:`Rule`\\ s, plus the fast path that skips them."""

    def __init__(self, rules: Iterable[Rule]) -> None:
        self._rules = tuple(rules)
        self._prefixes = tuple(sorted({_namespace(s) for r in self._rules for s in r.sources}))

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    @property
    def prefixes(self) -> tuple[str, ...]:
        return self._prefixes

    def applies(self, attributes: Mapping[str, Any] | None) -> bool:
        """True when the span carries at least one key under a source namespace."""
        if not attributes or not self._prefixes:
            return False
        return any(key.startswith(self._prefixes) for key in attributes)

    def additions(self, attributes: Mapping[str, Any]) -> dict[str, Any]:
        """Canonical keys this table would add, without removing anything."""
        added: dict[str, Any] = {}
        for rule in self._rules:
            if rule.target in attributes or rule.target in added:
                continue
            present = [attributes[s] for s in rule.sources if s in attributes]
            if not present:
                continue
            try:
                value = rule.convert(present)
            except Exception:
                # Fail safe: one bad converter must not cost the other rules
                # or the span.
                logger.warning(
                    "normalization converter raised for %r; rule skipped",
                    rule.target,
                    exc_info=True,
                )
                continue
            if value is not SKIP and value is not None:
                added[rule.target] = value
        return added

    def normalize(self, attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """A normalized copy of ``attributes``, or None when nothing changes."""
        if not self.applies(attributes):
            return None
        assert attributes is not None
        added = self.additions(attributes)
        # A covered source key goes whether or not its target was taken: a
        # native key wins, and the source is then a duplicate of it.
        removed = {s for rule in self._rules for s in rule.sources if s in attributes}
        if not added and not removed:
            return None
        new_attributes = {k: v for k, v in attributes.items() if k not in removed}
        new_attributes.update(added)
        return new_attributes


class NormalizingSpanProcessor(SpanProcessor):
    """Add canonical keys at span start, so pending snapshots carry them.

    Add-only: a live ``Span`` exposes no delete, and the source keys are
    removed by :class:`NormalizingSpanExporter` before the span leaves the
    process.
    """

    def __init__(self, table: NormalizationTable | None = None) -> None:
        self._table = table if table is not None else DEFAULT_TABLE

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        attributes = span.attributes
        if not self._table.applies(attributes):
            return
        assert attributes is not None
        for key, value in self._table.additions(attributes).items():
            span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:  # pragma: no cover - nothing to do
        pass

    def shutdown(self) -> None:  # pragma: no cover - nothing to hold
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True


class NormalizingSpanExporter(SpanExporter):
    """Map third-party keys onto canonical ones before delegating to ``inner``.

    Works on copies; see the module docstring for why in-place is both
    impossible and wrong.
    """

    def __init__(self, inner: SpanExporter, *, table: NormalizationTable | None = None) -> None:
        self._inner = inner
        self._table = table if table is not None else DEFAULT_TABLE

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._normalized(span) for span in spans])

    def _normalized(self, span: ReadableSpan) -> ReadableSpan:
        new_attributes = self._table.normalize(span.attributes)
        if new_attributes is None:
            return span
        normalized = copy.copy(span)
        normalized._attributes = new_attributes
        return normalized

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()


#: **Deliberately empty.** See the module docstring before adding a rule.
DEFAULT_TABLE = NormalizationTable([])
