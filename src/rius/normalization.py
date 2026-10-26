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
  so start-time mapping never removes a key: it adds canonical keys, and the
  only existing key it overwrites is an :class:`ExpandingRule`'s own source
  (``llm.invocation_parameters`` is rewritten without the members it
  promoted). The other source keys ride along and the exporter removes them.
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
no key under any source namespace at all; on an empty table it short-circuits
on the first check. Our OWN spans no longer take that path and now match a
source outright: every one carries ``openinference.span.kind``, which the
taxonomy rules read. They are the reason ``normalize`` ends with a no-op
check. The rules produce exactly the two keys the span already has, so the
rebuilt mapping is equal to the original, and returning None there is what
keeps a native span from being copied on every export. The copy, which is the
part that costs, still only happens for a span normalization actually
changed.

A rule is not a casual addition. Normalization is wired into ``init()``
unconditionally, so anything in ``DEFAULT_TABLE`` is live in every process on
the next release — and a rule DELETES its source key, so a wrong mapping is
unrecoverable: the original is gone and a wrongly-shaped value sits under a
canonical key that the sink and console read as conventional. Half-migrating a
concept is worse than not migrating it. A rule therefore belongs here only
once the mapping is known to be both correct and TOTAL for that source — the
canonical key can represent everything the source could hold. Where it is not,
the source stays unmapped and rides through untouched, which costs nothing:
the sink still sees it under its own name. The deliberate omissions are listed
with their reasons at ``OPENINFERENCE_RULES`` below.

The shipped table covers the OpenInference ``llm.*`` model-call and usage
families (RIUS-921), plus the GenAI-shaped usage spellings that third-party
instrumentations still emit (``cache_creation``, and pydantic-ai's
``details.reasoning_tokens`` / ``thinking_tokens`` / ``thoughts_tokens``).
Rules used to exercise the machinery itself live in the
tests, injected through ``table=``.

Not everything a dialect says is an attribute. OpenInference records the first
streamed chunk as a span EVENT, so it cannot go through the rule table at all;
``first_chunk_from_event`` derives the canonical streaming attributes from
it, leaving the event as recorded, and the exporter applies it alongside the
table. It lives in the exporter for two reasons that agree: the event does not
exist yet at ``on_start``, and the exporter is rebuilding the attribute dict
anyway, so the derived attribute is still settable there. A failure is the
same shape: an auto-instrumented span records the OTel ``exception`` event and
an ERROR status but no ``error.type``, and
``error_type_from_exception_event`` derives it there for the same reasons.
``response_id_from_output_value`` promotes an LLM span's response id out of
``output.value`` there too: the value only exists once the call returned, and
masking, which runs after, strips it under capture-off.
``normalize_tool_definitions`` is applied there too, for a different reason:
its source is an indexed key family (``llm.tools.N.tool.json_schema``), which
the exact-key rule table cannot match, and its target is content, which must
never be written at span start.
``reassemble_openinference_messages`` sits beside it for the same two
reasons: the flattened ``llm.input_messages.N.message.*`` and
``llm.output_messages.N.message.*`` families become ``gen_ai.input.messages``
and ``gen_ai.output.messages``, which are content.
``drop_embedding_vectors`` removes the indexed
``embedding.embeddings.N.embedding.vector`` family, which is never exported.

Ordering: the normalizing exporter must run BEFORE the masking exporter
(i.e. it wraps it), so masking only has to recognise canonical content keys.
Reversed, masking would strip ``llm.input_messages`` before it could be
mapped and the canonical key would arrive empty.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Link, StatusCode

from ._attributes import rebuilt_item_attributes, replacement_attributes
from ._media import (
    rebuild_native_parts,
    url_part,
)
from ._native_spans import native_error_type
from ._provider_compaction import ProviderCompactions, compactions_in_response
from ._provider_response import response_json
from ._serde import (
    LONE_SURROGATE,
    UNREADABLE,
    attribute_value,
    loads_json,
    loads_object,
    serialize,
    strip_within_budget,
)
from .aggregated_usage import WrappedCallTracker, move_aggregated_usage
from .compaction import CompactedContextProcessor
from .semconv import (
    EMBEDDING_MODEL_NAME,
    ERROR_TYPE,
    EXCEPTION_EVENT,
    EXCEPTION_TYPE,
    GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK,
    GEN_AI_CONVERSATION_COMPACTED,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_CHOICE_COUNT,
    GEN_AI_REQUEST_ENCODING_FORMATS,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID,
    GEN_AI_REQUEST_REASONING_LEVEL,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_REQUEST_STREAM_CURSOR,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
    GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_DETAILS_AUDIO_CACHE_TOKENS,
    GEN_AI_USAGE_DETAILS_AUDIO_CANDIDATES_TOKENS,
    GEN_AI_USAGE_DETAILS_AUDIO_PROMPT_TOKENS,
    GEN_AI_USAGE_DETAILS_AUDIO_RESPONSE_TOKENS,
    GEN_AI_USAGE_DETAILS_CACHE_AUDIO_READ_TOKENS,
    GEN_AI_USAGE_DETAILS_IMAGE_CACHE_TOKENS,
    GEN_AI_USAGE_DETAILS_IMAGE_CANDIDATES_TOKENS,
    GEN_AI_USAGE_DETAILS_IMAGE_PROMPT_TOKENS,
    GEN_AI_USAGE_DETAILS_INPUT_AUDIO_TOKENS,
    GEN_AI_USAGE_DETAILS_OUTPUT_AUDIO_TOKENS,
    GEN_AI_USAGE_DETAILS_REASONING_TOKENS,
    GEN_AI_USAGE_DETAILS_THINKING_TOKENS,
    GEN_AI_USAGE_DETAILS_THOUGHTS_TOKENS,
    GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
    GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKEN_DETAILS_AUDIO_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_PREFIX,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_TEXT_INPUT_TOKENS,
    GEN_AI_USAGE_TEXT_OUTPUT_TOKENS,
    INVOCATION_PARAMETERS_CONTENT_MEMBERS,
    LLM_FINISH_REASON,
    LLM_INVOCATION_PARAMETERS,
    LLM_PROVIDER,
    LLM_REQUEST_MODEL_NAME,
    LLM_RESPONSE_MODEL_NAME,
    LLM_SYSTEM,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
    LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
    METADATA,
    OPENINFERENCE_SPAN_KIND,
    RIUS_AGGREGATED_USAGE_PREFIX,
    RIUS_COMPACTION_EVENT,
    RIUS_REQUEST_TOOL_CHOICE,
    TOOL_NAME,
    SpanKind,
    kind_for_operation,
    operation_for_kind,
)

logger = logging.getLogger(__name__)

#: A converter's way of saying "no value"; the target key is then not set.
SKIP: Any = object()

#: Takes the values of the rule's present source keys, in rule order.
Converter = Callable[[list[Any]], Any]


def copy_value(values: list[Any]) -> Any:
    """The source value, unchanged."""
    return values[0]


def to_int(values: list[Any]) -> Any:
    """The source value as a count (see ``_count``); SKIP when it is none.

    A string is read as the integer it spells, as some instrumentors write one.
    """
    value = values[0]
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            return SKIP
    return _count(value)


def wrap_in_list(values: list[Any]) -> Any:
    """The source scalar as a one-element list (e.g. a lone finish reason)."""
    value = values[0]
    return list(value) if isinstance(value, (list, tuple)) else [value]


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

    def apply(self, values: list[Any], attributes: Mapping[str, Any]) -> Mapping[str, Any]:
        """The canonical key(s) this rule produces from its present sources."""
        return {self.target: self.convert(values)}


class ExpandingRule:
    """One source key that fans out over SEVERAL canonical keys.

    For a source that is a bag rather than a value — ``llm.invocation_parameters``
    is a JSON object whose members are separate facts. A plain :class:`Rule`
    cannot express it: it has one target, and the generic delete would then
    take the whole bag away for the sake of one member.

    ``expand`` receives the raw source value and returns EVERYTHING the span
    should carry in its place. That may include the source key itself, holding
    the members no canonical key claimed — the only way to promote part of a
    bag without dropping the rest. Returning the source key is the one exempt
    from native-wins, since the rule is rewriting its own input rather than
    competing with an instrumentation that already speaks the convention.

    ``reads`` names other keys the expansion depends on without consuming
    them: they are not sources, so they are never deleted. When it is set,
    ``expand`` also receives those keys' values (the ones present).
    """

    __slots__ = ("sources", "expand", "reads")

    def __init__(
        self,
        source: str,
        expand: Callable[..., Mapping[str, Any]],
        *,
        reads: Sequence[str] = (),
    ) -> None:
        self.sources: tuple[str, ...] = (source,)
        self.expand = expand
        self.reads: tuple[str, ...] = tuple(reads)

    def apply(self, values: list[Any], attributes: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.reads:
            return self.expand(values[0])
        return self.expand(values[0], {k: attributes[k] for k in self.reads if k in attributes})


#: A rule of either shape. Both expose ``sources`` and ``apply``.
AnyRule = Rule | ExpandingRule


def _namespace(key: str) -> str:
    """The fast-path prefix a source key belongs to (``llm.model_name`` → ``llm.``)."""
    head, dot, _ = key.partition(".")
    return head + dot if dot else key


class NormalizationTable:
    """A set of :class:`Rule`\\ s, plus the fast path that skips them."""

    def __init__(self, rules: Iterable[AnyRule]) -> None:
        self._rules = tuple(rules)
        self._prefixes = tuple(sorted({_namespace(s) for r in self._rules for s in r.sources}))

    @property
    def rules(self) -> tuple[AnyRule, ...]:
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
        """Keys this table would set, without removing anything.

        Canonical keys, plus an :class:`ExpandingRule`'s rewrite of its own
        source key.
        """
        added: dict[str, Any] = {}
        for rule in self._rules:
            present = [attributes[s] for s in rule.sources if s in attributes]
            if not present:
                continue
            try:
                produced = rule.apply(present, attributes)
            except Exception:
                # Fail safe: one bad converter must not cost the other rules
                # or the span.
                logger.warning(
                    "normalization rule raised for %r; rule skipped",
                    rule.sources,
                    exc_info=True,
                )
                continue
            for key, value in produced.items():
                if value is SKIP or value is None or key in added:
                    continue
                # Native wins — except for the rule's own source, which an
                # ExpandingRule rewrites rather than competes with.
                if key in attributes and key not in rule.sources:
                    continue
                added[key] = value
        # Only a model this table produced: a native one is the producer's
        # own choice of name. Here, not at export, because the processor
        # writes these additions onto the live span at start, and by export
        # the model already counts as native.
        model = added.get(GEN_AI_REQUEST_MODEL)
        if isinstance(model, str):
            provider = added.get(GEN_AI_PROVIDER_NAME, attributes.get(GEN_AI_PROVIDER_NAME))
            added[GEN_AI_REQUEST_MODEL] = without_provider_route(model, provider)
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
        # Every span we emit ourselves reaches here, because the taxonomy
        # rules take openinference.span.kind as a source and we always set it.
        # For those the rules re-derive what is already there, so the result
        # is equal to the input and the only honest answer is "unchanged" —
        # otherwise the exporter rebuilds every native span for nothing. One
        # mapping comparison is much cheaper than the span copy it avoids.
        if new_attributes == attributes:
            return None
        return new_attributes


class NormalizingSpanProcessor(SpanProcessor):
    """Add canonical keys at span start, so pending snapshots carry them.

    Never removes a key: a live ``Span`` exposes no delete, and the source
    keys are removed by :class:`NormalizingSpanExporter` before the span
    leaves the process. It adds canonical keys and overwrites one kind of
    existing key, an :class:`ExpandingRule`'s own source, with what that rule
    left of it (``llm.invocation_parameters`` minus its promoted members).
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


#: The name OpenInference's instrumentors give the first streamed chunk.
#: Confirmed in ``openinference-instrumentation-openai`` 0.1.52,
#: ``openinference/instrumentation/openai/_stream.py::_Stream._process_chunk``:
#: on the first iteration it calls ``add_event("First Token Stream Event")``
#: with no attributes and no explicit timestamp, so the SDK stamps the moment
#: the chunk arrived. That is the whole signal — the instrumentors set no
#: streaming attribute and no time-to-first-chunk of their own.
OPENINFERENCE_FIRST_TOKEN_EVENT = "First Token Stream Event"


def first_chunk_from_event(span: ReadableSpan) -> dict[str, Any]:
    """The canonical streaming attributes for a span marked by OpenInference.

    Returns the attributes to add: ``gen_ai.request.stream`` and
    ``gen_ai.response.time_to_first_chunk``, the seconds from the span's start
    to its first marker. The marker event itself is left as recorded: no
    ``gen_ai.first_token`` event is ever written, since the backend derives
    the one it stores from the duration. This does NOT go through
    :class:`NormalizationTable`: the table maps attribute keys, and the source
    here is an event.

    It is the exporter's job, not the processor's, for two reasons that point
    the same way. The event does not exist at ``on_start`` (it is added
    mid-stream), so a start-time processor has nothing to read. And the
    derived duration is an attribute, which the exporter can still set
    because it rebuilds ``_attributes`` on a copy of the span (see
    :class:`NormalizingSpanExporter`); it is only the *live* span that is out
    of reach by then, and we do not need it.

    Native wins, as for attributes: a key the span already carries is left
    out, and the caller adds the rest with setdefault, so a value the table
    produced is never overwritten either. The duration is never negative. A
    marker after the span's end is not clamped: it is when the chunk arrived
    (LlamaIndex ends its stream_chat span before the first chunk), and the
    backend keeps a recorded marker's time as it is.

    Unit: ``gen_ai.response.time_to_first_chunk`` is **seconds** (a float),
    matching what ``GenerationSpan.record_first_token`` emits on the native
    path and what the conventions specify.
    """
    events = span.events or ()
    first_token_ns = next(
        (e.timestamp for e in events if e.name == OPENINFERENCE_FIRST_TOKEN_EVENT), None
    )
    if first_token_ns is None:
        return {}
    attributes = span.attributes or {}
    added: dict[str, Any] = {}
    if GEN_AI_REQUEST_STREAM not in attributes:
        # A first chunk arriving is what proves the request streamed: the
        # same inference record_first_token makes.
        added[GEN_AI_REQUEST_STREAM] = True
    start_ns = span.start_time
    if (
        GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK not in attributes
        and isinstance(first_token_ns, int)
        and isinstance(start_ns, int)
    ):
        added[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] = max(first_token_ns - start_ns, 0) / 1e9
    return added


#: Kept for compatibility: the module's former public name for the registry's
#: GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK, the conventions' metric name
#: for time to first chunk that pydantic-ai and @ai-sdk/otel write onto the
#: span. A source spelling: mapped, never emitted.
GEN_AI_CLIENT_TIME_TO_FIRST_CHUNK = GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK


def first_chunk_from_metric(raw: Any) -> Mapping[str, Any]:
    """The span attribute the metric spelling stands for, plus the stream flag.

    Same unit (seconds). Only a streamed response has a first chunk, so the
    request streamed: the inference ``record_first_token`` makes.
    """
    seconds = _number(raw)
    if seconds is SKIP or seconds < 0:
        return {}
    return {GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK: seconds, GEN_AI_REQUEST_STREAM: True}


def error_type_from_exception_event(span: ReadableSpan) -> dict[str, Any]:
    """``error.type`` for a failed span, from its OTel ``exception`` event.

    A third-party instrumentor that fails a span records the exception event
    and an ERROR status and stops there, so the failure never carries the
    ``error.type`` the GenAI conventions make Conditionally Required. Our own
    helpers set it natively (``_errors.record_error``); this is the same fact
    for the spans we did not write.

    Like :func:`first_chunk_from_event` it reads an EVENT, so it sits
    outside :class:`NormalizationTable`, and it is export-stage only: neither
    the event nor the ERROR status exists at ``on_start``. ``error.type`` is
    metadata, not identity, so it never needs to reach a pending snapshot.

    Four cases:

    * ERROR status, an exception event, no ``error.type`` -> the event's
      ``exception.type``, spelled exactly as the event spells it, so one span
      never carries two spellings of the same failure. With several exception
      events the first usable one names it, the choice the TypeScript SDK and
      the sink make too.
    * ERROR status, NO exception event -> nothing. An error with no exception
      is not classifiable, and a guessed value is worse than an absent one.
    * An exception event on a span that did not fail -> nothing. A recorded
      and handled exception is not a failure.
    * ``error.type`` already present -> nothing; native wins.
    """
    if span.status.status_code is not StatusCode.ERROR:
        return {}
    if ERROR_TYPE in (span.attributes or {}):
        return {}
    for event in span.events:
        if event.name != EXCEPTION_EVENT:
            continue
        value = (event.attributes or {}).get(EXCEPTION_TYPE)
        if isinstance(value, str) and value:
            return {ERROR_TYPE: value}
    return {}


#: OpenInference's indexed tool-definition family: ``llm.tools.{i}.tool.json_schema``.
#: A source spelling the registry has no constant for (a pattern over indexed
#: keys, not one key), so it is spelled here.
_LLM_TOOL_SCHEMA = re.compile(r"llm\.tools\.(\d+)\.tool\.json_schema")


def normalize_tool_definitions(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Reassemble a span's tool definitions into one ``gen_ai.tool.definitions``.

    Returns the rebuilt attribute dict, or None when nothing changes. Like the
    event passes this sits outside :class:`NormalizationTable`, for a reason of
    its own: the source is an INDEXED family, ``llm.tools.N.tool.json_schema``,
    and a :class:`Rule`'s sources are exact keys. It is export-stage only,
    which is also right on the merits — definitions are content, so they never
    ride a pending snapshot, and only a start-time pass could put them there.

    Two sources, in order:

    * ``llm.tools.N.tool.json_schema``, which every bundled OpenInference
      instrumentor that sees tools writes (openai pops ``tools`` out of the
      request bag to do it). Each value is a JSON string; the schemas are
      parsed and re-serialized as one array, in index order, VERBATIM — an
      Anthropic ``input_schema`` stays an Anthropic ``input_schema``, as on
      the native ``set_tool_definitions``. The indexed keys are then deleted.
      If any one schema does not parse, the family is left untouched rather
      than reassembled without it: it is content by prefix already, so
      nothing escapes, and a partial array would be a silent loss.
    * Only when there is no ``llm.tools.*`` schema: the tool-definition
      members of ``llm.invocation_parameters`` (``tools`` and the legacy
      ``functions``), where litellm and langchain leave them. Both present
      are concatenated, tools first. The members leave the bag; a bag left
      empty goes too, rather than riding as ``{}``. When the indexed schemas
      were the source, the bag's members are duplicates of them and leave
      the bag all the same.

    Native wins: a span already carrying ``gen_ai.tool.definitions`` keeps it,
    and the sources are removed anyway because they are then duplicates.

    This runs before masking, so the promoted key is then stripped under
    ``capture_content=False`` exactly like a native one — it is on the
    content allowlist.
    """
    if not attributes:
        return None
    native = GEN_AI_TOOL_DEFINITIONS in attributes

    indexed: dict[int, str] = {}
    for key in attributes:
        # The cheap prefix test first: this runs on every exported span.
        match = _LLM_TOOL_SCHEMA.fullmatch(key) if key.startswith("llm.tools.") else None
        if match is not None:
            indexed[int(match.group(1))] = key
    if indexed:
        schemas: list[Any] = []
        for index in sorted(indexed):
            raw = attributes[indexed[index]]
            if not isinstance(raw, str):
                return None
            schema = loads_json(raw)
            if schema is UNREADABLE:
                return None
            schemas.append(schema)
        drop = set(indexed.values())
        rebuilt = {k: v for k, v in attributes.items() if k not in drop}
        # litellm and langchain write the same definitions into the request
        # bag as well. They are duplicates of the ones just reassembled, and
        # go now: left behind, a second normalization would find a native
        # gen_ai.tool.definitions and remove them then, so normalizing twice
        # would differ from normalizing once.
        stripped = _bag_without_definitions(rebuilt)
        if stripped is not None:
            rebuilt = stripped[0]
        if not native:
            rebuilt[GEN_AI_TOOL_DEFINITIONS] = serialize(schemas)
        return rebuilt

    stripped = _bag_without_definitions(attributes)
    if stripped is None:
        return None
    rebuilt, definitions = stripped
    if not native:
        rebuilt[GEN_AI_TOOL_DEFINITIONS] = serialize(definitions)
    return rebuilt


def _bag_without_definitions(
    attributes: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Any]] | None:
    """The attributes with the bag's tool-definition members removed, and those definitions.

    None when the bag holds none, or holds them in a shape we cannot vouch is
    a list of definitions: the bag is content already, so leaving it is safe.
    A bag left empty goes too, rather than riding as ``{}``.
    """
    bag = attributes.get(LLM_INVOCATION_PARAMETERS)
    if not isinstance(bag, str):
        return None
    payload = loads_object(bag)
    if payload is None:
        return None
    members = [m for m in INVOCATION_PARAMETERS_CONTENT_MEMBERS if m in payload]
    if not members or not all(isinstance(payload[m], list) for m in members):
        return None
    definitions = [definition for m in members for definition in payload[m]]
    leftover = {k: v for k, v in payload.items() if k not in members}
    rebuilt = {k: v for k, v in attributes.items() if k != LLM_INVOCATION_PARAMETERS}
    if leftover:
        rebuilt[LLM_INVOCATION_PARAMETERS] = serialize(leftover)
    return rebuilt, definitions


#: OpenInference's flattened message families and the canonical key each one
#: is reassembled into. Source spellings the registry generates no Python
#: constant for, so they live here, like ``_LLM_TOOL_SCHEMA`` above.
_MESSAGE_FAMILIES = (
    ("llm.input_messages.", GEN_AI_INPUT_MESSAGES),
    ("llm.output_messages.", GEN_AI_OUTPUT_MESSAGES),
)
_MESSAGE_PREFIXES = tuple(prefix for prefix, _ in _MESSAGE_FAMILIES)

#: The message fields that are read, besides the tool-call and contents
#: families. Any other ``message.*`` field (``message.name``,
#: ``message.function_call_name``, ...) is not consumed and rides through.
_MESSAGE_ROLE = "message.role"
_MESSAGE_CONTENT = "message.content"
_MESSAGE_TOOL_CALL_ID = "message.tool_call_id"
_TOOL_CALLS = "message.tool_calls."
_TOOL_CALL = "tool_call."
_CONTENTS = "message.contents."
_CONTENT_ITEM = "message_content."

#: Tool-call fields and the part field each one fills. Any other
#: ``tool_call.*`` field is consumed and dropped, as the sink does.
_TOOL_CALL_FIELDS = (("id", "id"), ("function.name", "name"), ("function.arguments", "arguments"))

#: An index as the sink's ``strconv.Atoi`` reads one: an optional sign and
#: ASCII digits, within int64. The sink is the other producer of this key, and
#: the two must agree on which keys are messages.
_INDEX = re.compile(r"[+-]?[0-9]+", re.ASCII)


def _index(text: str) -> int | None:
    if _INDEX.fullmatch(text) is None:
        return None
    value = int(text)
    return value if 0 <= value < 2**63 else None


class _FlatMessage:
    """One flattened message's fields, gathered before it is encoded."""

    def __init__(self) -> None:
        self.role = ""
        self.content: str | None = None
        self.tool_call_id = ""
        self.tool_calls: dict[int, dict[str, str]] = {}
        self.contents: dict[int, dict[str, str]] = {}

    def encode(self) -> dict[str, Any]:
        """The message in the GenAI role/parts shape, as the sink encodes it.

        The content first (a ``tool_call_response`` when the message carries
        a tool-call id, a text part otherwise), then the multimodal contents,
        then the tool calls, each family in index order. An empty role, id,
        name or arguments is omitted rather than written empty, as the sink
        does; empty content is still a part, because empty content was said.
        """
        message: dict[str, Any] = {}
        if self.role:
            message["role"] = self.role
        parts: list[dict[str, Any]] = []
        if self.content is not None:
            if self.tool_call_id:
                parts.append(
                    {
                        "type": "tool_call_response",
                        "id": self.tool_call_id,
                        "response": self.content,
                    }
                )
            else:
                parts.append({"type": "text", "content": self.content})
        for k in sorted(self.contents):
            part = _content_part(self.contents[k], self.tool_calls)
            if part is not None:
                parts.append(part)
        for j in sorted(self.tool_calls):
            call = self.tool_calls[j]
            part = {"type": "tool_call"}
            for field, name in _TOOL_CALL_FIELDS:
                if call.get(field):
                    part[name] = call[field]
            parts.append(part)
        message["parts"] = parts
        return message


def _content_part(
    item: dict[str, str], tool_calls: dict[int, dict[str, str]]
) -> dict[str, Any] | None:
    """One multimodal contents item as a message part, or None for no part.

    * A text item, and nothing more, is a text part. This is how the
      Anthropic instrumentors write every text block, replies and block-list
      system prompts alike.
    * A ``tool_use`` item that repeats a tool call the message already carries
      adds no part. The Anthropic instrumentors write each tool_use block
      twice, under ``message.tool_calls.J`` and as a contents item, so the call
      is already a ``tool_call`` part and a second copy would only be noise.
      It must hold nothing but the call's fields, and each of them must equal
      one tool call's; otherwise it is not provably a copy, and is serialized.
    * An image item is a media part (see :func:`_image_part`), never a text
      part holding the item. One whose url is not a string has no target to
      point at, so it is left as it came, serialized like any other item.
    * Anything else (a reasoning block, a text item carrying an id or a
      signature, an item without a type) becomes a text part holding the
      item serialized, as the generation helper does with content it has no
      part type for. The item is its fields by name (``message_content.`` cut,
      ``tool_call.`` kept), sorted, so no field is lost and every producer
      orders it the same way.
    """
    if item.keys() == {"type", "text"} and item["type"] == "text":
        return {"type": "text", "content": item["text"]}
    if item.get("type") == "tool_use" and _repeats_a_tool_call(item, tool_calls):
        return None
    if item.get("type") == "image" and isinstance(item.get(_IMAGE_URL, ""), str):
        return _image_part(item)
    return {"type": "text", "content": serialize({k: item[k] for k in sorted(item)})}


#: An image contents item's fields, ``message_content.`` cut: its URL, and
#: the media type OpenInference's JS Anthropic instrumentor leaves instead.
_IMAGE_URL = "image.image.url"
_IMAGE_MEDIA_TYPE = "image.media_type"


def _image_part(item: dict[str, str]) -> dict[str, Any]:
    """An image item as a semconv media part.

    A data URL is a blob (its bytes never kept), an http(s) URL a uri. What
    the instrumentor emptied is a blob with ``stripped_reason``
    ``redacted_upstream``: OpenInference's ``__REDACTED__`` for a data URL
    over its size limit, the Anthropic instrumentor's ``data:None;url,None``
    for a url source, and an item with no URL at all (the JS Anthropic
    instrumentor's, carrying ``image.media_type`` when the source had one).
    """
    return url_part(item.get(_IMAGE_URL, ""), "image", item.get(_IMAGE_MEDIA_TYPE))


#: The fields a tool_use contents item may carry to count as a copy of a tool
#: call, each with the tool-call field it must equal.
_TOOL_USE_ITEM_FIELDS = {f"{_TOOL_CALL}{field}": field for field, _ in _TOOL_CALL_FIELDS}


def _repeats_a_tool_call(item: dict[str, str], tool_calls: dict[int, dict[str, str]]) -> bool:
    fields = {k: v for k, v in item.items() if k != "type"}
    if not fields.keys() <= _TOOL_USE_ITEM_FIELDS.keys():
        return False
    return any(
        all(call.get(_TOOL_USE_ITEM_FIELDS[k], "") == v for k, v in fields.items())
        for call in tool_calls.values()
    )


def _reassemble_family(
    attributes: Mapping[str, Any], prefix: str
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """The family's messages in index order and the keys they consumed.

    None when the family has no message, or holds a value that is not a
    string: OpenInference writes every message field as one, so anything
    else is a shape we cannot vouch for, and leaving it is safe because the
    family is content by prefix already.
    """
    messages: dict[int, _FlatMessage] = {}
    consumed: list[str] = []
    for key, value in attributes.items():
        if not key.startswith(prefix):
            continue
        index_text, dot, field = key[len(prefix) :].partition(".")
        index = _index(index_text) if dot else None
        if index is None:
            continue
        call: tuple[int, str] | None = None
        item: tuple[int, str] | None = None
        if field.startswith(_TOOL_CALLS):
            sub_text, dot, rest = field[len(_TOOL_CALLS) :].partition(".")
            sub = _index(sub_text) if dot else None
            if sub is not None and rest.startswith(_TOOL_CALL):
                call = (sub, rest[len(_TOOL_CALL) :])
        elif field.startswith(_CONTENTS):
            sub_text, dot, rest = field[len(_CONTENTS) :].partition(".")
            sub = _index(sub_text) if dot else None
            if sub is not None and rest.startswith(_CONTENT_ITEM):
                item = (sub, rest[len(_CONTENT_ITEM) :])
            elif sub is not None and rest.startswith(_TOOL_CALL):
                # A tool_use item's call fields sit beside message_content.type
                # rather than under it; they are the item's too, prefix kept.
                item = (sub, rest)
        if (
            call is None
            and item is None
            and field
            not in (
                _MESSAGE_ROLE,
                _MESSAGE_CONTENT,
                _MESSAGE_TOOL_CALL_ID,
            )
        ):
            continue
        if not isinstance(value, str):
            return None
        message = messages.setdefault(index, _FlatMessage())
        if call is not None:
            message.tool_calls.setdefault(call[0], {})[call[1]] = value
        elif item is not None:
            message.contents.setdefault(item[0], {})[item[1]] = value
        elif field == _MESSAGE_ROLE:
            message.role = value
        elif field == _MESSAGE_TOOL_CALL_ID:
            message.tool_call_id = value
        else:
            message.content = value
        consumed.append(key)
    if not messages:
        return None
    return [messages[i].encode() for i in sorted(messages)], consumed


def reassemble_openinference_messages(
    attributes: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Rebuild OpenInference's flattened messages as ``gen_ai.*.messages``.

    Returns the rebuilt attribute dict, or None when nothing changes.
    OpenInference writes one attribute per message field:
    ``llm.input_messages.{i}.message.{role,content,tool_call_id}``,
    ``...message.tool_calls.{j}.tool_call.{id,function.name,function.arguments}``
    and the multimodal ``...message.contents.{k}.message_content.*``, and the
    same under ``llm.output_messages``. Each family becomes one JSON array
    under its canonical key, in the role/parts shape the generation helpers
    write, capped like every other JSON attribute, and every key it consumed
    is deleted.

    The output is BYTE-IDENTICAL to the sink's ``reassembleMessages`` for the
    same input, because a span normalized here and one normalized by the sink
    must be indistinguishable downstream, and
    ``tests/fixtures/openinference_messages.json`` pins it. That is why this
    does not route through ``_normalize_message``, which would default a
    missing role, write ``null`` for a missing tool-call field and drop the
    tool calls of a tool message, each of which the sink does not do. The
    encoding is the shared one (``serialize``). The one known difference is
    that the sink does not cap the attribute.

    The multimodal ``contents`` form is not optional: the Anthropic
    instrumentors write EVERY text block there, never in ``message.content``,
    so without it an Anthropic reply or a block-list system prompt arrives
    with empty parts.

    Native wins: a family whose canonical key is already present is left
    entirely as it came, flattened keys included, as the sink leaves it.
    Export-stage only, like the tool definitions: messages are content, so
    they never ride a pending snapshot, and masking, which runs after this,
    strips the canonical key under ``capture_content=False``.
    """
    if not attributes or not any(key.startswith(_MESSAGE_PREFIXES) for key in attributes):
        return None
    rebuilt: dict[str, Any] | None = None
    for prefix, target in _MESSAGE_FAMILIES:
        if target in attributes:
            continue
        family = _reassemble_family(attributes, prefix)
        if family is None:
            continue
        messages, consumed = family
        if rebuilt is None:
            rebuilt = dict(attributes)
        for key in consumed:
            del rebuilt[key]
        rebuilt[target] = serialize(messages)
    return rebuilt


def _sorted_deep(value: Any) -> Any:
    """``value`` with every object's keys sorted, at every depth."""
    if isinstance(value, dict):
        return {k: _sorted_deep(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_deep(item) for item in value]
    return value


def _system_part(block: Any) -> dict[str, str]:
    """One element of a block-list system prompt as a message part.

    A text block's text, and nothing else of it (its ``cache_control``,
    citations and so on are request plumbing, not what the model read), or
    any other block as its compact JSON with keys sorted at every depth, as the
    TypeScript SDK and the sink spell it.
    """
    if (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ):
        return {"type": "text", "content": block["text"]}
    return {"type": "text", "content": _compact_json(_sorted_deep(block))}


def promote_system_instruction(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The attributes with the bag's ``system`` member as the first input message.

    Returns the rebuilt attribute dict, or None when nothing changes. The
    OpenInference Anthropic instrumentation for JavaScript records the request
    body minus ``messages`` as ``llm.invocation_parameters``, so Anthropic's
    top-level ``system`` prompt stays in the bag and never becomes a message.
    The TypeScript SDK and the sink apply this same rule:

    * Only when the span has no system-role input message: one an
      instrumentation already wrote wins, like every native key.
    * A non-empty string becomes ``{"role":"system","parts":[{"type":"text",
      "content":<string>}]}``; a non-empty list becomes ONE system message with
      a part per element (see :func:`_system_part`). Any other shape, an empty
      string and an empty list included, is not something we can vouch is a
      prompt, so the bag is left as it came.
    * Prepended to ``gen_ai.input.messages``, created when absent. Messages
      that are present but not a JSON array are left alone with the span.
    * ``system`` leaves the bag, and a bag left empty goes too.

    Export-stage only and after the reassembly, which produces the messages it
    extends: a system prompt is content, so it never rides a pending snapshot.
    Before masking, so under ``capture_content=False`` the system message is
    stripped with the rest of the messages.
    """
    if not attributes:
        return None
    bag = attributes.get(LLM_INVOCATION_PARAMETERS)
    if not isinstance(bag, str):
        return None
    payload = loads_object(bag)
    if payload is None or "system" not in payload:
        return None
    system = payload["system"]
    if isinstance(system, str) and system:
        parts = [{"type": "text", "content": system}]
    elif isinstance(system, list) and system:
        parts = [_system_part(block) for block in system]
    else:
        return None

    messages: list[Any] = []
    existing = attributes.get(GEN_AI_INPUT_MESSAGES)
    if existing is not None:
        if not isinstance(existing, str):
            return None
        messages = loads_json(existing)
        if not isinstance(messages, list):
            return None
    if any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        return None

    rebuilt = {k: v for k, v in attributes.items() if k != LLM_INVOCATION_PARAMETERS}
    rebuilt[GEN_AI_INPUT_MESSAGES] = serialize([{"role": "system", "parts": parts}, *messages])
    leftover = {k: v for k, v in payload.items() if k != "system"}
    if leftover:
        rebuilt[LLM_INVOCATION_PARAMETERS] = serialize(leftover)
    return rebuilt


#: OpenInference's embedding vectors: ``embedding.embeddings.{i}.embedding.vector``.
#: A source spelling, like the message families above. The index is any run of
#: ASCII digits, with no sign and no bound, the same pattern the TypeScript SDK
#: uses, so the two drop exactly the same keys.
_EMBEDDING_VECTOR = re.compile(r"embedding\.embeddings\.[0-9]+\.embedding\.vector", re.ASCII)


def drop_embedding_vectors(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The attributes without ``embedding.embeddings.N.embedding.vector``.

    Returns the rebuilt dict, or None when there is no vector to drop. An
    EMBEDDING span from OpenInference carries every returned vector as its own
    attribute, 1536 floats each for ``text-embedding-3-small``. Nothing
    downstream reads them, and the provider response under ``output.value``
    already holds them, so they are never exported. The neighbouring
    ``...embedding.text`` keys are the INPUT and are kept; they are content,
    and masking strips them under ``capture_content=False``.

    Not a table rule: the family is indexed and a rule's sources are exact
    keys. Export-stage only: the vectors come from the response, so they do
    not exist at ``on_start``.
    """
    if not attributes:
        return None
    # The cheap prefix test first: this runs on every exported span.
    drop = [
        key
        for key in attributes
        if key.startswith("embedding.embeddings.") and _EMBEDDING_VECTOR.fullmatch(key)
    ]
    if not drop:
        return None
    rebuilt = dict(attributes)
    for key in drop:
        del rebuilt[key]
    return rebuilt


#: The usage breakdowns the conventions recommend only when available and
#: non-zero: the cache and reasoning subsets and the per-modality counts, as
#: suffixes after ``gen_ai.usage.``. The input and output totals are not among
#: them; they are reported even at 0.
USAGE_BREAKDOWN_SUFFIXES: tuple[str, ...] = tuple(
    key[len(GEN_AI_USAGE_PREFIX) :]
    for key in (
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        GEN_AI_USAGE_TEXT_INPUT_TOKENS,
        GEN_AI_USAGE_TEXT_OUTPUT_TOKENS,
        GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS,
        GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
        GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
        GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
        GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
        GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
        GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
    )
)

_ZERO_DROP_KEYS = tuple(
    prefix + suffix
    for prefix in (GEN_AI_USAGE_PREFIX, RIUS_AGGREGATED_USAGE_PREFIX)
    for suffix in USAGE_BREAKDOWN_SUFFIXES
)


def _is_zero(value: Any) -> bool:
    """The number 0, as the TypeScript SDK's ``=== 0``: never a bool or a string."""
    return type(value) in (int, float) and value == 0


def drop_zero_usage_breakdowns(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The attributes without a usage breakdown whose count is 0.

    Returns the rebuilt dict, or None when there is no zero to drop. Under
    ``gen_ai.usage.*`` and under the ``rius.aggregated_usage.*`` a run's usage
    moves to. Whoever wrote it, the producer natively or the table mapping a
    dialect's count: the source of a mapped one is already gone, so the zero
    leaves the span entirely. Only the number 0 (``-0.0`` included; NaN is
    not 0); a count of another shape is the producer's and stays.

    Runs on every exported span, the SDK's own generations included: a
    breakdown ``set_usage`` wrote and a later call set to 0 leaves here too.
    """
    if not attributes:
        return None
    drop = [key for key in _ZERO_DROP_KEYS if key in attributes and _is_zero(attributes[key])]
    if not drop:
        return None
    rebuilt = dict(attributes)
    for key in drop:
        del rebuilt[key]
    return rebuilt


def _llm_response_json(attributes: Mapping[str, Any]) -> str | None:
    """``output.value`` of an LLM span, when it can be the provider's JSON response object.

    Only an LLM span qualifies, by the taxonomy (the kind the taxonomy rules
    derive from either key), because a tool's or chain's ``output.value`` is
    the user's data and its ``id`` is not a response id.

    Hot path: every exported LLM span reaches here. The cheap checks all run
    before any ``json.loads``, and a value that does not start with ``{`` is
    never parsed.
    """
    if attributes.get(OPENINFERENCE_SPAN_KIND) != SpanKind.LLM.value:
        return None
    return response_json(attributes)


def response_id_from_output_value(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    """``gen_ai.response.id`` for an LLM span, from the provider object in ``output.value``.

    OpenInference writes no response-id attribute. The id (``chatcmpl-...``
    from OpenAI chat, ``resp_...`` from the Responses API, ``msg_...`` from
    Anthropic) exists only as the top-level ``id`` of the provider object the
    instrumentors serialize under ``output.value``, and masking strips
    ``output.value`` under ``capture_content=False``. It runs in the exporter,
    ahead of masking, so the id survives capture-off. ``output.value`` itself
    is never modified or removed.

    Only an LLM span qualifies (see :func:`_llm_response_json`). Native wins: a
    span already carrying ``gen_ai.response.id`` is left alone.

    Streaming, as recorded from the real instrumentors: openai's
    ``stream=True`` serializes the ACCUMULATED completion, whose ``id`` is the
    one every chunk carries, so the id is real. anthropic's ``.stream()``
    helper writes the final message, id included. anthropic's
    ``messages.create(stream=True)`` writes no ``output.value`` at all, so
    that span gets no id rather than an invented one.
    """
    if not attributes or GEN_AI_RESPONSE_ID in attributes:
        return {}
    raw = _llm_response_json(attributes)
    if raw is None:
        return {}
    payload = loads_object(raw)
    if payload is None:
        # Not JSON (truncated by a length cap, or nested too deep): no id to vouch for.
        return {}
    response_id = payload.get("id")
    if not isinstance(response_id, str) or not response_id:
        return {}
    return {GEN_AI_RESPONSE_ID: response_id}


def compactions_from_output_value(attributes: Mapping[str, Any] | None) -> ProviderCompactions:
    """The ``rius.compaction`` events an LLM span's response reports.

    Anthropic's context editing and compaction blocks and OpenAI's compaction
    items are in the provider object under ``output.value`` and nowhere else
    (see :mod:`rius._provider_compaction`). Like the response id, they are read
    in the exporter, ahead of masking, so a compaction survives capture-off;
    the summary text is a content attribute on the event and masking handles it
    there.

    The gate is :func:`_llm_response_json`, and a value that cannot report a
    compaction is not parsed at all.
    """
    raw = _llm_response_json(attributes) if attributes else None
    if raw is None:
        return ProviderCompactions([], False)
    return compactions_in_response(raw)


#: The message keys whose parts a normalizer builds (the contract's
#: ``raw_payloads.message_keys`` that hold semconv JSON). The system
#: instructions are read and rewritten here, never emitted.
_NATIVE_MESSAGE_KEYS = (GEN_AI_INPUT_MESSAGES, GEN_AI_OUTPUT_MESSAGES, GEN_AI_SYSTEM_INSTRUCTIONS)


def native_media_messages(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The native message keys with every media part made conformant.

    Returns the rebuilt attribute dict, or None when nothing changes. An
    instrumentation that already speaks the conventions can still write a
    blob WITH its bytes (pydantic-ai does), label a PDF an image
    (``@ai-sdk/otel`` does), or put a data URL in a uri part. Each such part
    is rewritten in place as the contract shapes it (see
    ``_media.native_part``); every other byte of the key stays as it came.
    The raw payloads (``pydantic_ai.all_messages`` and the like) are not
    touched here: they are never mined for parts, and the strip replaces
    their media in place.
    """
    if not attributes:
        return None
    rebuilt: dict[str, Any] | None = None
    for key in _NATIVE_MESSAGE_KEYS:
        raw = attributes.get(key)
        if not isinstance(raw, str):
            continue
        fixed = rebuild_native_parts(raw, parts_only=key == GEN_AI_SYSTEM_INSTRUCTIONS)
        if fixed != raw:
            if rebuilt is None:
                rebuilt = dict(attributes)
            rebuilt[key] = fixed
    return rebuilt


def _stripped_value(value: Any) -> Any:
    """A value with its media stripped, the same object when nothing changes.

    A string is bounded by the strip budget, as a value this SDK records is: a
    third-party payload of several MB is cut and stripped conservatively
    instead of parsed whole on the export thread. A byte object in it, at any
    depth, is the bare marker (see ``_serde.attribute_value``): the SDK's own
    setters already wrote it so, and this catches another instrumentation's.
    """
    return attribute_value(value, strip_within_budget)


def strip_media(attributes: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Every string value (and string array element) with its media stripped.

    Returns the rebuilt attribute dict, or None when nothing changes. The
    contract's raw-payload strip, run LAST, over every attribute: a raw
    request or response (``input.value``, ``output.value`` and the other raw
    keys), the message keys this exporter built or kept, and anything else a
    string can hide a data URL in. Each media token is replaced in place and
    every other byte stays (see ``_media.strip_raw_payload``), so a span the
    sink strips again comes out the same. A value past the strip budget is cut
    and stripped conservatively instead (see ``_serde.strip_within_budget``).
    """
    if not attributes:
        return None
    rebuilt: dict[str, Any] | None = None
    for key, value in attributes.items():
        stripped = _stripped_value(value)
        if stripped is not value and stripped != value:
            if rebuilt is None:
                rebuilt = dict(attributes)
            rebuilt[key] = stripped
    return rebuilt


def _stripped_events(span: ReadableSpan) -> tuple[Event, ...] | None:
    """The span's events with their attributes stripped, or None if unchanged."""
    events = tuple(span.events)
    rebuilt = [(event, strip_media(event.attributes)) for event in events]
    if all(attributes is None for _, attributes in rebuilt):
        return None
    return tuple(
        event
        if attributes is None
        else Event(event.name, rebuilt_item_attributes(event, attributes), event.timestamp)
        for event, attributes in rebuilt
    )


def _compaction_events(
    span: ReadableSpan, attributes: Mapping[str, Any] | None
) -> tuple[list[Event], bool]:
    """The ``rius.compaction`` events the span's response reports, and whether it is flagged.

    Nothing for a span that already carries one: a producer that records its
    own compaction is authoritative, and the same compaction is never reported
    twice. Stamped at the end of the call, when the response that reported it
    arrived.
    """
    found = compactions_from_output_value(attributes)
    if not found.events or any(event.name == RIUS_COMPACTION_EVENT for event in span.events):
        return [], False
    timestamp = span.end_time if span.end_time is not None else span.start_time
    events = [
        Event(RIUS_COMPACTION_EVENT, rebuilt_item_attributes(None, event), timestamp)
        for event in found.events
    ]
    return events, found.compacted_context


def _stripped_links(span: ReadableSpan) -> tuple[Link, ...] | None:
    """The span's links with their attributes stripped, or None if unchanged."""
    links = tuple(span.links)
    rebuilt = [(link, strip_media(link.attributes)) for link in links]
    if all(attributes is None for _, attributes in rebuilt):
        return None
    return tuple(
        link
        if attributes is None
        else Link(link.context, rebuilt_item_attributes(link, attributes))
        for link, attributes in rebuilt
    )


class NormalizingSpanExporter(SpanExporter):
    """Map third-party keys onto canonical ones before delegating to ``inner``.

    Works on copies; see the module docstring for why in-place is both
    impossible and wrong.
    """

    def __init__(
        self,
        inner: SpanExporter,
        *,
        table: NormalizationTable | None = None,
        wrapped_calls: WrappedCallTracker | None = None,
        compacted_context: CompactedContextProcessor | None = None,
    ) -> None:
        self._inner = inner
        self._table = table if table is not None else DEFAULT_TABLE
        # Fed by the same provider's processors; without it no wrapper's usage
        # moves (a span on its own cannot say whether it repeats a child's).
        self._wrapped_calls = wrapped_calls
        # The same for a model call that was not known to be one when it
        # started (see CompactedContextProcessor).
        self._compacted_context = compacted_context

    def _takes_mark(self, span: ReadableSpan) -> bool:
        return self._compacted_context is not None and self._compacted_context.take(span)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._normalized(span) for span in spans])

    def _normalized(self, span: ReadableSpan) -> ReadableSpan:
        new_attributes = self._table.normalize(span.attributes)
        # After the table, which maps the dialects' usage onto the canonical
        # keys, so a zero it mapped goes with the native ones. On every span,
        # the SDK's own generations included, not only third-party ones.
        zeroless = drop_zero_usage_breakdowns(
            span.attributes if new_attributes is None else new_attributes
        )
        if zeroless is not None:
            new_attributes = zeroless
        # After the table, on its output: the table has already taken the
        # request knobs out of the bag, so what this pass re-serializes is
        # the remainder, and it never has to re-derive what the table did.
        tooled = normalize_tool_definitions(
            span.attributes if new_attributes is None else new_attributes
        )
        if tooled is not None:
            new_attributes = tooled
        messages = reassemble_openinference_messages(
            span.attributes if new_attributes is None else new_attributes
        )
        if messages is not None:
            new_attributes = messages
        # After the reassembly, whose messages it extends.
        system = promote_system_instruction(
            span.attributes if new_attributes is None else new_attributes
        )
        if system is not None:
            new_attributes = system
        vectorless = drop_embedding_vectors(
            span.attributes if new_attributes is None else new_attributes
        )
        if vectorless is not None:
            new_attributes = vectorless
        # On the rebuilt attributes, so the taxonomy the table derived
        # decides whether this is an LLM span.
        llm_attributes = span.attributes if new_attributes is None else new_attributes
        compactions, compacted = _compaction_events(span, llm_attributes)
        # Always asked, so a mark is consumed even when the response flags the span.
        marked = self._takes_mark(span)
        event_attributes = {
            **first_chunk_from_event(span),
            **error_type_from_exception_event(span),
            **response_id_from_output_value(llm_attributes),
            **({GEN_AI_CONVERSATION_COMPACTED: True} if compacted or marked else {}),
        }
        if event_attributes:
            # setdefault, not update: the event passes already skipped the
            # keys the span carried natively, and a table rule that produced
            # one wins over the derived value for the same reason — it read
            # the span's own data rather than inferring.
            base = dict(span.attributes or {}) if new_attributes is None else new_attributes
            for key, value in event_attributes.items():
                base.setdefault(key, value)
            new_attributes = base
        # Replaces, unlike the passes above: the anthropic SDK's own value is
        # the API's error code, which no other span uses for the failure.
        renamed = native_error_type(span)
        if renamed:
            base = dict(span.attributes or {}) if new_attributes is None else new_attributes
            new_attributes = {**base, **renamed}
        # Last, on the canonical keys: a LiteLLM span whose child recorded the
        # same call keeps its usage only as rius.aggregated_usage.*.
        if self._wrapped_calls is not None and self._wrapped_calls.take(span):
            moved = move_aggregated_usage(
                (span.attributes or {}) if new_attributes is None else new_attributes
            )
            if moved is not None:
                new_attributes = moved
        # Media parts in the native message keys, after every pass that
        # builds or extends them, so each part is shaped once.
        native = native_media_messages(
            span.attributes if new_attributes is None else new_attributes
        )
        if native is not None:
            new_attributes = native
        # The strip runs last, over everything the passes above produced.
        stripped = strip_media(span.attributes if new_attributes is None else new_attributes)
        if stripped is not None:
            new_attributes = stripped
        events = _stripped_events(span)
        if compactions:
            events = (*(tuple(span.events) if events is None else events), *compactions)
        links = _stripped_links(span)
        if new_attributes is None and events is None and links is None:
            return span
        normalized = copy.copy(span)
        if new_attributes is not None:
            # Carrying the dropped count: a plain dict would report 0 and hide
            # a span that hit the attribute-count limit.
            normalized._attributes = replacement_attributes(span, new_attributes)
        if events is not None:
            normalized._events = events
        if links is not None:
            normalized._links = links
        return normalized

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._inner.shutdown()


# --- OpenInference (openai, anthropic, langchain, llama-index, litellm) -----
#
# Source spellings were read from the installed instrumentors, not from
# memory; the evidence for each is in the comments below and in
# tests/test_normalization_openinference.py.

#: Provider spellings the GenAI registry writes differently from the source.
#:
#: Two dialects feed ``gen_ai.provider.name`` and both have values the registry
#: renamed. The pairs come from the registry itself
#: (``opentelemetry.semconv._incubating.attributes.gen_ai_attributes``), whose
#: ``GenAiSystemValues`` members carry "Deprecated: Replaced by ``X``"
#: docstrings, and whose ``GenAiProviderNameValues`` spells xAI ``x_ai``
#: against OpenInference's ``xai``.
#:
#: Mistral is the exception: the GenAI registry has since renamed its value
#: from ``mistral_ai`` to ``mistral``, ahead of that Python package, which still
#: spells it ``mistral_ai``. Both ``mistral_ai`` and OpenInference's
#: ``mistralai`` map to ``mistral``, so an instrumentation that still writes
#: the old value reads the same as one that writes the new.
#:
#: Only renames of the SAME provider are listed. OpenInference's ``azure``,
#: ``aws`` and ``google`` are NOT translated: each covers several registry
#: values (``azure.ai.openai`` vs ``azure.ai.inference``, ``aws.bedrock`` vs
#: the rest of AWS, three ``gcp.*``), so a translation would be a guess.
#: They pass through verbatim, which is allowed — the attribute's values are
#: "well-known", not closed, and langchain already forwards any
#: ``ls_provider`` string it is given (``_tracer.py:1076``).
#:
#: The registry's own former spellings of a provider value. Unlike the
#: aliases below they are rewritten even under a native
#: ``gen_ai.provider.name`` (see ``PROVIDER_RENAME_RULES``): an
#: instrumentation written against the earlier registry, @ai-sdk/otel among
#: them, still emits them.
PROVIDER_NAME_RENAMES: Mapping[str, str] = {
    "mistral_ai": "mistral",
}

PROVIDER_NAME_ALIASES: Mapping[str, str] = {
    # OpenInference enum values (openinference.semconv.trace)
    "mistralai": "mistral",
    "xai": "x_ai",
    "vertexai": "gcp.vertex_ai",
    # legacy gen_ai.system values
    "vertex_ai": "gcp.vertex_ai",
    "gemini": "gcp.gemini",
    "az.ai.inference": "azure.ai.inference",
    "az.ai.openai": "azure.ai.openai",
    # the registry's own former spellings
    **PROVIDER_NAME_RENAMES,
}


def provider_name(values: list[Any]) -> Any:
    """The provider, under the registry's spelling where it differs."""
    value = values[0]
    if not isinstance(value, str):
        return SKIP
    return PROVIDER_NAME_ALIASES.get(value.strip().lower(), value)


def renamed_provider(raw: Any) -> Mapping[str, Any]:
    """A native ``gen_ai.provider.name`` under the registry's current spelling, in place.

    Only a former spelling of the registry's own changes; anything else is the
    producer's and stays as written. Always returns the key, so the rule
    rewrites its own source and never deletes it.
    """
    if not isinstance(raw, str):
        return {GEN_AI_PROVIDER_NAME: raw}
    return {GEN_AI_PROVIDER_NAME: PROVIDER_NAME_RENAMES.get(raw.strip().lower(), raw)}


def without_provider_route(model: str, provider: Any) -> str:
    """The model without a leading ``<provider>/`` that names the span's own provider.

    LiteLLM routes on a prefix (``openai/gpt-4o-mini``) and its instrumentor
    records the model exactly as passed, so the same model would otherwise
    carry two names depending on whether LiteLLM was in the path. The prefix
    is dropped only when it IS the provider (compared under the registry's
    spelling, so ``vertex_ai/...`` matches ``gcp.vertex_ai``): then it says
    nothing ``gen_ai.provider.name`` does not, and the routed string can be
    rebuilt from the two. Any other prefix is part of the name, as in
    ``openai/google/gemini-2.5-flash`` routed through an OpenAI-compatible
    gateway, and stays.
    """
    if not isinstance(provider, str) or not provider:
        return model
    route, slash, rest = model.partition("/")
    if not slash or not rest or provider_name([route]) != provider:
        return model
    return rest


def provider_from_system(raw: Any) -> Mapping[str, Any]:
    """``llm.system`` kept, plus the provider it implies.

    The fallback for a span with no ``llm.provider``: an OpenAI client pointed
    at a gateway, whose host the instrumentor does not recognize. The registry
    defines the provider as the instrumentation's best knowledge, allowed to
    differ from whoever served the request behind a proxy, and as the
    discriminator for the telemetry flavor; the API the client speaks is both.

    Kept rather than consumed: on Azure ``llm.system`` (openai) differs from
    ``llm.provider`` (azure), so it is a fact of its own, not a duplicate.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}
    produced: dict[str, Any] = {LLM_SYSTEM: raw}
    provider = provider_name([raw])
    if provider is not SKIP:
        produced[GEN_AI_PROVIDER_NAME] = provider
    return produced


def _number(value: Any) -> Any:
    """A finite double, or SKIP. Booleans are not numbers here, and NaN and
    the infinities are not a value a model is sent (``json`` reads them)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return SKIP
    try:
        number = float(value)
    except OverflowError:  # an int beyond a double's range
        return SKIP
    return number if math.isfinite(number) else SKIP


def _count(value: Any) -> Any:
    """A non-negative whole number as an int, or SKIP.

    Booleans are not counts, nor is a fractional float (truncating 12.9 would
    invent a value) or a non-finite one (``int()`` of one raises). A negative
    count is no count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return SKIP
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        return SKIP
    count = int(value)
    return count if count >= 0 else SKIP


def _configured_count(value: Any) -> Any:
    """``_count``, with a negative value read as "not configured".

    LlamaIndex writes ``num_output: -1`` when the request sets no output
    budget; that is no value, not a budget of -1.
    """
    return _count(value)  # which already reads a negative value as none


def _text(value: Any) -> Any:
    return value if isinstance(value, str) and value else SKIP


def _flag(value: Any) -> Any:
    return value if isinstance(value, bool) else SKIP


def nonempty_text(values: list[Any]) -> Any:
    """The source value when it is a non-empty string; SKIP otherwise."""
    return _text(values[0])


def _compact_json(value: Any) -> str:
    """``value`` as compact JSON in raw UTF-8, member order kept.

    Spelled out here, not taken from ``_serde``, because the result is matched
    byte for byte: the sink promotes the same member from the same bag and
    the TypeScript SDK writes the same string. U+2028/U+2029 stay RAW, as the
    sink writes them (its ``encodeJSON`` undoes Go's escapes for exactly these
    two) and as ``JSON.stringify`` does. An unpaired surrogate becomes U+FFFD,
    as the sink's decoder makes it.
    """
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return LONE_SURROGATE.sub("\ufffd", text)


def _tool_choice(value: Any) -> Any:
    """A mode kept as the non-empty string it is, or an object as compact JSON.

    The two shapes a provider's ``tool_choice`` takes. Any other shape is not
    one we can vouch for, so it SKIPs and stays in the bag.

    A mode's unpaired surrogate becomes U+FFFD, as it does inside an object
    and as the sink's decoder makes it. Kept, it could not be encoded as UTF-8,
    and the OTLP exporter would drop the whole attribute.
    """
    if isinstance(value, dict):
        return _compact_json(value)
    text = _text(value)
    return text if text is SKIP else LONE_SURROGATE.sub("\ufffd", text)


def _text_sequence(value: Any) -> Any:
    """A string list. A lone stop string is the one-element list of itself."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    return SKIP


#: Members of ``llm.invocation_parameters`` that a canonical key represents
#: TOTALLY (a ``gen_ai.request.*`` key, or ``rius.request.tool_choice``), in
#: precedence order (the first spelling to produce a
#: target keeps it). Everything else stays in the blob: see
#: :func:`openinference_invocation_parameters`.
INVOCATION_PARAMETER_MEMBERS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    # Every instrumentor but anthropic's keeps `model` in the bag, and it is
    # the literal request field — unlike `llm.model_name`, see below.
    ("model", GEN_AI_REQUEST_MODEL, _text),
    # LlamaIndex fills the bag with the model's metadata, not the request's
    # parameters, and names the model `model_name` there. LangChain's bag has
    # `model_name` too, next to `model` with the same value; listed after
    # `model`, it is promoted only where `model` is absent.
    ("model_name", GEN_AI_REQUEST_MODEL, _text),
    ("temperature", GEN_AI_REQUEST_TEMPERATURE, _number),
    ("top_p", GEN_AI_REQUEST_TOP_P, _number),
    ("top_k", GEN_AI_REQUEST_TOP_K, _count),
    ("max_tokens", GEN_AI_REQUEST_MAX_TOKENS, _count),
    # OpenAI's replacement for max_tokens on the reasoning models: "an upper
    # bound for the number of tokens that can be generated for a completion",
    # which is what gen_ai.request.max_tokens means. Listed second so that a
    # request carrying both keeps the one the provider would honour.
    ("max_completion_tokens", GEN_AI_REQUEST_MAX_TOKENS, _count),
    # LlamaIndex's metadata spelling of the same setting: the number of
    # tokens the model is configured to generate, -1 when nothing is.
    ("num_output", GEN_AI_REQUEST_MAX_TOKENS, _configured_count),
    ("frequency_penalty", GEN_AI_REQUEST_FREQUENCY_PENALTY, _number),
    ("presence_penalty", GEN_AI_REQUEST_PRESENCE_PENALTY, _number),
    ("seed", GEN_AI_REQUEST_SEED, _count),
    ("n", GEN_AI_REQUEST_CHOICE_COUNT, _count),
    ("stop", GEN_AI_REQUEST_STOP_SEQUENCES, _text_sequence),
    ("stop_sequences", GEN_AI_REQUEST_STOP_SEQUENCES, _text_sequence),
    ("stream", GEN_AI_REQUEST_STREAM, _flag),
    # Not a convention key: the GenAI conventions define no tool_choice, so it
    # goes where an unnamed request parameter goes, rius.request.*. Promoted
    # because context attribution reads it to tell a forced tool call from an
    # automatic one, and the bag is content: left inside, it would be dropped
    # with the bag under capture_content=False. It is a routing parameter, not
    # content, and this runs before masking.
    ("tool_choice", RIUS_REQUEST_TOOL_CHOICE, _tool_choice),
)

#: The guard for every canonical ``gen_ai.request.*`` key, by the key's type
#: in the GenAI registry: the value to record, or SKIP when the value is the
#: wrong shape for that key. The native ``model_parameters`` path
#: (``generation._request_attributes``) and the members above use the SAME
#: function per key, so one parameter has one shape on the wire whichever way
#: it arrived; a test pins that the two stay wired to this table.
REQUEST_PARAMETER_GUARDS: Mapping[str, Callable[[Any], Any]] = {
    GEN_AI_REQUEST_MODEL: _text,  # string
    GEN_AI_REQUEST_MAX_TOKENS: _count,  # int
    GEN_AI_REQUEST_CHOICE_COUNT: _count,  # int
    GEN_AI_REQUEST_TEMPERATURE: _number,  # double
    GEN_AI_REQUEST_TOP_P: _number,  # double
    GEN_AI_REQUEST_TOP_K: _count,  # int
    GEN_AI_REQUEST_STOP_SEQUENCES: _text_sequence,  # string[]
    GEN_AI_REQUEST_FREQUENCY_PENALTY: _number,  # double
    GEN_AI_REQUEST_PRESENCE_PENALTY: _number,  # double
    GEN_AI_REQUEST_ENCODING_FORMATS: _text_sequence,  # string[]
    GEN_AI_REQUEST_SEED: _count,  # int
    GEN_AI_REQUEST_STREAM: _flag,  # boolean
    GEN_AI_REQUEST_REASONING_LEVEL: _text,  # string
    GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID: _text,  # string
    GEN_AI_REQUEST_STREAM_CURSOR: _text,  # string
}


#: The prefix LangChain gives its own run-metadata keys (ls_provider,
#: ls_model_name, ls_integration, ...), in both its Python and JS releases.
_LANGCHAIN_METADATA_PREFIX = "ls_"


def _is_langchain_run(payload: Mapping[str, Any], read: Mapping[str, Any]) -> bool:
    """True when the bag came from a LangChain model object.

    The Python integration marks its bag with LangChain's ``_type``; the JS one
    does not, but both write LangChain's ``ls_*`` keys into the run metadata.
    """
    if "_type" in payload:
        return True
    raw = read.get(METADATA)
    if not isinstance(raw, str):
        return False
    metadata = loads_object(raw)
    return metadata is not None and any(
        isinstance(key, str) and key.startswith(_LANGCHAIN_METADATA_PREFIX) for key in metadata
    )


def _langchain_provider(read: Mapping[str, Any]) -> Any:
    """The provider LangChain names in the run metadata's ``ls_provider``, or SKIP.

    Through the same spellings as ``llm.provider``, which the Python LangChain
    instrumentor fills from this very key: it is the chat model class's own
    name for the API it calls (``openai`` for ``ChatOpenAI``, also behind a
    gateway).
    """
    raw = read.get(METADATA)
    if not isinstance(raw, str):
        return SKIP
    metadata = loads_object(raw)
    named = metadata.get("ls_provider") if metadata is not None else None
    if not isinstance(named, str) or not named.strip():
        return SKIP
    return provider_name([named])


def openinference_invocation_parameters(
    raw: Any, read: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Promote the spec-defined members of the request bag; keep the rest.

    The members left over go back under ``llm.invocation_parameters``, and
    that is deliberate rather than a half-measure. The bag's membership is
    open and provider-defined, and the whole bag is content to masking.
    Fanning unknown members out into keys of our own would move them out from
    under that, so a member is promoted only when a canonical key represents
    it totally, and the bag survives to carry everything else. ``tool_choice``
    is promoted too although no convention names it: attribution reads it,
    and it must not be dropped with the bag when content capture is off. A
    bag left empty is dropped rather than kept as ``{}``. The request's
    ``tools`` / ``functions`` arrays, which litellm and langchain leave here,
    are the one exception, and they are not promoted HERE: their canonical key
    is content, it must not be written at span start, and whether to promote
    them depends on another key (``llm.tools.*``) that an expander cannot see.
    :func:`normalize_tool_definitions` does it at export.

    A member that produced a canonical key is removed from the bag even when
    a native key beat it — at that point it is a duplicate, which is the same
    reason a mapped source key is deleted.

    LangChain's ``stream`` member is never promoted. LangChain serializes the
    model object's settings into the bag, so ``stream`` is the model's
    default, not what the call did: a ``ChatOpenAI.stream()`` call records
    ``false`` there. An absent key is honest; that one is not. ``read`` holds
    the span's run metadata, which identifies a LangChain run.

    A LangChain run also gets its provider from that metadata
    (``ls_provider``). The JS instrumentor records no provider key at all, so
    this is the only place a Node LangChain span can get one. The provider
    rules come earlier in the table and win, so ``llm.provider`` (which the
    Python instrumentor copies from ``ls_provider``) and ``llm.system`` keep
    precedence, and a native ``gen_ai.provider.name`` wins as always.
    """
    if not isinstance(raw, str):
        return {LLM_INVOCATION_PARAMETERS: raw}
    payload = loads_object(raw)
    if payload is None:
        # Unreadable is exactly when a pass-through is right: the caller's
        # value is the only record of it.
        return {LLM_INVOCATION_PARAMETERS: raw}

    langchain = _is_langchain_run(payload, read or {})
    produced: dict[str, Any] = {}
    leftover = dict(payload)
    for member, target, convert in INVOCATION_PARAMETER_MEMBERS:
        if member not in leftover:
            continue
        if target in produced:
            # A second spelling with the same value (LangChain's model_name
            # next to model) is a duplicate, and goes like a promoted member.
            # Left in the bag, it would be promoted on the next pass, so
            # normalizing twice would differ from normalizing once.
            if convert(leftover[member]) == produced[target]:
                del leftover[member]
            continue
        if langchain and target == GEN_AI_REQUEST_STREAM:
            continue
        value = convert(leftover[member])
        if value is SKIP:
            # Wrong shape for the canonical key; leave it where it was.
            continue
        produced[target] = value
        del leftover[member]
    provider = _langchain_provider(read or {}) if langchain else SKIP
    if not produced:
        unchanged: dict[str, Any] = {LLM_INVOCATION_PARAMETERS: raw}  # byte-identical
        if provider is not SKIP:
            unchanged[GEN_AI_PROVIDER_NAME] = provider
        return unchanged
    if leftover:
        produced[LLM_INVOCATION_PARAMETERS] = serialize(leftover)
    if provider is not SKIP:
        produced[GEN_AI_PROVIDER_NAME] = provider
    return produced


#: The shipped rules. Order matters where two sources share a target: the
#: first to produce it wins.
#:
#: DELIBERATE OMISSIONS — mappings that are not TOTAL, left unmapped so the
#: source rides through under its own name:
#:
#: * ``llm.model_name``. Its meaning varies by instrumentor and, in langchain,
#:   within one: openai sets it from the RESPONSE object
#:   (``_response_attributes_extractor.py:96``), langchain prefers the
#:   response's ``llm_output`` and falls back to the request metadata
#:   (``_tracer.py:1085-1113``), anthropic writes the request model and then
#:   overwrites it with the response model (``_wrappers.py:595``, ``:602``).
#:   Neither ``gen_ai.request.model`` nor ``gen_ai.response.model`` can hold
#:   all of that, and guessing would put a response model on a request key for
#:   a call that never got a response. The request model is recovered from the
#:   request bag's ``model`` member instead, which is unambiguous.
#: * ``llm.system`` as a consumed source. OpenInference emits both, and for
#:   Azure OpenAI they differ (``llm.provider`` = azure, ``llm.system`` =
#:   openai), so it is only a fallback for the provider and stays on the span
#:   (see ``provider_from_system``).
#: * ``llm.token_count.total``. No canonical key: the conventions record input
#:   and output and leave the sum to the reader.
#: * ``llm.token_count.prompt_details.cache_input`` ("input tokens in the
#:   prompt that were cached"). It overlaps cache_read and cache_write without
#:   saying how, so neither canonical cache key can hold it.
#: * ``llm.cost.*``. We deliberately do not put cost on the wire; the backend
#:   prices from the token counts.
#: * A SUM rule for the input tokens. Every bundled instrumentor already
#:   reports ``llm.token_count.prompt`` INCLUSIVE of the cache counts —
#:   anthropic ``_utils.py:29``, llama-index ``_callback.py:684``, langchain
#:   ``_tracer.py:1154`` and its Bedrock heuristic at ``:1204``, litellm via
#:   litellm's own Anthropic transformation (``chat/transformation.py:2358``),
#:   and openai by the provider's definition of ``prompt_tokens``. Summing
#:   again would double-count every cached token.
def taxonomy_from_kind(raw: Any) -> Mapping[str, Any]:
    """``openinference.span.kind`` kept, plus the operation it implies."""
    if not isinstance(raw, str):
        return {}
    operation = operation_for_kind(raw)
    produced: dict[str, Any] = {OPENINFERENCE_SPAN_KIND: raw}
    if operation is not None:
        produced[GEN_AI_OPERATION_NAME] = operation
    return produced


def taxonomy_from_operation(raw: Any) -> Mapping[str, Any]:
    """``gen_ai.operation.name`` kept, plus the taxonomy value it implies."""
    if not isinstance(raw, str):
        return {}
    kind = kind_for_operation(raw)
    produced: dict[str, Any] = {GEN_AI_OPERATION_NAME: raw}
    if kind is not None:
        produced[OPENINFERENCE_SPAN_KIND] = kind
    return produced


#: Both taxonomy keys on every span, whichever one the instrumentation speaks.
#:
#: These are ``ExpandingRule``s rather than plain ``Rule``s for one reason: a
#: ``Rule`` DELETES its source after mapping, and here the source must stay.
#: The two keys carry different information and the contract requires both, so
#: neither may be consumed to produce the other. An expander returning its own
#: source key is the shape that says "rewrite, do not consume".
#:
#: Both directions are needed. OpenInference instrumentors set only the kind;
#: a GenAI-native instrumentation sets only the operation. A span carrying
#: both is left alone by native-wins.
TAXONOMY_RULES: tuple[AnyRule, ...] = (
    ExpandingRule(OPENINFERENCE_SPAN_KIND, taxonomy_from_kind),
    ExpandingRule(GEN_AI_OPERATION_NAME, taxonomy_from_operation),
)

OPENINFERENCE_RULES: tuple[AnyRule, ...] = (
    # Provider. One rule, two spellings, first present wins: llm.provider is
    # OpenInference's and the more specific (azure/aws/google rather than the
    # product); gen_ai.system is the deprecated GenAI key, which we map here
    # and never emit. Like the llm.* keys, both are source spellings: the
    # registry names them so they are not typed here, and the upstream check
    # skips them because we never emit them.
    Rule((LLM_PROVIDER, GEN_AI_SYSTEM), GEN_AI_PROVIDER_NAME, provider_name),
    # Last resort, AFTER the rule above so a known host always wins: the API
    # the client speaks. Covers OpenAI clients pointed at a gateway.
    ExpandingRule(LLM_SYSTEM, provider_from_system),
    # Model. Only the anthropic instrumentor emits this unambiguous pair
    # (_wrappers.py:596, :603); see the omission note on llm.model_name.
    Rule(LLM_REQUEST_MODEL_NAME, GEN_AI_REQUEST_MODEL, copy_value),
    Rule(LLM_RESPONSE_MODEL_NAME, GEN_AI_RESPONSE_MODEL, copy_value),
    # Tool identity. OpenInference writes the bare key only on TOOL spans (on
    # an LLM span the same name sits under llm.tools.N.tool.name instead), so
    # the rule needs no kind guard. A rule rather than an export pass because
    # gen_ai.tool.name is IDENTITY: the processor adds it at start, and a
    # still-running tool call is then named on its pending snapshot, as the
    # native one is.
    #
    # Deliberately NO fallback to the span name. The native path stopped
    # deriving the tool name from the span name because a span name is not a
    # tool name, and a wrong name silently groups unrelated calls — worse than
    # an absent one. A third-party span offers no better guarantee. (It could
    # not be a rule anyway: the span name is not an attribute.)
    Rule(TOOL_NAME, GEN_AI_TOOL_NAME, nonempty_text),
    # Why the model stopped. The source is a SCALAR and the canonical key is
    # an array (one entry per generation), so wrap rather than copy.
    #
    # The VALUE is passed through untouched, deliberately. The registry
    # defines this key as a free-form string array with no enum, and says
    # instrumentations report whatever the provider supplied. OpenInference's
    # own converter lowercases and folds tool_calls/function_call into
    # tool_call; following it would replace the string OpenAI actually
    # returned with one neither the provider nor the conventions use, while
    # leaving Anthropic's end_turn and tool_use alone — so it would cost
    # fidelity and unify nothing. Grouping "stop" with "end_turn" is a
    # question for a reader who still has both, not for the SDK that would
    # destroy one of them.
    Rule(LLM_FINISH_REASON, GEN_AI_RESPONSE_FINISH_REASONS, wrap_in_list),
    # Usage. to_int rather than copy: a count under a canonical key must be a
    # count, and a converter that SKIPs is how a wrongly-shaped value stays
    # off the wire.
    Rule(LLM_TOKEN_COUNT_PROMPT, GEN_AI_USAGE_INPUT_TOKENS, to_int),
    Rule(LLM_TOKEN_COUNT_COMPLETION, GEN_AI_USAGE_OUTPUT_TOKENS, to_int),
    Rule(
        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        to_int,
    ),
    # The audio split of each count: a subset of it, never added to it, as
    # the per-modality usage keys are defined.
    Rule(LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO, GEN_AI_USAGE_AUDIO_INPUT_TOKENS, to_int),
    Rule(LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO, GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS, to_int),
    # @ai-sdk/otel's flattening of a provider's input_token_details on a
    # transcription span: the audio subset of the input, after the
    # OpenInference spelling and before pydantic-ai's, below.
    Rule(
        GEN_AI_USAGE_INPUT_TOKEN_DETAILS_AUDIO_TOKENS,
        GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
        to_int,
    ),
    # Two GenAI-shaped spellings of the same counts, each its own rule AFTER
    # the OpenInference one for its target: the OpenInference count keeps the
    # target where a span somehow carries both, and a separate rule rather
    # than a second source means an OpenInference count that won't parse
    # still lets a usable alternate through (a rule converts only its first
    # present source). Like gen_ai.system, they are source spellings we map
    # and never emit.
    #
    # cache_creation is the cache-write count's name before the upstream
    # rename to cache_write. Permanent, not a transition aid: current
    # third-party releases still emit it (pydantic-ai, and @ai-sdk/otel for
    # every Vercel AI SDK app), and the backend prices cache writes from the
    # canonical key alone. The rename is exact, so nothing is lost.
    Rule(
        GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        to_int,
    ),
    # pydantic-ai writes each provider's usage details under
    # gen_ai.usage.details.<name>, and three of those names are the same
    # reasoning split: OpenAI's reasoning_tokens, Anthropic's thinking_tokens
    # and Google's thoughts_tokens. One per provider, so a span should carry
    # at most one; if it carries several, this order decides and all of them
    # go, as in the sink. Separate rules so one that won't parse does not
    # block the next.
    Rule(
        GEN_AI_USAGE_DETAILS_REASONING_TOKENS,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_THINKING_TOKENS,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_THOUGHTS_TOKENS,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        to_int,
    ),
    # pydantic-ai also writes Google's per-modality counts there, as
    # <modality>_<prompt|cache|candidates|response>_tokens (response on the
    # Gemini Live path), and its own typed audio counts, as
    # <input|cache|output>_audio_tokens, which genai-prices fills for Google,
    # OpenAI and the OpenAI realtime API. All of them are subsets of the input,
    # cache-read and output totals, never added to them. The audio ones sit
    # after OpenInference's and @ai-sdk/otel's, which keep their targets where
    # a span carries both. The typed counts come before the Gemini splits: on
    # Gemini both arrive with the same value, except that the typed input count
    # also holds the tool-use prompt's audio, so it is the whole audio input.
    # Either way the first wins and both go, so nothing is counted twice. The
    # rest of the namespace has no canonical key; its old audio_tokens is the
    # output audio count again, under no direction.
    Rule(
        GEN_AI_USAGE_DETAILS_IMAGE_PROMPT_TOKENS,
        GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_IMAGE_CACHE_TOKENS,
        GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_IMAGE_CANDIDATES_TOKENS,
        GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_INPUT_AUDIO_TOKENS,
        GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_CACHE_AUDIO_READ_TOKENS,
        GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_OUTPUT_AUDIO_TOKENS,
        GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_AUDIO_PROMPT_TOKENS,
        GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_AUDIO_CACHE_TOKENS,
        GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_AUDIO_CANDIDATES_TOKENS,
        GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
        to_int,
    ),
    Rule(
        GEN_AI_USAGE_DETAILS_AUDIO_RESPONSE_TOKENS,
        GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
        to_int,
    ),
    # Time to first chunk under the conventions' metric name, which
    # pydantic-ai and @ai-sdk/otel write as a span attribute. The duration is
    # the whole signal: no first-token event is added.
    ExpandingRule(GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK, first_chunk_from_metric),
    # The request bag: the dedicated model rules above take precedence over
    # its `model` member.
    ExpandingRule(
        LLM_INVOCATION_PARAMETERS, openinference_invocation_parameters, reads=(METADATA,)
    ),
    # The embedding model, AFTER every LLM model spelling, so it can never
    # change which value an existing rule gives gen_ai.request.model; it only
    # fills the key on a span that has no other source for it, which is every
    # OpenInference EMBEDDING span. Unlike llm.model_name this one is mapped:
    # an embedding call has no separate response model to confuse it with, and
    # OpenInference's own GenAI converter reads it as the REQUEST model
    # (openinference.instrumentation._genai_conversion._get_request_model).
    # The openai instrumentor sets it from the response's `model`; OpenAI's
    # embedding models have no dated snapshots for a response to resolve to,
    # so it is the model asked for.
    Rule(EMBEDDING_MODEL_NAME, GEN_AI_REQUEST_MODEL, nonempty_text),
)

#: A native provider value the registry has renamed since its producer was
#: written, under the new name. Last, so a value the rules above produced is
#: already current and passes through. An ``ExpandingRule`` returning its own
#: source, so the processor writes it at start and a pending snapshot shows
#: the same name the ended span does.
PROVIDER_RENAME_RULES: tuple[AnyRule, ...] = (
    ExpandingRule(GEN_AI_PROVIDER_NAME, renamed_provider),
)

#: Live in every process. See the module docstring before adding a rule.
#: Taxonomy first: it is the only family whose rules are pure additions, and
#: putting it ahead of the mapping rules keeps the order easy to read. The
#: provider rename is last, so what the rules above produced is current.
DEFAULT_TABLE = NormalizationTable(TAXONOMY_RULES + OPENINFERENCE_RULES + PROVIDER_RENAME_RULES)
