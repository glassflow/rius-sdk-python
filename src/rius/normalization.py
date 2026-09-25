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
families (RIUS-921), plus two GenAI-shaped usage spellings that third-party
instrumentations still emit (``cache_creation``, ``details.reasoning_tokens``).
Rules used to exercise the machinery itself live in the
tests, injected through ``table=``.

Not everything a dialect says is an attribute. OpenInference records the first
streamed chunk as a span EVENT, so it cannot go through the rule table at all;
``normalize_first_token_event`` maps it onto ``gen_ai.first_token`` plus the
canonical streaming attributes, and the exporter applies it alongside the
table. It lives in the exporter for two reasons that agree: the event does not
exist yet at ``on_start``, and the exporter is rebuilding the attribute dict
anyway, so the derived attribute is still settable there. A failure is the
same shape: an auto-instrumented span records the OTel ``exception`` event and
an ERROR status but no ``error.type``, and
``error_type_from_exception_event`` derives it there for the same reasons.
``normalize_tool_definitions`` is applied there too, for a different reason:
its source is an indexed key family (``llm.tools.N.tool.json_schema``), which
the exact-key rule table cannot match, and its target is content, which must
never be written at span start.
``reassemble_openinference_messages`` sits beside it for the same two
reasons: the flattened ``llm.input_messages.N.message.*`` and
``llm.output_messages.N.message.*`` families become ``gen_ai.input.messages``
and ``gen_ai.output.messages``, which are content.

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
from opentelemetry.trace import StatusCode

from ._serde import serialize
from .semconv import (
    ERROR_TYPE,
    EXCEPTION_EVENT,
    EXCEPTION_TYPE,
    GEN_AI_FIRST_TOKEN_EVENT,
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
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    INVOCATION_PARAMETERS_CONTENT_MEMBERS,
    LLM_INVOCATION_PARAMETERS,
    OPENINFERENCE_SPAN_KIND,
    RIUS_REQUEST_TOOL_CHOICE,
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

    def apply(self, values: list[Any]) -> Mapping[str, Any]:
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
    """

    __slots__ = ("sources", "expand")

    def __init__(self, source: str, expand: Callable[[Any], Mapping[str, Any]]) -> None:
        self.sources: tuple[str, ...] = (source,)
        self.expand = expand

    def apply(self, values: list[Any]) -> Mapping[str, Any]:
        return self.expand(values[0])


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
        """Canonical keys this table would add, without removing anything."""
        added: dict[str, Any] = {}
        for rule in self._rules:
            present = [attributes[s] for s in rule.sources if s in attributes]
            if not present:
                continue
            try:
                produced = rule.apply(present)
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


#: The name OpenInference's instrumentors give the first streamed chunk.
#: Confirmed in ``openinference-instrumentation-openai`` 0.1.52,
#: ``openinference/instrumentation/openai/_stream.py::_Stream._process_chunk``:
#: on the first iteration it calls ``add_event("First Token Stream Event")``
#: with no attributes and no explicit timestamp, so the SDK stamps the moment
#: the chunk arrived. That is the whole signal — the instrumentors set no
#: streaming attribute and no time-to-first-chunk of their own.
OPENINFERENCE_FIRST_TOKEN_EVENT = "First Token Stream Event"


def normalize_first_token_event(
    span: ReadableSpan,
) -> tuple[tuple[Event, ...] | None, dict[str, Any]]:
    """Map OpenInference's first-token event onto the canonical shape.

    Returns the rebuilt event tuple (None when nothing changes) and the
    canonical attributes to add. This does NOT go through
    :class:`NormalizationTable`: the table maps attribute keys, and the source
    here is an event.

    It is the exporter's job, not the processor's, for two reasons that point
    the same way. The event does not exist at ``on_start`` — it is added
    mid-stream — so a start-time processor has nothing to read. And the
    derived ``gen_ai.response.time_to_first_chunk`` is an attribute, which the
    exporter can still set because it rebuilds ``_attributes`` on a copy of
    the span (see :class:`NormalizingSpanExporter`); it is only the *live*
    span that is out of reach by then, and we do not need it.

    Native wins, as for attributes: a span already carrying
    ``gen_ai.first_token`` keeps it, and canonical attributes already present
    are never overwritten. The source event is dropped either way — kept, it
    would double-count as a second first-token marker.

    Unit: ``gen_ai.response.time_to_first_chunk`` is **seconds** (a float),
    matching what ``GenerationSpan.record_first_token`` emits on the native
    path and what the conventions specify.
    """
    events = span.events
    if not events or not any(e.name == OPENINFERENCE_FIRST_TOKEN_EVENT for e in events):
        return None, {}

    attributes = span.attributes or {}
    canonical_seen = any(e.name == GEN_AI_FIRST_TOKEN_EVENT for e in events)
    first_token_ns: int | None = None
    rebuilt: list[Event] = []
    for event in events:
        if event.name != OPENINFERENCE_FIRST_TOKEN_EVENT:
            rebuilt.append(event)
            continue
        if first_token_ns is None:
            first_token_ns = event.timestamp
        if canonical_seen:
            # A duplicate marker; the canonical one is authoritative.
            continue
        canonical_seen = True
        rebuilt.append(Event(GEN_AI_FIRST_TOKEN_EVENT, event.attributes, event.timestamp))

    added: dict[str, Any] = {}
    if GEN_AI_REQUEST_STREAM not in attributes:
        # A first chunk arriving is what proves the request streamed — the
        # same inference record_first_token makes.
        added[GEN_AI_REQUEST_STREAM] = True
    start_ns = span.start_time
    if (
        GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK not in attributes
        and isinstance(first_token_ns, int)
        and isinstance(start_ns, int)
    ):
        added[GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] = max(first_token_ns - start_ns, 0) / 1e9
    return tuple(rebuilt), added


def error_type_from_exception_event(span: ReadableSpan) -> dict[str, Any]:
    """``error.type`` for a failed span, from its OTel ``exception`` event.

    A third-party instrumentor that fails a span records the exception event
    and an ERROR status and stops there, so the failure never carries the
    ``error.type`` the GenAI conventions make Conditionally Required. Our own
    helpers set it natively (``_errors.record_error``); this is the same fact
    for the spans we did not write.

    Like :func:`normalize_first_token_event` it reads an EVENT, so it sits
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
#: A source spelling, so it lives here rather than in ``semconv.py`` (the set
#: of keys we emit), like the ``llm.*`` keys in the rule table below.
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
      empty goes too, rather than riding as ``{}``.

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
            try:
                schemas.append(json.loads(raw))
            except ValueError:
                return None
        drop = set(indexed.values())
        rebuilt = {k: v for k, v in attributes.items() if k not in drop}
        if not native:
            rebuilt[GEN_AI_TOOL_DEFINITIONS] = serialize(schemas)
        return rebuilt

    bag = attributes.get(LLM_INVOCATION_PARAMETERS)
    if not isinstance(bag, str):
        return None
    try:
        payload = json.loads(bag)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    members = [m for m in INVOCATION_PARAMETERS_CONTENT_MEMBERS if m in payload]
    if not members or not all(isinstance(payload[m], list) for m in members):
        # Absent, or a shape we cannot vouch is a list of definitions: the
        # bag is content already, so leaving it is safe.
        return None
    definitions = [definition for m in members for definition in payload[m]]
    leftover = {k: v for k, v in payload.items() if k not in members}
    rebuilt = {k: v for k, v in attributes.items() if k != LLM_INVOCATION_PARAMETERS}
    if leftover:
        rebuilt[LLM_INVOCATION_PARAMETERS] = serialize(leftover)
    if not native:
        rebuilt[GEN_AI_TOOL_DEFINITIONS] = serialize(definitions)
    return rebuilt


#: OpenInference's flattened message families and the canonical key each one
#: is reassembled into. Source spellings, so they live here rather than in
#: ``semconv.py``, like ``_LLM_TOOL_SCHEMA`` above.
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
    * Anything else (an image, a reasoning block, a text item carrying an id
      or a signature, an item without a type) becomes a text part holding the
      item serialized, as the generation helper does with content it has no
      part type for. The item is its fields by name (``message_content.`` cut,
      ``tool_call.`` kept), sorted, so no field is lost and every producer
      orders it the same way.
    """
    if item.keys() == {"type", "text"} and item["type"] == "text":
        return {"type": "text", "content": item["text"]}
    if item.get("type") == "tool_use" and _repeats_a_tool_call(item, tool_calls):
        return None
    return {"type": "text", "content": serialize({k: item[k] for k in sorted(item)})}


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
        new_events, event_attributes = normalize_first_token_event(span)
        event_attributes = {**event_attributes, **error_type_from_exception_event(span)}
        if event_attributes:
            # setdefault, not update: the event passes already skipped the
            # keys the span carried natively, and a table rule that produced
            # one wins over the derived value for the same reason — it read
            # the span's own data rather than inferring.
            base = dict(span.attributes or {}) if new_attributes is None else new_attributes
            for key, value in event_attributes.items():
                base.setdefault(key, value)
            new_attributes = base
        if new_attributes is None and new_events is None:
            return span
        normalized = copy.copy(span)
        if new_attributes is not None:
            normalized._attributes = new_attributes
        if new_events is not None:
            normalized._events = new_events
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
#: docstrings, and whose ``GenAiProviderNameValues`` spells xAI ``x_ai`` and
#: Mistral ``mistral_ai`` against OpenInference's ``xai`` / ``mistralai``.
#:
#: Only renames of the SAME provider are listed. OpenInference's ``azure``,
#: ``aws`` and ``google`` are NOT translated: each covers several registry
#: values (``azure.ai.openai`` vs ``azure.ai.inference``, ``aws.bedrock`` vs
#: the rest of AWS, three ``gcp.*``), so a translation would be a guess.
#: They pass through verbatim, which is allowed — the attribute's values are
#: "well-known", not closed, and langchain already forwards any
#: ``ls_provider`` string it is given (``_tracer.py:1076``).
PROVIDER_NAME_ALIASES: Mapping[str, str] = {
    # OpenInference enum values (openinference.semconv.trace)
    "mistralai": "mistral_ai",
    "xai": "x_ai",
    # legacy gen_ai.system values
    "vertex_ai": "gcp.vertex_ai",
    "gemini": "gcp.gemini",
    "az.ai.inference": "azure.ai.inference",
    "az.ai.openai": "azure.ai.openai",
}


def provider_name(values: list[Any]) -> Any:
    """The provider, under the registry's spelling where it differs."""
    value = values[0]
    if not isinstance(value, str):
        return SKIP
    return PROVIDER_NAME_ALIASES.get(value.strip().lower(), value)


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
    """An int, or SKIP. A non-finite float is SKIP: ``int()`` of one raises."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return SKIP
    if isinstance(value, float) and not math.isfinite(value):
        return SKIP
    return int(value)


def _text(value: Any) -> Any:
    return value if isinstance(value, str) and value else SKIP


def _flag(value: Any) -> Any:
    return value if isinstance(value, bool) else SKIP


def text_value(values: list[Any]) -> Any:
    """The source value when it is a non-empty string; SKIP otherwise."""
    return _text(values[0])


#: Lone UTF-16 surrogates. ``json.loads`` pairs the valid ones into one code
#: point, so any left in a decoded ``str`` are unpaired.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def _compact_json(value: Any) -> str:
    """``value`` as compact JSON in raw UTF-8, member order kept.

    Spelled out here, not taken from ``_serde``, because the result is matched
    byte for byte: the sink promotes the same member from the same bag and
    the TypeScript SDK writes the same string. Both escape U+2028/U+2029 and
    turn an unpaired surrogate into U+FFFD, as Go's encoder does.
    """
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    text = text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return _LONE_SURROGATE.sub("\ufffd", text)


def _tool_choice(value: Any) -> Any:
    """A mode kept as the non-empty string it is, or an object as compact JSON.

    The two shapes a provider's ``tool_choice`` takes. Any other shape is not
    one we can vouch for, so it SKIPs and stays in the bag.
    """
    if isinstance(value, dict):
        return _compact_json(value)
    return _text(value)


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
    ("temperature", GEN_AI_REQUEST_TEMPERATURE, _number),
    ("top_p", GEN_AI_REQUEST_TOP_P, _number),
    ("top_k", GEN_AI_REQUEST_TOP_K, _count),
    ("max_tokens", GEN_AI_REQUEST_MAX_TOKENS, _count),
    # OpenAI's replacement for max_tokens on the reasoning models: "an upper
    # bound for the number of tokens that can be generated for a completion",
    # which is what gen_ai.request.max_tokens means. Listed second so that a
    # request carrying both keeps the one the provider would honour.
    ("max_completion_tokens", GEN_AI_REQUEST_MAX_TOKENS, _count),
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


def openinference_invocation_parameters(raw: Any) -> Mapping[str, Any]:
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
    """
    if not isinstance(raw, str):
        return {LLM_INVOCATION_PARAMETERS: raw}
    try:
        payload = json.loads(raw)
    except ValueError:
        # Unreadable is exactly when a pass-through is right: the caller's
        # value is the only record of it.
        return {LLM_INVOCATION_PARAMETERS: raw}
    if not isinstance(payload, dict):
        return {LLM_INVOCATION_PARAMETERS: raw}

    produced: dict[str, Any] = {}
    leftover = dict(payload)
    for member, target, convert in INVOCATION_PARAMETER_MEMBERS:
        if member not in leftover or target in produced:
            continue
        value = convert(leftover[member])
        if value is SKIP:
            # Wrong shape for the canonical key; leave it where it was.
            continue
        produced[target] = value
        del leftover[member]
    if not produced:
        return {LLM_INVOCATION_PARAMETERS: raw}  # byte-identical
    if leftover:
        produced[LLM_INVOCATION_PARAMETERS] = serialize(leftover)
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
#: * ``llm.system``. NOT the provider: OpenInference emits both, and for Azure
#:   OpenAI they differ (``llm.provider`` = azure, ``llm.system`` = openai).
#:   ``gen_ai.provider.name`` is the provider.
#: * ``llm.token_count.total``. No canonical key: the conventions record input
#:   and output and leave the sum to the reader.
#: * ``llm.token_count.prompt_details.cache_input`` ("input tokens in the
#:   prompt that were cached"). It overlaps cache_read and cache_write without
#:   saying how, so neither canonical cache key can hold it.
#: * ``llm.token_count.*_details.audio``. No canonical key.
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
    # and never emit.
    # GEN_AI_SYSTEM is spelled here, not in semconv.py: that module is the
    # set of keys we EMIT (and a CI check holds it against the upstream
    # registry, which has dropped the deprecated name). This is a source
    # spelling, like the llm.* ones.
    Rule(("llm.provider", "gen_ai.system"), GEN_AI_PROVIDER_NAME, provider_name),
    # Model. Only the anthropic instrumentor emits this unambiguous pair
    # (_wrappers.py:596, :603); see the omission note on llm.model_name.
    Rule("llm.request.model_name", GEN_AI_REQUEST_MODEL, copy_value),
    Rule("llm.response.model_name", GEN_AI_RESPONSE_MODEL, copy_value),
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
    Rule("tool.name", GEN_AI_TOOL_NAME, text_value),
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
    Rule("llm.finish_reason", GEN_AI_RESPONSE_FINISH_REASONS, wrap_in_list),
    # Usage. to_int rather than copy: a count under a canonical key must be a
    # count, and a converter that SKIPs is how a wrongly-shaped value stays
    # off the wire.
    Rule("llm.token_count.prompt", GEN_AI_USAGE_INPUT_TOKENS, to_int),
    Rule("llm.token_count.completion", GEN_AI_USAGE_OUTPUT_TOKENS, to_int),
    Rule(
        "llm.token_count.prompt_details.cache_read",
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        "llm.token_count.prompt_details.cache_write",
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        to_int,
    ),
    Rule(
        "llm.token_count.completion_details.reasoning",
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        to_int,
    ),
    # Two GenAI-shaped spellings of the same counts, each its own rule AFTER
    # the OpenInference one for its target: the OpenInference count keeps the
    # target where a span somehow carries both, and a separate rule rather
    # than a second source means an OpenInference count that won't parse
    # still lets a usable alternate through (a rule converts only its first
    # present source). Spelled here rather than in semconv.py for the reason
    # gen_ai.system is: they are source spellings we map and never emit.
    #
    # cache_creation is the cache-write count's name before the upstream
    # rename to cache_write. Permanent, not a transition aid: current
    # third-party releases still emit it (pydantic-ai, and @ai-sdk/otel for
    # every Vercel AI SDK app), and the backend prices cache writes from the
    # canonical key alone. The rename is exact, so nothing is lost.
    Rule(
        "gen_ai.usage.cache_creation.input_tokens",
        GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
        to_int,
    ),
    # pydantic-ai writes OpenAI's reasoning count as one entry of its usage
    # details namespace. The other providers' names there for the same split
    # (Anthropic's thinking_tokens, Google's thoughts_tokens) are not mapped,
    # matching the sink; the rest of the namespace has no canonical key.
    Rule(
        "gen_ai.usage.details.reasoning_tokens",
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        to_int,
    ),
    # The request bag, last: the dedicated model rules above take precedence
    # over its `model` member.
    ExpandingRule(LLM_INVOCATION_PARAMETERS, openinference_invocation_parameters),
)

#: Live in every process. See the module docstring before adding a rule.
#: Taxonomy first: it is the only family whose rules are pure additions, and
#: putting it ahead of the mapping rules keeps the order easy to read.
DEFAULT_TABLE = NormalizationTable(TAXONOMY_RULES + OPENINFERENCE_RULES)
