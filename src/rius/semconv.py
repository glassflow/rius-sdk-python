"""Semantic conventions for the GlassFlow SDK.

Centralizes the OpenTelemetry instrumentation-scope name, the span-kind taxonomy,
and span attribute keys. Span kinds use the OpenInference `openinference.span.kind`
values (understood across the ecosystem); LLM specifics use OTel GenAI `gen_ai.*`.

The constants and tables come from the rius-semconv registry: they are
generated there and vendored, at a pinned release, as ``_semconv_generated``
(see scripts/sync-semconv.sh), and re-exported here. A key, its value and the
reason for it change in the registry, not here. This module adds the helper
functions that use them, because logic is not registry data.
"""

from __future__ import annotations

from collections.abc import Mapping

from opentelemetry.trace import Span
from opentelemetry.trace import SpanKind as OtelSpanKind

from rius._semconv_generated import *  # noqa: F403
from rius._semconv_generated import (
    _KIND_BY_OPERATION,
    _NAME_TARGET_BY_KIND,
    _OPERATION_BY_KIND,
    _OTEL_KIND_BY_KIND,
    CHAIN_SPAN_NAME,
    GEN_AI_AGENT_ID,
    GEN_AI_AGENT_NAME,
    GEN_AI_AGENT_VERSION,
    GEN_AI_DATA_SOURCE_ID,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_PARAMETERS,
    GEN_AI_RETRIEVAL_TOP_K,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    OPENINFERENCE_SPAN_KIND,
    RIUS_REQUEST_PREFIX,
    SpanKind,
)


def request_attribute_key(parameter: str) -> str:
    """The attribute key one caller-supplied request parameter is recorded under.

    A parameter the GenAI conventions define — under its canonical name or a
    recognised provider spelling — normalises to its canonical
    ``gen_ai.request.*`` key. Everything else keeps its key verbatim under
    ``rius.request.``.
    """
    canonical = GEN_AI_REQUEST_PARAMETERS.get(parameter)
    if canonical is not None:
        return canonical
    return f"{RIUS_REQUEST_PREFIX}{parameter}"


def operation_for_kind(kind: str) -> str | None:
    """``gen_ai.operation.name`` for a taxonomy value, or None if it has none.

    Takes the raw attribute string rather than the enum: the caller is
    normalizing a third-party span, where the value is whatever the
    instrumentation wrote. An unrecognised one yields None rather than raising,
    because a span carrying a kind we do not know is not a reason to lose the
    span.
    """
    try:
        return _OPERATION_BY_KIND.get(SpanKind(kind))
    except ValueError:
        return None


def kind_for_operation(operation: str) -> str | None:
    """The taxonomy value for a ``gen_ai.operation.name``, or None if unknown."""
    kind = _KIND_BY_OPERATION.get(operation)
    return kind.value if kind is not None else None


def otel_span_kind(kind: SpanKind) -> OtelSpanKind:
    """The OpenTelemetry ``SpanKind`` field a span of taxonomy ``kind`` should carry.

    The generated table holds the member's name, so the generated module needs
    no OpenTelemetry import; the member is looked up here.
    """
    return OtelSpanKind[_OTEL_KIND_BY_KIND[kind]]


def compose_span_name(kind: SpanKind, attributes: Mapping[str, str | int]) -> str:
    """The span name for a span of ``kind``: ``"{operation} {target}"``.

    The GenAI conventions give every operation they define a SHOULD-level span
    name of the operation followed by the one identifier saying what it acted
    on: ``chat gpt-4o``, ``embeddings text-embedding-3-small``,
    ``execute_tool get_weather``, ``invoke_agent planner``, ``retrieval kb``.

    Composed from the span's OWN creation attributes rather than from the
    caller's arguments, which is what makes the name and the attributes agree
    by construction instead of by discipline. Three things follow for free: an
    operation overridden per generation is picked up, because the override is
    already in the map; an agent name that fell back to the configured one is
    picked up for the same reason; and the rendered ``execute_tool x`` can
    never become ``gen_ai.tool.name``, because this function only ever reads.
    The TypeScript SDK composes the identical way, so the two cannot drift as
    new identifiers are added.

    The model read here is the REQUEST model, since that is the only one in
    the creation attributes. A response model arriving later would make a
    pending snapshot and its final span disagree on their name, and that
    equality is part of the pending wire contract.

    The identifier is optional on every kind, and the name degrades to the
    bare operation rather than to a name with a hole in it. ``CHAIN`` has no
    operation at all and falls back to :data:`CHAIN_SPAN_NAME`.
    """
    operation = attributes.get(GEN_AI_OPERATION_NAME)
    if not isinstance(operation, str):
        return CHAIN_SPAN_NAME
    target_key = _NAME_TARGET_BY_KIND.get(kind)
    target = attributes.get(target_key) if target_key is not None else None
    return f"{operation} {target}" if isinstance(target, str) and target else operation


def kind_attributes(
    kind: SpanKind,
    tool_name: str | None = None,
    *,
    data_source_id: str | None = None,
    top_k: int | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_version: str | None = None,
    executing_agent_name: str | None = None,
    tool_call_id: str | None = None,
    tool_type: str | None = None,
) -> dict[str, str | int]:
    """Identity attributes for a span of ``kind``, for setting at CREATION.

    Pending snapshots (pending.py) are built at ``on_start``, so taxonomy set
    via ``set_attribute`` afterwards is invisible to them; passing these at
    span creation is what makes a pending span classifiable.

    ``tool_name`` is the tool's own name, which the GenAI execute-tool
    convention requires as ``gen_ai.tool.name`` on a TOOL span. It is
    deliberately NOT the span name: the two coincide today but stop
    coinciding as soon as span names carry an operation prefix, and the tool
    name feeds the MCP detection predicate, the tool-loop analysis and the
    tool-arguments fingerprint, none of which tolerate a prefix.

    ``data_source_id`` is the index or collection a RETRIEVER span searched
    (``gen_ai.data_source.id``), and ``top_k`` how many documents it asked
    for (``gen_ai.retrieval.top_k``). Only the caller knows either, so both
    are arguments rather than something derived. What the search RETURNED is
    not here: it is unknown at creation, and set through the observation.

    ``agent_name`` and ``agent_id`` identify the agent an AGENT span invokes,
    which the conventions make Conditionally Required on an invoke-agent span.
    Neither is ever derived from the span name: unlike a tool, whose name and
    span name coincided historically, a function name is not an agent's
    identity, and guessing one mislabels every span beneath it. The caller
    supplies them, or the span helper falls back to the configured agent name.

    ``agent_version`` is ``gen_ai.agent.version``, the version of that SAME
    invoked agent's definition. AGENT-only, caller-supplied, taken verbatim,
    and never derived: not from ``service.version``, which versions the
    deployed process, and not from the main agent's version, which versions
    the agent this process IS rather than the one it just called. Different
    agent, different scope; borrowing one for the other is confidently wrong
    data. It does not touch the span NAME either — ``_NAME_TARGET_BY_KIND``
    maps AGENT to ``gen_ai.agent.name`` alone.

    ``tool_call_id`` is ``gen_ai.tool.call.id``, the identifier the MODEL put
    on the tool-call message this execution answers, and ``tool_type`` is
    ``gen_ai.tool.type`` ("function", "extension", "datastore", ...). Both are
    TOOL-only, both are caller-supplied, and neither is ever guessed: only the
    caller has seen the model's response, and a tool's type is a fact about
    its declaration rather than about the call. Neither touches the span NAME,
    because ``_NAME_TARGET_BY_KIND`` maps TOOL to ``gen_ai.tool.name`` alone.

    ``executing_agent_name`` is a SEPARATE argument writing the SAME key,
    ``gen_ai.agent.name``, on a TOOL span — where the conventions define it as
    "the human-readable name of the agent executing the tool". One key, two
    meanings, told apart by ``gen_ai.operation.name``: on ``invoke_agent`` it
    is the agent being invoked, on ``execute_tool`` the agent doing the call.
    Two arguments rather than one precisely so the two can never be fed from
    the same source by accident; the TOOL one is never passed by a caller, it
    is resolved from the enclosing agent scope (see ``_agent``). It does not
    touch the span NAME: ``_NAME_TARGET_BY_KIND`` maps TOOL to
    ``gen_ai.tool.name``, so a tool span stays ``execute_tool {tool}``.
    """
    attributes: dict[str, str | int] = {OPENINFERENCE_SPAN_KIND: kind.value}
    operation = _OPERATION_BY_KIND.get(kind)
    if operation is not None:
        attributes[GEN_AI_OPERATION_NAME] = operation
    if kind is SpanKind.TOOL:
        if tool_name is not None:
            attributes[GEN_AI_TOOL_NAME] = tool_name
        # The agent EXECUTING the tool — not one being invoked. See the
        # docstring; the AGENT branch below writes the same key from the
        # other, caller-supplied source.
        if executing_agent_name is not None:
            attributes[GEN_AI_AGENT_NAME] = executing_agent_name
        if tool_call_id is not None:
            attributes[GEN_AI_TOOL_CALL_ID] = tool_call_id
        if tool_type is not None:
            attributes[GEN_AI_TOOL_TYPE] = tool_type
    if kind is SpanKind.RETRIEVER:
        if data_source_id is not None:
            attributes[GEN_AI_DATA_SOURCE_ID] = data_source_id
        if top_k is not None:
            attributes[GEN_AI_RETRIEVAL_TOP_K] = top_k
    if kind is SpanKind.AGENT:
        if agent_name is not None:
            attributes[GEN_AI_AGENT_NAME] = agent_name
        if agent_id is not None:
            attributes[GEN_AI_AGENT_ID] = agent_id
        if agent_version is not None:
            attributes[GEN_AI_AGENT_VERSION] = agent_version
    return attributes


def set_span_kind(span: Span, kind: SpanKind) -> None:
    """Stamp a span with its OpenInference kind and (if applicable) gen_ai operation."""
    span.set_attribute(OPENINFERENCE_SPAN_KIND, kind.value)
    operation = _OPERATION_BY_KIND.get(kind)
    if operation is not None:
        span.set_attribute(GEN_AI_OPERATION_NAME, operation)
