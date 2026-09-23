"""Manual span API.

Two surfaces, following the OpenTelemetry / Langfuse / Laminar convention:

- ``start_as_current_span`` is the context manager: it activates the span in
  the OTel context (so children nest under it) and auto-ends it.
- ``start_span`` is manual: it returns an ``Observation`` you must ``.end()``. The span
  is parented to the current span at creation but is NOT set as current and does
  NOT auto-record exceptions. For lifetimes a ``with`` block can't express
  (streaming, callbacks, passing a span across boundaries).

``start_generation`` / ``start_as_current_generation`` are the LLM-specialized
equivalents.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from opentelemetry.trace import Span

from ._agent import (
    executing_agent_name,
    executing_agent_scope,
    invoked_agent_name,
    resolve_agent_name,
)
from ._errors import error_type, record_error
from ._serde import serialize
from ._tracer import sdk_tracer
from .semconv import (
    ERROR_TYPE,
    GEN_AI_RETRIEVAL_DOCUMENTS,
    INPUT_VALUE,
    OUTPUT_VALUE,
    USER_ID,
    SpanKind,
    compose_span_name,
    kind_attributes,
    otel_span_kind,
)
from .user import user


class Observation:
    """Handle for annotating a span from ``start_span`` / ``start_as_current_span``.

    Wraps an OpenTelemetry span and exposes the annotation surface for generic
    (non-LLM) spans: input, output, and arbitrary attributes. Inputs and
    outputs are serialized to JSON (with a ``repr`` fallback) and truncated at 32 KB.
    """

    def __init__(self, span: Span) -> None:
        self._span = span

    def set_input(self, value: Any) -> None:
        """Record the span input (the ``input.value`` attribute).

        Args:
            value: Any value; serialized to JSON with a ``repr`` fallback.
        """
        self._span.set_attribute(INPUT_VALUE, serialize(value))

    def set_output(self, value: Any) -> None:
        """Record the span output (the ``output.value`` attribute).

        Args:
            value: Any value; serialized to JSON with a ``repr`` fallback.
        """
        self._span.set_attribute(OUTPUT_VALUE, serialize(value))

    def set_retrieved_documents(self, documents: Any) -> None:
        """Record what a retrieval returned (``gen_ai.retrieval.documents``).

        The conventions define this as an array of objects, each carrying an
        optional ``id`` and an optional ``score``. Identifiers and relevance,
        never document text, which is why the spec does not mark it sensitive
        and why it is not treated as content here: it survives
        ``capture_content=False`` the way token counts do. Put the retrieved
        text in :meth:`set_output` if you want it captured, and it will be
        masked and stripped like every other content attribute.

        Unlike the data source and ``top_k``, which describe the request and
        are passed at span creation, this is only knowable once the search has
        run, so it never reaches a pending snapshot.

        Args:
            documents: A sequence of ``{"id": ..., "score": ...}`` mappings;
                serialized to JSON, since OTel attributes carry no structure.
        """
        self._span.set_attribute(GEN_AI_RETRIEVAL_DOCUMENTS, serialize(documents))

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an arbitrary attribute on the underlying span.

        Args:
            key: Attribute name.
            value: An OpenTelemetry-compatible attribute value.
        """
        self._span.set_attribute(key, value)

    def update(self, *, input: Any = None, output: Any = None) -> None:
        """Record input and/or output in one call.

        Args:
            input: When not ``None``, forwarded to :meth:`set_input`.
            output: When not ``None``, forwarded to :meth:`set_output`.
        """
        if input is not None:
            self.set_input(input)
        if output is not None:
            self.set_output(output)

    def record_exception(self, exc: BaseException) -> None:
        """Record a failure on this span: event, ERROR status and ``error.type``.

        ``start_span`` does not auto-record exceptions, so a manual handle has
        to be told. Recording the failure through the underlying span object
        instead leaves ``error.type`` unset, and the span then groups
        differently from every other error span the SDK emits.

        Args:
            exc: The exception that ended the operation. ``error.type`` is set
                to its class, module-qualified unless it is a builtin, which
                is the same spelling the exception event uses.
        """
        record_error(self._span, exc)

    def end(self) -> None:
        """End the underlying span.

        Required for spans created with ``start_span``; spans from
        ``start_as_current_span`` end automatically when the block exits.
        """
        self._span.end()


def _configure(observation: Observation, input: Any) -> None:
    # Kind (and user id) are already on the span from _creation_attributes;
    # writing them again cost a locked BoundedAttributes update per span.
    if input is not None:
        observation.set_input(input)


def _creation(
    name: str | None,
    kind: SpanKind,
    user_id: str | None,
    tool_name: str | None = None,
    *,
    data_source_id: str | None = None,
    top_k: int | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_version: str | None = None,
    tool_call_id: str | None = None,
    tool_type: str | None = None,
) -> tuple[str, dict[str, str | int]]:
    """The span name and the identity attributes, resolved together.

    One function because the two must agree: the name is composed from the
    attribute map itself, so it cannot read an identifier the span does not
    carry, and the rendered ``execute_tool x`` can never travel back into
    ``gen_ai.tool.name``.
    """
    # Identity at CREATION so pending snapshots (on_start) carry it; the
    # user id is set here as well as via the user() scope so it reaches the
    # span even on a provider without UserSpanProcessor installed.
    # The tool name falls back to the span name, which is all a caller of this
    # surface gives us; an explicit one keeps the two independent. An unnamed
    # span has nothing to fall back to, and composes from the tool name alone.
    resolved_tool_name = tool_name if tool_name is not None else name
    resolved_agent_name = resolve_agent_name(agent_name, kind)
    # The OTHER meaning of gen_ai.agent.name: on a TOOL span it is the agent
    # DOING the call, taken from the enclosing agent scope rather than from
    # any argument. Resolved only for TOOL, so no other kind pays for a
    # context lookup it would discard.
    resolved_executing_agent = executing_agent_name() if kind is SpanKind.TOOL else None
    attributes: dict[str, str | int] = dict(
        kind_attributes(
            kind,
            resolved_tool_name,
            data_source_id=data_source_id,
            top_k=top_k,
            agent_name=resolved_agent_name,
            agent_id=agent_id,
            agent_version=agent_version,
            executing_agent_name=resolved_executing_agent,
            tool_call_id=tool_call_id,
            tool_type=tool_type,
        )
    )
    if user_id is not None:
        attributes[USER_ID] = user_id
    return (name if name is not None else compose_span_name(kind, attributes)), attributes


def start_span(
    name: str | None = None,
    *,
    kind: SpanKind = SpanKind.CHAIN,
    input: Any = None,
    user_id: str | None = None,
    tool_name: str | None = None,
    data_source_id: str | None = None,
    top_k: int | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_version: str | None = None,
    tool_call_id: str | None = None,
    tool_type: str | None = None,
) -> Observation:
    """Create a span and return an ``Observation``. You MUST call ``.end()``.

    The span is parented to the current span at creation, but is not set as the
    current span and does not auto-record exceptions. Use ``start_as_current_span``
    for block-scoped tracing.

    ``name`` is optional. Left out, the span is named the way the GenAI
    conventions say it should be: the operation the kind maps to, followed by
    what it acted on — ``execute_tool get_weather``, ``invoke_agent planner``,
    ``retrieval product-kb`` — falling back to the bare operation when that
    identifier is unknown. A ``CHAIN`` span is the one degenerate case: the
    conventions define no operation for a generic step and this surface has no
    function to borrow a qualname from, so an unnamed one is called ``chain``.
    An explicit name always wins.

    ``user_id`` stamps ``user.id`` on this span only; it is sugar for a span
    that has no children of its own. To attribute a whole request, including
    auto-instrumented spans, use the ``user()`` scope instead.

    ``tool_name`` sets ``gen_ai.tool.name`` on a ``TOOL`` span; it defaults to
    the span name, and exists so a span name that is not the bare tool name
    does not become one.

    ``tool_call_id`` is the id the MODEL put on the tool-call message this
    ``TOOL`` span answers (``gen_ai.tool.call.id``), which is what joins the
    tool span back to the generation that requested it; ``tool_type`` is what
    kind of tool ran (``gen_ai.tool.type``: ``"function"``, ``"extension"``,
    ``"datastore"``, ...). Both are set at creation, so a still-running tool
    call is already attributable, and neither is ever derived: only the caller
    has seen the model's response, and neither changes the span name. Both are
    ignored on every other kind.

    ``data_source_id`` names the index, collection or knowledge base a
    ``RETRIEVER`` span searched (``gen_ai.data_source.id``), and ``top_k`` how
    many documents it asked for (``gen_ai.retrieval.top_k``). Both are set at
    creation, so a still-running retrieval is already attributable to its
    source. What came back is recorded afterwards with
    ``Observation.set_retrieved_documents``.

    ``agent_name`` names the agent an ``AGENT`` span INVOKES, which is what
    ``gen_ai.agent.name`` means on an invoke-agent span. Unset, it falls back
    to the agent name ``init()`` was given, and is never taken from the span
    name. ``agent_id`` is for a HOSTED agent resource, such as a Bedrock agent
    ARN; the conventions advise against putting a transient in-memory instance
    id there, so an in-process agent leaves it unset. ``agent_version`` is
    that same invoked agent's definition version (``gen_ai.agent.version``),
    taken verbatim — the conventions' examples are ``"1.0.0"`` and
    ``"2025-05-01"``, so there is no format to validate — and never derived
    from ``service.version`` or from the process's own main-agent version:
    different agent, different scope. All three are ignored on every other
    kind.

    A ``TOOL`` span carries ``gen_ai.agent.name`` too, and there it means the
    agent EXECUTING the tool: the innermost enclosing agent scope, else the
    configured agent name. This surface does not OPEN such a scope, because it
    does not activate context at all — an agent span from ``start_span`` does
    not lend its name to the tool spans that follow it, the same limitation
    ``user_id`` has here. Use ``start_as_current_span`` for that.
    """
    span_name, attributes = _creation(
        name,
        kind,
        user_id,
        tool_name,
        data_source_id=data_source_id,
        top_k=top_k,
        agent_name=agent_name,
        agent_id=agent_id,
        agent_version=agent_version,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
    )
    span = sdk_tracer().start_span(
        span_name,
        kind=otel_span_kind(kind),
        attributes=attributes,
    )
    observation = Observation(span)
    _configure(observation, input)
    return observation


@contextmanager
def start_as_current_span(
    name: str | None = None,
    *,
    kind: SpanKind = SpanKind.CHAIN,
    input: Any = None,
    user_id: str | None = None,
    tool_name: str | None = None,
    data_source_id: str | None = None,
    top_k: int | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_version: str | None = None,
    tool_call_id: str | None = None,
    tool_type: str | None = None,
) -> Iterator[Observation]:
    """Open a span as the current span and yield an ``Observation``; auto-ends.

    Exceptions raised in the block are recorded and set the span status to ERROR
    (OpenTelemetry's ``start_as_current_span`` default), then re-raised.

    ``name`` is optional. Left out, the span is named the way the GenAI
    conventions say it should be: the operation the kind maps to, followed by
    what it acted on — ``execute_tool get_weather``, ``invoke_agent planner``,
    ``retrieval product-kb`` — falling back to the bare operation when that
    identifier is unknown. A ``CHAIN`` span is the one degenerate case: the
    conventions define no operation for a generic step and this surface has no
    function to borrow a qualname from, so an unnamed one is called ``chain``.
    An explicit name always wins.

    ``user_id`` is sugar for wrapping the block in ``user(user_id)``: this span
    and every span opened inside the block carry ``user.id``.

    ``tool_name`` sets ``gen_ai.tool.name`` on a ``TOOL`` span; it defaults to
    the span name, and exists so a span name that is not the bare tool name
    does not become one.

    ``tool_call_id`` is the id the MODEL put on the tool-call message this
    ``TOOL`` span answers (``gen_ai.tool.call.id``), which is what joins the
    tool span back to the generation that requested it; ``tool_type`` is what
    kind of tool ran (``gen_ai.tool.type``: ``"function"``, ``"extension"``,
    ``"datastore"``, ...). Both are set at creation, so a still-running tool
    call is already attributable, and neither is ever derived: only the caller
    has seen the model's response, and neither changes the span name. Both are
    ignored on every other kind.

    ``data_source_id`` names the index, collection or knowledge base a
    ``RETRIEVER`` span searched (``gen_ai.data_source.id``), and ``top_k`` how
    many documents it asked for (``gen_ai.retrieval.top_k``). Both are set at
    creation, so a still-running retrieval is already attributable to its
    source. What came back is recorded afterwards with
    ``Observation.set_retrieved_documents``.

    ``agent_name`` names the agent an ``AGENT`` span INVOKES, which is what
    ``gen_ai.agent.name`` means on an invoke-agent span. Unset, it falls back
    to the agent name ``init()`` was given, and is never taken from the span
    name. ``agent_id`` is for a HOSTED agent resource, such as a Bedrock agent
    ARN; the conventions advise against putting a transient in-memory instance
    id there, so an in-process agent leaves it unset. ``agent_version`` is
    that same invoked agent's definition version (``gen_ai.agent.version``),
    taken verbatim — the conventions' examples are ``"1.0.0"`` and
    ``"2025-05-01"``, so there is no format to validate — and never derived
    from ``service.version`` or from the process's own main-agent version:
    different agent, different scope. All three are ignored on every other
    kind.

    A ``TOOL`` span carries ``gen_ai.agent.name`` too, and there it means the
    agent EXECUTING the tool. An ``AGENT`` block opened here scopes its
    resolved agent name over everything inside it, so every tool span in the
    block — including MCP tool calls — names it as the executor; nested agent
    blocks override outer ones. Without any enclosing block the configured
    agent name applies, and a process that named nothing emits nothing.
    """
    tracer = sdk_tracer()
    span_name, attributes = _creation(
        name,
        kind,
        user_id,
        tool_name,
        data_source_id=data_source_id,
        top_k=top_k,
        agent_name=agent_name,
        agent_id=agent_id,
        agent_version=agent_version,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
    )
    with (
        user(user_id) if user_id is not None else nullcontext(),
        # An AGENT block is the scope every TOOL span inside it reads to say
        # who executed it. Only this surface can open one: start_span does
        # not activate context, so its agent spans do not scope their tools,
        # the same limitation user_id has there.
        executing_agent_scope(invoked_agent_name(kind, attributes)),
        tracer.start_as_current_span(
            span_name,
            kind=otel_span_kind(kind),
            attributes=attributes,
        ) as span,
    ):
        observation = Observation(span)
        _configure(observation, input)
        try:
            yield observation
        except BaseException as exc:
            # The OTel context manager above records the exception event and
            # ERROR status as the block unwinds; error.type is the one thing
            # it does not set, and the GenAI conventions require it on a span
            # that ends in an error. Set it before re-raising so the event and
            # this attribute land on the same span.
            span.set_attribute(ERROR_TYPE, error_type(exc))
            raise
