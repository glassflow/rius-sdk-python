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

import numbers
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any

from opentelemetry.trace import Span

from ._agent import (
    executing_agent_name,
    executing_agent_scope,
    invoked_agent_name,
    resolve_agent_name,
)
from ._errors import record_error
from ._serde import attribute_value, serialize, text_value
from ._tracer import sdk_tracer
from .semconv import (
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


def _structured(value: Any) -> bool:
    """Whether ``value`` is a mapping or an array holding one.

    OpenTelemetry would export either as an OTLP kvlist (or drop the array as
    inhomogeneous); everywhere else the SDK writes structure as JSON.
    """
    if isinstance(value, Mapping):
        return True
    return isinstance(value, (list, tuple)) and any(_structured(item) for item in value)


def _is_bool(item: Any) -> bool:
    """A Python bool, or NumPy's bool scalar, which is no Python bool and no
    number."""
    kind = type(item)
    return isinstance(item, bool) or (
        kind.__module__ == "numpy" and kind.__name__ in ("bool_", "bool")
    )


def _plain_array(items: list[Any] | tuple[Any, ...]) -> tuple[Any, ...] | None:
    """The array as OpenTelemetry holds it, or None when it cannot hold it.

    An array is held when, ``None`` aside, its elements are all strings, all
    numbers or all booleans (empty included), as the TypeScript SDK reads
    them: a bool is no number, an int and a float are both numbers. Each
    element becomes the plain Python value (a NumPy scalar included). Ints
    beside floats become floats, since OpenTelemetry Python before 1.25 drops
    an array mixing the two; an int too large for a float leaves the array
    one OpenTelemetry cannot hold.
    """
    kinds: set[str] = set()
    out: list[Any] = []
    for item in items:
        if item is None:
            out.append(None)
            continue
        if _is_bool(item):
            kinds.add("bool")
            out.append(bool(item))
        elif isinstance(item, numbers.Integral):
            kinds.add("number")
            out.append(int(item))
        elif isinstance(item, numbers.Real):
            kinds.add("number")
            out.append(float(item))
        elif isinstance(item, str):
            kinds.add("str")
            out.append(item)
        else:
            return None
        if len(kinds) > 1:
            return None
    present = [item for item in out if item is not None]
    if any(isinstance(item, float) for item in present) and any(
        isinstance(item, int) for item in present
    ):
        try:
            return tuple(None if item is None else float(item) for item in out)
        except OverflowError:
            return None
    return tuple(out)


class Observation:
    """Handle for annotating a span from ``start_span`` / ``start_as_current_span``.

    Wraps an OpenTelemetry span and exposes the annotation surface for generic
    (non-LLM) spans: input, output, and arbitrary attributes. A string input or
    output is recorded as-is; anything else is serialized to JSON (with a
    ``repr`` fallback). Both are truncated at 32 KB.
    """

    def __init__(self, span: Span) -> None:
        self._span = span

    def set_input(self, value: Any) -> None:
        """Record the span input (the ``input.value`` attribute).

        Args:
            value: Any value. A string is recorded as-is; anything else is
                serialized to JSON with a ``repr`` fallback.
        """
        self._span.set_attribute(INPUT_VALUE, text_value(value))

    def set_output(self, value: Any) -> None:
        """Record the span output (the ``output.value`` attribute).

        Args:
            value: Any value. A string is recorded as-is; anything else is
                serialized to JSON with a ``repr`` fallback.
        """
        self._span.set_attribute(OUTPUT_VALUE, text_value(value))

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
            value: An OpenTelemetry-compatible attribute value. An array
                whose elements, ``None`` aside, are all strings, all numbers
                or all booleans is kept as an array (ints beside floats become
                floats). A mapping, or any other array (mixed types, nested
                arrays, one holding a mapping), is recorded as its JSON
                string, the way every other structured value this SDK writes
                is. A bytes, bytearray or memoryview in it, at any depth, is
                recorded as ``[stripped unknown <N> bytes]``, never its
                content.
        """
        # Structure is serialized once from the raw value (serialize writes a
        # byte object as its marker); converting first would turn a mixed
        # array into its JSON string and encode it a second time.
        if _structured(value):
            self._span.set_attribute(key, serialize(value))
            return
        written = attribute_value(value)
        if isinstance(written, (list, tuple)):
            # An array OpenTelemetry cannot hold as one (mixed types, nested
            # arrays) is its JSON, one string, as in the TypeScript SDK.
            plain = _plain_array(written)
            if plain is None:
                self._span.set_attribute(key, serialize(value))
                return
            written = plain
        self._span.set_attribute(key, written)

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


def _warn_on_span_name_as_tool_name(
    surface: str, name: str | None, kind: SpanKind, tool_name: str | None, stacklevel: int
) -> None:
    """Warn when a TOOL span's name is standing in for its tool name.

    The same ambiguity, and the same warning, as ``@observe(name=..., kind=TOOL)``:
    a caller labelling the span also renames the tool, and ``gen_ai.tool.name``
    is a Required metric dimension. ``stacklevel`` counts from this function.
    """
    if kind is SpanKind.TOOL and name is not None and tool_name is None:
        warnings.warn(
            f"{surface}(name={name!r}, kind=TOOL) is naming the tool as well as the span; "
            f"gen_ai.tool.name will be {name!r}. Pass tool_name= to set them separately. "
            "A future major release will stop deriving the tool name from the span name.",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )


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
    does not become one. Leaving it to that default on a named span raises a
    ``DeprecationWarning``, as ``@observe(name=..., kind=TOOL)`` does: a future
    major release stops deriving the tool name from the span name.

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
    _warn_on_span_name_as_tool_name("start_span", name, kind, tool_name, stacklevel=2)
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
    and ``error.type``, then re-raised unchanged.

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
    does not become one. Leaving it to that default on a named span raises a
    ``DeprecationWarning``, as ``@observe(name=..., kind=TOOL)`` does: a future
    major release stops deriving the tool name from the span name.

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
    # Two frames above this body: contextlib's __enter__, then the caller.
    _warn_on_span_name_as_tool_name("start_as_current_span", name, kind, tool_name, stacklevel=3)
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
            # Recorded below through record_error instead: OTel's own recording
            # calls str(exc), which raises for an exception with a broken
            # __str__ and would replace the caller's exception with that error.
            record_exception=False,
            set_status_on_exception=False,
        ) as span,
    ):
        observation = Observation(span)
        _configure(observation, input)
        try:
            yield observation
        except Exception as exc:
            # Only an Exception is a failure. Cancellation, KeyboardInterrupt
            # and SystemExit unwind through here unrecorded (UNSET, no event,
            # no error.type), as they do through @observe and as OTel's own
            # context manager treats them.
            record_error(span, exc)
            raise
