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

from ._errors import error_type
from ._serde import serialize
from ._tracer import sdk_tracer
from .semconv import (
    ERROR_TYPE,
    GEN_AI_RETRIEVAL_DOCUMENTS,
    INPUT_VALUE,
    OUTPUT_VALUE,
    USER_ID,
    SpanKind,
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


def _creation_attributes(
    name: str,
    kind: SpanKind,
    user_id: str | None,
    data_source_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, str | int]:
    # Identity at CREATION so pending snapshots (on_start) carry it; the
    # user id is set here as well as via the user() scope so it reaches the
    # span even on a provider without UserSpanProcessor installed.
    attributes: dict[str, str | int] = dict(kind_attributes(kind, name, data_source_id, top_k))
    if user_id is not None:
        attributes[USER_ID] = user_id
    return attributes


def start_span(
    name: str,
    *,
    kind: SpanKind = SpanKind.CHAIN,
    input: Any = None,
    user_id: str | None = None,
    data_source_id: str | None = None,
    top_k: int | None = None,
) -> Observation:
    """Create a span and return an ``Observation``. You MUST call ``.end()``.

    The span is parented to the current span at creation, but is not set as the
    current span and does not auto-record exceptions. Use ``start_as_current_span``
    for block-scoped tracing.

    ``user_id`` stamps ``user.id`` on this span only; it is sugar for a span
    that has no children of its own. To attribute a whole request, including
    auto-instrumented spans, use the ``user()`` scope instead.

    ``data_source_id`` names the index, collection or knowledge base a
    ``RETRIEVER`` span searched (``gen_ai.data_source.id``), and ``top_k`` how
    many documents it asked for (``gen_ai.retrieval.top_k``). Both are set at
    creation, so a still-running retrieval is already attributable to its
    source. What came back is recorded afterwards with
    ``Observation.set_retrieved_documents``.
    """
    span = sdk_tracer().start_span(
        name,
        kind=otel_span_kind(kind),
        attributes=_creation_attributes(name, kind, user_id, data_source_id, top_k),
    )
    observation = Observation(span)
    _configure(observation, input)
    return observation


@contextmanager
def start_as_current_span(
    name: str,
    *,
    kind: SpanKind = SpanKind.CHAIN,
    input: Any = None,
    user_id: str | None = None,
    data_source_id: str | None = None,
    top_k: int | None = None,
) -> Iterator[Observation]:
    """Open a span as the current span and yield an ``Observation``; auto-ends.

    Exceptions raised in the block are recorded and set the span status to ERROR
    (OpenTelemetry's ``start_as_current_span`` default), then re-raised.

    ``user_id`` is sugar for wrapping the block in ``user(user_id)``: this span
    and every span opened inside the block carry ``user.id``.

    ``data_source_id`` names the index, collection or knowledge base a
    ``RETRIEVER`` span searched (``gen_ai.data_source.id``), and ``top_k`` how
    many documents it asked for (``gen_ai.retrieval.top_k``). Both are set at
    creation, so a still-running retrieval is already attributable to its
    source. What came back is recorded afterwards with
    ``Observation.set_retrieved_documents``.
    """
    tracer = sdk_tracer()
    with (
        user(user_id) if user_id is not None else nullcontext(),
        tracer.start_as_current_span(
            name,
            kind=otel_span_kind(kind),
            attributes=_creation_attributes(name, kind, user_id, data_source_id, top_k),
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
