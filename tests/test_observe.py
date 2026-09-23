import asyncio
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind as OtelSpanKind
from opentelemetry.trace import StatusCode

from rius import observe
from rius.semconv import SpanKind


@observe
def add(a: int, b: int) -> int:
    return a + b


@observe(name="custom-name")
def named() -> str:
    return "ok"


@observe
def boom() -> None:
    raise ValueError("nope")


@observe
async def async_add(a: int, b: int) -> int:
    return a + b


@observe
def gen(n: int):
    yield from range(n)


@observe(capture_input=False, capture_output=False)
def secret(password: str) -> str:
    return "redacted-return"


@observe
def outer() -> int:
    return add(1, 2)


def test_sync_span_named_after_function(exported_spans: InMemorySpanExporter) -> None:
    assert add(2, 3) == 5
    spans = exported_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "add"


def test_custom_name(exported_spans: InMemorySpanExporter) -> None:
    named()
    assert exported_spans.get_finished_spans()[0].name == "custom-name"


def test_captures_input_and_output(exported_spans: InMemorySpanExporter) -> None:
    add(2, 3)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "2" in attrs["input.value"] and "3" in attrs["input.value"]
    assert attrs["output.value"] == "5"


def test_exception_records_error_and_reraises(exported_spans: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError, match="nope"):
        boom()
    span = exported_spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)


def test_async_function(exported_spans: InMemorySpanExporter) -> None:
    assert asyncio.run(async_add(2, 3)) == 5
    assert exported_spans.get_finished_spans()[0].name == "async_add"


def test_generator_spans_iteration(exported_spans: InMemorySpanExporter) -> None:
    assert list(gen(3)) == [0, 1, 2]
    spans = exported_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "gen"


def test_capture_flags_disable_io(exported_spans: InMemorySpanExporter) -> None:
    secret("hunter2")
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert "input.value" not in attrs
    assert "output.value" not in attrs


def test_default_kind_is_chain(exported_spans: InMemorySpanExporter) -> None:
    add(1, 1)
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["openinference.span.kind"] == "CHAIN"


def test_kind_override(exported_spans: InMemorySpanExporter) -> None:
    @observe(kind=SpanKind.TOOL)
    def search(q: str) -> str:
        return "result"

    search("hi")
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["openinference.span.kind"] == "TOOL"


def test_nested_spans_have_parent(exported_spans: InMemorySpanExporter) -> None:
    assert outer() == 3
    spans = {s.name: s for s in exported_spans.get_finished_spans()}
    assert set(spans) == {"outer", "add"}
    assert spans["add"].parent.span_id == spans["outer"].context.span_id


class Service:
    @observe
    def work(self, x: int) -> int:
        return x * 2


@observe
async def agen(n: int):
    for i in range(n):
        yield i


def test_method_target(exported_spans: InMemorySpanExporter) -> None:
    assert Service().work(21) == 42
    assert exported_spans.get_finished_spans()[0].name == "Service.work"


def test_async_generator(exported_spans: InMemorySpanExporter) -> None:
    async def consume() -> list[int]:
        return [i async for i in agen(3)]

    assert asyncio.run(consume()) == [0, 1, 2]
    spans = exported_spans.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "agen"


# --- exception recording and generator context hygiene ---


def test_exception_recorded_exactly_once(exported_spans: InMemorySpanExporter) -> None:
    @observe
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    span = exported_spans.get_finished_spans()[0]
    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1
    assert span.status.status_code == trace.StatusCode.ERROR


def test_generator_does_not_leak_context_into_caller(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe
    def stream() -> "Iterator[int]":
        yield 1
        yield 2

    g = stream()
    next(g)
    # between yields, the caller's context must NOT have the generator span active
    assert trace.get_current_span() is trace.INVALID_SPAN
    tracer = trace.get_tracer("caller")
    with tracer.start_as_current_span("caller-work"):
        pass
    next(g, None)
    g.close()

    spans = {s.name: s for s in exported_spans.get_finished_spans()}
    assert spans["caller-work"].parent is None  # not parented under the generator


def test_generator_body_spans_still_nest_under_generator_span(
    exported_spans: InMemorySpanExporter,
) -> None:
    tracer = trace.get_tracer("inner")

    @observe
    def stream() -> "Iterator[int]":
        with tracer.start_as_current_span("inner-work"):
            yield 1

    list(stream())
    spans = {s.name: s for s in exported_spans.get_finished_spans()}
    gen_span = spans[next(n for n in spans if n.endswith("stream"))]
    assert spans["inner-work"].parent is not None
    assert spans["inner-work"].parent.span_id == gen_span.context.span_id


# --- error.type ---
# GenAI semconv: error.type is Conditionally Required on every span that ends
# in an error; the class only (low cardinality), never the message.


def test_sync_exception_sets_error_type(exported_spans: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError, match="nope"):
        boom()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["error.type"] == "ValueError"


def test_async_exception_sets_error_type(exported_spans: InMemorySpanExporter) -> None:
    @observe
    async def async_boom() -> None:
        raise KeyError("missing")

    with pytest.raises(KeyError):
        asyncio.run(async_boom())
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["error.type"] == "KeyError"


def test_generator_exception_sets_error_type(exported_spans: InMemorySpanExporter) -> None:
    @observe
    def gen_boom():
        yield 1
        raise ValueError("mid-stream")

    with pytest.raises(ValueError, match="mid-stream"):
        list(gen_boom())
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["error.type"] == "ValueError"


def test_error_type_is_module_qualified_for_user_exceptions(
    exported_spans: InMemorySpanExporter,
) -> None:
    class ToolBroke(RuntimeError):
        pass

    @observe
    def custom_boom() -> None:
        raise ToolBroke("details that must not leak into error.type")

    with pytest.raises(ToolBroke):
        custom_boom()
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["error.type"] == f"{ToolBroke.__module__}.{ToolBroke.__qualname__}"
    assert "details" not in attrs["error.type"]


def test_success_sets_no_error_type(exported_spans: InMemorySpanExporter) -> None:
    add(1, 2)
    assert "error.type" not in exported_spans.get_finished_spans()[0].attributes


def test_tool_kind_takes_the_tool_name_from_its_own_argument(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe(name="search-docs", kind=SpanKind.TOOL, tool_name="search")
    def search(q: str) -> str:
        return "result"

    search("hi")
    attrs = exported_spans.get_finished_spans()[0].attributes
    assert attrs["gen_ai.tool.name"] == "search"


def test_a_custom_name_still_names_the_tool_and_warns(
    exported_spans: InMemorySpanExporter,
) -> None:
    """A custom name on a TOOL decorator names the tool, as it always has.

    Every comparable decorator resolves tool identity as "explicit name, else
    function name", so a caller who wrote ``name=`` is naming the tool, not
    labelling the span. Dropping that would rename their tool on upgrade, and
    ``gen_ai.tool.name`` is a Required metric dimension, so the rename would
    split their series silently. The warning is what makes the coupling
    visible without breaking them.
    """

    # The resolution happens once when the decorator is applied, not per call,
    # so the warning fires at decoration time and costs nothing at run time.
    with pytest.warns(DeprecationWarning, match="naming the tool as well as the span"):

        @observe(name="search-docs", kind=SpanKind.TOOL)
        def search(q: str) -> str:
            return "result"

    search("hi")
    span = exported_spans.get_finished_spans()[0]
    assert span.name == "search-docs"
    assert span.attributes["gen_ai.tool.name"] == "search-docs"


def test_an_explicit_tool_name_silences_the_warning(
    exported_spans: InMemorySpanExporter,
) -> None:
    """Separating the two is the shape this converges on, so it must be quiet."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        @observe(name="search-docs", kind=SpanKind.TOOL, tool_name="search")
        def search(q: str) -> str:
            return "result"

    search("hi")
    span = exported_spans.get_finished_spans()[0]
    assert span.name == "search-docs"
    assert span.attributes["gen_ai.tool.name"] == "search"


def test_a_custom_name_on_a_non_tool_kind_does_not_warn(
    exported_spans: InMemorySpanExporter,
) -> None:
    """Only TOOL spans carry a tool name, so only they can be ambiguous."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        @observe(name="step-one", kind=SpanKind.CHAIN)
        def step() -> str:
            return "ok"

    step()

    assert "gen_ai.tool.name" not in exported_spans.get_finished_spans()[0].attributes


def test_tool_name_never_carries_the_operation_prefix(
    exported_spans: InMemorySpanExporter,
) -> None:
    """A spec-form span name must not leak its operation into the tool name.

    The tool name feeds the MCP detection predicate, the tool-loop analysis
    and the tool-arguments fingerprint, none of which tolerate a prefix.
    """

    @observe(name="execute_tool get_weather", kind=SpanKind.TOOL, tool_name="get_weather")
    def weather(city: str) -> str:
        return "sunny"

    weather("Berlin")
    span = exported_spans.get_finished_spans()[0]
    assert span.name == "execute_tool get_weather"
    assert span.attributes["gen_ai.tool.name"] == "get_weather"


def test_tool_kind_sets_gen_ai_tool_name_from_qualname(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe(kind=SpanKind.TOOL)
    def lookup(q: str) -> str:
        return "result"

    lookup("hi")
    span = exported_spans.get_finished_spans()[0]
    assert span.attributes["gen_ai.tool.name"] == span.name
    assert span.name.endswith("lookup")


def test_tool_kind_sets_gen_ai_tool_name_on_generators(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe(kind=SpanKind.TOOL)
    def stream():  # noqa: ANN202
        yield 1

    @observe(kind=SpanKind.TOOL)
    async def astream():  # noqa: ANN202
        yield 1

    @observe(kind=SpanKind.TOOL)
    async def acall() -> int:
        return 1

    async def drive() -> None:
        async for _ in astream():
            pass
        await acall()

    list(stream())
    asyncio.run(drive())
    for span in exported_spans.get_finished_spans():
        assert span.attributes["gen_ai.tool.name"] == span.name


def test_non_tool_kind_has_no_gen_ai_tool_name(exported_spans: InMemorySpanExporter) -> None:
    add(1, 1)
    assert "gen_ai.tool.name" not in exported_spans.get_finished_spans()[0].attributes


# --- the OTel SpanKind FIELD, derived from the taxonomy ---


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (SpanKind.LLM, OtelSpanKind.CLIENT),
        (SpanKind.EMBEDDING, OtelSpanKind.CLIENT),
        (SpanKind.RETRIEVER, OtelSpanKind.CLIENT),
        (SpanKind.TOOL, OtelSpanKind.INTERNAL),
        (SpanKind.AGENT, OtelSpanKind.INTERNAL),
        (SpanKind.CHAIN, OtelSpanKind.INTERNAL),
    ],
)
def test_otel_kind_field_follows_taxonomy_on_sync_path(
    exported_spans: InMemorySpanExporter, kind: SpanKind, expected: OtelSpanKind
) -> None:
    @observe(kind=kind)
    def step() -> None:
        pass

    step()
    span = exported_spans.get_finished_spans()[0]
    assert span.kind is expected
    assert span.attributes["openinference.span.kind"] == kind.value


def test_otel_kind_field_follows_taxonomy_on_async_path(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe(kind=SpanKind.RETRIEVER)
    async def search() -> None:
        pass

    asyncio.run(search())
    assert exported_spans.get_finished_spans()[0].kind is OtelSpanKind.CLIENT


def test_otel_kind_field_follows_taxonomy_on_generator_paths(
    exported_spans: InMemorySpanExporter,
) -> None:
    @observe(kind=SpanKind.EMBEDDING)
    def embed():
        yield 1

    @observe(kind=SpanKind.EMBEDDING)
    async def aembed():
        yield 1

    async def drain() -> None:
        async for _ in aembed():
            pass

    list(embed())
    asyncio.run(drain())
    assert [s.kind for s in exported_spans.get_finished_spans()] == [
        OtelSpanKind.CLIENT,
        OtelSpanKind.CLIENT,
    ]
