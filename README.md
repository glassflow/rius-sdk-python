<p align="center">
  <img src="https://raw.githubusercontent.com/glassflow/rius-sdk-python/main/docs/assets/banner.png" alt="Rius by GlassFlow: trace your AI agents with OpenTelemetry" width="100%">
</p>

# Rius Python SDK

OpenTelemetry-native tracing for AI agents and LLM applications. `glassflow-rius`
emits [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
traces over OTLP to the managed GlassFlow observability platform (or any
OTLP-compatible backend).

<p>
  <a href="https://docs.glassflow.ai/rius/sdk"><b>Docs</b></a> ·
  <a href="https://docs.glassflow.ai/rius/sdk/reference/python">API reference</a> ·
  <a href="https://console.rius-glassflow.com">Console</a> ·
  <a href="https://github.com/glassflow/rius-sdk-python/blob/main/CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/glassflow/rius-sdk-python/issues">Issues</a>
</p>

[![PyPI](https://img.shields.io/pypi/v/glassflow-rius?style=flat-square)](https://pypi.org/project/glassflow-rius/)
[![Python versions](https://img.shields.io/pypi/pyversions/glassflow-rius?style=flat-square)](https://pypi.org/project/glassflow-rius/)
[![CI](https://img.shields.io/github/actions/workflow/status/glassflow/rius-sdk-python/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/glassflow/rius-sdk-python/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-6b7280?style=flat-square)](https://github.com/glassflow/rius-sdk-python/blob/main/LICENSE)
[![OpenTelemetry GenAI](https://img.shields.io/badge/OpenTelemetry-GenAI-425cc7?style=flat-square&logo=opentelemetry&logoColor=white)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
[![Downloads](https://img.shields.io/pypi/dm/glassflow-rius?style=flat-square)](https://pypistats.org/packages/glassflow-rius)

## Install

```bash
pip install glassflow-rius
```

## Quickstart

```python
import rius
from rius import observe, start_as_current_generation, start_as_current_span
from rius.semconv import SpanKind

rius.init(
    api_key="glassflow_...",          # or set RIUS_API_KEY
    service_name="my-agent",          # or set RIUS_SERVICE_NAME
)

# 1. Decorator — trace a whole function
@observe
def handle(query: str) -> str: ...

# 2. Context manager — trace a block
with start_as_current_span("retrieve", kind=SpanKind.RETRIEVER) as obs:
    obs.set_output(docs)

# 3. LLM generations — gen_ai-native
with start_as_current_generation("chat", model="gpt-4o", input=messages) as gen:
    gen.set_output(reply)
    gen.set_usage(input_tokens=42, output_tokens=17)
```

Span names are optional. Left out, a span is named the way the OpenTelemetry
GenAI conventions say it should be: the operation followed by what it acted on,
such as `chat gpt-4o`, `embeddings text-embedding-3-small`, `execute_tool get_weather`,
`invoke_agent planner` or `retrieval product-kb`. When that identifier is unknown,
the name falls back to the bare operation (`chat`). `@observe` keeps the function's
qualified name for a plain `CHAIN` step, which has no operation; an unnamed
`start_span()` of that kind is called `chain`. An explicit name always wins.

Each surface has a **manual** variant for lifetimes a `with` block can't express
(streaming, callbacks): `start_span(...)` / `start_generation(...)` return a handle
you `.update()` and must `.end()` yourself. A manual handle does not auto-record
exceptions, so report a failure with `.record_exception(exc)`, which records the
exception event, sets the ERROR status and sets `error.type` together:

```python
gen = start_generation("chat", model="gpt-4o", input=messages)
try:
    reply = call_the_model(messages)
except Exception as exc:
    gen.record_exception(exc)
    raise
finally:
    gen.end()
```

Configuration is resolved from explicit arguments first, then environment
variables:

| Argument       | Environment variable     | Default                        | Description                                                          |
| -------------- | ------------------------ | ------------------------------ | -------------------------------------------------------------------- |
| `endpoint`     | `RIUS_ENDPOINT`     | `https://ingest.eu.console.rius-glassflow.com` | Base OTLP endpoint. Traces are sent to `<endpoint>/v1/traces`.       |
| `api_key`      | `RIUS_API_KEY`      | none                           | Injected as an `Authorization: Bearer <key>` header on every export. |
| `service_name` | `RIUS_SERVICE_NAME` | `unknown_service`              | Sets the OpenTelemetry `service.name` resource attribute.            |
| `disabled`     | `RIUS_DISABLED`     | `false`                        | Kill switch. When true, spans are created but never exported.        |
| `sample_rate`  | `RIUS_SAMPLE_RATE`  | `1.0`                          | Head sampling ratio `0.0`–`1.0` (whole-trace; children follow root). |
| `capture_content` | `RIUS_CAPTURE_CONTENT` | `true`                    | When false, prompt/response content is stripped at export (metadata still sent). |
| `bridge_foreign_provider` | `RIUS_BRIDGE_FOREIGN_PROVIDER` | `false`        | Also export the spans of another OpenTelemetry SDK that owns the global provider (see below). |
| `agent_name`   | `RIUS_AGENT_NAME`        | `service_name`                 | The agent this process is. Also the `gen_ai.main_agent.id` when `main_agent_id` is unset, so renaming it changes that id. |
| `main_agent_id` | `RIUS_MAIN_AGENT_ID`    | a `gen_ai.main_agent.id` in `OTEL_RESOURCE_ATTRIBUTES`, else the agent name | A stable id for the agent (a registry id or an ARN), stamped as `gen_ai.main_agent.id`. Never a per-process value. |
| `main_agent_description` | `RIUS_MAIN_AGENT_DESCRIPTION` | none             | What the agent does, stamped as `gen_ai.main_agent.description`. |

The agent this process *is* is described on the resource under the GenAI
conventions' `gen_ai.main_agent.*`. `gen_ai.main_agent.id` is the entity's
identity: it is `main_agent_id` when you set it (the argument, then
`RIUS_MAIN_AGENT_ID`), then a `gen_ai.main_agent.id` you already put in
`OTEL_RESOURCE_ATTRIBUTES`, and otherwise it falls back to the agent name, so
**renaming the agent changes the id unless `main_agent_id` is set** (a one-time
DEBUG log on the `rius.client` logger says so). `gen_ai.main_agent.name` is the
resolved `agent_name`, and `gen_ai.main_agent.description` is
`main_agent_description` when set; a name or description you set in
`OTEL_RESOURCE_ATTRIBUTES` is kept instead. When there is neither an id nor a
name (the `unknown_service` placeholder), the SDK stamps none of the
`gen_ai.main_agent.*` keys (any you set yourself pass through), because the conventions do not allow the entity without an
identifier. The deprecated `rius.main_agent.name`, `.id` and `.description`
keep their old behaviour through their deprecation window
(`rius.main_agent.id` only ever carries a configured `main_agent_id`), and
`rius.main_agent.version` (`main_agent_version`, `RIUS_MAIN_AGENT_VERSION`),
which has no upstream twin, carries the optional agent version.
`gen_ai.agent.name` is still stamped alongside them, unchanged: the keys are
additive so that no deployment loses its agent identity between an SDK upgrade
and a backend one.

`mask` is a code-only option (no env var): pass a callable to `init(mask=...)` and
it is applied to every content attribute value at export, across our spans and any
bundled third-party instrumentation. A mask that accepts a `key` keyword also
receives the attribute key, for per-attribute decisions.

Both controls cover the whole span, not just its attributes: event and link
attributes, the `exception.message` / `exception.stacktrace` of a recorded
exception, and the ERROR status description, since provider errors routinely
echo the rejected request back. With `capture_content=False` those are
stripped and the status code and `exception.type` are kept, so failures stay
visible; with a `mask`, the mask runs over them (the status description is
offered under `key="status.description"`).

```python
rius.init(mask=lambda value: "[REDACTED]")   # redact all captured content
rius.init(mask=lambda value, *, key: hash_pii(value) if "input" in key else value)
rius.init(capture_content=False)             # drop content entirely, keep metadata
```

### Running next to another OpenTelemetry SDK

OpenTelemetry has one global tracer provider per process. If another SDK
(Langfuse v3, for one) claims it before `rius.init()`, your own job and tool
spans go to that SDK only, while the LLM spans inside them still reach Rius, so
Rius receives traces whose parents it never gets. rius detects this: those LLM
spans carry `rius.parent.foreign=true`, the resource records
`rius.sdk.global_provider=foreign:<class>`, and the first one logs a warning.

To send the other SDK's spans to Rius as well, opt in with
`init(bridge_foreign_provider=True)` or `RIUS_BRIDGE_FOREIGN_PROVIDER=true`.
rius then exports that SDK's spans too, subject to rius's `sample_rate`,
`capture_content` and `mask`, and it never modifies what the other SDK
exports: the other vendor receives exactly what it would without rius.

## Auto-instrumentation

The SDK bundles existing OTel instrumentors (OpenInference) as optional extras,
so a single install captures your LLM provider and framework calls. Install the
extras you need and `init()` enables whatever it finds; the instrumentation
spans nest under your `@observe` / `start_as_current_span` traces automatically.

```bash
pip install "glassflow-rius[openai]"        # one provider
pip install "glassflow-rius[instruments]"   # everything supported
```

An extra installs the instrumentation **for** a library, never the library
itself, so the SDK never pins or upgrades the versions your code runs against:
`glassflow-rius[anthropic]` expects `anthropic` to be your own dependency,
which it already is in any project that calls Anthropic. When an
instrumentation's library is missing or too old, `init()` logs a warning naming
the version it needs and leaves that one integration off. For example,
`openinference-instrumentation-anthropic` 3.x needs `anthropic>=1.8.0`; with
an older `anthropic`, upgrade it or keep the instrumentation on 2.x.

```python
rius.init()                          # auto-enables installed instrumentors
rius.init(instruments=["openai"])    # restrict to specific ones
rius.init(instruments=[])            # disable auto-instrumentation
```

Supported instruments: `openai`, `anthropic`, `langchain`, `llama-index`,
`litellm`, and `mcp`. Content captured by instrumentors is covered by the same
`mask` / `capture_content` controls as our own spans.

The `mcp` instrument is built in and has **no extra**: there is no third-party
instrumentation to install, and an extra that installed `mcp` for you would
break the rule above. If the [`mcp`](https://pypi.org/project/mcp/)
package is present, every `ClientSession.call_tool()` your agent makes becomes
a first-class TOOL span (`execute_tool <name>`) with `gen_ai.tool.name`, the
arguments and result as `input.value`/`output.value`, latency, and error status
(including tools that return `isError` results).

Instrumentors patch libraries process-wide, so a scoped client
(`init(set_global=False)`) only enables them when `instruments=[...]` is passed
explicitly. Calling `init()` again while a client is active logs a warning and
returns the existing client unchanged; call `client.shutdown()` first to
reconfigure.

### Span attribute limit

OpenTelemetry caps a span at **128 attributes** by default and, once a span is
full, silently evicts its OLDEST keys. The OpenInference instrumentors write
one attribute per message field and per tool field, so an agent loop with ten
tools passes 128 within about six (Anthropic) to nine (OpenAI) turns, and what
goes first is what they wrote first: the model and request parameters, the
tool definitions, and the system prompt.

`init()` therefore gives its tracer provider a limit of **4096** attributes per
span. The value-length limit is left alone (the SDK caps its own JSON
attributes). If you set `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT` or
`OTEL_ATTRIBUTE_COUNT_LIMIT` to a non-negative integer, your value wins (a blank value
is ignored, and an invalid one is ignored with a warning) and OpenTelemetry resolves the
limits as usual. A span that still hits the limit reports how many attributes
it dropped (`dropped_attributes_count` on the wire), so a loss is visible
rather than silent.

The limit applies to the provider `init()` builds. `init()` does not accept a
provider of your own; if you run OpenInference instrumentors **without** this
SDK, on your own `TracerProvider`, raise the limit yourself, either with
`OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` in the environment or in code:

```python
from opentelemetry.sdk.trace import SpanLimits, TracerProvider

provider = TracerProvider(span_limits=SpanLimits(max_span_attributes=4096))
```

## Reliability

Export is designed to never block or crash your application:

- **Async batched export.** Spans are queued in-process and exported in batches
  from a background thread (`BatchSpanProcessor`). Span creation stays fast even
  when the backend is slow or unreachable.
- **Retries.** Transient failures (connection errors, 429/5xx) are retried with
  exponential backoff and jitter, bounded by the export timeout.
- **Graceful degradation.** If the backend stays down, spans are dropped and an
  error is logged. Exceptions never propagate into application code. A failing
  `mask` callable drops only the affected attribute value (fail closed), never
  the batch.
- **Flush on shutdown.** Pending spans are flushed automatically at interpreter
  exit. Call `client.flush()` to force an export, or `client.shutdown()` to
  drain and stop.

Batching and backpressure are tunable via the standard OpenTelemetry env vars:
`OTEL_BSP_MAX_QUEUE_SIZE` (default 2048; spans beyond this are dropped),
`OTEL_BSP_SCHEDULE_DELAY` (default 5000 ms), `OTEL_BSP_MAX_EXPORT_BATCH_SIZE`
(default 512), and `OTEL_BSP_EXPORT_TIMEOUT` (default 30000 ms).

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

See [CONTRIBUTING.md](https://github.com/glassflow/rius-sdk-python/blob/main/CONTRIBUTING.md)
for the full workflow, and [SECURITY.md](https://github.com/glassflow/rius-sdk-python/blob/main/SECURITY.md) to report a vulnerability.

## Releasing

Releases are automated with [release-please](https://github.com/googleapis/release-please)
and published to PyPI via Trusted Publishing.

1. Merge changes to `main` using [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:` → minor, `fix:` → patch). Mark a breaking change with `!` in the PR
   title (`feat!:` → major); PRs are squash-merged with a blank body, so a
   `BREAKING CHANGE:` footer is lost.
2. release-please keeps a **Release PR** open that bumps `__version__` and updates
   `CHANGELOG.md`. Merge it when you want to cut a release.
3. Merging tags `vX.Y.Z`, creates a GitHub Release, and publishes to PyPI automatically.

Non-conventional commits are ignored for versioning.

## License

MIT

