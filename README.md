# Rius Python SDK

OpenTelemetry-native tracing for AI agents and LLM applications. `glassflow-rius`
emits [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
traces over OTLP to the managed GlassFlow observability platform (or any
OTLP-compatible backend).

> Status: alpha. APIs may change.

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

Each surface has a **manual** variant for lifetimes a `with` block can't express
(streaming, callbacks): `start_span(...)` / `start_generation(...)` return a handle
you `.update()` and must `.end()` yourself.

Configuration is resolved from explicit arguments first, then environment
variables:

| Argument       | Environment variable     | Default                        | Description                                                          |
| -------------- | ------------------------ | ------------------------------ | -------------------------------------------------------------------- |
| `endpoint`     | `RIUS_ENDPOINT`     | `https://ingest.eu.console.rius-glassflow.com` | Base OTLP endpoint. Traces are sent to `<endpoint>/v1/traces`.       |
| `api_key`      | `RIUS_API_KEY`      | —                              | Injected as an `Authorization: Bearer <key>` header on every export. |
| `service_name` | `RIUS_SERVICE_NAME` | `unknown_service`              | Sets the OpenTelemetry `service.name` resource attribute.            |
| `disabled`     | `RIUS_DISABLED`     | `false`                        | Kill switch. When true, spans are created but never exported.        |
| `sample_rate`  | `RIUS_SAMPLE_RATE`  | `1.0`                          | Head sampling ratio `0.0`–`1.0` (whole-trace; children follow root). |
| `capture_content` | `RIUS_CAPTURE_CONTENT` | `true`                    | When false, prompt/response content is stripped at export (metadata still sent). |

Every variable also accepts its former `GLASSFLOW_*` spelling (for example
`GLASSFLOW_API_KEY`). Those names are deprecated: they keep working for now,
log a warning naming the `RIUS_*` replacement, and will be removed in a
future release. When both spellings are set, `RIUS_*` wins.

`mask` is a code-only option (no env var): pass a callable to `init(mask=...)` and
it is applied to every content attribute value at export, across our spans and any
bundled third-party instrumentation. A mask that accepts a `key` keyword also
receives the attribute key, for per-attribute decisions.

```python
rius.init(mask=lambda value: "[REDACTED]")   # redact all captured content
rius.init(mask=lambda value, *, key: hash_pii(value) if "input" in key else value)
rius.init(capture_content=False)             # drop content entirely, keep metadata
```

## Auto-instrumentation

The SDK bundles existing OTel instrumentors (OpenInference) as optional extras,
so a single install captures your LLM provider and framework calls. Install the
extras you need and `init()` enables whatever it finds; the instrumentation
spans nest under your `@observe` / `start_as_current_span` traces automatically.

```bash
pip install "glassflow-rius[openai]"        # one provider
pip install "glassflow-rius[instruments]"   # everything supported
```

```python
rius.init()                          # auto-enables installed instrumentors
rius.init(instruments=["openai"])    # restrict to specific ones
rius.init(instruments=[])            # disable auto-instrumentation
```

Supported instruments: `openai`, `anthropic`, `langchain`, `llama-index`,
`litellm`, and `mcp`. Content captured by instrumentors is covered by the same
`mask` / `capture_content` controls as our own spans.

The `mcp` instrument is built in: if the [`mcp`](https://pypi.org/project/mcp/)
package is installed, every `ClientSession.call_tool()` your agent makes becomes
a first-class TOOL span (`execute_tool <name>`) with `gen_ai.tool.name`, the
arguments and result as `input.value`/`output.value`, latency, and error status
(including tools that return `isError` results).

Instrumentors patch libraries process-wide, so a scoped client
(`init(set_global=False)`) only enables them when `instruments=[...]` is passed
explicitly. Calling `init()` again while a client is active logs a warning and
returns the existing client unchanged; call `client.shutdown()` first to
reconfigure.

## Reliability

Export is designed to never block or crash your application:

- **Async batched export.** Spans are queued in-process and exported in batches
  from a background thread (`BatchSpanProcessor`). Span creation stays fast even
  when the backend is slow or unreachable.
- **Retries.** Transient failures (connection errors, 429/5xx) are retried with
  exponential backoff and jitter, bounded by the export timeout.
- **Graceful degradation.** If the backend stays down, spans are dropped and an
  error is logged — exceptions never propagate into application code. A failing
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

## Releasing

Releases are automated with [release-please](https://github.com/googleapis/release-please)
and published to PyPI via Trusted Publishing.

1. Merge changes to `main` using [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:` → minor, `fix:` → patch, `feat!:`/`BREAKING CHANGE` → major).
2. release-please keeps a **Release PR** open that bumps `__version__` and updates
   `CHANGELOG.md`. Merge it when you want to cut a release.
3. Merging tags `vX.Y.Z`, creates a GitHub Release, and publishes to PyPI automatically.

Non-conventional commits are ignored for versioning.

## License

MIT

