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
    api_key="glassflow_...",  # or set RIUS_API_KEY
    service_name="my-agent",  # or set RIUS_SERVICE_NAME
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
| `endpoint`     | `RIUS_ENDPOINT`     | `https://ingest.eu.console.rius-glassflow.com` | Base OTLP endpoint. Traces are sent to `<endpoint>/v1/traces`. Use `https://`: an `http://` endpoint sends traces and the API key unencrypted, and logs a warning. |
| `api_key`      | `RIUS_API_KEY`      | none                           | Injected as an `Authorization: Bearer <key>` header on every export. |
| `service_name` | `RIUS_SERVICE_NAME` | `unknown_service`              | Sets the OpenTelemetry `service.name` resource attribute.            |
| `service_version` | `RIUS_SERVICE_VERSION` | none                       | Sets `service.version` on the resource. Unset leaves it off, so `OTEL_RESOURCE_ATTRIBUTES` can supply it. |
| `disabled`     | `RIUS_DISABLED`     | `false`                        | Kill switch. When true, spans are created but never exported.        |
| `sample_rate`  | `RIUS_SAMPLE_RATE`  | `1.0`                          | Head sampling ratio `0.0`–`1.0` (whole-trace; children follow root). |
| `capture_content` | `RIUS_CAPTURE_CONTENT` | `true`                    | When false, prompt/response content is stripped at export (metadata still sent). |
| `bridge_foreign_provider` | `RIUS_BRIDGE_FOREIGN_PROVIDER` | `false`        | Also export the spans of another OpenTelemetry SDK that owns the global provider (see below). |
| `agent_name`   | `RIUS_AGENT_NAME`        | `service_name`                 | The agent this process is. Also the `gen_ai.main_agent.id` when `main_agent_id` is unset, so renaming it changes that id. |
| `main_agent_id` | `RIUS_MAIN_AGENT_ID`    | a `gen_ai.main_agent.id` in `OTEL_RESOURCE_ATTRIBUTES`, else the agent name | A stable id for the agent (a registry id or an ARN), stamped as `gen_ai.main_agent.id`. Never a per-process value. |
| `main_agent_description` | `RIUS_MAIN_AGENT_DESCRIPTION` | none             | What the agent does, stamped as `gen_ai.main_agent.description`. |
| `heartbeat`    | `RIUS_HEARTBEAT`         | `true`                         | Send liveness pings to `<endpoint>/v1/heartbeat` (see [Heartbeat](#heartbeat)). |
| `heartbeat_interval` | `RIUS_HEARTBEAT_INTERVAL` | `15`                   | Seconds between heartbeat pings, clamped to `5`–`300`. |
| `partial_spans` | `RIUS_PARTIAL_SPANS`    | `false`                        | Also export a content-free snapshot of each span when it starts (see [Partial spans](#partial-spans)). |
| `partial_spans_delay` | `RIUS_PARTIAL_SPANS_DELAY` | `0`                  | Seconds a span must stay open before its snapshot is sent, clamped to `0`–`60`. `0` sends it at start. |
| `session_id`   | `RIUS_SESSION_ID`        | none                           | A process-wide `session.id` for every span; a `rius.session()` scope overrides it. |

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
and a backend one. It is omitted on the `unknown_service` placeholder too.

Boolean environment variables accept `1`/`true`/`yes`/`on` and
`0`/`false`/`no`/`off` (case-insensitive). Any other value, including an empty
one, keeps the default and logs a warning on the `rius.config` logger.

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
rius.init(mask=lambda value: "[REDACTED]")  # redact all captured content
rius.init(mask=lambda value, *, key: hash_pii(value) if "input" in key else value)
rius.init(capture_content=False)  # drop content entirely, keep metadata
```

### Sessions, users and workspaces

Three context managers scope every span started inside them, including the
spans of auto-instrumented libraries. Each stamps its value at span start, so
it reaches partial spans too, and each follows async tasks the way
OpenTelemetry context does.

```python
with rius.session(conversation_id):  # session.id on every span of the turn
    handle_turn(message)

with rius.session() as session_id:  # no argument: a fresh UUID for the block
    run_job()

with rius.user(request.user_id):  # user.id on every span of the request
    handle(request)
```

Without a `session()` scope or `session_id`, spans carry no session id and the
backend groups each trace as its own session. `session_id` (or
`RIUS_SESSION_ID`) suits an agent that runs one session per process; a server
handling many sessions should use the scope. Pass `user()` an opaque id rather
than an email: it is stored on every span and shown in the console.

To send different spans to different workspaces (one per customer, for
example), opt in with `init(workspaces={alias: api_key, ...})` and pick the
destination with `rius.workspace(alias)`. Spans outside a scope, or under an
alias that is not registered (with a warning), go to the default workspace of
`api_key`. `rius.register_workspace(alias, api_key)` adds a destination after
`init()`, or rotates the key of an existing one; it needs `workspaces=` at
`init()`, and an empty dict is enough.

```python
rius.init(api_key="glassflow_...", workspaces={})
rius.register_workspace("acme", acme_key)

with rius.workspace("acme"):
    handle(request)  # exported with acme's key
```

A trace must stay in one workspace, so enter the scope at a request boundary,
before the root span starts.

### Heartbeat

`init()` starts a background thread that sends a small liveness ping to
`<endpoint>/v1/heartbeat` right away and then every 15 seconds
(`heartbeat_interval`), so the console shows whether the agent is running even
when it records no traces. `client.shutdown()`, or interpreter exit, sends a
final ping that marks the agent as stopped. The heartbeat needs credentials:
without `api_key` (or an `Authorization` header) it does not start. Turn it off
with `init(heartbeat=False)` or `RIUS_HEARTBEAT=false`.

### Partial spans

A span normally leaves the process when it ends, so a long agent run shows
nothing until it finishes, and a crashed one shows nothing at all. With
`partial_spans=True` (or `RIUS_PARTIAL_SPANS=true`) each sampled span is also
exported once when it starts, as a snapshot that carries its identity and
`rius.span.pending=true` but never its content. The finished span replaces it;
a snapshot that is never replaced records what a crashed agent was doing.
`partial_spans_delay` sends a snapshot only for spans still open after that
many seconds, so short spans cost nothing extra.

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
rius.init()  # auto-enables installed instrumentors
rius.init(instruments=["openai"])  # restrict to specific ones
rius.init(instruments=[])  # disable auto-instrumentation
```

Supported instruments: `openai`, `anthropic`, `langchain`, `llama-index`,
`litellm`, and `mcp`. Content captured by instrumentors is covered by the same
`mask` / `capture_content` controls as our own spans.

From `anthropic` 1.14.0 the Anthropic SDK also starts a span of its own for
each API call. While the `anthropic` instrument covers the call, rius exports
the SDK's chat span as a plain client span (its HTTP and `anthropic.*`
attributes, without the model, usage or message content), so each call counts
once in tokens and cost. Without the instrument, the SDK's span is the call's
LLM span. Two things to expect:

- A top-level `messages.stream()` (outside any `@observe` or span of yours)
  lands in two traces: the SDK starts its span outside the instrumentor's, so
  the plain client span is the root of a trace with no LLM span. Inside a span
  of yours, both are in its trace.
- The client span keeps `gen_ai.conversation.id` when the SDK sets one. If you
  set no `session_id`, the backend can group by that id, so the client span
  may land in a different session from the LLM span.

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

## Compactions

When an agent's history is shrunk, the trace says where. Each compaction is a
`rius.compaction` event (or, for your own, a `compaction` span) carrying what
it did, who did it (`provider` or `client`), and the token and message counts
that are known. The model call that ran on the shrunk context is flagged
`gen_ai.conversation.compacted`.

Reported automatically, from the provider's response:

- **Anthropic** context editing (`clear_tool_uses`, `clear_thinking`) and
  server-side compaction blocks.
- **OpenAI** `compaction` items in a Responses `output`. `truncation: "auto"`
  reports nothing, because the response does not say what was cut. The
  `/responses/compact` call is reported as the compaction it performed, with
  the tokens it read; it is not flagged itself, since it ran on the full
  context, and the model call that follows it is.

For compaction your own code does, call `mark_compaction` once the history has
been replaced:

```python
import rius

rius.mark_compaction(
    "summarize",
    trigger="auto",
    tokens_before=92_000,
    tokens_after=11_000,
    messages_compacted=48,
    messages_kept=6,
    summary=summary_text,
)
```

It records the span under the current span (a root when there is none) and
flags the next model call made in the same context, once, whenever it
comes: the flag does not expire. A mark made in a child task or thread
(`create_task`, `to_thread`, an executor, a graph node) is not seen by the code
that started it; it reaches only copies of the context made after the call.
`kind` is
`"summarize"` (the default), `"clear_tool_results"`, `"clear_thinking"` or
`"truncate"`, and `trigger` is `"auto"` or `"manual"`; anything else raises
`ValueError`. Counts you do not know are left out.

The summary is content: its size in bytes is always sent, and its text follows
`capture_content` and `mask` like a prompt does.

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

