# Recorded spans

Real spans from the instrumentors Rius normalizes, recorded once and shared by
every normalizer: the Python SDK, the TypeScript SDK and the sink. Each
recording comes with the result the normalizers are expected to produce from
it. Every consumer runs the recordings through its own normalizer and must
produce exactly that result, so the three agree with each other without ever
being compared directly. When one disagrees, the expected result says which
side changes.

## Layout

```
fixtures/
  _recording.py        shared by every Python recorder: OTLP/JSON encoding,
                       deterministic ids and times, file writing
  _replay.py           a stored scenario back into OpenTelemetry ReadableSpans
  <family>/
    record.py          drives the instrumentor against a scripted provider
    manifest.json      pins, one line per scenario, where the expected result
                       came from, when it was last re-recorded
    <family>-<version>.json            the raw recording
    <family>-<version>.expected.json   the expected normalized result
```

A family is one instrumentor (`openinference-openai`), named after the package
and keyed by its version, so a new instrumentor release is a new recording
beside the old one rather than a replacement.

## The recording file

```json
{"recordings": [
  {"scenario": "chat",
   "resource": {"attributes": [...]},
   "spans": [{"scope": {"name": "...", "version": "..."}, "span": {...}}]}
]}
```

Each `span` is the protobuf JSON mapping of
`opentelemetry.proto.trace.v1.Span` (OTLP/JSON), so any language can decode it
with its own OTLP types. Attributes and event attributes are sorted by key. To
make a re-recording byte-identical in the pinned environment:

- **ids** are renumbered: each trace id becomes `…0001`, `…0002` and each span
  id `…0001`, `…0002`, in start order. Parent links are kept.
- **times** are replaced by their rank in the scenario's timeline, one
  millisecond apart from 1700000000000000000. Ordering and nesting are kept,
  and a first-token event still lands after its span's start, so time to first
  chunk is still derivable (it reads as 1 ms per step).
- an `exception` event's `exception.stacktrace` is replaced by
  `<stacktrace elided>`.
- a span's status message is cut where a Python traceback starts
  (`Traceback (most recent call last):`). LangChain appends the formatted
  traceback there, with machine paths in it. The Node recorders cut a status
  message and an `exception.message` at the first stack frame (`\n    at `)
  for the same reason: LangChain JS appends the error's stack to both.
- every UUID in a string value becomes `00000000-0000-0000-0000-<n>`, numbered
  by first appearance in the scenario (frameworks such as LangChain write
  per-run ids into JSON payloads).
- the resource's `service.instance.id` is fixed (`recorder`).

A value the instrumentor *measures* (pydantic-ai writes a time-to-first-chunk
in seconds into an attribute) is made deterministic in the recorder itself, by
pinning the clock it reads, never by editing the recording afterwards. The
manifest says so when a recorder does this.

The rius SDK is never involved in a recording: these are the instrumentor's
raw spans. The control families below are the one exception.

## Controls

`rius-python` and `rius-typescript` record what each SDK itself exports: its
native spans (`observe`, spans, generations, sessions, users) and its MCP
client wrapper, captured at the exporter the SDK was given
(`init(span_exporter=...)`, `init({ spanExporter })`). Those spans are already
normalized, so normalizing them again must change nothing. Their manifests
say `"control": true`, and `check_fixtures` then requires the expected result
to equal the recording exactly. A normalizer change that alters an SDK's own
output fails there, and every consumer's conformance test checks the same
property for its own normalizer.

Both families have the same scenarios, so the two SDKs' output can also be
compared directly. They pin a released SDK version, like every other family
pins its instrumentor.

## The expected result

The same shape, plus the normalizer that produced it:

```json
{"normalizer": {"package": "glassflow-rius", "version": "1.0.0", "commit": "<sha>"},
 "recordings": [...]}
```

Span ids, parents and times are exactly the recording's, so a consumer matches
each expected span to its raw span by id. What it holds is the span after
normalization and before masking: what the normalizer hands to the exporter.

Where it comes from (`expected.source` in the manifest):

- `python-sdk`: families the SDKs cover. Produced by `tools/expect_python.py`
  with the Python SDK (the reference implementation) installed from a named
  commit, then **reviewed by hand** before it is committed.
- `typescript-sdk`: families only the TypeScript SDK covers (the Vercel AI
  SDK). Produced by `tools/expect_typescript.mts`, which replays each scenario
  through a real tracer with the SDK's processor chain (the Vercel `ai.*`
  transform, then the normalizer), the recorded ids handed out by an id
  generator. A manifest entry's `replay.aggregating_producer` makes it start
  spans through the SDK's `aggregatingTracer`, as the SDK does live for
  `@ai-sdk/otel`. It imports the SDK's source from a checkout, so it runs under
  `tsx`:
  `npx tsx tools/expect_typescript.mts --sdk <checkout> --sdk-commit <sha> <recording.json>`
- `sink`: families only the sink normalizes (OpenLLMetry, Laminar). No SDK
  output exists to agree with, so the expected result is the **rows** the sink
  stores, in `<recording>.expected-row.json`: for each scenario, the
  ClickHouse rows (after the map and enrich stages, cost included) in span id
  order, and the argus-core commit that produced them. Written by argus-core's
  `go run ./internal/conformance/cmd/expect-rows -commit <sha> <recording.json>`
  (run from `apps/sink`), with the conformance price table, and reviewed by
  hand. argus-core's conformance test recomputes them.

JavaScript recordings of families the Python SDK also covers (OpenInference's
JS instrumentors) take the Python SDK's output as expected, like their Python
counterparts. A JavaScript-specific step the TypeScript SDK takes live (for
example qualifying a recorded exception's generic `Error` type at the moment it
is recorded) cannot run on a replay, so it is not part of any expected result.

## Writing a recorder

`_recording.py` holds everything a recorder shares:

- `ScriptedProvider(httpx)`: the scripted provider. `answer(url, *responses)`
  queues responses, each answering exactly one request in order, and a
  scenario fails if any is left unused. `transport()` gives a mock transport
  for clients that accept one; `intercept_all()` answers every request in the
  process, for libraries that build their HTTP clients internally (passing
  `client=` would end up serialized in the span). Pass the httpx module the
  client uses: openai >= 3 and anthropic >= 1 use the `httpx2` fork.
- `sse(events)`: a server-sent-events body for streaming responses.
- `recorder_resource()`: the provider's resource, with a fixed instance id.
- `record_all(scenarios, exporter, script, packages, argv)`: runs every
  scenario and writes the recording and the environment.
- `spans_from_otlp(captured)`: for a recorder that captures OTLP protobuf off
  the wire instead of holding an in-memory exporter (Laminar's runs the
  instrumented app as a child process), turns the captured (resource, scope,
  span) triples into spans `scenario_json` can make canonical.

Enable only the instrumentor under test. A combination (LangChain with the
openai instrumentor, LiteLLM with it) produces extra spans for the same call
and is recorded as its own family.

### Node recorders

JavaScript instrumentors are recorded with `record.mjs` and
`_recording.mjs`, which writes exactly the same shape as `_recording.py` (the
round-trip test decodes Node recordings with the Python replay to prove it).
Each Node family has its own `package.json` with exact pins and a committed
`package-lock.json`; run `npm ci` in the family directory before recording.
`node_modules` is never committed. The helpers mirror the Python ones
(`ScriptedProvider` with a `fetch`, `sse`, `json`, `recordAll`), plus:

- `deterministicClock()`: replaces `Date.now()` and `performance.now()` with
  clocks that advance one tick per reading. Needed where an instrumentor mixes
  clocks (LangChain stamps run records with `Date.now()` at millisecond
  resolution while OpenTelemetry reads `performance.now()`), so that which of
  two close moments came first cannot change between runs.
- Register an `AsyncLocalStorageContextManager` when the instrumented library
  nests spans through the active context (the AI SDK v5 does), as every Node
  app's provider does.

`_vercel.mjs` holds the Vercel AI SDK scenarios both Vercel families share;
each family passes in its own `ai` and `@ai-sdk/openai` modules.

### Timeline ties

Where two moments share a timestamp, ties are broken by structure, the same
way in both encoders: a start before an event before an end; among starts the
parent first; among ends the child first. A span's start therefore always
precedes its own end.

## Recording a family

1. Make a virtualenv holding exactly the manifest's pins (Python version
   included).
2. `python record.py <family>-<version>.json <environment.json>`. The second
   file gets the Python version and package pins for the manifest.
3. Run it twice and compare: the two outputs must be byte-identical.
4. Produce the expected result, in a virtualenv with the Python SDK installed
   from the commit you name:
   `python tools/expect_python.py --sdk-commit <sha> fixtures/<family>/<file>.json`
5. Review the expected result against the recording, scenario by scenario:
   `python tools/review_expected.py --unchanged fixtures/<family>/<file>.json`
   lists what the normalizer removed, added, changed and left alone. Write
   anything surprising into the pull request.
6. `python tools/check_fixtures.py` must be clean.
