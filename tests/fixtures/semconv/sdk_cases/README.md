# Shared SDK cases

Behaviour both Rius SDKs must share that no recording exercises: what an SDK
writes for a value a caller hands it, what its masking exporter keeps, and how
it reads its environment. Each file is a list of cases with one expected
result. The Python SDK and the TypeScript SDK both vendor this directory
(`scripts/sync-semconv-fixtures.sh` copies it to
`tests/fixtures/semconv/sdk_cases/`) and run every case through their own
code, so the two SDKs agree with each other by agreeing with these files.

| File | What it checks | Through |
| -- | -- | -- |
| `values.json` | `input.value` / `output.value` of a generic span: a string as it is, anything else its compact JSON, numbers and booleans included. | `set_input` / `setInput`, `set_output` / `setOutput` |
| `attributes.json` | A span attribute set by the SDK: primitives and homogeneous arrays of them kept; a mapping, a mixed or nested array, or an array holding a mapping one JSON encoding, byte objects as markers. | `set_attribute` / `setAttribute` |
| `messages.json` | `gen_ai.input.messages`, `gen_ai.output.messages` and `rius.context.sizes`: a lone string or mapping is one message, a missing role is the default, a null role is kept and sized as `user`. | a generation's `set_input` / `set_output` |
| `masking.json` | What reaches the exporter with content capture off, and with a mask that returns a value, returns null or throws: span attributes, `exception` event text and the status description. | `MaskingSpanExporter` |
| `env.json` | `RIUS_*` variables: booleans (blank or unrecognised keeps the default), blank strings (unset), numbers (as Python's `float()` reads them, clamped to range). | `resolve_config()` / `resolveConfig({}, env)` |
| `resource.json` | `OTEL_RESOURCE_ATTRIBUTES` merged under the SDK's own resource keys, and the `unknown_service` placeholder never stamped as an agent name. | `init()` |
| `anthropic_error_type.json` | `error.type` on a span the Anthropic SDK made itself: the exception class it raises for the HTTP status, for the statuses every Anthropic client in both languages maps alike (400, 401, 403, 404, 409, 422, 429, and every 5xx but 503, 504 and 529); anything else kept. | the SDK's span normalization |

Each file holds an `about`, which says what a case's fields mean, and
`cases`, each with a unique `name`. A field an SDK spells differently is named
the Python way (`capture_content`, `service_name`); each SDK's test maps it.

## Optional fields

- `warns` (`env.json` only): whether resolving the case logs a warning.
  `true` means at least one, `false` means none. The Python SDK reads WARNING
  records on the `rius.config` logger, the TypeScript SDK `console.warn` calls.
  A case without `warns` does not check warnings at all, so leave it out only
  where the warning is not part of the contract.

## Tagged values

JSON cannot hold some values a caller can pass. A case writes them as a
mapping with one tagged key, which each SDK's test decodes:

- `{"$bytes": N}`: a byte object of N bytes (Python `bytes(N)`, TypeScript
  `new Uint8Array(N)`).
- `{"$number": "NaN"}`, `{"$number": "Infinity"}`, `{"$number": "-Infinity"}`:
  that number.

## What is left out

A case belongs here only when both SDKs must give the same result. Where the
languages differ by nature it is left out: JavaScript has one number type, so
`1.0` is `"1"` in TypeScript and `"1.0"` in Python; Python writes a boolean a
mask returns for the status description as `True`, TypeScript as `true`.
Behaviour only one SDK has is tested in that SDK alone.

## Changing a case

Change the expected result here first, then make both SDKs pass it. An SDK
that cannot pass a case yet is a disagreement to fix, not a case to weaken.
`tests/test_sdk_cases.py` checks that every file is well formed.
