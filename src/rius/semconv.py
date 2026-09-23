"""Semantic conventions for the GlassFlow SDK.

Centralizes the OpenTelemetry instrumentation-scope name, the span-kind taxonomy,
and span attribute keys. Span kinds use the OpenInference `openinference.span.kind`
values (understood across the ecosystem); LLM specifics use OTel GenAI `gen_ai.*`.
"""

from __future__ import annotations

from enum import Enum

from opentelemetry.trace import Span
from opentelemetry.trace import SpanKind as OtelSpanKind

# Instrumentation scope name (stamped on every span as otel.scope.name).
# "rius" since the vendor keys were normalized under the product name; it was
# "glassflow" before. Wire-visible, but nothing in the backend keys on it: the
# sink stores the scope name verbatim and no reader filters on it, so the two
# values coexist in stored data without any special handling.
TRACER_NAME = "rius"

# --- Resource attribute keys ---
# OTel standard identity of one process lifetime (one uuid per client, minted
# at init). The heartbeat payload's instance_id carries the SAME value, which
# is what lets the backend join heartbeats to traces and count replicas.
SERVICE_INSTANCE_ID = "service.instance.id"
# The agent this process IS, stamped once on the resource so every span it
# emits groups under the same name its heartbeats do. Without it the backend
# falls through to service.name per span, and a process whose agent name
# differs from its service name has its agents view and its trace list
# disagreeing about what it is called.
GEN_AI_AGENT_NAME = "gen_ai.agent.name"

# The identifier of a HOSTED agent resource, not of an agent in general: the
# conventions give an AWS Bedrock agent ARN and a GCP Agent Registry id as the
# examples, and say it is NOT RECOMMENDED to record in-memory agent instance
# ids here, because those are transient. So an in-process agent leaves it
# unset and there is no configured default to fall back to — a name is the
# only identity such an agent has. Adopted rather than deferred because the
# callers who do run hosted agents have nowhere else to put the id.
GEN_AI_AGENT_ID = "gen_ai.agent.id"

# --- Attribute keys ---
# OpenInference
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
# The session grouping key (see session.py). OpenInference's spelling, and the
# one the sink reads first; gen_ai.conversation.id is deliberately not emitted
# alongside it, one name for one fact.
SESSION_ID = "session.id"
# The end-user identity (see user.py). OpenInference's spelling, also what
# Langfuse reads and what the OTel registry lists; the sink additionally
# accepts OTel's enduser.id / enduser.pseudo.id from third-party instrumentors,
# but this SDK emits one name for one fact.
USER_ID = "user.id"

# Process-local routing marker for multi-workspace export (see workspace.py).
# Stamped at span start so pending snapshots route too, and ALWAYS stripped by
# the routing exporter before spans leave the process: the destination's API
# key is what tells the backend which workspace a span belongs to.
WORKSPACE_ROUTE = "rius.workspace"

# OTel GenAI (subset we emit)
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_REASONING_LEVEL = "gen_ai.request.reasoning.level"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
# Streaming, per the GenAI inference-span conventions: gen_ai.request.stream
# (boolean, Conditionally Required when streaming) and
# gen_ai.response.time_to_first_chunk (double, seconds, "measured from request
# issuance", Recommended for streaming requests). Both are set by
# record_first_token: a first chunk arriving is what proves the request
# streamed, and the SDK has no earlier hook for it.
GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
# semconv-genai renamed cache_creation -> cache_write (PR #440) before our
# cache fields first shipped; SDKs 0.12.x emitted the pre-rename name.
GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
# The request's tool/function definitions, serialized verbatim (provider
# shapes differ; the backend reads names and sizes from either). Content,
# not identity — listed in CONTENT_ATTRIBUTES below.
GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
# The index, collection or knowledge base a retrieval ran against. Identity,
# set at span creation: it is what a retriever span is named after and the
# only thing that distinguishes two otherwise identical searches.
GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"
# How many documents the retriever was ASKED for, the spec's own framing
# ("also known as k, limit, or max_num_results"). Identity: it is a property
# of the request, known before the search runs, so it rides pending snapshots.
GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"
# What the retriever RETURNED, as the conventions define it: a JSON array of
# objects each with an optional `id` and an optional `score`. Identifiers and
# relevance only, never document text, which is why the spec does not mark it
# sensitive and why it is NOT on the content allowlist. It is metadata: known
# only once the search has run, so it never reaches a pending snapshot, and
# it survives capture_content=False because a document id is not content.
GEN_AI_RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"
# OTel MCP semantic conventions (semantic-conventions-genai, Development
# stability). mcp.method.name is the REQUIRED attribute of an MCP client span
# and the marker everything downstream keys on: a local TOOL span has the same
# kind, name and I/O shape, so this is what tells the two apart.
MCP_METHOD_NAME = "mcp.method.name"
MCP_METHOD_TOOLS_CALL = "tools/call"
# The version the initialize handshake negotiated — not the one we asked for.
MCP_PROTOCOL_VERSION = "mcp.protocol.version"
# OTel general error.type: on an MCP tools/call it is "tool_error" when the
# result carries isError (the tool ran and reported failure), else the
# exception class when the call itself raised.
ERROR_TYPE = "error.type"
ERROR_TYPE_TOOL_ERROR = "tool_error"
# MCP spec 2026-07-28: a tools/call round can end with an interim
# "input_required" result (MRTR) instead of a final one. Set ONLY on interim
# rounds. NOT an OTel semconv attribute, unlike the two above: the key
# borrows the mcp SDK's own `mcp.*` spelling for its result-type field, and
# no convention defines it. Kept as-is because the backend reads it.
MCP_RESULT_TYPE = "mcp.result_type"
GEN_AI_REQUEST_PREFIX = "gen_ai.request."
# Per-part byte sizes of a generation's context, as compact JSON with
# readable keys: tool_definitions (name, bytes), input_messages and
# output_messages (literal role, parts typed text / tool_call /
# tool_call_response with bytes and tool name), cache_marker, and a folded
# summary of older input; see _context_sizes.py for the full shape.
# A Rius vendor attribute, NOT a convention:
# the backend reads it to attribute gen_ai.usage.input_tokens across the
# context, and no convention describes context composition. Deliberately
# absent from CONTENT_ATTRIBUTES (it is sizes, not content, and must survive
# masking and capture_content=False — that is its whole point when the
# messages are truncated or stripped) and from PENDING_IDENTITY_ATTRIBUTES
# (content is unknown at span start, so a pending snapshot never has it).
RIUS_CONTEXT_SIZES = "rius.context.sizes"

# --- Span event names ---
# First streamed token/chunk arrived: the exact timestamp the backend reads to
# derive TTFT (event time minus span start). The GenAI conventions express the
# same signal as the derived span attribute
# gen_ai.response.time_to_first_chunk (see above), which record_first_token
# emits alongside this event; the event stays because it is what the backend
# keys on, and an absolute timestamp is not recoverable from the duration
# once the span is stored. The name follows the gen_ai.* style (precedent
# Langfuse's completion_start_time); no convention defines an event for this.
GEN_AI_FIRST_TOKEN_EVENT = "gen_ai.first_token"

# --- Pending (partial) spans ---
# Marks the content-free snapshot exported at span START; the backend maps it
# to Finished=0 and the real span replaces it at end. This key knowingly bends
# the convention-native rule (no vendor namespace): OpenTelemetry has NO
# pending-span mechanism to align with (spec #3732/#4646, semconv #2133, all
# open, none planned), and the only shipping precedent (Logfire's
# logfire.span_type) is equally vendor-namespaced. Spelled glassflow.span.pending
# before the vendor keys moved under rius.*; the backend reads both spellings
# for as long as pre-rename SDK versions are in the field.
RIUS_SPAN_PENDING = "rius.span.pending"

# Attributes allowed to ride a pending snapshot: identity/taxonomy known at
# span start. An ALLOWLIST on purpose: content exclusion must hold for
# third-party instrumentors' attribute families too, and a blocklist would
# have to enumerate all of them.
PENDING_IDENTITY_ATTRIBUTES = frozenset(
    {
        OPENINFERENCE_SPAN_KIND,
        GEN_AI_OPERATION_NAME,
        GEN_AI_PROVIDER_NAME,
        GEN_AI_TOOL_NAME,
        # Which index a still-running retrieval is searching is identity, and
        # the live view has nothing else to tell two searches apart by.
        GEN_AI_DATA_SOURCE_ID,
        # How many documents were asked for is a property of the request, so
        # it is known before the search returns and belongs on the snapshot.
        # Its counterpart, gen_ai.retrieval.documents, deliberately is not:
        # what came back cannot be known while the span is still open.
        GEN_AI_RETRIEVAL_TOP_K,
        # Which agent a span invokes is chosen by the caller before the work
        # starts, so a still-running agent is attributable in the live view.
        # Distinct from the resource key of the same name, which says which
        # PROCESS is running; this says which agent that process invoked here.
        GEN_AI_AGENT_NAME,
        GEN_AI_AGENT_ID,
        # Protocol identity, not content: a still-running MCP call must be
        # distinguishable from a local tool in the live view — the one place
        # setting the marker at creation pays off.
        MCP_METHOD_NAME,
        MCP_PROTOCOL_VERSION,
        # Identity, not content: a pending span must be groupable into its
        # session while still running, that is the live view's whole point.
        SESSION_ID,
        # Same reason: a crashed run's partial spans must still be
        # attributable to the user they served.
        USER_ID,
        # Routing, not content: a crashed run's snapshot must land in the same
        # workspace its final span would have. Stripped at export either way.
        WORKSPACE_ROUTE,
    }
)
# gen_ai.request.* (model, temperature, ...) is identity, not content.
PENDING_IDENTITY_PREFIXES = (GEN_AI_REQUEST_PREFIX,)

# Attribute keys carrying user content, masked/stripped at export (see masking.py).
CONTENT_ATTRIBUTES = frozenset(
    {
        INPUT_VALUE,
        OUTPUT_VALUE,
        GEN_AI_INPUT_MESSAGES,
        GEN_AI_OUTPUT_MESSAGES,
        # Sensitive per the GenAI conventions (semconv-genai#431): tool
        # definitions routinely embed proprietary prompt engineering, and
        # sometimes credentials or internal URLs in parameter defaults.
        # gen_ai.tool.name stays: it is identity, not content.
        "gen_ai.tool.description",
        GEN_AI_TOOL_DEFINITIONS,
        # GenAI semconv keys emitted by OTel-native instrumentations (the
        # Vercel AI SDK's @ai-sdk/otel among them, pinned 2026-09-14): the
        # system prompt and each tool call's input and output are content in
        # the same sense messages are. gen_ai.tool.call.id stays: identity.
        "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        # OpenInference TOOL-kind spans carry the definition under these two
        # bare keys, sensitive for the reason gen_ai.tool.description is.
        "tool.description",
        "tool.parameters",
        # common third-party content keys (bundled instrumentation)
        "gen_ai.prompt",
        "gen_ai.completion",
        "llm.input_messages",
        "llm.output_messages",
        # bare (unflattened) forms of the prefix-covered families below:
        # instrumentations vary on whether they flatten these into indexed
        # keys, and a prefix match never covers its own bare key
        "llm.prompts",
        "llm.prompt_template",
        "mlflow.spanInputs",
        "mlflow.spanOutputs",
        # OpenLLMetry workflow/task spans carry full I/O here
        "traceloop.entity.input",
        "traceloop.entity.output",
        # Every bundled OpenInference instrumentor emits tool definitions as
        # llm.tools.{i}.tool.json_schema (pinned empirically 2026-09-14);
        # covered by prefix below, bare key listed per the bare-key rule.
        "llm.tools",
        # The Vercel AI SDK's own telemetry keys survive the OpenInference
        # transform untouched, and they carry the full prompt (messages AND
        # tool definitions), response content, and tool-call I/O. Names, ids
        # and models (ai.toolCall.name, ai.response.model, ...) are identity
        # and stay.
        "ai.prompt",
        "ai.response.text",
        "ai.response.object",
        "ai.toolCall.args",
        "ai.toolCall.result",
        # Same family, keys enumerated from the ai v7 telemetry surface
        # (2026-09-14): model reasoning, tool calls in the response, response
        # files, embedding inputs, rerank documents, and the generateObject
        # schema (a tool definition by another name).
        "ai.response.reasoning",
        "ai.response.toolCalls",
        "ai.response.files",
        "ai.value",
        "ai.values",
        "ai.documents",
        "ai.schema",
        "ai.schema.description",
    }
)

# The request-parameters bag OpenInference instrumentors emit. Not wholly
# content — sampling parameters are identity — but the litellm and langchain
# instrumentations embed the request's tools/functions arrays inside it, so
# masking redacts those members and keeps the rest (see masking.py).
LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
# JSON members of LLM_INVOCATION_PARAMETERS that carry tool definitions.
INVOCATION_PARAMETERS_CONTENT_MEMBERS = ("tools", "functions")

# OpenInference/OpenLLMetry instrumentors flatten message content into indexed
# keys (e.g. `llm.input_messages.0.message.content`), matched by prefix.
CONTENT_ATTRIBUTE_PREFIXES = (
    "llm.input_messages.",
    "llm.output_messages.",
    "gen_ai.prompt.",
    "gen_ai.completion.",
    "llm.prompts.",
    "llm.prompt_template.",
    "llm.tools.",
    "ai.prompt.",
)

# Indexed families where only the content leaf is sensitive (siblings like
# `.document.id` / `.document.score` are metadata), matched by suffix.
CONTENT_ATTRIBUTE_SUFFIXES = (
    ".document.content",
    ".embedding.text",
)


class SpanKind(str, Enum):
    """Observation kind. Values are OpenInference ``openinference.span.kind`` values.

    - ``AGENT``: an agent invocation or run
    - ``LLM``: a model call (generations use this)
    - ``TOOL``: a tool execution
    - ``RETRIEVER``: a retrieval / search step
    - ``EMBEDDING``: an embedding computation
    - ``CHAIN``: a generic processing step (the default)
    """

    AGENT = "AGENT"
    LLM = "LLM"
    TOOL = "TOOL"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    CHAIN = "CHAIN"


# SpanKind -> OTel GenAI gen_ai.operation.name, where a canonical operation exists.
# CHAIN is deliberately absent: the conventions define no operation for a
# generic workflow step, and inventing one would put a non-spec value in a
# spec-defined enum. Revisit only if the upstream invoke_node proposal lands.
_OPERATION_BY_KIND: dict[SpanKind, str] = {
    SpanKind.LLM: "chat",
    SpanKind.TOOL: "execute_tool",
    SpanKind.EMBEDDING: "embeddings",
    SpanKind.AGENT: "invoke_agent",
    SpanKind.RETRIEVER: "retrieval",
}


# SpanKind (our taxonomy ATTRIBUTE) -> the OpenTelemetry SpanKind FIELD, set
# centrally at span creation. The GenAI conventions decide the field per
# operation: inference, embeddings and retrieval call out of the process
# (CLIENT); execute_tool is INTERNAL; invoke_agent is CLIENT only for a hosted
# agent. Hence:
# - LLM / EMBEDDING / RETRIEVER -> CLIENT: a remote model or index is called.
# - TOOL -> INTERNAL: a bare TOOL span is an in-process function, and the
#   execute-tool convention says INTERNAL. The MCP wrapper overrides this to
#   CLIENT at its own call site because a tools/call crosses a process
#   boundary and the MCP client convention says CLIENT.
# - AGENT -> INTERNAL: `@observe(kind=AGENT)` wraps an in-process agent, the
#   convention's INTERNAL case; a hosted agent would need a call-site override.
# - CHAIN -> INTERNAL: a workflow step by definition.
# Nothing in the backend groups on this field (it reads the attributes), so
# the mapping is free to follow the conventions exactly.
_OTEL_KIND_BY_KIND: dict[SpanKind, OtelSpanKind] = {
    SpanKind.LLM: OtelSpanKind.CLIENT,
    SpanKind.EMBEDDING: OtelSpanKind.CLIENT,
    SpanKind.RETRIEVER: OtelSpanKind.CLIENT,
    SpanKind.TOOL: OtelSpanKind.INTERNAL,
    SpanKind.AGENT: OtelSpanKind.INTERNAL,
    SpanKind.CHAIN: OtelSpanKind.INTERNAL,
}


def otel_span_kind(kind: SpanKind) -> OtelSpanKind:
    """The OpenTelemetry ``SpanKind`` field a span of taxonomy ``kind`` should carry."""
    return _OTEL_KIND_BY_KIND[kind]


def kind_attributes(
    kind: SpanKind,
    tool_name: str | None = None,
    *,
    data_source_id: str | None = None,
    top_k: int | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
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
    """
    attributes: dict[str, str | int] = {OPENINFERENCE_SPAN_KIND: kind.value}
    operation = _OPERATION_BY_KIND.get(kind)
    if operation is not None:
        attributes[GEN_AI_OPERATION_NAME] = operation
    if kind is SpanKind.TOOL and tool_name is not None:
        attributes[GEN_AI_TOOL_NAME] = tool_name
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
    return attributes


def set_span_kind(span: Span, kind: SpanKind) -> None:
    """Stamp a span with its OpenInference kind and (if applicable) gen_ai operation."""
    span.set_attribute(OPENINFERENCE_SPAN_KIND, kind.value)
    operation = _OPERATION_BY_KIND.get(kind)
    if operation is not None:
        span.set_attribute(GEN_AI_OPERATION_NAME, operation)
