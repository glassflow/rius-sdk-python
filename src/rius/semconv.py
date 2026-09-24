"""Semantic conventions for the GlassFlow SDK.

Centralizes the OpenTelemetry instrumentation-scope name, the span-kind taxonomy,
and span attribute keys. Span kinds use the OpenInference `openinference.span.kind`
values (understood across the ecosystem); LLM specifics use OTel GenAI `gen_ai.*`.
"""

from __future__ import annotations

from collections.abc import Mapping
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
# OTel standard version of the deployed service. Metadata about the process,
# not about any one span. Stamped only when a version is actually known: there
# is deliberately no placeholder default, because a fake one merges every
# deployment into a single bucket — the same mistake `unknown_service` makes
# for service.name, which _agent.py has to suppress for exactly this reason.
SERVICE_VERSION = "service.version"
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

# The version of the agent an AGENT span INVOKED — the same callee
# ``gen_ai.agent.name`` and ``gen_ai.agent.id`` describe on that span, never
# the process's own version and never ``service.version``. Registry type is a
# plain string and the conventions' own examples are "1.0.0" and "2025-05-01",
# so the caller's value is taken verbatim: there is no format to validate.
GEN_AI_AGENT_VERSION = "gen_ai.agent.version"

# --- The MAIN agent: the agent this PROCESS is ---
# Our own namespace, because the question is ours. ``gen_ai.agent.name`` has no
# resource-level meaning in the conventions, and ON A SPAN it means the agent
# being INVOKED — one key answering two different questions, told apart only by
# which of a resource and a span you happen to be reading. These four keys say
# "the process emitting this telemetry IS this agent", once per OTLP batch.
#
# ``gen_ai.agent.name`` stays on the resource alongside them, deliberately and
# indefinitely: a swap would be a flag day, breaking agent identity for every
# deployment running an SDK newer than the sink. Resource attributes ride once
# per batch, not per span, so carrying both costs essentially nothing.
#
# Named so a future rename to ``gen_ai.main_agent.*`` is a prefix substitution,
# should the conventions ever grow the concept.
RIUS_MAIN_AGENT_NAME = "rius.main_agent.name"
# The SAME stability constraint the conventions put on ``gen_ai.agent.id``,
# adopted rather than loosened because we own the namespace: a transient
# in-memory or process-local id here would mint a new "agent" every restart and
# make the key useless for grouping. Taking the constraint now is what keeps
# that future rename mechanical instead of a data-quality problem.
RIUS_MAIN_AGENT_ID = "rius.main_agent.id"
RIUS_MAIN_AGENT_DESCRIPTION = "rius.main_agent.description"
# The version of the agent DEFINITION — its prompt, tools and policy — which is
# deliberately NOT ``service.version``. A service can sit at 2.3.1 while its
# agent definition is at 7, and the two move independently; deriving either
# from the other produces confidently wrong data on every deployment where
# they disagree, which is most of them.
RIUS_MAIN_AGENT_VERSION = "rius.main_agent.version"

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
# The provider's own identifier for the completion ("chatcmpl-123"),
# Recommended on an inference span. Metadata: it exists only once the provider
# has answered, so it is a post-call setter and never reaches a pending
# snapshot. Not content — an opaque id says nothing about what was said — so
# it survives capture_content=False.
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
# The output modality the client ASKED for (text / json / image / speech in
# the registry today), Conditionally Required when the request specifies one.
# Identity: it describes the request, so it is known at span creation and must
# ride pending snapshots. Note the key is NOT under gen_ai.request., so
# PENDING_IDENTITY_PREFIXES does not cover it and it needs its own allowlist
# entry below. Recorded verbatim rather than validated against the enum: the
# conventions are free to extend it, and a value we do not recognise is still
# the caller's truth. Not content, for the reason the response id is not.
GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
# Streaming, per the GenAI inference-span conventions: gen_ai.request.stream
# (boolean, Conditionally Required when streaming) and
# gen_ai.response.time_to_first_chunk (double, seconds, "measured from request
# issuance", Recommended for streaming requests). Both are set by
# record_first_token: a first chunk arriving is what proves the request
# streamed, and the SDK has no earlier hook for it.
GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"
# Sampling parameters. The SDK's own API takes no sampling parameters, so
# these are produced only by normalization, which promotes them out of a
# third-party instrumentor's request bag (`llm.invocation_parameters`). Listed
# here anyway because they go on the wire, and the wire's keys live in one
# place. Names and types are the GenAI registry's
# (opentelemetry.semconv._incubating.attributes.gen_ai_attributes).
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
# "The target number of candidate completions to return" — OpenAI's `n`.
GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
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
# The identifier of the tool call this execution answers — the id the MODEL
# minted on its tool-call message, which is what joins a tool span back to the
# generation that asked for it. Identity, set at span creation, so it needs an
# allowlist entry below. Never derived: only the caller holds the model's
# response. Its sibling gen_ai.tool.call.arguments IS content and is listed in
# CONTENT_ATTRIBUTES; an opaque id is not.
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
# What KIND of tool ran, the conventions' own examples being "function"
# (client-side), "extension" (agent-side) and "datastore". Identity for the
# same reason the call id is: caller-supplied, known before the tool runs.
# A free string, not an enum, in the registry.
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
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
# The OTel exception event, as record_exception() writes it, and its attribute
# naming the exception's class. Read, never emitted by us directly: an
# auto-instrumented failure carries these but no error.type, so normalization
# derives error.type from EXCEPTION_TYPE (see normalization.py).
EXCEPTION_EVENT = "exception"
EXCEPTION_TYPE = "exception.type"
# MCP spec 2026-07-28: a tools/call round can end with an interim
# "input_required" result (MRTR) instead of a final one. Set ONLY on interim
# rounds. NOT an OTel semconv attribute, unlike the two above: the key
# borrows the mcp SDK's own `mcp.*` spelling for its result-type field, and
# no convention defines it. Kept as-is because the backend reads it.
MCP_RESULT_TYPE = "mcp.result_type"
GEN_AI_REQUEST_PREFIX = "gen_ai.request."
# Where a request parameter the GenAI registry does NOT define goes. Our own
# namespace rather than gen_ai.request.<key>, per OTel's naming guidance: an
# existing semantic-convention namespace must not be used as a prefix for
# application-specific attributes, because the convention is free to define
# that exact key later and mean something else. The GenAI conventions offer
# no catch-all of their own, so we own one. Settled in RIUS-917.
#
# Masking: rius.request.* follows gen_ai.request.* exactly. Today neither is
# in CONTENT_ATTRIBUTES, so both export in clear. The TIE is the rule, not
# the current value — if request parameters ever become maskable, both
# namespaces change together, and neither may join the content allowlist
# alone. test_request_namespaces_are_tied_for_masking pins it.
RIUS_REQUEST_PREFIX = "rius.request."

# Request parameters the GenAI conventions define, mapped from every spelling
# we recognise to the canonical attribute key. Verified against
# open-telemetry/semantic-conventions-genai at commit
# 8ffdf568e1b4391a99adb081db16e8102e36918e (2026-09-22),
# model/gen-ai/registry.yaml — the repo cuts no releases, so a commit is the
# only citable pin. Each canonical key appears under its own bare spelling
# plus the provider spellings that mean the SAME parameter; anything absent
# here is a caller parameter and goes to RIUS_REQUEST_PREFIX verbatim.
#
# A provider spelling normalises to ONE key, the canonical one: the provider
# spelling is not also emitted. Two keys for one parameter would make every
# consumer de-duplicate, and the point of a convention is that there is one
# place to look.
GEN_AI_REQUEST_PARAMETERS: dict[str, str] = {
    # model — the request's model; also settable via the `model=` argument.
    "model": GEN_AI_REQUEST_MODEL,
    # max_tokens — OpenAI Chat Completions `max_tokens`, its successor
    # `max_completion_tokens`, the Responses API's `max_output_tokens`, and
    # Google's `maxOutputTokens`.
    "max_tokens": "gen_ai.request.max_tokens",
    "maxTokens": "gen_ai.request.max_tokens",
    "max_completion_tokens": "gen_ai.request.max_tokens",
    "maxCompletionTokens": "gen_ai.request.max_tokens",
    "max_output_tokens": "gen_ai.request.max_tokens",
    "maxOutputTokens": "gen_ai.request.max_tokens",
    # choice.count — "the target number of candidate completions to return":
    # OpenAI `n`, Google `candidateCount`, Cohere `num_generations`.
    "choice.count": "gen_ai.request.choice.count",
    "n": "gen_ai.request.choice.count",
    "candidate_count": "gen_ai.request.choice.count",
    "candidateCount": "gen_ai.request.choice.count",
    "num_generations": "gen_ai.request.choice.count",
    "temperature": "gen_ai.request.temperature",
    # top_p — Google `topP`, Cohere `p`.
    "top_p": "gen_ai.request.top_p",
    "topP": "gen_ai.request.top_p",
    "p": "gen_ai.request.top_p",
    # top_k — the registry's own note names Anthropic `top_k`, Cohere `k` and
    # Google `topK`, and says OpenAI's `top_logprobs` MUST NOT be reported
    # here (it shapes the response, not the sampling). So top_logprobs is
    # deliberately absent and lands in rius.request.*.
    "top_k": "gen_ai.request.top_k",
    "topK": "gen_ai.request.top_k",
    "k": "gen_ai.request.top_k",
    # stop_sequences — OpenAI `stop`, Google `stopSequences`.
    "stop_sequences": "gen_ai.request.stop_sequences",
    "stopSequences": "gen_ai.request.stop_sequences",
    "stop": "gen_ai.request.stop_sequences",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "frequencyPenalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "presencePenalty": "gen_ai.request.presence_penalty",
    # encoding_formats — plural in the registry; OpenAI's embeddings endpoint
    # sends the singular `encoding_format`, and the registry's note says some
    # systems call these "embedding types" (Cohere `embedding_types`).
    "encoding_formats": "gen_ai.request.encoding_formats",
    "encoding_format": "gen_ai.request.encoding_formats",
    "encodingFormat": "gen_ai.request.encoding_formats",
    "embedding_types": "gen_ai.request.encoding_formats",
    "seed": "gen_ai.request.seed",
    "stream": GEN_AI_REQUEST_STREAM,
    # reasoning.level — "the exact string value sent to the provider";
    # OpenAI sends it as `reasoning_effort`.
    "reasoning.level": GEN_AI_REQUEST_REASONING_LEVEL,
    "reasoning_level": GEN_AI_REQUEST_REASONING_LEVEL,
    "reasoning_effort": GEN_AI_REQUEST_REASONING_LEVEL,
    "reasoningEffort": GEN_AI_REQUEST_REASONING_LEVEL,
    # previous_response.id — the registry names OpenAI's
    # `previous_response_id` and Google's `previous_interaction_id`.
    "previous_response.id": "gen_ai.request.previous_response.id",
    "previous_response_id": "gen_ai.request.previous_response.id",
    "previousResponseId": "gen_ai.request.previous_response.id",
    "previous_interaction_id": "gen_ai.request.previous_response.id",
    # stream_cursor — the registry names OpenAI's `starting_after` and
    # Google's `last_event_id`.
    "stream_cursor": "gen_ai.request.stream_cursor",
    "starting_after": "gen_ai.request.stream_cursor",
    "last_event_id": "gen_ai.request.stream_cursor",
}


def request_attribute_key(parameter: str) -> str:
    """The attribute key one caller-supplied request parameter is recorded under.

    A parameter the GenAI conventions define — under its canonical name or a
    recognised provider spelling — normalises to its canonical
    ``gen_ai.request.*`` key. Everything else keeps its key verbatim under
    ``rius.request.``.
    """
    canonical = GEN_AI_REQUEST_PARAMETERS.get(parameter)
    if canonical is not None:
        return canonical
    return f"{RIUS_REQUEST_PREFIX}{parameter}"


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
        # Which tool call a still-running execution answers, and what kind of
        # tool it is. Both are caller-supplied at creation, and the call id is
        # what joins the pending tool span back to the generation that asked
        # for it while the tool is still running.
        GEN_AI_TOOL_CALL_ID,
        GEN_AI_TOOL_TYPE,
        # The requested output modality describes the request, so it is known
        # before the model answers. Listed explicitly because the key sits
        # outside gen_ai.request. and the prefix rule below does not reach it.
        GEN_AI_OUTPUT_TYPE,
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
        # On a TOOL span the same key instead names the agent EXECUTING the
        # call, known from the enclosing scope at creation for the same
        # reason: which agent is stuck is the live view's first question.
        # Distinct from the resource key of the same name, which says which
        # PROCESS is running; this says which agent that process invoked here.
        GEN_AI_AGENT_NAME,
        GEN_AI_AGENT_ID,
        # Which VERSION of that agent definition was invoked. Chosen by the
        # caller at creation like the other two, and the live view's follow-up
        # question once it knows which agent is stuck.
        GEN_AI_AGENT_VERSION,
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
# gen_ai.request.* (model, temperature, ...) is identity, not content, and so
# is rius.request.* (the caller parameters the conventions do not define):
# both are chosen before the call runs, so a live view of a still-running
# generation must already show them. The two namespaces are deliberately
# treated alike here as well as in CONTENT_ATTRIBUTES.
PENDING_IDENTITY_PREFIXES = (GEN_AI_REQUEST_PREFIX, RIUS_REQUEST_PREFIX)

# Parameter names that carry TOOL DEFINITIONS rather than a sampling knob.
# Named here once and consumed by three places, so the routes to the same
# definitions cannot drift apart:
#   - as gen_ai.request.<member>, and
#   - as rius.request.<member>, when a caller passes `tools=` through
#     model_parameters and it reaches either namespace.
# Tool definitions are content in the same sense messages are — see
# GEN_AI_TOOL_DEFINITIONS below and semconv-genai#431 — and which of the two
# routes they arrived by cannot be what decides whether they are protected.
# A third route, as JSON members of llm.invocation_parameters, is covered by
# the whole bag being content; there normalization reads this same list to
# PROMOTE those members onto gen_ai.tool.definitions (normalization.py,
# normalize_tool_definitions), so the bag route and the native routes agree on
# which parameter names are tool definitions.
INVOCATION_PARAMETERS_CONTENT_MEMBERS = ("tools", "functions")

# The request-parameters bag OpenInference instrumentors emit.
LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"

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
        # The OpenInference request-parameters bag, whole. Its membership is
        # open and provider-defined — litellm and langchain put the request's
        # tools array in it, and providers are free to add anything else —
        # so a per-member blocklist is always one provider behind. It is only
        # safe to treat the whole bag as content because normalization runs
        # BEFORE masking and has already promoted every member the rule table
        # recognises onto gen_ai.request.*, so what reaches here is exactly
        # the set nothing has classified. A knob that matters is recovered by
        # teaching the rule table its name, which is a reviewed decision;
        # the default is that unclassified request data does not leave a
        # process that asked for no content.
        LLM_INVOCATION_PARAMETERS,
    }
    # The NAMED EXCEPTION to the rule that request parameters export in clear
    # (RIUS-917). That rule is right for scalar knobs — temperature, top_p,
    # seed — and was written before a tools array could land in either
    # request namespace. A caller passing
    # model_parameters={"tools": [...]} would otherwise export proprietary
    # prompt engineering verbatim with capture_content=False explicitly set,
    # while the very same array is stripped when it arrives as
    # gen_ai.tool.definitions or as a member of llm.invocation_parameters.
    # Per-KEY, not per-namespace: the namespaces stay non-content wholesale,
    # which is what keeps the RIUS-917 tie intact.
    | {
        f"{prefix}{member}"
        for prefix in (GEN_AI_REQUEST_PREFIX, RIUS_REQUEST_PREFIX)
        for member in INVOCATION_PARAMETERS_CONTENT_MEMBERS
    }
)

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

# The inverse, for spans that speak only the conventions. A GenAI-native
# instrumentation sets gen_ai.operation.name and never heard of
# openinference.span.kind, and both keys must be present on every span, so the
# derivation has to run in both directions.
#
# Not a mechanical inversion of the map above, and it cannot be generated from
# it, for two reasons:
#   - several operations share one kind. text_completion and generate_content
#     are LLM calls the same way chat is, and create_agent is an AGENT span
#     just as invoke_agent is, so the mapping is many-to-one and only the
#     chat/invoke_agent entries round-trip.
#   - CHAIN has no operation of its own, but two operations land on it.
#     invoke_workflow and plan are real GenAI operations with no taxonomy
#     value of their own; CHAIN is the honest home for both, and the reverse
#     direction is the only one that can say so.
# The structural test asserts the round-trip where one exists, not equality of
# the two maps. Mirrors the console's deriveSpanKind.
_KIND_BY_OPERATION: dict[str, SpanKind] = {
    "chat": SpanKind.LLM,
    "text_completion": SpanKind.LLM,
    "generate_content": SpanKind.LLM,
    "execute_tool": SpanKind.TOOL,
    "embeddings": SpanKind.EMBEDDING,
    "invoke_agent": SpanKind.AGENT,
    "create_agent": SpanKind.AGENT,
    "retrieval": SpanKind.RETRIEVER,
    "invoke_workflow": SpanKind.CHAIN,
    "plan": SpanKind.CHAIN,
}


def operation_for_kind(kind: str) -> str | None:
    """``gen_ai.operation.name`` for a taxonomy value, or None if it has none.

    Takes the raw attribute string rather than the enum: the caller is
    normalizing a third-party span, where the value is whatever the
    instrumentation wrote. An unrecognised one yields None rather than raising,
    because a span carrying a kind we do not know is not a reason to lose the
    span.
    """
    try:
        return _OPERATION_BY_KIND.get(SpanKind(kind))
    except ValueError:
        return None


def kind_for_operation(operation: str) -> str | None:
    """The taxonomy value for a ``gen_ai.operation.name``, or None if unknown."""
    kind = _KIND_BY_OPERATION.get(operation)
    return kind.value if kind is not None else None


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


# The name of a span whose kind has no operation and nothing else to compose
# from. The conventions define no operation for a generic workflow step, and
# the manual span helpers — unlike @observe, which has a function to borrow a
# qualname from — see nothing but the kind. A literal beats an empty name.
CHAIN_SPAN_NAME = "chain"


# Which attribute names the target, per kind. One key each, so a tool name can
# never surface in an agent's name and the composed name stays a reliable
# filter value. CHAIN is absent: it has no operation and so no composed name.
_NAME_TARGET_BY_KIND: dict[SpanKind, str] = {
    SpanKind.LLM: GEN_AI_REQUEST_MODEL,
    SpanKind.EMBEDDING: GEN_AI_REQUEST_MODEL,
    SpanKind.TOOL: GEN_AI_TOOL_NAME,
    SpanKind.AGENT: GEN_AI_AGENT_NAME,
    SpanKind.RETRIEVER: GEN_AI_DATA_SOURCE_ID,
}


def compose_span_name(kind: SpanKind, attributes: Mapping[str, str | int]) -> str:
    """The span name for a span of ``kind``: ``"{operation} {target}"``.

    The GenAI conventions give every operation they define a SHOULD-level span
    name of the operation followed by the one identifier saying what it acted
    on: ``chat gpt-4o``, ``embeddings text-embedding-3-small``,
    ``execute_tool get_weather``, ``invoke_agent planner``, ``retrieval kb``.

    Composed from the span's OWN creation attributes rather than from the
    caller's arguments, which is what makes the name and the attributes agree
    by construction instead of by discipline. Three things follow for free: an
    operation overridden per generation is picked up, because the override is
    already in the map; an agent name that fell back to the configured one is
    picked up for the same reason; and the rendered ``execute_tool x`` can
    never become ``gen_ai.tool.name``, because this function only ever reads.
    The TypeScript SDK composes the identical way, so the two cannot drift as
    new identifiers are added.

    The model read here is the REQUEST model, since that is the only one in
    the creation attributes. A response model arriving later would make a
    pending snapshot and its final span disagree on their name, and that
    equality is part of the pending wire contract.

    The identifier is optional on every kind, and the name degrades to the
    bare operation rather than to a name with a hole in it. ``CHAIN`` has no
    operation at all and falls back to :data:`CHAIN_SPAN_NAME`.
    """
    operation = attributes.get(GEN_AI_OPERATION_NAME)
    if not isinstance(operation, str):
        return CHAIN_SPAN_NAME
    target_key = _NAME_TARGET_BY_KIND.get(kind)
    target = attributes.get(target_key) if target_key is not None else None
    return f"{operation} {target}" if isinstance(target, str) and target else operation


def kind_attributes(
    kind: SpanKind,
    tool_name: str | None = None,
    *,
    data_source_id: str | None = None,
    top_k: int | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_version: str | None = None,
    executing_agent_name: str | None = None,
    tool_call_id: str | None = None,
    tool_type: str | None = None,
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

    ``agent_version`` is ``gen_ai.agent.version``, the version of that SAME
    invoked agent's definition. AGENT-only, caller-supplied, taken verbatim,
    and never derived: not from ``service.version``, which versions the
    deployed process, and not from the main agent's version, which versions
    the agent this process IS rather than the one it just called. Different
    agent, different scope; borrowing one for the other is confidently wrong
    data. It does not touch the span NAME either — ``_NAME_TARGET_BY_KIND``
    maps AGENT to ``gen_ai.agent.name`` alone.

    ``tool_call_id`` is ``gen_ai.tool.call.id``, the identifier the MODEL put
    on the tool-call message this execution answers, and ``tool_type`` is
    ``gen_ai.tool.type`` ("function", "extension", "datastore", ...). Both are
    TOOL-only, both are caller-supplied, and neither is ever guessed: only the
    caller has seen the model's response, and a tool's type is a fact about
    its declaration rather than about the call. Neither touches the span NAME,
    because ``_NAME_TARGET_BY_KIND`` maps TOOL to ``gen_ai.tool.name`` alone.

    ``executing_agent_name`` is a SEPARATE argument writing the SAME key,
    ``gen_ai.agent.name``, on a TOOL span — where the conventions define it as
    "the human-readable name of the agent executing the tool". One key, two
    meanings, told apart by ``gen_ai.operation.name``: on ``invoke_agent`` it
    is the agent being invoked, on ``execute_tool`` the agent doing the call.
    Two arguments rather than one precisely so the two can never be fed from
    the same source by accident; the TOOL one is never passed by a caller, it
    is resolved from the enclosing agent scope (see ``_agent``). It does not
    touch the span NAME: ``_NAME_TARGET_BY_KIND`` maps TOOL to
    ``gen_ai.tool.name``, so a tool span stays ``execute_tool {tool}``.
    """
    attributes: dict[str, str | int] = {OPENINFERENCE_SPAN_KIND: kind.value}
    operation = _OPERATION_BY_KIND.get(kind)
    if operation is not None:
        attributes[GEN_AI_OPERATION_NAME] = operation
    if kind is SpanKind.TOOL:
        if tool_name is not None:
            attributes[GEN_AI_TOOL_NAME] = tool_name
        # The agent EXECUTING the tool — not one being invoked. See the
        # docstring; the AGENT branch below writes the same key from the
        # other, caller-supplied source.
        if executing_agent_name is not None:
            attributes[GEN_AI_AGENT_NAME] = executing_agent_name
        if tool_call_id is not None:
            attributes[GEN_AI_TOOL_CALL_ID] = tool_call_id
        if tool_type is not None:
            attributes[GEN_AI_TOOL_TYPE] = tool_type
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
        if agent_version is not None:
            attributes[GEN_AI_AGENT_VERSION] = agent_version
    return attributes


def set_span_kind(span: Span, kind: SpanKind) -> None:
    """Stamp a span with its OpenInference kind and (if applicable) gen_ai operation."""
    span.set_attribute(OPENINFERENCE_SPAN_KIND, kind.value)
    operation = _OPERATION_BY_KIND.get(kind)
    if operation is not None:
        span.set_attribute(GEN_AI_OPERATION_NAME, operation)
