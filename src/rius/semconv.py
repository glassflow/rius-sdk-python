"""Semantic conventions for the GlassFlow SDK.

Centralizes the OpenTelemetry instrumentation-scope name, the span-kind taxonomy,
and span attribute keys. Span kinds use the OpenInference `openinference.span.kind`
values (understood across the ecosystem); LLM specifics use OTel GenAI `gen_ai.*`.
"""

from __future__ import annotations

from enum import Enum

from opentelemetry.trace import Span

# Instrumentation scope name (stamped on every span as otel.scope.name).
# Deliberately still "glassflow" after the Rius rebrand: the value
# is wire-visible and the backend keys on it; renaming needs backend
# coordination, tracked separately.
TRACER_NAME = "glassflow"

# --- Attribute keys ---
# OpenInference
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"

# OTel GenAI (subset we emit)
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
# MCP spec 2026-07-28: a tools/call round can end with an interim
# "input_required" result (MRTR) instead of a final one. Set ONLY on interim
# rounds; the key follows the mcp SDK's own `mcp.*` attribute namespace.
MCP_RESULT_TYPE = "mcp.result_type"
GEN_AI_REQUEST_PREFIX = "gen_ai.request."

# --- Span event names ---
# First streamed token/chunk arrived: the TTFT anchor (event time minus span
# start). No client-side semconv exists for this yet (OTel only standardizes
# the server-side gen_ai.server.time_to_first_token metric); this follows the
# gen_ai.* naming style, precedent Langfuse's completion_start_time.
GEN_AI_FIRST_TOKEN_EVENT = "gen_ai.first_token"

# --- Pending (partial) spans ---
# Marks the content-free snapshot exported at span START; the backend maps it
# to Finished=0 and the real span replaces it at end. This key knowingly bends
# the convention-native rule (no glassflow.* namespace): OpenTelemetry has NO
# pending-span mechanism to align with (spec #3732/#4646, semconv #2133, all
# open, none planned), and the only shipping precedent (Logfire's
# logfire.span_type) is equally vendor-namespaced.
GLASSFLOW_SPAN_PENDING = "glassflow.span.pending"

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
        # common third-party content keys (bundled instrumentation)
        "gen_ai.prompt",
        "gen_ai.completion",
        "llm.input_messages",
        "llm.output_messages",
        "mlflow.spanInputs",
        "mlflow.spanOutputs",
        # OpenLLMetry workflow/task spans carry full I/O here
        "traceloop.entity.input",
        "traceloop.entity.output",
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
_OPERATION_BY_KIND: dict[SpanKind, str] = {
    SpanKind.LLM: "chat",
    SpanKind.TOOL: "execute_tool",
    SpanKind.EMBEDDING: "embeddings",
    SpanKind.AGENT: "invoke_agent",
}


def kind_attributes(kind: SpanKind) -> dict[str, str]:
    """Identity attributes for a span of ``kind``, for setting at CREATION.

    Pending snapshots (pending.py) are built at ``on_start``, so taxonomy set
    via ``set_attribute`` afterwards is invisible to them; passing these at
    span creation is what makes a pending span classifiable.
    """
    attributes = {OPENINFERENCE_SPAN_KIND: kind.value}
    operation = _OPERATION_BY_KIND.get(kind)
    if operation is not None:
        attributes[GEN_AI_OPERATION_NAME] = operation
    return attributes


def set_span_kind(span: Span, kind: SpanKind) -> None:
    """Stamp a span with its OpenInference kind and (if applicable) gen_ai operation."""
    span.set_attribute(OPENINFERENCE_SPAN_KIND, kind.value)
    operation = _OPERATION_BY_KIND.get(kind)
    if operation is not None:
        span.set_attribute(GEN_AI_OPERATION_NAME, operation)
