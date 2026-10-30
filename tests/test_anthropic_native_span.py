"""The anthropic SDK's own chat span, next to OpenInference's, counts once.

From 1.14.0 the anthropic SDK starts its own CLIENT span per API call (scope
``com.anthropic.sdk.python``) on the OpenTelemetry global provider. Under a
global ``init()`` with the OpenInference Anthropic instrumentor, that span
arrived beside OpenInference's: two spans normalized as LLM per call, tokens
and cost counted twice, a second ``error.type`` spelled ``not_found_error``,
and an unflagged duplicate after ``mark_compaction()``. When rius registered
that instrumentor, the SDK's chat span is exported as a plain client span.
From openinference-instrumentation-anthropic 3.2.0, OpenInference stops the
SDK from starting that chat span at all, so the demotion has nothing to do.

These run each scenario in a fresh interpreter (``anthropic_native_span_run.py``):
the global provider is set once per process, and with ``set_global=False``,
which the rest of the suite uses, the SDK's spans never reach rius.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind as OtelSpanKind

from rius._native_spans import NativeChatSpanDemoter

pytest.importorskip("anthropic")
pytest.importorskip("openinference.instrumentation.anthropic")

_RUNNER = Path(__file__).with_name("anthropic_native_span_run.py")
_SCOPE = "com.anthropic.sdk.python"
_SYNC_CASES = ("create", "create_streamed", "stream_helper", "not_found")
# The same calls through AsyncAnthropic, which OpenInference wraps separately.
_CASES = (*_SYNC_CASES, *(f"async_{case}" for case in _SYNC_CASES))
_NOT_FOUND = ("not_found", "async_not_found")
_SECRETS = ("secret-prompt-text", "secret-reply-text", "secret-system-text")
_KEPT_GEN_AI = {"gen_ai.provider.name", "gen_ai.conversation.id"}


@cache
def _run(
    instruments: tuple[str, ...] = ("anthropic",),
    content_mode: bool = False,
    partial_spans: bool = False,
    foreign: bool = False,
    capture_content: bool = True,
) -> dict[str, Any]:
    env = {**os.environ}
    if content_mode:
        env["ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE"] = "content"
    options = {
        "instruments": list(instruments),
        "partial_spans": partial_spans,
        "foreign": foreign,
        "capture_content": capture_content,
    }
    done = subprocess.run(
        [sys.executable, str(_RUNNER), json.dumps(options)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    result: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


def _llm(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in spans if s["attributes"].get("openinference.span.kind") == "LLM"]


def _native(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in spans if s["scope"] == _SCOPE]


def _sdk_traces() -> bool:
    """Whether this anthropic starts spans of its own (from 1.14.0), seen without OpenInference."""
    return bool(_native(_run(instruments=())["cases"]["create"]))


def _openinference_leaves_the_sdk_chat_span() -> bool:
    """Whether the SDK's chat span still starts beside OpenInference's.

    From openinference-instrumentation-anthropic 3.2.0 it does not: OpenInference
    stops the SDK from starting a chat span for a call it traces, so there is
    nothing to demote. The SDK's tool-runner and other API spans still start.
    """
    return bool(_native(_run()["cases"]["create"]))


_NO_SDK_TRACING = "this anthropic has no tracing of its own (it starts in 1.14.0)"
_SDK_CHAT_SPAN_LEFT = "this OpenInference leaves the SDK's chat span; the demotion tests cover it"
_NO_SDK_CHAT_SPAN = (
    "OpenInference stops the SDK's chat span (3.2.0 on): nothing to demote; "
    "see test_openinference_that_stops_the_sdk_chat_span_leaves_one_llm_span"
)


def _needs_the_sdk_chat_span() -> None:
    if not _sdk_traces():
        pytest.skip(_NO_SDK_TRACING)
    if not _openinference_leaves_the_sdk_chat_span():
        pytest.skip(_NO_SDK_CHAT_SPAN)


_ARGUS_USAGE = tuple(
    f"gen_ai.usage.{k}_tokens" for k in ("input", "output", "cache_read.input", "cache_write.input")
)


def _argus_badges_llm(attributes: dict[str, Any]) -> bool:
    """argus-core's llm badge: the OpenInference kind, else a request model."""
    kind = attributes.get("openinference.span.kind")
    return kind == "LLM" if kind is not None else "gen_ai.request.model" in attributes


def _argus_prices(attributes: dict[str, Any]) -> bool:
    """argus-core's enrich.costFor: a request model and any token count, whatever the kind."""
    return "gen_ai.request.model" in attributes and any(k in attributes for k in _ARGUS_USAGE)


@pytest.mark.integration
def test_the_runner_uses_the_anthropic_under_test() -> None:
    # A `uv run --with anthropic==X` overlay must reach the subprocess too.
    import anthropic

    assert _run()["anthropic"] == anthropic.__version__


@pytest.mark.integration
@pytest.mark.parametrize("case", _CASES)
def test_one_call_exports_one_llm_span(case: str) -> None:
    (llm,) = _llm(_run()["cases"][case])
    assert llm["scope"] == "openinference.instrumentation.anthropic"


@pytest.mark.integration
@pytest.mark.parametrize("case", _NOT_FOUND)
def test_a_provider_404_has_one_error_type_the_exception_class(case: str) -> None:
    spans = _run()["cases"][case]
    (llm,) = _llm(spans)
    assert llm["attributes"]["error.type"] in ("anthropic.NotFoundError", "NotFoundError")


@pytest.mark.integration
def test_after_mark_compaction_only_the_next_call_is_flagged_and_has_no_duplicate() -> None:
    spans = _run()["cases"]["compaction"]
    llm = sorted(
        _llm(spans), key=lambda s: s["attributes"].get("gen_ai.conversation.compacted") is None
    )
    assert [s["attributes"].get("gen_ai.conversation.compacted") for s in llm] == [True, None]


@pytest.mark.integration
@pytest.mark.parametrize("case", [*_CASES, "compaction"])
def test_the_sdk_chat_span_is_a_plain_client_span(case: str) -> None:
    _needs_the_sdk_chat_span()
    spans = _run()["cases"][case]
    native = _native(spans)
    assert native, "the SDK's span is kept, demoted, not dropped"
    for span in native:
        attributes = span["attributes"]
        assert span["kind"] == "CLIENT"
        assert "openinference.span.kind" not in attributes
        assert not _argus_badges_llm(attributes)
        assert not _argus_prices(attributes)
        assert "gen_ai.request.model" not in attributes
        assert not [k for k in attributes if k.startswith("gen_ai.usage.")]
        assert {k for k in attributes if k.startswith("gen_ai.")} <= _KEPT_GEN_AI
        assert attributes["gen_ai.provider.name"] == "anthropic"
        assert attributes["server.address"] == "127.0.0.1"
        assert "http.response.status_code" in attributes


@pytest.mark.integration
@pytest.mark.parametrize("case", _CASES)
def test_argus_badges_and_prices_each_call_once(case: str) -> None:
    attributes = [s["attributes"] for s in _run()["cases"][case]]
    assert sum(_argus_badges_llm(a) for a in attributes) == 1
    if case not in _NOT_FOUND:
        assert sum(_argus_prices(a) for a in attributes) == 1


@pytest.mark.integration
def test_a_bedrock_call_counts_once_and_its_sdk_span_keeps_its_provider() -> None:
    spans = _run()["cases"].get("bedrock")
    if spans is None:
        pytest.skip("the Bedrock client needs botocore")
    (llm,) = _llm(spans)
    assert llm["scope"] == "openinference.instrumentation.anthropic"
    for span in [s for s in spans if s["scope"].startswith("com.anthropic.sdk.")]:
        assert span["scope"] == "com.anthropic.sdk.python.bedrock"
        assert span["attributes"]["gen_ai.provider.name"] == "aws.bedrock"
        assert not _argus_badges_llm(span["attributes"])
        assert not _argus_prices(span["attributes"])


@pytest.mark.integration
@pytest.mark.parametrize("case", _NOT_FOUND)
def test_the_demoted_404_keeps_its_http_failure(case: str) -> None:
    _needs_the_sdk_chat_span()
    (native,) = _native(_run()["cases"][case])
    assert native["attributes"]["http.response.status_code"] == 404
    # Named as OpenInference's span names it, so both group as one failure.
    assert native["attributes"]["error.type"] == "anthropic.NotFoundError"
    assert native["status"] == "ERROR"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("case", "status", "expected"),
    [
        ("not_found", 404, "anthropic.NotFoundError"),
        ("async_not_found", 404, "anthropic.NotFoundError"),
        ("rate_limited", 429, "anthropic.RateLimitError"),
    ],
)
def test_without_openinference_the_sdk_span_names_the_exception_class(
    case: str, status: int, expected: str
) -> None:
    # The SDK writes the API's error code (not_found_error); exported, it
    # carries the class the SDK raised, as OpenInference's span would.
    if not _sdk_traces():
        pytest.skip(_NO_SDK_TRACING)
    (native,) = _native(_run(instruments=())["cases"][case])
    assert native["attributes"]["http.response.status_code"] == status
    assert native["attributes"]["error.type"] == expected
    assert native["status"] == "ERROR"


@pytest.mark.integration
def test_the_stream_helper_span_in_a_trace_of_its_own_is_demoted_too() -> None:
    # The SDK's span for messages.stream() has no parent: it is not under
    # OpenInference's span, so nothing keyed on parentage can catch it.
    _needs_the_sdk_chat_span()
    spans = _run()["cases"]["stream_helper"]
    (native,) = _native(spans)
    assert native["parent_id"] is None
    assert "gen_ai.request.model" not in native["attributes"]


@pytest.mark.integration
def test_no_content_survives_on_the_demoted_span() -> None:
    _needs_the_sdk_chat_span()
    run = _run(content_mode=True)
    for case in _CASES:
        for span in _native(run["cases"][case]):
            text = json.dumps([span["attributes"], span["events"]])
            assert not [s for s in _SECRETS if s in text], (case, span["attributes"])
    # The OpenInference span still carries the conversation.
    assert "secret-prompt-text" in json.dumps(_llm(run["cases"]["create"])[0]["attributes"])


@pytest.mark.integration
def test_the_pending_snapshot_of_the_sdk_span_is_not_an_llm_span() -> None:
    _needs_the_sdk_chat_span()
    spans = _run(partial_spans=True)["cases"]["create"]
    pending = [s for s in _native(spans) if s["attributes"].get("rius.span.pending")]
    assert pending
    for span in pending:
        assert "openinference.span.kind" not in span["attributes"]
        assert "gen_ai.request.model" not in span["attributes"]


@pytest.mark.integration
def test_without_openinference_the_sdk_span_stays_the_llm_span() -> None:
    if not _sdk_traces():
        pytest.skip(_NO_SDK_TRACING)
    run = _run(instruments=())
    for case in _CASES:
        (llm,) = _llm(run["cases"][case])
        assert llm["scope"] == _SCOPE
        assert llm["attributes"]["gen_ai.request.model"]


@pytest.mark.integration
@pytest.mark.parametrize("case", _CASES)
def test_bridged_to_another_sdk_each_call_still_counts_once(case: str) -> None:
    # The anthropic SDK's span then belongs to the other provider: rius's
    # pipeline sees a twin at start and a separate copy at end, and must
    # demote the copy it exports without touching the other SDK's span.
    run = _run(foreign=True)
    attributes = [s["attributes"] for s in run["cases"][case]]
    assert sum(_argus_badges_llm(a) for a in attributes) == 1
    if case not in _NOT_FOUND:
        assert sum(_argus_prices(a) for a in attributes) == 1
    if not (_sdk_traces() and _openinference_leaves_the_sdk_chat_span()):
        return
    (ours,) = _native(run["cases"][case])
    assert "gen_ai.request.model" not in ours["attributes"]
    assert "gen_ai.operation.name" not in ours["attributes"]
    (theirs,) = _native(run["foreign"][case])
    assert theirs["span_id"] == ours["span_id"]
    assert theirs["attributes"]["gen_ai.operation.name"] == "chat"
    assert theirs["attributes"]["gen_ai.request.model"].startswith("claude-")
    assert "openinference.span.kind" not in theirs["attributes"]
    if case in _NOT_FOUND:
        # Renamed on the copy rius exports; the other provider's span keeps its own.
        assert ours["attributes"]["error.type"] == "anthropic.NotFoundError"
        assert theirs["attributes"]["error.type"] == "not_found_error"


@pytest.mark.integration
@pytest.mark.parametrize("case", [*_CASES, "compaction", "rate_limited"])
def test_openinference_that_stops_the_sdk_chat_span_leaves_one_llm_span(case: str) -> None:
    # The SDK traces on its own, but OpenInference (3.2.0 on) keeps it from
    # starting a chat span for the calls it traces: no duplicate to demote.
    if not _sdk_traces():
        pytest.skip(_NO_SDK_TRACING)
    if _openinference_leaves_the_sdk_chat_span():
        pytest.skip(_SDK_CHAT_SPAN_LEFT)
    spans = _run()["cases"][case]
    # Any SDK span at all, not just one named chat: the demoter strips the
    # operation name, so a chat span that leaked would no longer carry it.
    assert _native(spans) == []
    assert [s["scope"] for s in _llm(spans)] == ["openinference.instrumentation.anthropic"] * (
        2 if case == "compaction" else 1
    )


_PATH_PREFIX = "secret-path-prefix"
_MEMORIES_PATH = "/v1/memory_stores/memstore_1/memories"


def _memories_list_span(capture_content: bool) -> dict[str, Any]:
    if not _sdk_traces():
        pytest.skip(_NO_SDK_TRACING)
    spans = _run(content_mode=True, capture_content=capture_content)["cases"].get("memories_list")
    if spans is None:
        pytest.skip("this anthropic has no memory stores")
    (native,) = _native(spans)
    return native


@pytest.mark.integration
def test_capture_off_drops_the_query_from_the_sdk_url() -> None:
    # In its content mode the SDK keeps a query parameter it treats as content
    # (path_prefix) in url.query and url.full. With capture_content=False
    # neither may carry it; the URL without its query stays.
    attributes = _memories_list_span(capture_content=False)["attributes"]
    assert _PATH_PREFIX not in json.dumps(attributes), attributes
    assert "url.query" not in attributes
    assert attributes["url.path"] == _MEMORIES_PATH
    assert attributes["url.full"].endswith(_MEMORIES_PATH)


@pytest.mark.integration
def test_with_capture_on_the_sdk_url_keeps_its_query() -> None:
    # The same call with capture on: the SDK does write the query, so the test
    # above is not passing on a run that never recorded it.
    attributes = _memories_list_span(capture_content=True)["attributes"]
    assert _PATH_PREFIX in attributes["url.query"]
    assert _PATH_PREFIX in attributes["url.full"]


_TOOL_RUNNER_CASES = ("tool_runner", "async_tool_runner")
_TOOL_SECRETS = (*_SECRETS, "secret-tool-argument", "secret-tool-output")


@pytest.mark.integration
@pytest.mark.parametrize("case", _TOOL_RUNNER_CASES)
def test_capture_off_drops_the_tool_runner_content_from_every_span(case: str) -> None:
    # In its content mode the SDK records the tool's arguments and output on
    # its execute_tool span, and the conversation on its chat spans. With
    # capture_content=False none of it may leave the process.
    if not _sdk_traces():
        pytest.skip(_NO_SDK_TRACING)
    spans = _run(content_mode=True, capture_content=False)["cases"].get(case)
    if spans is None:
        pytest.skip("this anthropic has no tool runner")
    for span in spans:
        text = json.dumps([span["attributes"], span["events"]])
        assert not [s for s in _TOOL_SECRETS if s in text], (span["name"], span["attributes"])
    tools = [
        s for s in _native(spans) if s["attributes"].get("gen_ai.operation.name") == "execute_tool"
    ]
    (tool,) = tools
    assert tool["attributes"]["gen_ai.tool.name"] == "lookup"
    assert tool["attributes"]["gen_ai.tool.call.id"] == "toolu_1"


@pytest.mark.integration
@pytest.mark.parametrize("case", _TOOL_RUNNER_CASES)
def test_with_capture_on_the_tool_runner_content_is_kept(case: str) -> None:
    # The same run with capture on: the SDK's execute_tool span does carry the
    # tool's arguments and output under its own keys, so the test above is not
    # passing on a run that never recorded them.
    if not _sdk_traces():
        pytest.skip(_NO_SDK_TRACING)
    spans = _run(content_mode=True)["cases"].get(case)
    if spans is None:
        pytest.skip("this anthropic has no tool runner")
    (tool,) = [
        s for s in _native(spans) if s["attributes"].get("gen_ai.operation.name") == "execute_tool"
    ]
    assert "secret-tool-output" in tool["attributes"]["anthropic.tool_result.content"]
    assert "secret-tool-argument" in tool["attributes"]["gen_ai.tool.call.arguments"]


# --- the processor on its own -------------------------------------------------


def _pipeline(active: bool) -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(NativeChatSpanDemoter(lambda: active))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


_CHAT = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.conversation.id": "conv_1",
    "gen_ai.request.model": "claude-sonnet-5",
    "openinference.span.kind": "LLM",
    "server.address": "api.anthropic.com",
    "anthropic.request.service_tier": "auto",
}


def _chat_span(provider: TracerProvider, scope: str = _SCOPE, **attributes: Any) -> None:
    span = provider.get_tracer(scope).start_span(
        "anthropic.messages.create", kind=OtelSpanKind.CLIENT, attributes={**_CHAT, **attributes}
    )
    span.set_attribute("gen_ai.usage.input_tokens", 10)
    span.set_attribute("gen_ai.response.model", "claude-sonnet-5")
    span.set_attribute("http.response.status_code", 200)
    span.end()


@pytest.mark.parametrize(
    ("scope", "provider_name"),
    [
        ("com.anthropic.sdk.python.bedrock", "aws.bedrock"),
        ("com.anthropic.sdk.python.vertex", "gcp.vertex_ai"),
        ("com.anthropic.sdk.python.foundry", "anthropic"),
        ("com.anthropic.sdk.typescript", "anthropic"),
    ],
)
def test_the_platform_clients_chat_spans_are_demoted_too(scope: str, provider_name: str) -> None:
    provider, exporter = _pipeline(active=True)
    _chat_span(provider, scope=scope, **{"gen_ai.provider.name": provider_name})
    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert "gen_ai.request.model" not in attributes
    assert attributes["gen_ai.provider.name"] == provider_name


def test_the_processor_keeps_only_the_non_llm_keys() -> None:
    provider, exporter = _pipeline(active=True)
    _chat_span(provider)
    (span,) = exporter.get_finished_spans()
    assert dict(span.attributes or {}) == {
        "gen_ai.provider.name": "anthropic",
        "gen_ai.conversation.id": "conv_1",
        "server.address": "api.anthropic.com",
        "anthropic.request.service_tier": "auto",
        "http.response.status_code": 200,
    }


def test_the_processor_does_nothing_unless_openinference_is_ours() -> None:
    provider, exporter = _pipeline(active=False)
    _chat_span(provider)
    (span,) = exporter.get_finished_spans()
    assert (span.attributes or {})["gen_ai.request.model"] == "claude-sonnet-5"


@pytest.mark.parametrize("operation", ["execute_tool", "invoke_agent"])
def test_the_sdk_tool_and_agent_spans_are_left_alone(operation: str) -> None:
    provider, exporter = _pipeline(active=True)
    _chat_span(provider, **{"gen_ai.operation.name": operation})
    (span,) = exporter.get_finished_spans()
    assert (span.attributes or {})["gen_ai.request.model"] == "claude-sonnet-5"


@pytest.mark.parametrize(
    "scope", ["openinference.instrumentation.anthropic", "rius", "com.anthropic"]
)
def test_a_chat_span_from_another_scope_is_left_alone(scope: str) -> None:
    provider, exporter = _pipeline(active=True)
    _chat_span(provider, scope=scope)
    (span,) = exporter.get_finished_spans()
    assert (span.attributes or {})["gen_ai.request.model"] == "claude-sonnet-5"


def test_a_failing_check_never_reaches_the_caller() -> None:
    def broken() -> bool:
        raise RuntimeError("boom")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(NativeChatSpanDemoter(broken))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _chat_span(provider)
    (span,) = exporter.get_finished_spans()
    assert (span.attributes or {})["gen_ai.request.model"] == "claude-sonnet-5"


def test_a_chat_span_seen_only_at_its_end_is_demoted_from_its_own_attributes() -> None:
    # Decided again at on_end from the span's own scope and operation, so a
    # copy whose start this processor never saw is demoted all the same.
    finished = InMemorySpanExporter()
    elsewhere = TracerProvider()
    elsewhere.add_span_processor(SimpleSpanProcessor(finished))
    _chat_span(elsewhere)
    (span,) = finished.get_finished_spans()
    NativeChatSpanDemoter(lambda: True).on_end(span)
    attributes = dict(span.attributes or {})
    assert "gen_ai.request.model" not in attributes
    assert "gen_ai.usage.input_tokens" not in attributes
    assert attributes["gen_ai.provider.name"] == "anthropic"


def test_a_tool_span_seen_only_at_its_end_is_left_alone() -> None:
    finished = InMemorySpanExporter()
    elsewhere = TracerProvider()
    elsewhere.add_span_processor(SimpleSpanProcessor(finished))
    _chat_span(elsewhere, **{"gen_ai.operation.name": "execute_tool"})
    (span,) = finished.get_finished_spans()
    NativeChatSpanDemoter(lambda: True).on_end(span)
    assert (span.attributes or {})["gen_ai.request.model"] == "claude-sonnet-5"
