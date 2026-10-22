"""The anthropic SDK's own chat span, next to OpenInference's, counts once.

From 1.14.0 the anthropic SDK starts its own CLIENT span per API call (scope
``com.anthropic.sdk.python``) on the OpenTelemetry global provider. Under a
global ``init()`` with the OpenInference Anthropic instrumentor, that span
arrived beside OpenInference's: two spans normalized as LLM per call, tokens
and cost counted twice, a second ``error.type`` spelled ``not_found_error``,
and an unflagged duplicate after ``mark_compaction()``. When rius registered
that instrumentor, the SDK's chat span is exported as a plain client span.

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
_CASES = ("create", "create_streamed", "stream_helper", "not_found")
_SECRETS = ("secret-prompt-text", "secret-reply-text", "secret-system-text")
_KEPT_GEN_AI = {"gen_ai.provider.name", "gen_ai.conversation.id"}


@cache
def _run(
    instruments: tuple[str, ...] = ("anthropic",),
    content_mode: bool = False,
    partial_spans: bool = False,
) -> dict[str, Any]:
    env = {**os.environ}
    if content_mode:
        env["ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE"] = "content"
    options = {"instruments": list(instruments), "partial_spans": partial_spans}
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


def _has_native_tracing() -> bool:
    return bool(_native(_run()["cases"]["create"]))


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
def test_a_provider_404_has_one_error_type_the_exception_class() -> None:
    spans = _run()["cases"]["not_found"]
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
    if not _has_native_tracing():
        pytest.skip("this anthropic has no tracing of its own (it starts in 1.14.0)")
    spans = _run()["cases"][case]
    native = _native(spans)
    assert native, "the SDK's span is kept, demoted, not dropped"
    for span in native:
        attributes = span["attributes"]
        assert span["kind"] == "CLIENT"
        assert "openinference.span.kind" not in attributes
        # argus-core badges a span llm on the model and prices it on model + usage.
        assert "gen_ai.request.model" not in attributes
        assert not [k for k in attributes if k.startswith("gen_ai.usage.")]
        assert {k for k in attributes if k.startswith("gen_ai.")} <= _KEPT_GEN_AI
        assert attributes["gen_ai.provider.name"] == "anthropic"
        assert attributes["server.address"] == "127.0.0.1"
        assert "http.response.status_code" in attributes


@pytest.mark.integration
def test_the_demoted_404_keeps_its_http_failure() -> None:
    if not _has_native_tracing():
        pytest.skip("this anthropic has no tracing of its own (it starts in 1.14.0)")
    (native,) = _native(_run()["cases"]["not_found"])
    assert native["attributes"]["http.response.status_code"] == 404
    assert native["attributes"]["error.type"] == "not_found_error"
    assert native["status"] == "ERROR"


@pytest.mark.integration
def test_the_stream_helper_span_in_a_trace_of_its_own_is_demoted_too() -> None:
    # The SDK's span for messages.stream() has no parent: it is not under
    # OpenInference's span, so nothing keyed on parentage can catch it.
    if not _has_native_tracing():
        pytest.skip("this anthropic has no tracing of its own (it starts in 1.14.0)")
    spans = _run()["cases"]["stream_helper"]
    (native,) = _native(spans)
    assert native["parent_id"] is None
    assert "gen_ai.request.model" not in native["attributes"]


@pytest.mark.integration
def test_no_content_survives_on_the_demoted_span() -> None:
    if not _has_native_tracing():
        pytest.skip("this anthropic has no tracing of its own (it starts in 1.14.0)")
    run = _run(content_mode=True)
    for case in _CASES:
        for span in _native(run["cases"][case]):
            text = json.dumps([span["attributes"], span["events"]])
            assert not [s for s in _SECRETS if s in text], (case, span["attributes"])
    # The OpenInference span still carries the conversation.
    assert "secret-prompt-text" in json.dumps(_llm(run["cases"]["create"])[0]["attributes"])


@pytest.mark.integration
def test_the_pending_snapshot_of_the_sdk_span_is_not_an_llm_span() -> None:
    if not _has_native_tracing():
        pytest.skip("this anthropic has no tracing of its own (it starts in 1.14.0)")
    spans = _run(partial_spans=True)["cases"]["create"]
    pending = [s for s in _native(spans) if s["attributes"].get("rius.span.pending")]
    assert pending
    for span in pending:
        assert "openinference.span.kind" not in span["attributes"]
        assert "gen_ai.request.model" not in span["attributes"]


@pytest.mark.integration
def test_without_openinference_the_sdk_span_stays_the_llm_span() -> None:
    if not _has_native_tracing():
        pytest.skip("this anthropic has no tracing of its own (it starts in 1.14.0)")
    run = _run(instruments=())
    for case in _CASES:
        (llm,) = _llm(run["cases"][case])
        assert llm["scope"] == _SCOPE
        assert llm["attributes"]["gen_ai.request.model"]


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
