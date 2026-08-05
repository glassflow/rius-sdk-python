"""Reliable export pipeline.

The SDK must never block or crash the host application: span creation stays
fast when the exporter is slow, backend failures never raise into app code,
transient errors are retried, pending spans are flushed at interpreter exit,
and a broken user mask fails closed without dropping the batch.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rius import init


class _SlowExporter(SpanExporter):
    """Blocks each export until shutdown (bounded), like a hung backend."""

    def __init__(self) -> None:
        self._stop = threading.Event()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self._stop.wait(0.5)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def shutdown(self) -> None:
        self._stop.set()


class _FailingExporter(SpanExporter):
    """Raises on every export, like an unreachable backend."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise ConnectionError("backend down")

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def shutdown(self) -> None:
        pass


class _OtlpHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        server: Any = self.server
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        server.requests.append(self.path)
        status = server.responses.pop(0) if server.responses else 200
        self.send_response(status)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def _start_server(responses: list[int]) -> Any:
    server: Any = HTTPServer(("127.0.0.1", 0), _OtlpHandler)
    server.responses = list(responses)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_span_creation_is_nonblocking_when_exporter_is_slow() -> None:
    client = init(span_exporter=_SlowExporter(), set_global=False)
    tracer = client.get_tracer()
    started = time.perf_counter()
    for i in range(100):
        with tracer.start_as_current_span(f"op{i}"):
            pass
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"span creation blocked on export: {elapsed:.2f}s for 100 spans"
    client.shutdown()


def test_backend_down_never_raises_into_app_code() -> None:
    client = init(span_exporter=_FailingExporter(), set_global=False)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.flush()  # must complete despite the exporter raising
    client.shutdown()


def test_shutdown_drains_pending_spans() -> None:
    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False)
    with client.get_tracer().start_as_current_span("op"):
        pass
    client.shutdown()  # no explicit flush()
    assert len(inner.get_finished_spans()) == 1


def test_mask_error_fails_closed_without_dropping_batch() -> None:
    def bad_mask(value: Any) -> Any:
        if value == "boom":
            raise ValueError("mask failed")
        return "***"

    inner = InMemorySpanExporter()
    client = init(span_exporter=inner, set_global=False, mask=bad_mask)
    with client.get_tracer().start_as_current_span("op") as span:
        span.set_attribute("input.value", "boom")
        span.set_attribute("output.value", "fine")
        span.set_attribute("gen_ai.request.model", "gpt-4o")
    client.flush()

    spans = inner.get_finished_spans()
    assert len(spans) == 1, "a failing mask must not drop the batch"
    attrs = spans[0].attributes
    assert attrs is not None
    assert "input.value" not in attrs, "unmaskable content must be dropped, not leaked"
    assert attrs["output.value"] == "***"
    assert attrs["gen_ai.request.model"] == "gpt-4o"


@pytest.mark.integration
def test_transient_server_errors_are_retried() -> None:
    server = _start_server(responses=[503, 200])
    client = init(endpoint=f"http://127.0.0.1:{server.server_port}", set_global=False)
    with client.get_tracer().start_as_current_span("op"):
        pass
    assert client.flush() is True
    client.shutdown()
    assert len(server.requests) >= 2, f"expected a retry, got {len(server.requests)} request(s)"


@pytest.mark.integration
def test_pending_spans_are_flushed_at_interpreter_exit() -> None:
    server = _start_server(responses=[200])
    code = (
        "import rius\n"
        f'client = rius.init(endpoint="http://127.0.0.1:{server.server_port}")\n'
        'with client.get_tracer().start_as_current_span("exit-op"):\n'
        "    pass\n"
        "# exit without flush() — the atexit hook must drain the queue\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
    assert len(server.requests) == 1
