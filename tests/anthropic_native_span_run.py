"""Run Anthropic calls under a GLOBAL ``rius.init()`` and print the exported spans as JSON.

Run in a fresh interpreter by ``test_anthropic_native_span.py``: the
OpenTelemetry global tracer provider can be set once per process, and the
anthropic SDK's own spans go to that global provider. The suite's other tests
use ``init(set_global=False)``, where those spans never reach rius.

Usage: ``python anthropic_native_span_run.py '<json options>'`` with
``instruments`` (list), ``capture_content`` (bool), ``partial_spans`` (bool) and
``foreign`` (bool: another provider holds the global and rius bridges to it;
what that provider exports is printed under ``foreign``).
"""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import json
import sys
import threading
from typing import Any

import anthropic
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import rius

PROMPT = "secret-prompt-text"
REPLY = "secret-reply-text"
SYSTEM = "secret-system-text"

_MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": REPLY}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _sse() -> bytes:
    start = {**_MESSAGE, "content": [], "stop_reason": None}
    events = [
        ("message_start", {"type": "message_start", "message": start}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": REPLY},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events).encode()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        # Bedrock names the model in the path, not the body.
        if request.get("model") == "claude-nope":
            code, content_type = 404, "application/json"
            error = {"type": "not_found_error", "message": "model: claude-nope"}
            body = json.dumps({"type": "error", "error": error}).encode()
        elif request.get("stream"):
            code, content_type, body = 200, "text/event-stream", _sse()
        else:
            code, content_type, body = 200, "application/json", json.dumps(_MESSAGE).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def _span(span: ReadableSpan) -> dict[str, Any]:
    parent = span.parent
    return {
        "name": span.name,
        "scope": span.instrumentation_scope.name if span.instrumentation_scope else None,
        "kind": span.kind.name,
        "span_id": span.context.span_id,
        "trace_id": span.context.trace_id,
        "parent_id": parent.span_id if parent is not None else None,
        "attributes": {
            k: v if isinstance(v, str | int | float | bool) else list(v)
            for k, v in (span.attributes or {}).items()
        },
        "events": [
            {"name": e.name, "attributes": {k: str(v) for k, v in (e.attributes or {}).items()}}
            for e in span.events
        ],
        "status": span.status.status_code.name,
    }


def main() -> None:
    options = json.loads(sys.argv[1])
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    exporter = InMemorySpanExporter()
    foreign_exporter: InMemorySpanExporter | None = None
    if options.get("foreign"):
        # Another SDK claims the global first; rius bridges to it, so the
        # anthropic SDK's spans are that provider's, reaching rius as copies.
        foreign_exporter = InMemorySpanExporter()
        foreign = TracerProvider()
        foreign.add_span_processor(SimpleSpanProcessor(foreign_exporter))
        trace.set_tracer_provider(foreign)
    client = rius.init(
        span_exporter=exporter,
        instruments=options["instruments"],
        capture_content=options.get("capture_content", True),
        partial_spans=options.get("partial_spans", False),
        bridge_foreign_provider=bool(options.get("foreign")),
        heartbeat=False,
    )
    messages = anthropic.Anthropic(
        api_key="test-key", base_url=f"http://127.0.0.1:{server.server_port}", max_retries=0
    ).messages
    request: dict[str, Any] = {
        "model": "claude-sonnet-5",
        "max_tokens": 5,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": PROMPT}],
    }

    def create() -> None:
        messages.create(**request)

    def create_streamed() -> None:
        for _ in messages.create(stream=True, **request):
            pass

    def stream_helper() -> None:
        with messages.stream(**request) as stream:
            for _ in stream:
                pass

    def not_found() -> None:
        with contextlib.suppress(anthropic.NotFoundError):
            messages.create(**{**request, "model": "claude-nope"})

    def compaction() -> None:
        rius.mark_compaction()
        messages.create(**request)
        messages.create(**request)

    cases = {
        "create": create,
        "create_streamed": create_streamed,
        "stream_helper": stream_helper,
        "not_found": not_found,
        "compaction": compaction,
    }
    if importlib.util.find_spec("botocore") is not None:
        # The Bedrock client goes through the same Messages resource, so
        # OpenInference wraps it too; its own span has a platform scope
        # (com.anthropic.sdk.python.bedrock) and provider aws.bedrock.
        bedrock = anthropic.AnthropicBedrock(
            aws_access_key="AKIAEXAMPLE",
            aws_secret_key="test-secret",
            aws_region="us-east-1",
            base_url=f"http://127.0.0.1:{server.server_port}",
            max_retries=0,
        )

        def bedrock_create() -> None:
            bedrock.messages.create(**request)

        cases["bedrock"] = bedrock_create
    out: dict[str, Any] = {"anthropic": anthropic.__version__, "cases": {}, "foreign": {}}
    for name, call in cases.items():
        exporter.clear()
        if foreign_exporter is not None:
            foreign_exporter.clear()
        call()
        client.flush()
        out["cases"][name] = [_span(s) for s in exporter.get_finished_spans()]
        if foreign_exporter is not None:
            out["foreign"][name] = [_span(s) for s in foreign_exporter.get_finished_spans()]
    client.shutdown()
    server.shutdown()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
