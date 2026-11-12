"""Test agent for hints: a docs agent with a silently broken tool.

``search_docs`` always returns zero results (no error), and the system prompt
tells the model to keep searching, so the agent loops on it: a tool-repetition
silent failure. ``lookup_index`` works. A hint from Rius arrives on a
heartbeat, and the agent drains it before its next model call and shows its
text to the model as a mid-conversation system message. The question is
whether the model switches tools.

    # the fake receiver, in another shell
    uv run python spike/hints/fake_receiver.py
    # baseline: no hint
    RIUS_ENDPOINT=http://127.0.0.1:8787 uv run python spike/hints/agent.py
    # the script plays the detector: after N empty searches it queues a hint
    # for this run, on the fake receiver (or in the real hint bucket, below)
    RIUS_ENDPOINT=http://127.0.0.1:8787 uv run python spike/hints/agent.py --auto-hint 3

Against a real receiver with hints on, hints normally come from the
alert-manager's detectors and ``--auto-hint`` is left off. To hand-write one
instead, set RIUS_HINT_WORKSPACE (and NATS_URL for the nats CLI): the script
then puts the hint in the ``argus_hints`` bucket itself. That put overwrites
the instance's pending hints and skips the writer's compare-and-set, which is
fine for a test instance and nothing else.

The default scenario changes its query from search to search, so it does not
meet CF-01's identical-arguments condition. ``--scenario identical`` is the
variant the real detector needs (RIUS-1727): search_docs is called again and
again with the same query, each call a TOOL span with its arguments captured,
sequential and successful, while the run stays open. Run it against a real
stack (receiver, sink, alert-manager with the tool_loop rule and the hint
writer), with no ``--auto-hint``: the hint then comes from the real detector
and writer.

    RIUS_ENDPOINT=... RIUS_API_KEY=... MODEL=scripted \
        uv run python spike/hints/agent.py --scenario identical --session s-1 --hold 120

``MODEL=scripted`` replaces the model with a fixed script and makes no API
call: it searches until a hint arrives, then looks the page up and answers;
with no hint it searches until ``--hold`` seconds after its first call and
gives up. Its model calls are LLM spans with the conversation captured (or,
with ``--no-transcript``, none), as an instrumented provider records them.
Scripted runs validate transport, detection, assignment and the measurement
report; they say nothing about whether a model follows a hint.

Telemetry faults, for the measurement's coverage checks: ``--duplicate-export``
sends every batch twice (an exporter retry after a lost response);
``--late-after A --late-by B`` holds every batch exported from A seconds after
start for B seconds (late telemetry). ``--runs N`` runs N runs one after the
other in the same session, so a session keeps its arm across runs.

Hints need the run's root stored while it is open, so every task turns
partial spans on; with them off the writer finds no open run to address.

The last line printed is a JSON summary, for collecting repeated runs. It is
printed for every run, including one that failed, with its outcome (answer,
max_turns, refusal or error) and whether the answer was correct.

Env: RIUS_ENDPOINT, RIUS_API_KEY, RIUS_HINT_WORKSPACE, ANTHROPIC_API_KEY,
MODEL (default claude-opus-5-5; ``scripted`` for no model), EFFORT (default
low), HINT_VARIANT (slug, tool or vague; default slug).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import cycle
from typing import Any

import anthropic
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from tasks import TASKS, WEBHOOK_QUESTION, Task

import rius
from rius import SpanKind
from rius.client import build_span_exporter

MODEL = os.environ.get("MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("EFFORT", "low")
MAX_TURNS = 12
TOOL_LATENCY_S = 3.0
HINT_CODE = "CF-TEST"

HINT_TEXTS = {
    # evidence + an alternative tool + an example slug (which happens to be
    # the page holding the answer)
    "slug": (
        "Rius monitoring: search_docs has returned 0 results {n} times in this run, for "
        "different queries. It looks broken, not empty. lookup_index can find the "
        "same pages by topic slug (for example 'payments/webhooks')."
    ),
    # evidence + an alternative tool, no hint about where the answer is
    "tool": (
        "Rius monitoring: search_docs has returned 0 results {n} times in this run, for "
        "different queries. It looks broken, not empty. lookup_index can find the "
        "same pages by topic slug."
    ),
    # no evidence, no alternative
    "vague": "Rius monitoring: something may be wrong with this run.",
}
HINT_VARIANT = os.environ.get("HINT_VARIANT", "slug")
HINT_TEXT = HINT_TEXTS[HINT_VARIANT]

SYSTEM = (
    "You answer questions about our payments platform from the internal docs. "
    "Use search_docs to find information. The docs are known to contain the answer, "
    "so if a search finds nothing, rephrase the query and search again."
)

# The scripted model's pause before each call, in seconds. Irregular, so the
# loop is no polling; --regular uses a fixed pause, which is.
IRREGULAR_GAPS_S = (2.0, 5.5, 3.0, 8.0, 4.0, 6.5, 2.5, 7.0)
REGULAR_GAP_S = 5.0

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": "Full-text search over the internal documentation. The main way to "
        "find docs.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "lookup_index",
        "description": "Fetch a page from the legacy docs index by exact topic slug, such as "
        "'billing/invoices'.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

PAGES = {
    "payments/webhooks": "Webhooks: failed deliveries are retried up to 7 times with "
    "exponential backoff, starting at 30 seconds.",
}


def run_tool(name: str, args: dict[str, Any], latency: float = TOOL_LATENCY_S) -> str:
    time.sleep(latency)  # keeps the run long enough for heartbeats to land
    if name == "search_docs":
        return json.dumps({"results": [], "total": 0})  # the silent failure
    if name == "lookup_index":
        page = PAGES.get(args["topic"].strip("/ ").lower())
        return page or json.dumps({"error": f"no page for topic {args['topic']!r}"})
    return json.dumps({"error": f"unknown tool {name}"})


def current_trace_id() -> str:
    return format(trace.get_current_span().get_span_context().trace_id, "032x")


def queue_hint(endpoint: str, n: int) -> None:
    """Stand-in for the detector: queue a hint for this run.

    With RIUS_HINT_WORKSPACE set, put it in the real receiver's argus_hints
    bucket with the nats CLI; otherwise ask the fake receiver.
    """
    text = HINT_TEXT.format(n=n)
    workspace = os.environ.get("RIUS_HINT_WORKSPACE")
    if workspace:
        resource = trace.get_tracer_provider().resource  # type: ignore[attr-defined]
        instance = resource.attributes["service.instance.id"]
        now = datetime.now(timezone.utc)
        hint = {
            "id": hint_id(now),
            "code": HINT_CODE,
            "template_version": 1,
            "text": text,
            "trace_id": current_trace_id(),
            "created_at": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "expires_at": (now + timedelta(minutes=10))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
        }
        put = subprocess.run(
            [
                "nats",
                "kv",
                "put",
                "argus_hints",
                f"hint.{workspace}.{instance}",
                json.dumps({"hints": [hint]}),
            ],
            capture_output=True,
            text=True,
        )
        if put.returncode != 0:
            raise RuntimeError(f"nats kv put failed: {put.stderr.strip()}")
        log(f"[detector] put hint {hint['id']} in argus_hints after {n} empty searches")
        return
    body = json.dumps({"code": HINT_CODE, "text": text, "trace_id": current_trace_id()})
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/admin/hint", data=body.encode(), method="POST"
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        queued = json.load(response)
    log(f"[detector] queued hint {queued['id']} after {n} empty searches")


def hint_id(now: datetime) -> str:
    """h_ followed by a ULID, as the hint writer mints them."""
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    n = (int(now.timestamp() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    return "h_" + "".join(alphabet[(n >> shift) & 31] for shift in range(125, -1, -5))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class DuplicatingExporter(SpanExporter):
    """Sends every batch twice, as an exporter retry after a lost response does."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        first = self._inner.export(spans)
        self._inner.export(spans)
        return first

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


class LateExporter(SpanExporter):
    """Holds every batch exported from ``after`` seconds on for ``by`` seconds."""

    def __init__(self, inner: SpanExporter, after: float, by: float) -> None:
        self._inner, self._by = inner, by
        self._from = time.monotonic() + after
        self._held: list[tuple[float, Sequence[ReadableSpan]]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._flush_due, daemon=True)
        self._thread.start()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        now = time.monotonic()
        if now < self._from:
            return self._inner.export(spans)
        with self._lock:
            self._held.append((now + self._by, list(spans)))
        return SpanExportResult.SUCCESS

    def _flush_due(self, force: bool = False) -> None:
        while True:
            with self._lock:
                due = [b for b in self._held if force or b[0] <= time.monotonic()]
                self._held = [b for b in self._held if b not in due]
            for _, batch in due:
                self._inner.export(batch)
            if force or self._stop.wait(1.0):
                return

    def shutdown(self) -> None:
        # Exit waits for the held batches: late, not lost.
        with self._lock:
            wait = max((at for at, _ in self._held), default=0.0) - time.monotonic()
        if wait > 0:
            log(f"[export] waiting {wait:.0f}s for held batches")
            time.sleep(wait)
        self._stop.set()
        self._flush_due(force=True)
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


class DroppingExporter(SpanExporter):
    """Never exports the k-th span of a tool, nor any snapshot of it: a lost call."""

    def __init__(self, inner: SpanExporter, tool: str, k: int) -> None:
        self._inner, self._tool, self._k = inner, tool, k
        self._seen: list[int] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        kept = []
        for span in spans:
            if span.name == self._tool and span.context is not None:
                sid = span.context.span_id
                if sid not in self._seen:
                    self._seen.append(sid)
                if self._seen.index(sid) + 1 == self._k:
                    continue
            kept.append(span)
        return self._inner.export(kept) if kept else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reply:
    stop_reason: str
    content: list[Block]


class ScriptedModel:
    """A task's fixed script in place of a model, with no API call.

    It makes the task's opening calls, then repeats its loop call until a hint
    is in the conversation, then makes the resolving call and answers; with
    no hint it gives up ``hold`` seconds after the run's first call. It reads
    the conversation the way a model gets it, so the same script serves the
    in-process model (MODEL=scripted) and the stub API (MODEL=stub).
    """

    def __init__(self, task: Task, hold: float, loop_calls: int = 0, idle: float = 0.0) -> None:
        self._task, self._hold, self._n = task, hold, 0
        self._loop_calls, self._idle = loop_calls, idle
        self._first_call: float | None = None

    def reply(self, messages: list[Any]) -> Reply:
        self._n += 1
        if len(messages) == 1:  # a new run
            self._first_call = None
        uses = [
            b
            for m in messages
            if _field(m, "role") == "assistant" and isinstance(_field(m, "content"), list)
            for b in _field(m, "content")
            if _field(b, "type") == "tool_use"
        ]
        calls = [_field(b, "name") for b in uses]
        looped = sum(
            1 for b in uses[len(self._task.opening) :] if _field(b, "name") == self._task.loop[0]
        )
        hinted = any(_field(m, "role") == "system" for m in messages)
        t = self._task
        if t.resolve[0] in calls:
            return Reply("end_turn", [Block("text", text=t.answer)])
        if len(calls) < len(t.opening):
            return self._call(t.opening[len(calls)])
        if hinted:
            return self._call(t.resolve)
        now = time.monotonic()
        if self._first_call is None:
            self._first_call = now
        if now - self._first_call >= self._hold or (
            self._loop_calls and looped >= self._loop_calls
        ):
            # The run stays open while it "thinks": a loop that stops is
            # not a run that ends at once.
            time.sleep(self._idle)
            return Reply("end_turn", [Block("text", text=t.give_up)])
        return self._call(t.loop)

    def _call(self, call: tuple[str, dict[str, Any]]) -> Reply:
        name, args = call
        return Reply("tool_use", [Block("tool_use", id=f"call_{self._n}", name=name, input=args)])


def _field(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def start_stub(model: ScriptedModel) -> str:
    """Serve the script as the Messages API on localhost; returns its base URL.

    The real Anthropic client and its instrumentation run against it, so the
    LLM spans are the ones a paid run records, at no cost.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server's name
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = model.reply(body["messages"])
            content = [
                {"type": "text", "text": b.text}
                if b.type == "text"
                else {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                for b in reply.content
            ]
            out = json.dumps(
                {
                    "id": f"msg_stub_{time.monotonic_ns()}",
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model", "stub"),
                    "content": content,
                    "stop_reason": reply.stop_reason,
                    "stop_sequence": None,
                    "usage": {"input_tokens": len(json.dumps(body)) // 4, "output_tokens": 40},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def as_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation as OpenAI-style messages, the shape the SDK records."""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            out.append(reply_message(content))
            continue
        for r in content:  # tool results
            out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": r["content"]})
    return out


def reply_message(content: list[Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": " ".join(b.text for b in content if b.type == "text"),
    }
    calls = [
        {
            "id": b.id,
            "type": "function",
            "function": {"name": b.name, "arguments": json.dumps(b.input)},
        }
        for b in content
        if b.type == "tool_use"
    ]
    if calls:
        message["tool_calls"] = calls
    return message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-hint", type=int, metavar="N", default=0)
    parser.add_argument("--scenario", choices=("search", *TASKS), default="search")
    parser.add_argument("--session", help="session id every run is scoped to")
    parser.add_argument("--runs", type=int, default=1, help="runs, one after the other")
    parser.add_argument(
        "--hold", type=float, default=120.0, help="scripted/stub: seconds an unhinted run loops"
    )
    parser.add_argument("--regular", action="store_true", help="scripted/stub: fixed pauses")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS, help="a real model's turns")
    parser.add_argument(
        "--no-transcript", action="store_true", help="scripted: capture no messages"
    )
    parser.add_argument("--duplicate-export", action="store_true")
    parser.add_argument("--late-after", type=float, default=0.0)
    parser.add_argument("--late-by", type=float, default=0.0)
    parser.add_argument(
        "--loop-calls",
        type=int,
        default=0,
        help="scripted/stub: give up after this many loop calls",
    )
    parser.add_argument(
        "--idle", type=float, default=0.0, help="scripted/stub: seconds the run stays open after"
    )
    parser.add_argument(
        "--drop-call", type=int, default=0, help="never export the K-th loop-tool span (lost)"
    )
    parser.add_argument("--fail-first", type=int, default=0, help="the first K loop calls error")
    parser.add_argument("--fail-every", type=int, default=0, help="every K-th loop call errors")
    parser.add_argument("--no-hints", action="store_true", help="the SDK does not take hints")
    opts = parser.parse_args()
    scripted = MODEL in ("scripted", "stub")
    if scripted and opts.scenario == "search":
        parser.error(f"MODEL={MODEL} plays a task, not the search scenario")

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("rius").setLevel(logging.INFO)
    exporter: SpanExporter | None = None
    task = TASKS.get(opts.scenario)
    if opts.duplicate_export or opts.late_by or opts.drop_call:
        exporter = build_span_exporter(rius.resolve_config())
        if opts.drop_call:
            exporter = DroppingExporter(
                exporter, task.loop[0] if task else "search_docs", opts.drop_call
            )
        if opts.duplicate_export:
            exporter = DuplicatingExporter(exporter)
        if opts.late_by:
            exporter = LateExporter(exporter, opts.late_after, opts.late_by)
    sdk = rius.init(
        agent_name="docs-agent",
        heartbeat_interval=5,
        hints=not opts.no_hints,
        partial_spans=opts.scenario in TASKS or None,
        span_exporter=exporter,
    )

    summaries = []
    with rius.session(opts.session) if opts.session else nullcontext():
        for run in range(1, opts.runs + 1):
            summaries.append(run_once(opts, run, scripted))
    sdk.flush()
    for summary in summaries:
        print(json.dumps(summary), flush=True)


def run_once(opts: argparse.Namespace, run: int, scripted: bool) -> dict[str, Any]:
    task = TASKS.get(opts.scenario)
    system = task.system if task else SYSTEM
    tools = task.tools if task else TOOLS
    question = task.question if task else WEBHOOK_QUESTION
    model = (
        ScriptedModel(task, opts.hold, opts.loop_calls, opts.idle) if scripted and task else None
    )
    client = None
    if MODEL == "stub":
        client = anthropic.Anthropic(base_url=start_stub(model), api_key="stub")  # type: ignore[arg-type]
        model = None
    elif not scripted:
        client = anthropic.Anthropic()
    gaps = iter(cycle((REGULAR_GAP_S,) if opts.regular else IRREGULAR_GAPS_S))
    max_turns = 200 if scripted else opts.max_turns
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    calls: list[tuple[int, str]] = []  # (turn, tool name)
    hint_ids: list[str] = []
    hinted_at: int | None = None
    queued_at: int | None = None
    searches_at_queue: int | None = None
    queue_error: str | None = None
    trace_id = ""
    input_tokens = cache_read_tokens = output_tokens = 0
    loop_n, failed = 0, False
    # answer, max_turns, refusal or error; correct says whether the answer has
    # the retry count from the docs page.
    outcome, answer, turn = "max_turns", "", 0

    try:
        with rius.start_as_current_span(
            "docs-agent", kind=SpanKind.AGENT, agent_name="docs-agent"
        ) as run_span:
            trace_id = current_trace_id()
            for turn in range(1, max_turns + 1):
                # drain_hints() records rius.hint.delivered on the run's root span.
                for hint in rius.drain_hints():
                    # The hint is an operator note, not user input: the
                    # mid-conversation system role is the channel the model
                    # treats that way, and it leaves the cached prefix intact.
                    messages.append({"role": "system", "content": hint.text})
                    hint_ids.append(hint.id)
                    hinted_at = hinted_at or turn
                    log(f"[run {run} turn {turn}] hint {hint.id} injected: {hint.text}")

                if model is not None:
                    response: Any = scripted_call(model, messages, opts.no_transcript)
                else:
                    response = client.beta.messages.create(  # type: ignore[union-attr]
                        model=MODEL,
                        max_tokens=16000,
                        system=system,
                        tools=tools,  # type: ignore[arg-type]
                        messages=messages,  # type: ignore[arg-type]
                        output_config={"effort": EFFORT},  # type: ignore[typeddict-item]
                        betas=["server-side-fallback-2026-07-01"],
                        fallbacks="default",
                    )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    input_tokens += (usage.input_tokens or 0) + (
                        getattr(usage, "cache_creation_input_tokens", 0) or 0
                    )
                    cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
                    output_tokens += usage.output_tokens or 0
                if response.stop_reason == "refusal":
                    log(f"[run {run} turn {turn}] refusal: {response.stop_details}")
                    outcome = "refusal"
                    break
                messages.append({"role": "assistant", "content": response.content})
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if response.stop_reason != "tool_use" or not tool_uses:
                    answer = " ".join(b.text for b in response.content if b.type == "text")
                    log(f"[run {run} turn {turn}] final answer: {answer}")
                    outcome = "answer"
                    # The run's output on its root: what tells a run that
                    # ended with an answer from one that ended without.
                    run_span.set_output(answer)
                    break

                results = []
                for block in tool_uses:
                    args = dict(block.input)  # type: ignore[arg-type]
                    log(f"[run {run} turn {turn}] {block.name}({json.dumps(args)})")
                    calls.append((turn, block.name))
                    # One TOOL span per call, its arguments captured: what
                    # the tool_loop detector groups calls by.
                    with rius.start_as_current_span(
                        block.name,
                        kind=SpanKind.TOOL,
                        input=args,
                        tool_name=block.name,
                        tool_call_id=block.id,
                    ) as span:
                        latency = next(gaps) if scripted else TOOL_LATENCY_S
                        if task:
                            time.sleep(latency)
                            if (block.name, args) == task.loop:
                                loop_n += 1
                                if (opts.fail_first and loop_n <= opts.fail_first) or (
                                    opts.fail_every and loop_n % opts.fail_every == 0
                                ):
                                    failed = True
                            result = task.run(block.name, args)
                        else:
                            result = run_tool(block.name, args, latency)
                        span.set_output(result)
                        if failed:
                            # An error status on the span: a retry, which
                            # exclude_retries leaves out of the loop.
                            span.record_exception(TimeoutError("upstream timeout"))
                            result = json.dumps({"error": "upstream timeout"})
                            failed = False
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )
                messages.append({"role": "user", "content": results})

                looped = task.loop[0] if task else "search_docs"
                empty = sum(1 for _, name in calls if name == looped)
                if opts.auto_hint and queued_at is None and empty >= opts.auto_hint:
                    # The hint reaches the model a ping or more later, after
                    # more searches: queued_at_turn and search_docs_before_hint
                    # tell the two apart.
                    queued_at, searches_at_queue = turn, empty
                    try:
                        queue_hint(os.environ["RIUS_ENDPOINT"], empty)
                    except Exception as exc:  # noqa: BLE001 — the run goes on, unhinted
                        queue_error = f"{type(exc).__name__}: {exc}"
                        log(f"[detector] could not queue the hint: {queue_error}")
    except Exception as exc:  # noqa: BLE001 — every run ends with a summary line
        outcome = "error"
        log(f"[run {run} turn {turn}] run failed: {type(exc).__name__}: {exc}")
    searches = [t for t, n in calls if n == "search_docs"]
    return {
        "model": MODEL,
        "effort": None if scripted else EFFORT,
        "scenario": opts.scenario,
        "session": opts.session,
        "run": run,
        "trace_id": trace_id,
        "variant": HINT_VARIANT if opts.auto_hint else None,
        "outcome": outcome,
        "correct": outcome == "answer" and (task.correct(answer) if task else "7" in answer),
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "calls": {
            name: sum(1 for _, n in calls if n == name) for name in sorted({n for _, n in calls})
        },
        "turns": turn,
        "queued_at_turn": queued_at,
        "search_docs_at_queue": searches_at_queue,
        "queue_error": queue_error,
        "hint_ids": hint_ids,
        "hinted_at_turn": hinted_at,
        "search_docs_total": len(searches),
        "search_docs_before_hint": (
            sum(1 for t in searches if t < hinted_at) if hinted_at else None
        ),
        "search_docs_after_hint": (
            sum(1 for t in searches if t >= hinted_at) if hinted_at else None
        ),
        "first_lookup_index_turn": next((t for t, n in calls if n == "lookup_index"), None),
    }


def scripted_call(
    model: ScriptedModel, messages: list[dict[str, Any]], no_transcript: bool
) -> Reply:
    """One scripted model call, recorded as an LLM span."""
    with rius.start_as_current_generation(
        "chat scripted", model="scripted", provider="scripted"
    ) as generation:
        if not no_transcript:
            generation.set_input(as_messages(messages))
        response = model.reply(messages)
        if not no_transcript:
            generation.set_output([reply_message(response.content)])
        return response


if __name__ == "__main__":
    main()
