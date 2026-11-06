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

The queries change from search to search, so this loop does not meet CF-01's
identical-arguments condition; a real detector run needs a variant that
repeats the same call.

The last line printed is a JSON summary, for collecting repeated runs. It is
printed for every run, including one that failed, with its outcome (answer,
max_turns, refusal or error) and whether the answer was correct.

Env: RIUS_ENDPOINT, RIUS_API_KEY, RIUS_HINT_WORKSPACE, ANTHROPIC_API_KEY,
MODEL (default claude-opus-5-5), EFFORT (default low), HINT_VARIANT (slug,
tool or vague; default slug).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

import anthropic
from opentelemetry import trace

import rius
from rius import SpanKind

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


def run_tool(name: str, args: dict[str, Any]) -> str:
    time.sleep(TOOL_LATENCY_S)  # keeps the run long enough for heartbeats to land
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-hint", type=int, metavar="N", default=0)
    opts = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("rius").setLevel(logging.INFO)
    rius.init(agent_name="docs-agent", heartbeat_interval=5, hints=True)

    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "How many times are failed payment webhooks retried?"}
    ]
    calls: list[tuple[int, str]] = []  # (turn, tool name)
    hint_ids: list[str] = []
    hinted_at: int | None = None
    queued_at: int | None = None
    searches_at_queue: int | None = None
    queue_error: str | None = None
    # answer, max_turns, refusal or error; correct says whether the answer has
    # the retry count from the docs page.
    outcome, answer, turn = "max_turns", "", 0

    try:
        with rius.start_as_current_span("docs-agent", kind=SpanKind.AGENT, agent_name="docs-agent"):
            for turn in range(1, MAX_TURNS + 1):
                # drain_hints() records rius.hint.delivered on the run's root span.
                for hint in rius.drain_hints():
                    # The hint is an operator note, not user input: the
                    # mid-conversation system role is the channel the model
                    # treats that way, and it leaves the cached prefix intact.
                    messages.append({"role": "system", "content": hint.text})
                    hint_ids.append(hint.id)
                    hinted_at = hinted_at or turn
                    log(f"[turn {turn}] hint {hint.id} injected: {hint.text}")

                response = client.beta.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM,
                    tools=TOOLS,  # type: ignore[arg-type]
                    messages=messages,  # type: ignore[arg-type]
                    output_config={"effort": EFFORT},  # type: ignore[typeddict-item]
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
                if response.stop_reason == "refusal":
                    log(f"[turn {turn}] refusal: {response.stop_details}")
                    outcome = "refusal"
                    break
                messages.append({"role": "assistant", "content": response.content})
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if response.stop_reason != "tool_use" or not tool_uses:
                    answer = " ".join(b.text for b in response.content if b.type == "text")
                    log(f"[turn {turn}] final answer: {answer}")
                    outcome = "answer"
                    break

                results = []
                for block in tool_uses:
                    args = dict(block.input)  # type: ignore[arg-type]
                    log(f"[turn {turn}] {block.name}({json.dumps(args)})")
                    calls.append((turn, block.name))
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": run_tool(block.name, args),
                        }
                    )
                messages.append({"role": "user", "content": results})

                empty = sum(1 for _, name in calls if name == "search_docs")
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
        log(f"[turn {turn}] run failed: {type(exc).__name__}: {exc}")
    finally:
        searches = [t for t, n in calls if n == "search_docs"]
        summary = {
            "model": MODEL,
            "effort": EFFORT,
            "variant": HINT_VARIANT if opts.auto_hint else None,
            "outcome": outcome,
            "correct": outcome == "answer" and "7" in answer,
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
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
