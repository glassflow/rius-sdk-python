"""Test agent for heartbeat steering: a docs agent with a silently broken tool.

``search_docs`` always returns zero results (no error), and the system prompt
tells the model to keep searching, so the agent loops on it: a tool-repetition
silent failure. ``lookup_index`` works. An advisory from Rius arrives on a
heartbeat and is shown to the model as a mid-conversation system message
before its next call. The question is whether the model switches tools.

    # baseline: no advisory
    uv run python spike/heartbeat_advisories/agent.py
    # the script plays the detector: after N empty searches it queues an
    # advisory on the fake receiver (or on the dev cluster, by hand)
    uv run python spike/heartbeat_advisories/agent.py --auto-advise 3

Env: RIUS_ENDPOINT (fake receiver or port-forwarded receiver), RIUS_API_KEY,
RIUS_ADVISORY_WORKSPACE (real receiver: write advisories to NATS KV),
ANTHROPIC_API_KEY, MODEL (default claude-opus-5-5), EFFORT (default low).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
import urllib.request
from typing import Any

import anthropic
from opentelemetry import trace

import rius
from rius import SpanKind

MODEL = os.environ.get("MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("EFFORT", "low")
MAX_TURNS = 12
TOOL_LATENCY_S = 3.0

ADVISORY_TEXTS = {
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
ADVISORY_TEXT = ADVISORY_TEXTS[os.environ.get("ADVISORY_VARIANT", "slug")]

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


def queue_advisory(endpoint: str, n: int) -> None:
    """Stand-in for the detector: queue an advisory for this instance.

    With RIUS_ADVISORY_WORKSPACE set, write it into the real receiver's
    ADVISORIES bucket with the nats CLI (NATS_URL picks the server);
    otherwise ask the fake receiver.
    """
    workspace = os.environ.get("RIUS_ADVISORY_WORKSPACE")
    if workspace:
        resource = trace.get_tracer_provider().resource  # type: ignore[attr-defined]
        instance = resource.attributes["service.instance.id"]
        value = json.dumps(
            [{"id": f"a-{int(time.time())}", "code": "CF-01", "text": ADVISORY_TEXT.format(n=n)}]
        )
        subprocess.run(
            ["nats", "kv", "put", "ADVISORIES", f"{workspace}.{instance}", value],
            check=True,
            capture_output=True,
        )
        log(f"[detector] put advisory in NATS KV after {n} empty searches")
        return
    body = json.dumps({"code": "tool-repetition", "text": ADVISORY_TEXT.format(n=n)})
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/admin/advisory", data=body.encode(), method="POST"
    )
    urllib.request.urlopen(request, timeout=3).close()
    log(f"[detector] queued advisory after {n} empty searches")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-advise", type=int, metavar="N", default=0)
    opts = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("rius").setLevel(logging.INFO)
    rius.init(agent_name="docs-agent", heartbeat_interval=5)

    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "How many times are failed payment webhooks retried?"}
    ]
    calls: list[tuple[int, str]] = []  # (turn, tool name)
    advised_at: int | None = None
    queued = False

    with rius.start_as_current_span("docs-agent", kind=SpanKind.AGENT, agent_name="docs-agent"):
        for turn in range(1, MAX_TURNS + 1):
            for advisory in rius.drain_advisories():
                # The advisory is an operator note, not user input: the
                # mid-conversation system role is the channel the model
                # treats that way, and it leaves the cached prefix intact.
                messages.append({"role": "system", "content": advisory["text"]})
                trace.get_current_span().add_event(
                    "rius.advisory.delivered",
                    {"rius.advisory.id": advisory["id"], "rius.advisory.code": advisory["code"]},
                )
                advised_at = advised_at or turn
                log(f"[turn {turn}] advisory injected: {advisory['text']}")

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
                break
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                text = " ".join(b.text for b in response.content if b.type == "text")
                log(f"[turn {turn}] final answer: {text}")
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
            if opts.auto_advise and not queued and empty >= opts.auto_advise:
                queue_advisory(os.environ["RIUS_ENDPOINT"], empty)
                queued = True

    searches_after = sum(
        1 for t, n in calls if n == "search_docs" and advised_at and t >= advised_at
    )
    switched = next((t for t, n in calls if n == "lookup_index"), None)
    log(
        f"SUMMARY model={MODEL} effort={EFFORT} "
        f"variant={os.environ.get('ADVISORY_VARIANT', 'slug')} "
        f"turns={turn} advised_at_turn={advised_at} "
        f"search_docs_total={sum(1 for _, n in calls if n == 'search_docs')} "
        f"search_docs_after_advisory={searches_after} first_lookup_index_turn={switched}"
    )
    rius.drain_advisories()


if __name__ == "__main__":
    main()
