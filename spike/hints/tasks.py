"""Tasks for the hints harness, each with a known answer.

``identical`` is the pipeline check: its prompt tells the model to repeat the
same search, so a loop is guaranteed and says nothing about real agents. The
other tasks never ask for a repeat. Each gives the model a tool with a flaw a
real integration has, which an agent may or may not loop on, and a second
tool that leads to the answer:

* ``orders``: the order listing's cursor never advances, so the next page is
  always the same page. The monthly summary has the answer.
* ``refund``: the refund status is stuck on ``processing``. The processor's
  event history shows it was rejected.
* ``docs``: full-text search returns nothing for any query. The legacy index
  has the page. A model usually rephrases, so most loops here are not
  identical calls: this task shows how often a natural loop does NOT meet
  CF-01.

``correct`` is each task's success criterion, checked on the final answer.
``loop``, ``resolve`` and the answers drive the scripted model (MODEL=scripted
or MODEL=stub); a real model gets only ``system``, ``question`` and ``tools``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Call = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class Task:
    name: str
    system: str
    question: str
    tools: list[dict[str, Any]]
    run: Callable[[str, dict[str, Any]], str]
    correct: Callable[[str], bool]
    # The scripted model: the opening calls, then ``loop`` until a hint
    # arrives or it gives up, then ``resolve`` and ``answer``.
    loop: Call
    resolve: Call
    answer: str
    give_up: str
    opening: list[Call] = field(default_factory=list)


def _tool(name: str, description: str, **props: dict[str, Any]) -> dict[str, Any]:
    required = [k for k, v in props.items() if not v.pop("optional", False)]
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False,
        },
    }


SUPPORT = (
    "You are a support assistant for an online store. Use the tools to look up "
    "what you need, then answer the support agent's question. Be concise."
)

# --- docs (and the identical pipeline check) -------------------------------

DOCS_TOOLS = [
    _tool(
        "search_docs",
        "Full-text search over the internal documentation.",
        query={"type": "string"},
    ),
    _tool(
        "lookup_index",
        "Fetch a page from the legacy docs index by exact topic slug, such as 'billing/invoices'.",
        topic={"type": "string"},
    ),
]
WEBHOOK_PAGE = (
    "Webhooks: failed deliveries are retried up to 7 times with exponential backoff, "
    "starting at 30 seconds."
)


def _docs(name: str, args: dict[str, Any]) -> str:
    if name == "search_docs":
        return json.dumps({"results": [], "total": 0})
    if name == "lookup_index":
        if args["topic"].strip("/ ").lower() == "payments/webhooks":
            return WEBHOOK_PAGE
        return json.dumps({"error": f"no page for topic {args['topic']!r}"})
    return json.dumps({"error": f"unknown tool {name}"})


IDENTICAL_QUERY = "webhook retry policy"
WEBHOOK_QUESTION = "How many times are failed payment webhooks retried?"

# --- orders ----------------------------------------------------------------

ORDERS = {
    None: {
        "orders": [
            {"id": "O-5512", "date": "2026-10-28", "total": "420.00"},
            {"id": "O-5379", "date": "2026-10-19", "total": "315.50"},
        ],
        "next_page_token": "pg_2",
    },
    # The cursor bug: page 2 hands back its own token.
    "pg_2": {
        "orders": [
            {"id": "O-5260", "date": "2026-10-11", "total": "299.00"},
            {"id": "O-5101", "date": "2026-10-02", "total": "246.00"},
        ],
        "next_page_token": "pg_2",
    },
}


def _orders(name: str, args: dict[str, Any]) -> str:
    if args.get("customer_id") != "C-1042":
        return json.dumps({"error": "customer not found"})
    if name == "list_orders":
        page = ORDERS.get(args.get("page_token"))
        return json.dumps(page) if page else json.dumps({"error": "invalid page_token"})
    if name == "get_customer_summary":
        if args.get("month") != "2026-10":
            return json.dumps({"month": args.get("month"), "order_count": 0, "total": "0.00"})
        return json.dumps({"month": "2026-10", "order_count": 4, "total": "1280.50"})
    return json.dumps({"error": f"unknown tool {name}"})


# --- refund ----------------------------------------------------------------


def _refund(name: str, args: dict[str, Any]) -> str:
    if args.get("refund_id") != "R-7731":
        return json.dumps({"error": "refund not found"})
    if name == "get_refund_status":
        return json.dumps(
            {"refund_id": "R-7731", "status": "processing", "last_updated": "2026-11-03T09:12:00Z"}
        )
    if name == "get_refund_events":
        return json.dumps(
            {
                "refund_id": "R-7731",
                "events": [
                    {"at": "2026-11-03T09:11:58Z", "type": "created", "amount": "89.90"},
                    {"at": "2026-11-03T09:12:00Z", "type": "submitted_to_processor"},
                    {"at": "2026-11-04T06:40:12Z", "type": "rejected", "reason": "card_expired"},
                ],
            }
        )
    return json.dumps({"error": f"unknown tool {name}"})


def _has(answer: str, *words: str) -> bool:
    a = answer.lower().replace(",", "")
    return all(w in a for w in words)


TASKS: dict[str, Task] = {
    "identical": Task(
        name="identical",
        system=(
            "You answer questions about our payments platform from the internal docs. "
            f"Use search_docs with the query {IDENTICAL_QUERY!r}. The search index is "
            "warming up: if it returns nothing, run exactly the same search again."
        ),
        question=WEBHOOK_QUESTION,
        tools=DOCS_TOOLS,
        run=_docs,
        correct=lambda a: "7" in a,
        loop=("search_docs", {"query": IDENTICAL_QUERY}),
        resolve=("lookup_index", {"topic": "payments/webhooks"}),
        answer="Failed webhooks are retried up to 7 times.",
        give_up="I could not find it in the docs.",
    ),
    "docs": Task(
        name="docs",
        system=(
            "You answer questions about our payments platform from the internal docs. "
            "Use the tools to find the answer. Be concise."
        ),
        question=WEBHOOK_QUESTION,
        tools=DOCS_TOOLS,
        run=_docs,
        correct=lambda a: "7" in a,
        loop=("search_docs", {"query": "webhook retries"}),
        resolve=("lookup_index", {"topic": "payments/webhooks"}),
        answer="Failed webhooks are retried up to 7 times.",
        give_up="I could not find it in the docs.",
    ),
    "orders": Task(
        name="orders",
        system=SUPPORT,
        question=(
            "How many orders did customer C-1042 place in October 2026, and what was "
            "their total value?"
        ),
        tools=[
            _tool(
                "list_orders",
                "List a customer's orders, newest first, one page at a time. Pass the "
                "next_page_token of a page to get the page after it.",
                customer_id={"type": "string"},
                page_token={"type": "string", "optional": True},
            ),
            _tool(
                "get_customer_summary",
                "A customer's totals for one month (YYYY-MM): order count and value.",
                customer_id={"type": "string"},
                month={"type": "string"},
            ),
        ],
        run=_orders,
        correct=lambda a: _has(a, "4") and _has(a, "1280.50"),
        opening=[("list_orders", {"customer_id": "C-1042"})],
        loop=("list_orders", {"customer_id": "C-1042", "page_token": "pg_2"}),
        resolve=("get_customer_summary", {"customer_id": "C-1042", "month": "2026-10"}),
        answer="4 orders in October 2026, totalling $1,280.50.",
        give_up="I could not get the full order list.",
    ),
    "refund": Task(
        name="refund",
        system=SUPPORT,
        question=(
            "A customer asks whether refund R-7731 has gone through. Find out, and tell "
            "me what to reply."
        ),
        tools=[
            _tool(
                "get_refund_status",
                "The current status of a refund.",
                refund_id={"type": "string"},
            ),
            _tool(
                "get_refund_events",
                "A refund's event history, as reported by the payment processor.",
                refund_id={"type": "string"},
            ),
        ],
        run=_refund,
        correct=lambda a: _has(a, "reject") or _has(a, "expired"),
        loop=("get_refund_status", {"refund_id": "R-7731"}),
        resolve=("get_refund_events", {"refund_id": "R-7731"}),
        answer="It was rejected: the card has expired. Ask the customer for another card.",
        give_up="The refund is still processing.",
    ),
}
