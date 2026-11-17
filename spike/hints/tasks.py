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

Three more give a real integration's flaw with no instruction to repeat, where a
model is more likely to repeat itself on its own:

* ``polling``: the refund question asks for the final state, and the status
  stays ``processing`` with a "check again" field. The event history shows it
  was rejected.
* ``retry``: the invoice service always answers "temporarily unavailable,
  retry". The reporting replica's invoice list has the answer.
* ``cursor``: the orders task without the monthly summary. The model has to
  notice the cursor hands back the same page; pages 1 and 2 hold the answer.

Final task answers use a JSON response contract (``answer_instruction``),
identical in both arms and containing field names/types but no expected values.
``correct`` validates the exact facts, including decimal totals and status.
Malformed, incomplete, contradictory and unverified answers are not successes.
Earlier harness runs used substring grading; their completion rates are not
interchangeable with runs using this contract without inspecting the answers.

``correct`` is each task's success criterion, checked on the final answer.
``loop``, ``resolve`` and the answers drive the scripted model (MODEL=scripted
or MODEL=stub); a real model gets only ``system``, ``question`` and ``tools``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
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
    # A response contract, never the expected values: both experiment arms get it.
    answer_fields: str
    # The scripted model: the opening calls, then ``loop`` until a hint
    # arrives or it gives up, then ``resolve`` and ``answer``.
    loop: Call
    resolve: Call
    answer: str
    give_up: str
    opening: list[Call] = field(default_factory=list)

    @property
    def answer_instruction(self) -> str:
        return (
            " When you have verified the answer, your final response must be only a JSON object "
            f"with these fields: {self.answer_fields}. Do not include Markdown or other fields. "
            "If you cannot establish the answer, return a JSON object with an error field instead. "
            "Do not guess missing facts."
        )


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


# --- polling ---------------------------------------------------------------


def _polling(name: str, args: dict[str, Any]) -> str:
    if name == "get_refund_status" and args.get("refund_id") == "R-7731":
        return json.dumps(
            {
                "refund_id": "R-7731",
                "status": "processing",
                "final": False,
                "check_again_in_seconds": 5,
            }
        )
    return _refund(name, args)


# --- retry -----------------------------------------------------------------


def _retry(name: str, args: dict[str, Any]) -> str:
    if name == "get_invoice":
        if args.get("invoice_id") != "INV-2048":
            return json.dumps({"error": "invoice not found"})
        return json.dumps(
            {"error": "billing service temporarily unavailable, please retry", "retryable": True}
        )
    if name == "list_invoices":
        if args.get("customer_id") != "C-1042":
            return json.dumps({"error": "customer not found"})
        return json.dumps(
            {
                "invoices": [
                    {"id": "INV-2048", "date": "2026-10-31", "total": "312.40", "status": "paid"},
                    {"id": "INV-1977", "date": "2026-09-30", "total": "198.00", "status": "paid"},
                ],
                "as_of": "2026-11-17T08:00:00Z",
            }
        )
    return json.dumps({"error": f"unknown tool {name}"})


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate answer field")
        result[key] = value
    return result


def _answer_matches(answer: str, expected: dict[str, Any]) -> bool:
    """Grade explicit facts, not substrings in free text or explanations.

    Numeric totals accept a JSON number or a decimal string, compared exactly.
    Counts require integers (not booleans); status/reason fields require the
    exact requested enum. Missing, extra and duplicate fields are not answers.
    """
    try:
        actual = json.loads(answer, parse_float=Decimal, object_pairs_hook=_unique_fields)
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return False
        for key, want in expected.items():
            value = actual[key]
            if isinstance(want, Decimal):
                if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
                    return False
                if isinstance(value, str) and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                    return False
                amount = Decimal(value)
                if not amount.is_finite() or amount != want:
                    return False
            elif type(value) is not type(want) or value != want:
                return False
        return True
    except (ValueError, TypeError, InvalidOperation):
        return False


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
        correct=lambda a: _answer_matches(a, {"retry_count": 7}),
        answer_fields="retry_count (integer)",
        loop=("search_docs", {"query": IDENTICAL_QUERY}),
        resolve=("lookup_index", {"topic": "payments/webhooks"}),
        answer='{"retry_count": 7}',
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
        correct=lambda a: _answer_matches(a, {"retry_count": 7}),
        answer_fields="retry_count (integer)",
        loop=("search_docs", {"query": "webhook retries"}),
        resolve=("lookup_index", {"topic": "payments/webhooks"}),
        answer='{"retry_count": 7}',
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
        correct=lambda a: _answer_matches(a, {"order_count": 4, "total": Decimal("1280.50")}),
        answer_fields="order_count (integer), total (decimal amount without currency symbols)",
        opening=[("list_orders", {"customer_id": "C-1042"})],
        loop=("list_orders", {"customer_id": "C-1042", "page_token": "pg_2"}),
        resolve=("get_customer_summary", {"customer_id": "C-1042", "month": "2026-10"}),
        answer='{"order_count": 4, "total": "1280.50"}',
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
        correct=lambda a: _answer_matches(a, {"status": "rejected", "reason": "card_expired"}),
        answer_fields="status (string), reason (string using the processor event reason)",
        loop=("get_refund_status", {"refund_id": "R-7731"}),
        resolve=("get_refund_events", {"refund_id": "R-7731"}),
        answer='{"status": "rejected", "reason": "card_expired"}',
        give_up="The refund is still processing.",
    ),
    "polling": Task(
        name="polling",
        system=SUPPORT,
        question=(
            "Refund R-7731 was submitted earlier today. Wait until it reaches a final "
            "state, then tell me whether it went through and what to tell the customer."
        ),
        tools=[
            _tool(
                "get_refund_status",
                "The current status of a refund. A refund that is still processing is "
                "not final yet: check again later.",
                refund_id={"type": "string"},
            ),
            _tool(
                "get_refund_events",
                "A refund's event history, as reported by the payment processor.",
                refund_id={"type": "string"},
            ),
        ],
        run=_polling,
        correct=lambda a: _answer_matches(a, {"status": "rejected", "reason": "card_expired"}),
        answer_fields="status (string), reason (string using the processor event reason)",
        loop=("get_refund_status", {"refund_id": "R-7731"}),
        resolve=("get_refund_events", {"refund_id": "R-7731"}),
        answer='{"status": "rejected", "reason": "card_expired"}',
        give_up="The refund is still processing.",
    ),
    "retry": Task(
        name="retry",
        system=SUPPORT,
        question="What is the total of invoice INV-2048, and has it been paid?",
        tools=[
            _tool(
                "get_invoice",
                "Fetch an invoice from the billing service by id.",
                invoice_id={"type": "string"},
            ),
            _tool(
                "list_invoices",
                "A customer's invoices from the reporting replica, refreshed hourly. "
                "Invoices are numbered per customer: INV-2048 belongs to C-1042.",
                customer_id={"type": "string"},
            ),
        ],
        run=_retry,
        correct=lambda a: _answer_matches(a, {"total": Decimal("312.40"), "status": "paid"}),
        answer_fields="total (decimal amount without currency symbols), status (string)",
        loop=("get_invoice", {"invoice_id": "INV-2048"}),
        resolve=("list_invoices", {"customer_id": "C-1042"}),
        answer='{"total": "312.40", "status": "paid"}',
        give_up="The billing service is unavailable.",
    ),
    "cursor": Task(
        name="cursor",
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
        ],
        run=_orders,
        correct=lambda a: _answer_matches(a, {"order_count": 4, "total": Decimal("1280.50")}),
        answer_fields="order_count (integer), total (decimal amount without currency symbols)",
        opening=[("list_orders", {"customer_id": "C-1042"})],
        loop=("list_orders", {"customer_id": "C-1042", "page_token": "pg_2"}),
        resolve=("list_orders", {"customer_id": "C-1042"}),
        answer='{"order_count": 4, "total": "1280.50"}',
        give_up="I could not get the full order list.",
    ),
}
