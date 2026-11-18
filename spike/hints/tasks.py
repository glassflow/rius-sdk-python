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

Three more test whether steering helps, harms or lasts, rather than whether
a loop happens:

* ``refund_record``: a genuine loop with a reachable answer. The status stays
  ``processing``; the processor's event history is final and says it is the
  system of record, and the policy gives the resolution. Steering can only
  help completion here if the hint leads to the events sooner than the
  model's own turns would.
* ``export_poll``: legitimate polling. The export job's status really
  changes: it is done 40 seconds after the first status check, and only the
  status says how many rows it exported. A hint that says the repeated check
  returns the same result is wrong here, and stopping early fails the task.
* ``refund_batch``: a longer task with the same stuck status on three
  refunds. A model that stops polling the first after a hint may poll the
  next two all the same: the recurrence the comparison counts.

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
import time
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


# --- refund_case -----------------------------------------------------------

REFUND_POLICIES = {
    "card_expired": {
        "reason": "card_expired",
        "action": "Do not retry the expired card. Offer store credit, or refund to a new "
        "card the customer provides.",
    },
    "account_closed": {
        "reason": "account_closed",
        "action": "The customer's bank account is closed. Refund as store credit only; "
        "do not ask for new card details.",
    },
}


def _refund_case(name: str, args: dict[str, Any]) -> str:
    if name in ("get_refund_status", "get_refund_events"):
        if args.get("refund_id") != "R-7731":
            return json.dumps({"error": "refund not found"})
        if name == "get_refund_status":
            return json.dumps(
                {
                    "refund_id": "R-7731",
                    "status": "processing",
                    "final": False,
                    "check_again_in_seconds": 5,
                }
            )
        events = json.loads(_refund(name, args))
        events["customer_id"] = "C-1042"
        return json.dumps(events)
    if name == "list_customer_refunds":
        if args.get("customer_id") != "C-1042":
            return json.dumps({"error": "customer not found"})
        return json.dumps(
            {
                "refunds": [
                    {"id": "R-7731", "amount": "89.90", "status": "processing"},
                    {"id": "R-7012", "amount": "45.00", "status": "succeeded"},
                    {"id": "R-6950", "amount": "12.99", "status": "succeeded"},
                    {"id": "R-6611", "amount": "30.00", "status": "cancelled"},
                ]
            }
        )
    if name == "get_refund_policy":
        policy = REFUND_POLICIES.get(str(args.get("reason", "")).strip().lower())
        if policy is None:
            return json.dumps({"error": f"no policy for reason {args.get('reason')!r}"})
        return json.dumps(policy)
    return json.dumps({"error": f"unknown tool {name}"})


# --- refund_record and refund_batch -----------------------------------------

# Each refund's processor events: the status endpoint says ``processing`` for
# every one of them, whatever the events say.
REFUND_RECORDS: dict[str, dict[str, Any]] = {
    "R-7731": {
        "amount": "89.90",
        "events": [
            {"at": "2026-11-17T09:11:58Z", "type": "created"},
            {"at": "2026-11-17T09:12:00Z", "type": "submitted_to_processor"},
            {"at": "2026-11-17T09:14:31Z", "type": "rejected", "reason": "card_expired"},
        ],
    },
    "R-7744": {
        "amount": "24.50",
        "events": [
            {"at": "2026-11-17T09:20:05Z", "type": "created"},
            {"at": "2026-11-17T09:20:07Z", "type": "submitted_to_processor"},
            {"at": "2026-11-17T09:21:40Z", "type": "succeeded"},
        ],
    },
    "R-7759": {
        "amount": "152.00",
        "events": [
            {"at": "2026-11-17T09:31:12Z", "type": "created"},
            {"at": "2026-11-17T09:31:14Z", "type": "submitted_to_processor"},
            {"at": "2026-11-17T09:33:02Z", "type": "rejected", "reason": "account_closed"},
        ],
    },
}
RECORD_NOTE = (
    "The processor's event history is the system of record. The refund status "
    "endpoint is refreshed from it and can lag behind it."
)


def _refund_records(name: str, args: dict[str, Any]) -> str:
    if name in ("get_refund_status", "get_refund_events"):
        refund_id = args.get("refund_id")
        record = REFUND_RECORDS.get(str(refund_id))
        if record is None:
            return json.dumps({"error": "refund not found"})
        if name == "get_refund_status":
            return json.dumps(
                {
                    "refund_id": refund_id,
                    "status": "processing",
                    "final": False,
                    "check_again_in_seconds": 5,
                }
            )
        return json.dumps(
            {
                "refund_id": refund_id,
                "customer_id": "C-1042",
                "amount": record["amount"],
                "events": record["events"],
                "final": True,
                "note": RECORD_NOTE,
            }
        )
    if name == "get_refund_policy":
        policy = REFUND_POLICIES.get(str(args.get("reason", "")).strip().lower())
        if policy is None:
            return json.dumps({"error": f"no policy for reason {args.get('reason')!r}"})
        return json.dumps(policy)
    return json.dumps({"error": f"unknown tool {name}"})


# --- refund_lag ----------------------------------------------------------------

# The processor's rejection reaches the event history only LAG_EVENTS_AFTER_S
# after the run's first refund call; the status endpoint never moves. An agent
# that reads the events at once sees no outcome yet, and only a second look at
# them, not more status checks, finds the answer.
LAG_EVENTS_AFTER_S = 25.0
_lag_first_call: dict[str, float] = {}


def _refund_lag(name: str, args: dict[str, Any]) -> str:
    if name in ("get_refund_status", "get_refund_events"):
        first = _lag_first_call.setdefault("R-7731", time.monotonic())
        if args.get("refund_id") != "R-7731":
            return json.dumps({"error": "refund not found"})
        if name == "get_refund_status":
            return json.dumps(
                {
                    "refund_id": "R-7731",
                    "status": "processing",
                    "final": False,
                    "check_again_in_seconds": 5,
                }
            )
        record = REFUND_RECORDS["R-7731"]
        settled = time.monotonic() - first >= LAG_EVENTS_AFTER_S
        return json.dumps(
            {
                "refund_id": "R-7731",
                "customer_id": "C-1042",
                "amount": record["amount"],
                "events": record["events"] if settled else record["events"][:2],
                "final": settled,
                "note": RECORD_NOTE,
            }
        )
    return _refund_records(name, args)


# --- refund_consistency ----------------------------------------------------------

# The status endpoint reads a replica by default ("eventual") that never sees
# the processor's rejection; asked for "strong" consistency it reads the
# primary, which has it. The only way to the answer is to change the call's
# arguments, which is what a loop hint suggests. Each eventual read takes a
# different time, as a replica under load does, so the calls do not come at the
# fixed interval that reads as polling.
CONSISTENCY_DELAYS_S = (0.3, 4.0, 0.6, 5.5, 0.4, 3.5, 1.0, 6.0)
_consistency_reads: dict[str, int] = {}


def _refund_consistency(name: str, args: dict[str, Any]) -> str:
    if name == "get_refund_status":
        if args.get("refund_id") != "R-7731":
            return json.dumps({"error": "refund not found"})
        if str(args.get("consistency", "eventual")).strip().lower() == "strong":
            return json.dumps(
                {
                    "refund_id": "R-7731",
                    "status": "rejected",
                    "reason": "card_expired",
                    "final": True,
                }
            )
        n = _consistency_reads.get("R-7731", 0)
        _consistency_reads["R-7731"] = n + 1
        time.sleep(CONSISTENCY_DELAYS_S[n % len(CONSISTENCY_DELAYS_S)])
        return json.dumps(
            {
                "refund_id": "R-7731",
                "status": "processing",
                "final": False,
                "check_again_in_seconds": 5,
            }
        )
    if name == "get_refund_policy":
        return _refund_records(name, args)
    return json.dumps({"error": f"unknown tool {name}"})


REFUND_CONSISTENCY_TOOLS = [
    _tool(
        "get_refund_status",
        "The current status of a refund. A refund that is still processing is not final "
        "yet: check again later.",
        refund_id={"type": "string"},
        consistency={
            "type": "string",
            "enum": ["eventual", "strong"],
            "description": "eventual (default) reads a replica, which can lag the primary by "
            "up to ten minutes. strong reads the primary; it is rate-limited per account, "
            "so use it only when a replica read cannot answer.",
            "optional": True,
        },
    ),
    _tool(
        "get_refund_policy",
        "What our refund policy says to do for a rejection reason.",
        reason={"type": "string"},
    ),
]


REFUND_RECORD_TOOLS = [
    _tool(
        "get_refund_status",
        "The current status of a refund. A refund that is still processing is not final "
        "yet: check again later.",
        refund_id={"type": "string"},
    ),
    _tool(
        "get_refund_events",
        "A refund's event history, as reported by the payment processor.",
        refund_id={"type": "string"},
    ),
    _tool(
        "get_refund_policy",
        "What our refund policy says to do for a processor rejection reason.",
        reason={"type": "string"},
    ),
]
RESOLUTIONS = (
    "one of: succeeded, retry_same_card, store_credit_or_new_card, store_credit_only, escalate"
)


# --- export_poll -------------------------------------------------------------

EXPORT_DONE_AFTER_S = 40.0
WAIT_MAX_S = 15.0
# When each job's status was first checked, in this process: the job is done
# EXPORT_DONE_AFTER_S later, however often it is checked meanwhile.
_export_first_check: dict[str, float] = {}


def _export(name: str, args: dict[str, Any]) -> str:
    if name == "wait":
        seconds = args.get("seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
            return json.dumps({"error": "seconds must be a positive number"})
        slept = min(float(seconds), WAIT_MAX_S)
        time.sleep(slept)
        return json.dumps({"waited_seconds": slept})
    if name == "get_export_status":
        if args.get("job_id") != "EX-5521":
            return json.dumps({"error": "export job not found"})
        first = _export_first_check.setdefault("EX-5521", time.monotonic())
        elapsed = time.monotonic() - first
        if elapsed < EXPORT_DONE_AFTER_S:
            return json.dumps(
                {
                    "job_id": "EX-5521",
                    "status": "running",
                    "progress_percent": 35 + int(60 * elapsed / EXPORT_DONE_AFTER_S),
                    "check_again_in_seconds": 10,
                }
            )
        return json.dumps(
            {
                "job_id": "EX-5521",
                "status": "done",
                "row_count": 18234,
                "file": "orders-2026-11.csv",
            }
        )
    if name == "list_exports":
        return json.dumps(
            {
                "exports": [
                    {"job_id": "EX-5521", "kind": "orders", "started": "2026-11-17T10:02:00Z"},
                    {"job_id": "EX-5488", "kind": "customers", "started": "2026-11-16T08:15:00Z"},
                ]
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
    "refund_case": Task(
        name="refund_case",
        system=SUPPORT,
        question=(
            "A customer is asking about refund R-7731, submitted earlier today. Wait until "
            "it reaches a final state. Then work out what we should do for them under our "
            "refund policy, and how many of this customer's other refunds succeeded, so "
            "I can write the reply."
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
            _tool(
                "list_customer_refunds",
                "All refunds of a customer, with their current status.",
                customer_id={"type": "string"},
            ),
            _tool(
                "get_refund_policy",
                "What our refund policy says to do for a processor rejection reason.",
                reason={"type": "string"},
            ),
        ],
        run=_refund_case,
        correct=lambda a: _answer_matches(
            a,
            {
                "status": "rejected",
                "reason": "card_expired",
                "resolution": "store_credit_or_new_card",
                "other_succeeded_refunds": 2,
            },
        ),
        answer_fields=(
            "status (string), reason (string using the processor event reason), resolution "
            "(one of: retry_same_card, store_credit_or_new_card, escalate), "
            "other_succeeded_refunds (integer: this customer's succeeded refunds other than "
            "R-7731)"
        ),
        loop=("get_refund_status", {"refund_id": "R-7731"}),
        resolve=("get_refund_events", {"refund_id": "R-7731"}),
        answer=(
            '{"status": "rejected", "reason": "card_expired", '
            '"resolution": "store_credit_or_new_card", "other_succeeded_refunds": 2}'
        ),
        give_up="The refund is still processing.",
    ),
    "refund_record": Task(
        name="refund_record",
        system=SUPPORT,
        question=(
            "A customer is asking about refund R-7731, submitted earlier today. Wait until "
            "it reaches a final state, then tell me its outcome and what we should do for "
            "the customer under our refund policy."
        ),
        tools=REFUND_RECORD_TOOLS,
        run=_refund_records,
        correct=lambda a: _answer_matches(
            a,
            {
                "status": "rejected",
                "reason": "card_expired",
                "resolution": "store_credit_or_new_card",
            },
        ),
        answer_fields=(
            "status (string: the refund's final status), reason (string using the processor "
            f"event reason), resolution ({RESOLUTIONS})"
        ),
        loop=("get_refund_status", {"refund_id": "R-7731"}),
        resolve=("get_refund_events", {"refund_id": "R-7731"}),
        answer=(
            '{"status": "rejected", "reason": "card_expired", '
            '"resolution": "store_credit_or_new_card"}'
        ),
        give_up="The refund is still processing.",
    ),
    "refund_lag": Task(
        name="refund_lag",
        system=SUPPORT,
        question=(
            "A customer is asking about refund R-7731, submitted earlier today. Wait until "
            "it reaches a final state, then tell me its outcome and what we should do for "
            "the customer under our refund policy."
        ),
        tools=REFUND_RECORD_TOOLS,
        run=_refund_lag,
        correct=lambda a: _answer_matches(
            a,
            {
                "status": "rejected",
                "reason": "card_expired",
                "resolution": "store_credit_or_new_card",
            },
        ),
        answer_fields=(
            "status (string: the refund's final status), reason (string using the processor "
            f"event reason), resolution ({RESOLUTIONS})"
        ),
        loop=("get_refund_status", {"refund_id": "R-7731"}),
        resolve=("get_refund_events", {"refund_id": "R-7731"}),
        answer=(
            '{"status": "rejected", "reason": "card_expired", '
            '"resolution": "store_credit_or_new_card"}'
        ),
        give_up="The refund is still processing.",
    ),
    "refund_consistency": Task(
        name="refund_consistency",
        system=SUPPORT,
        question=(
            "A customer is asking about refund R-7731, submitted earlier today. Wait until "
            "it reaches a final state, then tell me its outcome and what we should do for "
            "the customer under our refund policy."
        ),
        tools=REFUND_CONSISTENCY_TOOLS,
        run=_refund_consistency,
        correct=lambda a: _answer_matches(
            a,
            {
                "status": "rejected",
                "reason": "card_expired",
                "resolution": "store_credit_or_new_card",
            },
        ),
        answer_fields=(
            "status (string: the refund's final status), reason (string: the rejection "
            f"reason), resolution ({RESOLUTIONS})"
        ),
        loop=("get_refund_status", {"refund_id": "R-7731"}),
        resolve=("get_refund_status", {"refund_id": "R-7731", "consistency": "strong"}),
        answer=(
            '{"status": "rejected", "reason": "card_expired", '
            '"resolution": "store_credit_or_new_card"}'
        ),
        give_up="The refund is still processing.",
    ),
    "export_poll": Task(
        name="export_poll",
        system=(
            "You are an operations assistant. Use the tools to look up what you need, then "
            "answer the question. Be concise."
        ),
        question=(
            "I started export job EX-5521 a moment ago. Wait for it to finish, then tell me "
            "how many rows it exported and the name of the file."
        ),
        tools=[
            _tool(
                "get_export_status",
                "The current status of an export job. A running job reports its progress.",
                job_id={"type": "string"},
            ),
            _tool(
                "wait",
                f"Wait the given number of seconds (at most {WAIT_MAX_S:.0f}) before continuing.",
                seconds={"type": "number"},
            ),
            _tool("list_exports", "Recent export jobs of this workspace."),
        ],
        run=_export,
        correct=lambda a: _answer_matches(
            a, {"status": "done", "row_count": 18234, "file": "orders-2026-11.csv"}
        ),
        answer_fields=(
            "status (string: the job's final status), row_count (integer), file (string)"
        ),
        loop=("get_export_status", {"job_id": "EX-5521"}),
        resolve=("get_export_status", {"job_id": "EX-5521"}),
        answer='{"status": "done", "row_count": 18234, "file": "orders-2026-11.csv"}',
        give_up="The export is still running.",
    ),
    "refund_batch": Task(
        name="refund_batch",
        system=SUPPORT,
        question=(
            "Customer C-1042 has three refunds from today: R-7731, R-7744 and R-7759. For "
            "each, wait until it reaches a final state, then tell me what we should do under "
            "our refund policy so I can write one reply."
        ),
        tools=REFUND_RECORD_TOOLS,
        run=_refund_records,
        correct=lambda a: _answer_matches(
            a,
            {
                "R-7731": "store_credit_or_new_card",
                "R-7744": "succeeded",
                "R-7759": "store_credit_only",
            },
        ),
        answer_fields=(
            "one field per refund id (R-7731, R-7744, R-7759), each a string: succeeded for "
            f"a refund that succeeded, otherwise the policy's resolution ({RESOLUTIONS})"
        ),
        loop=("get_refund_status", {"refund_id": "R-7731"}),
        resolve=("get_refund_events", {"refund_id": "R-7731"}),
        answer=(
            '{"R-7731": "store_credit_or_new_card", "R-7744": "succeeded", '
            '"R-7759": "store_credit_only"}'
        ),
        give_up="The refunds are still processing.",
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
