"""The harness grades task facts exactly before reporting completion rates."""

import json
import runpy
from pathlib import Path

import pytest

TASKS = runpy.run_path(str(Path(__file__).parents[1] / "spike/hints/tasks.py"))["TASKS"]


@pytest.mark.parametrize("name", sorted(TASKS))
def test_scripted_answer_satisfies_its_response_contract(name):
    task = TASKS[name]
    assert task.correct(task.answer)
    assert not task.correct(task.give_up)
    assert not task.correct('{"error": "Unable to establish the answer"}')
    assert "JSON object" in task.answer_instruction


@pytest.mark.parametrize(
    ("name", "answer"),
    [
        ("docs", "Webhooks are retried 17 times."),
        ("identical", '{"retry_count": 17}'),
        ("docs", '{"retry_count": "7"}'),
        ("docs", '{"retry_count": 7.0}'),
        ("docs", '{"retry_count": true}'),
        ("docs", '{"retry_count": 7, "retry_count": 17}'),
        ("docs", '{"retry_count": 17, "retry_count": 7}'),
        ("docs", '{"retry_count": 7, "error": "Not verified"}'),
        ("docs", '"7"'),
        ("docs", '[{"retry_count": 7}]'),
        ("docs", ""),
        ("docs", "not JSON"),
        ("orders", "There are 14 orders, totaling $1,280.50."),
        ("orders", '{"order_count": 14, "total": "1280.50"}'),
        ("cursor", '{"order_count": 4, "total": "1280.51"}'),
        ("cursor", '{"order_count": 4}'),
        ("orders", '{"order_count": 4, "total": "1280.50 USD pending confirmation"}'),
        ("refund", "It was not rejected and the card has not expired; it is still processing."),
        ("refund", '{"status": "not rejected", "reason": "card_expired"}'),
        ("polling", '{"status": "processing", "reason": "card_expired"}'),
        ("polling", '{"status": "rejected", "reason": "not_expired"}'),
        ("retry", "Invoice INV-2048 totals $312.40 and is unpaid."),
        ("retry", '{"total": "312.40", "status": "unpaid"}'),
        ("retry", '{"total": "1312.40", "status": "paid"}'),
        ("retry", '{"total": "NaN", "status": "paid"}'),
        ("retry", '{"total": NaN, "status": "paid"}'),
        ("retry", '{"total": Infinity, "status": "paid"}'),
        ("retry", '{"total": true, "status": "paid"}'),
    ],
)
def test_incorrect_or_unverified_answer_is_not_a_completion(name, answer):
    assert not TASKS[name].correct(answer)


@pytest.mark.parametrize("name,count,total", [("orders", 4, "1280.50"), ("cursor", 4, 1280.50)])
def test_order_total_accepts_exact_decimal_string_or_json_number(name, count, total):
    assert TASKS[name].correct(json.dumps({"order_count": count, "total": total}))


def test_invoice_total_accepts_json_number():
    assert TASKS["retry"].correct('{"total": 312.40, "status": "paid"}')
