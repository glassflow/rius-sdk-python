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


@pytest.mark.parametrize(
    ("name", "answer"),
    [
        ("refund_record", '{"status": "processing", "reason": "", "resolution": "escalate"}'),
        (
            "refund_record",
            '{"status": "rejected", "reason": "card_expired", "resolution": "retry_same_card"}',
        ),
        ("export_poll", '{"status": "running", "row_count": 0, "file": ""}'),
        ("export_poll", '{"status": "done", "row_count": "18234", "file": "orders-2026-11.csv"}'),
        (
            "refund_batch",
            '{"R-7731": "store_credit_or_new_card", "R-7744": "succeeded", '
            '"R-7759": "store_credit_or_new_card"}',
        ),
        ("refund_batch", '{"R-7731": "store_credit_or_new_card", "R-7744": "succeeded"}'),
    ],
)
def test_behavioural_tasks_grade_every_fact(name, answer):
    assert not TASKS[name].correct(answer)


def test_refund_records_say_the_events_are_final_and_the_status_never_moves():
    task = TASKS["refund_batch"]
    for refund_id, outcome in (("R-7731", "rejected"), ("R-7744", "succeeded")):
        status = json.loads(task.run("get_refund_status", {"refund_id": refund_id}))
        assert status["status"] == "processing" and status["final"] is False
        events = json.loads(task.run("get_refund_events", {"refund_id": refund_id}))
        assert events["final"] is True and events["events"][-1]["type"] == outcome
    policy = json.loads(task.run("get_refund_policy", {"reason": "account_closed"}))
    assert "store credit only" in policy["action"]


def test_the_export_is_done_only_after_its_time(monkeypatch):
    module = runpy.run_path(str(Path(__file__).parents[1] / "spike/hints/tasks.py"))
    now = [1000.0]
    monkeypatch.setattr(module["time"], "monotonic", lambda: now[0])
    run = module["TASKS"]["export_poll"].run
    assert json.loads(run("get_export_status", {"job_id": "EX-5521"}))["status"] == "running"
    now[0] += module["EXPORT_DONE_AFTER_S"] - 1
    assert json.loads(run("get_export_status", {"job_id": "EX-5521"}))["status"] == "running"
    now[0] += 1
    done = json.loads(run("get_export_status", {"job_id": "EX-5521"}))
    assert done == {
        "job_id": "EX-5521",
        "status": "done",
        "row_count": 18234,
        "file": "orders-2026-11.csv",
    }


def test_the_lagging_refund_shows_its_outcome_only_in_later_events(monkeypatch):
    module = runpy.run_path(str(Path(__file__).parents[1] / "spike/hints/tasks.py"))
    now = [1000.0]
    monkeypatch.setattr(module["time"], "monotonic", lambda: now[0])
    task = module["TASKS"]["refund_lag"]
    status = json.loads(task.run("get_refund_status", {"refund_id": "R-7731"}))
    assert status["status"] == "processing" and status["final"] is False
    early = json.loads(task.run("get_refund_events", {"refund_id": "R-7731"}))
    assert early["final"] is False
    assert [e["type"] for e in early["events"]] == ["created", "submitted_to_processor"]
    now[0] += module["LAG_EVENTS_AFTER_S"]
    status = json.loads(task.run("get_refund_status", {"refund_id": "R-7731"}))
    assert status["status"] == "processing"
    late = json.loads(task.run("get_refund_events", {"refund_id": "R-7731"}))
    assert late["final"] is True and late["events"][-1]["reason"] == "card_expired"
    assert task.correct(
        '{"status": "rejected", "reason": "card_expired", "resolution": "store_credit_or_new_card"}'
    )
    assert not task.correct('{"status": "processing", "reason": "", "resolution": "escalate"}')


def test_the_replica_never_settles_and_the_primary_has_the_answer(monkeypatch):
    module = runpy.run_path(str(Path(__file__).parents[1] / "spike/hints/tasks.py"))
    slept = []
    monkeypatch.setattr(module["time"], "sleep", slept.append)
    task = module["TASKS"]["refund_consistency"]
    for consistency in (None, "eventual", None):
        args = {"refund_id": "R-7731"} | ({"consistency": consistency} if consistency else {})
        status = json.loads(task.run("get_refund_status", args))
        assert status["status"] == "processing" and status["final"] is False
    assert slept == list(module["CONSISTENCY_DELAYS_S"][:3])
    gaps = slept
    mean = sum(gaps) / len(gaps)
    cv = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean
    assert cv > 0.2, "the replica's delays must not read as polling"
    strong = json.loads(
        task.run("get_refund_status", {"refund_id": "R-7731", "consistency": "strong"})
    )
    assert strong == {
        "refund_id": "R-7731",
        "status": "rejected",
        "reason": "card_expired",
        "final": True,
    }
    assert task.correct(
        '{"status": "rejected", "reason": "card_expired", "resolution": "store_credit_or_new_card"}'
    )
