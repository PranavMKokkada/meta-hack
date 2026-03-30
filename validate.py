#!/usr/bin/env python3
"""Pre-submission validation script.

Run against a live server to verify all OpenEnv requirements pass.

Usage:
    python validate.py [--api-url http://localhost:7860]
"""

from __future__ import annotations

import argparse
import json
import sys

import requests

TASK_IDS = ["task_easy", "task_medium", "task_hard"]
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:7860")
    args = parser.parse_args()
    url = args.api_url.rstrip("/")

    passed = 0
    failed = 0
    total = 0

    def tally(ok: bool):
        nonlocal passed, failed, total
        total += 1
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  OpenEnv Pre-Submission Validator")
    print(f"  Target: {url}")
    print(f"{'='*60}\n")

    # ── 1. Health check ────────────────────────────────────────────────────
    print("[1/7] Health Check")
    try:
        r = requests.get(f"{url}/", timeout=10)
        tally(check("GET / returns 200", r.status_code == 200))
        body = r.json()
        tally(check("Response has 'status' field", "status" in body, str(body)))
    except Exception as e:
        tally(check("Server reachable", False, str(e)))

    # ── 2. Tasks endpoint ──────────────────────────────────────────────────
    print("\n[2/7] Task Enumeration")
    try:
        r = requests.get(f"{url}/tasks", timeout=10)
        tally(check("GET /tasks returns 200", r.status_code == 200))
        tasks = r.json().get("tasks", [])
        tally(check("3+ tasks returned", len(tasks) >= 3, f"got {len(tasks)}"))
        for t in tasks:
            tally(check(
                f"Task '{t['task_id']}' has action_schema",
                "action_schema" in t,
            ))
    except Exception as e:
        tally(check("Tasks endpoint works", False, str(e)))

    # ── 3. Reset / Step / State for each task ──────────────────────────────
    print("\n[3/7] Environment Loop (reset -> step -> state)")
    for task_id in TASK_IDS:
        print(f"\n  --- {task_id} ---")
        try:
            # Reset
            r = requests.post(f"{url}/reset", json={"task_id": task_id}, timeout=10)
            tally(check(f"reset({task_id}) returns 200", r.status_code == 200))
            obs = r.json()
            tally(check("Observation has ticket_id", "ticket_id" in obs))
            tally(check("Observation has body", "body" in obs and len(obs["body"]) > 0))
            tally(check("Observation has ticket_history", "ticket_history" in obs))

            # State
            r = requests.get(f"{url}/state", timeout=10)
            tally(check("state() returns 200", r.status_code == 200))
            st = r.json()
            tally(check("State shows not done", st.get("done") is False))
            tally(check("State step == 0", st.get("current_step") == 0))

            # Step with a dummy action
            action = {
                "ticket_id": obs["ticket_id"],
                "classification": "bug_report",
                "priority": "medium",
                "assigned_team": "engineering",
                "labels": ["test"],
                "duplicate_of": None,
                "response_draft": "Thank you for contacting support. We are investigating this issue.",
            }
            r = requests.post(f"{url}/step", json=action, timeout=10)
            tally(check("step() returns 200", r.status_code == 200))
            result = r.json()
            tally(check("StepResult has reward", "reward" in result))
            tally(check("Reward is float", isinstance(result["reward"], (int, float))))
            tally(check("StepResult has done", "done" in result))
            tally(check("StepResult has reward_breakdown", "reward_breakdown" in result))

        except Exception as e:
            tally(check(f"{task_id} full loop", False, str(e)))

    # ── 4. Grader endpoint ─────────────────────────────────────────────────
    print("\n[4/7] Grader")
    for task_id in TASK_IDS:
        try:
            # Submit minimal actions
            r = requests.post(f"{url}/reset", json={"task_id": task_id}, timeout=10)
            obs = r.json()

            actions = []
            done = False
            while not done:
                action = {
                    "ticket_id": obs["ticket_id"],
                    "classification": "general_inquiry",
                    "priority": "medium",
                    "assigned_team": "general_support",
                    "labels": ["test"],
                    "duplicate_of": None,
                    "response_draft": "Thank you for contacting us. We will look into this.",
                }
                actions.append(action)
                r = requests.post(f"{url}/step", json=action, timeout=10)
                step_r = r.json()
                done = step_r.get("done", True)
                if not done and step_r.get("observation"):
                    obs = step_r["observation"]

            r = requests.post(f"{url}/grader", json={"task_id": task_id, "actions": actions}, timeout=10)
            tally(check(f"grader({task_id}) returns 200", r.status_code == 200))
            gr = r.json()
            score = gr.get("score", -1)
            tally(check(f"Score in [0.0, 1.0]", 0.0 <= score <= 1.0, f"score={score}"))
            tally(check(f"per_ticket has entries", len(gr.get("per_ticket", [])) > 0))

        except Exception as e:
            tally(check(f"Grader {task_id}", False, str(e)))

    # ── 5. Score variance check ────────────────────────────────────────────
    print("\n[5/7] Score Variance (graders don't always return same score)")
    try:
        # Run grader with perfect actions on task_easy
        from data import get_tickets_for_task
        tickets = get_tickets_for_task("task_easy")
        perfect = []
        for t in tickets:
            perfect.append({
                "ticket_id": t.ticket_id,
                "classification": t.ground_truth.classification.value,
                "priority": t.ground_truth.priority.value,
                "assigned_team": t.ground_truth.assigned_team.value,
                "labels": t.ground_truth.labels,
            })
        r1 = requests.post(f"{url}/grader", json={"task_id": "task_easy", "actions": perfect}, timeout=10)
        s1 = r1.json()["score"]

        # All-wrong actions
        bad = []
        for t in tickets:
            bad.append({
                "ticket_id": t.ticket_id,
                "classification": "feature_request",
                "priority": "low",
                "assigned_team": "product",
                "labels": ["wrong"],
            })
        r2 = requests.post(f"{url}/grader", json={"task_id": "task_easy", "actions": bad}, timeout=10)
        s2 = r2.json()["score"]

        tally(check("Perfect > bad score", s1 > s2, f"perfect={s1}, bad={s2}"))
        tally(check("Scores are different", s1 != s2))

    except Exception as e:
        tally(check("Score variance", False, str(e)))

    # ── 6. Episode boundary ────────────────────────────────────────────────
    print("\n[6/7] Episode Boundaries")
    try:
        # Reset, complete episode, then try stepping again
        requests.post(f"{url}/reset", json={"task_id": "task_easy"}, timeout=10)
        for i in range(10):  # more than 6 tickets
            r = requests.post(f"{url}/step", json={
                "ticket_id": f"TK-{1001+i}",
                "classification": "bug_report",
                "priority": "medium",
                "assigned_team": "engineering",
                "labels": [],
            }, timeout=10)
            if r.json().get("done"):
                break

        # Step after done — should return done=True
        r = requests.post(f"{url}/step", json={
            "ticket_id": "TK-9999",
            "classification": "bug_report",
            "priority": "medium",
            "assigned_team": "engineering",
            "labels": [],
        }, timeout=10)
        after = r.json()
        tally(check("Step after done returns done=True", after.get("done") is True))
        tally(check("Step after done returns reward=0", after.get("reward", 1) == 0.0))

    except Exception as e:
        tally(check("Episode boundary", False, str(e)))

    # ── 7. Internal validate endpoint ──────────────────────────────────────
    print("\n[7/7] Internal Validation (/validate)")
    try:
        r = requests.get(f"{url}/validate", timeout=30)
        tally(check("GET /validate returns 200", r.status_code == 200))
        vr = r.json()
        tally(check("all_passed is True", vr.get("all_passed") is True, json.dumps(vr.get("checks", {}), indent=2)[:300]))
    except Exception as e:
        tally(check("Internal validate", False, str(e)))

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print(f"  \033[92mALL CHECKS PASSED — ready for submission!\033[0m")
    else:
        print(f"  \033[91m{failed} CHECKS FAILED — fix before submitting\033[0m")
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
