#!/usr/bin/env python3
"""Baseline inference script — runs an OpenAI model against all 3 tasks.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py [--api-url http://localhost:7860]

Produces reproducible scores by using temperature=0 and a fixed seed.
Outputs a JSON summary as the last line of stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

TASK_IDS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """\
You are a technical support triage agent.  For each ticket you receive,
you must return a JSON object with these fields:

- ticket_id: string (echo back the ticket ID)
- classification: one of "bug_report", "feature_request", "account_issue", "billing", "general_inquiry", "performance"
- priority: one of "critical", "high", "medium", "low"
- assigned_team: one of "engineering", "billing", "account_management", "product", "general_support", "devops"
- labels: list of short keyword strings describing the ticket
- duplicate_of: ticket ID string if this is a duplicate of a ticket in the history, else null
- response_draft: a professional response to the customer (2-4 sentences)

Consider the sender tier when setting priority: enterprise customers with blocking
issues are critical.  Check ticket history for possible duplicates — if the symptoms
and root cause match, flag it.

Respond ONLY with valid JSON.  No markdown, no explanation.
"""


def build_user_prompt(observation: dict) -> str:
    """Format an observation into a prompt for the LLM."""
    lines = [
        f"**Ticket {observation['ticket_id']}**",
        f"Subject: {observation['subject']}",
        f"From: {observation['sender_email']} ({observation['sender_tier']})",
        f"Time: {observation['timestamp']}",
        f"Sentiment: {observation.get('sentiment', 'neutral')}",
        f"SLA Hours Remaining: {observation.get('sla_hours_remaining', 'N/A')}",
        f"Repeat Sender: {observation.get('is_repeat_sender', False)} (prior tickets: {observation.get('sender_ticket_count', 0)})",
        f"Attachments: {observation.get('attachments', [])}",
        "",
        observation["body"],
    ]

    history = observation.get("ticket_history", [])
    if history:
        lines.append("\n--- TICKET HISTORY (previously triaged) ---")
        for h in history:
            lines.append(
                f"  [{h['ticket_id']}] ({h['category']}) {h['subject']}"
            )
            lines.append(f"    {h['body'][:200]}...")

    lines.append(
        f"\nThis is ticket {observation['step_number']} of {observation['total_steps']} "
        f"in task '{observation['task_id']}'."
    )
    return "\n".join(lines)


def run_task(client: OpenAI, api_url: str, task_id: str, model: str) -> dict:
    """Run one task end-to-end and return the grader result."""
    # Reset
    resp = requests.post(f"{api_url}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    actions: list[dict] = []
    done = False

    while not done:
        prompt = build_user_prompt(obs)

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            seed=42,
            max_tokens=1024,
        )

        raw = completion.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: send a minimal action
            action = {
                "ticket_id": obs["ticket_id"],
                "classification": "general_inquiry",
                "priority": "medium",
                "assigned_team": "general_support",
                "labels": [],
                "duplicate_of": None,
                "response_draft": "Thank you for contacting support. We are looking into this.",
            }

        actions.append(action)

        # Step
        step_resp = requests.post(f"{api_url}/step", json=action)
        step_resp.raise_for_status()
        step_result = step_resp.json()

        done = step_result["done"]
        if not done and step_result.get("observation"):
            obs = step_result["observation"]

    # Grade
    grader_resp = requests.post(
        f"{api_url}/grader", json={"task_id": task_id, "actions": actions}
    )
    grader_resp.raise_for_status()
    return grader_resp.json()


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for Triage Hub")
    parser.add_argument("--api-url", default="http://localhost:7860")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    if OpenAI is None:
        print("ERROR: 'openai' package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    results = {}
    for task_id in TASK_IDS:
        print(f"Running {task_id}...", file=sys.stderr)
        result = run_task(client, args.api_url, task_id, args.model)
        results[task_id] = result
        print(f"  {task_id}: score={result['score']}", file=sys.stderr)

    summary = {
        "model": args.model,
        "scores": {tid: results[tid]["score"] for tid in TASK_IDS},
        "details": results,
    }

    # Final line is machine-readable JSON
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
