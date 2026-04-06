#!/usr/bin/env python3
"""
Inference Script — Technical Support & Bug Triage Hub
===================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

Runs the agent against all 3 tasks and outputs scores.
Uses OpenAI Client for all LLM calls.
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ── Environment variables (MANDATORY) ─────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")

# ── Environment config ────────────────────────────────────────────────────────

ENV_API_URL = os.getenv("ENV_API_URL", "http://localhost:7860")
TASK_IDS = ["task_easy", "task_medium", "task_hard", "task_expert"]
TEMPERATURE = 0.2
MAX_TOKENS = 1024
FALLBACK_ACTION = {
    "ticket_id": "",
    "classification": "general_inquiry",
    "priority": "medium",
    "assigned_team": "general_support",
    "labels": [],
    "duplicate_of": None,
    "response_draft": "Thank you for contacting support. We are looking into this.",
    "escalate": False,
    "related_to": None,
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a technical support triage agent. For each ticket you receive,
you must return a JSON object with these fields:

- ticket_id: string (echo back the ticket ID)
- classification: one of "bug_report", "feature_request", "account_issue", "billing", "general_inquiry", "performance"
- priority: one of "critical", "high", "medium", "low"
- assigned_team: one of "engineering", "billing", "account_management", "product", "general_support", "devops"
- labels: list of short keyword strings describing the ticket
- duplicate_of: ticket ID string if this is a duplicate of a ticket in the history, else null
- response_draft: a professional response to the customer (2-4 sentences)
- escalate: boolean — true if this ticket should be escalated to management (e.g. production outages, security breaches, churn risk, high-value deals)
- related_to: ticket ID string if this ticket is related (but not a duplicate) to a prior ticket, else null

Consider the sender tier when setting priority: enterprise customers with blocking
issues are critical. Check ticket history for possible duplicates AND related issues.

Consider the customer's sentiment and SLA deadline when crafting your response.
Angry or frustrated customers need strong empathy and urgency. Tight SLA means
higher priority. If internal docs/knowledge base is provided, reference relevant
information in your response.

Escalate when: production outages, security incidents, churn risk, VIP/exec involvement,
legal/compliance urgency, or revenue-impacting issues.

Respond ONLY with valid JSON. No markdown, no explanation.
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

    kb = observation.get("knowledge_base")
    if kb:
        lines.append(f"\n--- INTERNAL KNOWLEDGE BASE ---\n{kb}")

    lines.append(
        f"\nThis is ticket {observation['step_number']} of {observation['total_steps']} "
        f"in task '{observation['task_id']}'."
    )
    return "\n".join(lines)


def parse_model_response(raw: str, ticket_id: str) -> dict:
    """Parse the model's response into an action dict."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        action = json.loads(text)
        # Ensure ticket_id is correct
        action["ticket_id"] = ticket_id
        return action
    except (json.JSONDecodeError, KeyError):
        # Fallback action
        fallback = FALLBACK_ACTION.copy()
        fallback["ticket_id"] = ticket_id
        return fallback


def run_task(client: OpenAI, task_id: str) -> tuple[dict, list[float]]:
    """Run one task end-to-end and return grader result + rewards list."""
    # Reset
    resp = requests.post(f"{ENV_API_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    actions: list[dict] = []
    rewards: list[float] = []
    done = False
    step_count = 0

    while not done:
        step_count += 1
        prompt = build_user_prompt(obs)

        error_msg = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  Model request failed ({exc}). Using fallback.", file=sys.stderr)
            response_text = ""
            error_msg = str(exc)

        action = parse_model_response(response_text, obs["ticket_id"])
        actions.append(action)

        # Step
        try:
            step_resp = requests.post(f"{ENV_API_URL}/step", json=action)
            step_resp.raise_for_status()
            step_result = step_resp.json()
        except Exception as exc:
            error_msg = str(exc)
            step_result = {"done": True}

        done = step_result.get("done", False)
        reward = step_result.get("reward", 0.0)
        rewards.append(reward)
        
        # Emit [STEP] line to stdout
        action_str = json.dumps(action, separators=(',', ':'))
        error_json = json.dumps(error_msg) if error_msg else "null"
        done_str = "true" if done else "false"
        print(f"[STEP] step={step_count} action={action_str} reward={reward:.4f} done={done_str} error={error_json}")
        
        if not done and step_result.get("observation"):
            obs = step_result["observation"]

    # Grade
    grader_resp = requests.post(
        f"{ENV_API_URL}/grader", json={"task_id": task_id, "actions": actions}
    )
    grader_resp.raise_for_status()
    grader_result = grader_resp.json()
    
    return grader_result, rewards


def main() -> None:
    if OpenAI is None:
        print("ERROR: 'openai' package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    if not API_KEY:
        print(
            "ERROR: HF_TOKEN or API_KEY environment variable not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"API Base URL: {API_BASE_URL}", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"Environment: {ENV_API_URL}", file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = {}
    for task_id in TASK_IDS:
        # Emit [START] to stdout
        print(f"[START] task={task_id} env=triage_hub model={MODEL_NAME}")
        print(f"Running {task_id}...", file=sys.stderr)
        
        result, rewards = run_task(client, task_id)
        results[task_id] = result
        
        # Emit [END] to stdout
        success = result.get("score", 0.0) > 0.0
        steps = len(rewards)
        rewards_str = ",".join(f"{r:.4f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} score={result['score']:.4f} rewards={rewards_str}")
        
        print(f"  {task_id}: score={result['score']}", file=sys.stderr)

    summary = {
        "model": MODEL_NAME,
        "scores": {tid: results[tid]["score"] for tid in TASK_IDS},
        "details": results,
    }

    # Final line is machine-readable JSON
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
