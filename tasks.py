"""Task definitions and graders for the Technical Support & Bug Triage Hub.

Three tasks with increasing difficulty:
  task_easy   — Classify tickets and assign priority (clear-cut cases)
  task_medium — Route to teams + detect duplicate tickets
  task_hard   — Full triage: classify, route, deduplicate, AND draft responses

Graders:
  - Deterministic, 0.0–1.0 range
  - Penalize incomplete submissions (fewer actions than tickets)
  - Score is average reward across all tickets (missing = 0.0)
"""

from __future__ import annotations

from dataclasses import dataclass

from models import Action
from data import get_tickets_for_task
from environment import TriageEnv


@dataclass
class TaskDefinition:
    task_id: str
    name: str
    description: str
    difficulty: str
    num_tickets: int
    action_fields_required: list[str]
    scoring_summary: str


TASK_DEFINITIONS: dict[str, TaskDefinition] = {
    "task_easy": TaskDefinition(
        task_id="task_easy",
        name="Ticket Classification & Priority",
        description=(
            "Classify each support ticket into the correct category "
            "(bug_report, feature_request, account_issue, billing, "
            "general_inquiry, performance) and assign the right priority "
            "(critical, high, medium, low).  Tickets are clear-cut with "
            "obvious signals."
        ),
        difficulty="easy",
        num_tickets=6,
        action_fields_required=[
            "ticket_id", "classification", "priority", "assigned_team", "labels"
        ],
        scoring_summary=(
            "classification=40%, priority=35%, routing=15%, labels=10%. "
            "SLA urgency modifier for enterprise+critical tickets."
        ),
    ),
    "task_medium": TaskDefinition(
        task_id="task_medium",
        name="Smart Routing & Duplicate Detection",
        description=(
            "In addition to classifying and prioritizing, route each ticket "
            "to the correct team and detect if a ticket is a duplicate of a "
            "previously seen one.  Tickets are more ambiguous and some are "
            "duplicates of earlier tickets.  Penalty for empty labels."
        ),
        difficulty="medium",
        num_tickets=6,
        action_fields_required=[
            "ticket_id", "classification", "priority", "assigned_team",
            "labels", "duplicate_of"
        ],
        scoring_summary=(
            "classification=20%, priority=15%, routing=25%, labels=15%, "
            "duplicate=25%. Penalties: empty labels (-0.05). "
            "SLA urgency modifier for enterprise+critical tickets."
        ),
    ),
    "task_hard": TaskDefinition(
        task_id="task_hard",
        name="Full Triage & Response Drafting",
        description=(
            "Handle the complete triage pipeline: classify, prioritize, route, "
            "detect duplicates, AND draft an appropriate customer response. "
            "Tickets are ambiguous, multi-issue, and emotionally charged.  "
            "Response quality is scored on keyword coverage, professional tone, "
            "appropriate length, and absence of unprofessional language."
        ),
        difficulty="hard",
        num_tickets=6,
        action_fields_required=[
            "ticket_id", "classification", "priority", "assigned_team",
            "labels", "duplicate_of", "response_draft"
        ],
        scoring_summary=(
            "classification=15%, priority=10%, routing=20%, labels=10%, "
            "duplicate=15%, response=30%. Response graded on: keyword "
            "coverage (40%), professional tone (25%), length (15%), "
            "no unprofessional language (20%). Penalties: empty labels "
            "(-0.05), missing response (-0.10). SLA urgency modifier."
        ),
    ),
    "task_expert": TaskDefinition(
        task_id="task_expert",
        name="Expert Triage: Full Pipeline + Escalation + Cross-Ticket + Docs",
        description=(
            "The ultimate triage challenge: 12 mixed-difficulty tickets requiring "
            "every skill. Classify, prioritize, route, detect duplicates, draft "
            "responses, decide when to escalate to management, identify cross-ticket "
            "relationships (not just duplicates), and reference internal knowledge "
            "base docs in your responses.  Tickets include production outages, "
            "billing disputes, compliance requests, and feature evaluations."
        ),
        difficulty="expert",
        num_tickets=12,
        action_fields_required=[
            "ticket_id", "classification", "priority", "assigned_team",
            "labels", "duplicate_of", "response_draft", "escalate", "related_to"
        ],
        scoring_summary=(
            "classification=10%, priority=10%, routing=15%, labels=10%, "
            "duplicate=10%, response=25%, escalation=20%. "
            "Cross-ticket bonus: +0.04 for correct related_to link. "
            "Knowledge base referenced in response = higher keyword score. "
            "Sentiment-aware grading. SLA urgency modifier with tight-SLA amplifier."
        ),
    ),
}


def run_grader(task_id: str, actions: list[dict]) -> dict:
    """Run the grader for a task given a list of action dicts.

    Missing tickets (fewer actions than expected) score 0.0 each.

    Returns:
        {
            "task_id": str,
            "score": float (0.0 – 1.0),
            "per_ticket": [ { ticket_id, reward, breakdown }, ... ],
            "num_tickets": int,
            "num_submitted": int,
        }
    """
    env = TriageEnv()
    env.reset(task_id)
    per_ticket = []
    total_reward = 0.0
    tickets = get_tickets_for_task(task_id)

    for action_dict in actions:
        try:
            action = Action(**action_dict)
        except Exception as e:
            # Invalid action format — score 0 and skip
            per_ticket.append({
                "ticket_id": action_dict.get("ticket_id", "unknown"),
                "reward": 0.0,
                "breakdown": {},
                "error": str(e),
            })
            continue

        result = env.step(action)
        per_ticket.append({
            "ticket_id": action.ticket_id,
            "reward": result.reward,
            "breakdown": result.reward_breakdown.model_dump(),
        })
        total_reward += result.reward
        if result.done:
            break

    # Penalize missing tickets: each unsubmitted ticket scores 0.001 (minimum non-zero)
    num_submitted = len(per_ticket)
    num_missing = max(0, len(tickets) - num_submitted)
    for i in range(num_missing):
        missing_tid = tickets[num_submitted + i].ticket_id
        per_ticket.append({
            "ticket_id": missing_tid,
            "reward": 0.001,
            "breakdown": {},
            "error": "no_action_submitted",
        })

    # Score = average over ALL tickets (including missing ones)
    # Must be strictly between 0 and 1 (exclusive) — clamp to (0.001, 0.999)
    score = total_reward / len(tickets) if tickets else 0.0
    score = round(max(0.001, min(score, 0.999)), 4)

    return {
        "task_id": task_id,
        "score": score,
        "per_ticket": per_ticket,
        "num_tickets": len(tickets),
        "num_submitted": num_submitted,
    }
