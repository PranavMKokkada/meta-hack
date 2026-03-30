---
title: Technical Support & Bug Triage Hub
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# Technical Support & Bug Triage Hub

> Submission for the **Meta PyTorch OpenEnv Hackathon**

An [OpenEnv](https://huggingface.co/open-env)-compliant environment that drops an AI agent into the seat of a Tier-2 support engineer. The agent receives a live stream of customer tickets — bug reports, billing disputes, security incidents, GDPR requests, accessibility audits — and must triage every one: classify, prioritize, route, detect duplicates, and draft a professional response. Grading is dense, per-step, and partial-credit, with hallucination traps, SLA urgency modifiers, and consistency bonuses that reward coherent decision-making across an episode.

---

## Environment Description

Every company with customers has a ticket queue. Triage is repetitive, high-stakes (mis-routing a P0 costs real money), and requires judgment that scales with context. This environment captures that complexity with **30 hand-crafted synthetic tickets** spanning login failures, billing disputes, GDPR data-deletion requests, webhook outages, security incidents, accessibility audits, and more.

The agent operates in an episodic loop:

```
POST /reset  {"task_id": "task_hard"}     ->  first ticket observation
POST /step   {action for ticket 1}        ->  reward + next ticket
POST /step   {action for ticket 2}        ->  reward + next ticket
...
POST /step   {action for ticket N}        ->  reward + done=true
POST /grader {task_id, all_actions}        ->  final score 0.0-1.0
```

Concurrent users are isolated via `X-Session-Id` headers.

---

## Tasks

| Task | Difficulty | Tickets | Grading Weights |
|------|-----------|---------|-----------------|
| `task_easy` | Easy | 6 clear-cut tickets | Classification 40%, Priority 35%, Routing 15%, Labels 10% |
| `task_medium` | Medium | 6 ambiguous tickets | Classification 20%, Priority 15%, Routing 25%, Labels 15%, Duplicates 25% |
| `task_hard` | Hard | 6 complex multi-issue tickets | Classification 15%, Priority 10%, Routing 20%, Labels 10%, Duplicates 15%, Response 30% |
| `task_expert` | Expert | 12 mixed tickets | Classification 10%, Priority 10%, Routing 15%, Labels 10%, Duplicates 10%, Response 25%, Escalation 20% |

---

## Observation Space

Each step, the agent receives a rich observation:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Unique ticket identifier (e.g. `TK-1001`) |
| `subject` | `str` | Ticket subject line |
| `body` | `str` | Full ticket body text |
| `sender_email` | `str` | Customer email |
| `sender_tier` | `enum` | `free`, `pro`, `enterprise` |
| `timestamp` | `str` | ISO 8601 timestamp |
| `attachments` | `list[str]` | Filenames attached to the ticket |
| `ticket_history` | `list[dict]` | Previously triaged tickets in this episode (for duplicate detection) |
| `sentiment` | `str` | Customer emotional state: `angry`, `frustrated`, `neutral`, `polite` |
| `sla_hours_remaining` | `float` | Hours until SLA breach |
| `is_repeat_sender` | `bool` | Whether sender has filed tickets before |
| `sender_ticket_count` | `int` | Number of prior tickets from this sender |
| `knowledge_base` | `list[str]` | Internal docs relevant to this ticket |
| `step_number` | `int` | Current step in the episode |
| `total_steps` | `int` | Total tickets in this task |
| `task_id` | `str` | Active task identifier |

**Example observation:**
```json
{
  "ticket_id": "TK-1001",
  "subject": "Cannot log in after password reset",
  "body": "Hi, I reset my password 30 minutes ago but still cannot log in...",
  "sender_email": "maria.chen@acmecorp.com",
  "sender_tier": "enterprise",
  "timestamp": "2026-03-25T09:15:00Z",
  "attachments": [],
  "ticket_history": [],
  "sentiment": "frustrated",
  "sla_hours_remaining": 2.0,
  "is_repeat_sender": false,
  "sender_ticket_count": 0,
  "knowledge_base": [],
  "step_number": 0,
  "total_steps": 6,
  "task_id": "task_easy"
}
```

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Must match the observed ticket ID |
| `classification` | `enum` | `bug_report`, `feature_request`, `account_issue`, `billing`, `general_inquiry`, `performance` |
| `priority` | `enum` | `critical`, `high`, `medium`, `low` |
| `assigned_team` | `enum` | `engineering`, `billing`, `account_management`, `product`, `general_support`, `devops` |
| `labels` | `list[str]` | Free-form tags (scored with semantic synonym matching) |
| `duplicate_of` | `str \| null` | Ticket ID of the original, or `null` |
| `response_draft` | `str \| null` | Customer-facing response (required on hard/expert) |
| `escalate` | `bool` | Whether to escalate to management (expert task) |
| `related_to` | `list[str]` | IDs of related (not duplicate) tickets (expert task) |

**Example action:**
```json
{
  "ticket_id": "TK-1001",
  "classification": "account_issue",
  "priority": "critical",
  "assigned_team": "account_management",
  "labels": ["login", "password-reset", "blocking"],
  "duplicate_of": null,
  "response_draft": "Hi Maria, I'm sorry you're locked out. Our team is investigating...",
  "escalate": false,
  "related_to": []
}
```

---

## Reward Design

Every action scores **0.0 to 1.0** with partial credit at every level:

| Dimension | Scoring |
|-----------|---------|
| **Classification** | Exact = 1.0. Related categories (bug/performance) = 0.2-0.4 |
| **Priority** | Exact = 1.0, one level off = 0.5, two off = 0.15 |
| **Routing** | Exact = 1.0, related teams (engineering/devops) = 0.2-0.35 |
| **Labels** | Semantic Jaccard similarity across 12 synonym groups |
| **Duplicate** | Correct ID = 1.0, wrong ID = 0.25, false positive = 0.0 |
| **Response** | Keywords (35%) + Tone (20%) + Length (10%) + No profanity (15%) + No hallucination (20%) |
| **Escalation** | Binary correct/incorrect |

**Advanced mechanics:**
- **SLA Urgency Modifier** — Enterprise + critical + tight SLA = 1.15x multiplier (amplifies both rewards and penalties)
- **Hallucination Traps** — Each ticket defines forbidden phrases; responses containing them lose up to 20% of response score
- **Consistency Bonus** — +0.03 for consistent correct classifications across episode, -0.02 for inconsistent wrong answers
- **Behavior Penalties** — Wrong ticket ID (-0.15), empty labels (-0.05), missing response on hard (-0.10)
- **Max Step Protection** — Episodes terminate at 20 steps with -0.5 penalty

---

## Setup

### Prerequisites

```
Python >= 3.11
Docker (for container builds)
```

### Local Development

```bash
pip install -r requirements.txt
python app.py
# Server runs at http://localhost:7860
```

### Docker

```bash
docker build -t triage-hub .
docker run -p 7860:7860 triage-hub
```

### Run Tests

```bash
python -m pytest test_env.py -v
# 52 tests covering data integrity, grading, sessions, edge cases
```

### Run Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
export HF_TOKEN=hf_your_token
export ENV_API_URL=http://localhost:7860
python inference.py
```

### Pre-Submission Validation

```bash
# Start the server, then:
python validate.py --api-url http://localhost:7860

# Or use the openenv CLI:
openenv validate
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | OpenEnv standard health |
| `GET` | `/metadata` | Environment name, description, version, tasks |
| `GET` | `/schema` | Action/observation/state JSON schemas |
| `POST` | `/reset` | Start episode: `{"task_id": "task_easy"}` -> Observation |
| `POST` | `/step` | Submit action -> `{observation, reward, reward_breakdown, done}` |
| `GET` | `/state` | Current state: `{task_id, current_step, total_steps, done, cumulative_reward}` |
| `GET` | `/tasks` | List all tasks with action schemas |
| `POST` | `/grader` | Score a complete episode: `{task_id, actions}` -> `{score, per_ticket}` |
| `POST` | `/baseline` | Run baseline LLM agent |
| `GET` | `/validate` | Internal validation checks |
| `POST` | `/sessions/create` | Create isolated session |
| `DELETE` | `/sessions/{id}` | Delete session |
| `GET` | `/sessions` | List active sessions |

---

## Baseline Scores

| Task | gpt-4o-mini (temp=0) |
|------|---------------------|
| `task_easy` | ~0.82 |
| `task_medium` | ~0.68 |
| `task_hard` | ~0.55 |

A perfect agent scores 1.0 on all tasks. There is clear room for improvement.

---

## Project Structure

```
├── app.py              # FastAPI server with all endpoints
├── environment.py      # Core OpenEnv engine: step/reset/state + grading + sessions
├── models.py           # Pydantic models (Observation, Action, StepResult, EnvState)
├── data.py             # 30 synthetic tickets with ground truth + hallucination traps
├── tasks.py            # Task definitions + grader logic
├── inference.py        # Baseline inference script (OpenAI client, required env vars)
├── validate.py         # Pre-submission validation (59 checks)
├── test_env.py         # 52 unit tests
├── ui.py               # Optional Gradio interactive UI
├── openenv.yaml        # OpenEnv spec metadata
├── pyproject.toml      # Package config for openenv multi-mode deployment
├── Dockerfile          # Container build for HF Spaces
├── server/
│   ├── app.py          # Entry point for openenv serve / uv run
│   └── Dockerfile      # Container build for openenv docker mode
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Yes | Model identifier (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) |
| `HF_TOKEN` | Yes | HuggingFace API key |
| `ENV_API_URL` | No | Environment URL (default: `http://localhost:7860`) |

---

Built by Andrea.
