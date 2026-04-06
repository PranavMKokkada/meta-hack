---
title: Technical Support & Bug Triage Hub
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
app_port: 7860
---

<div align="center">

# 🎫 Technical Support & Bug Triage Hub

**An OpenEnv-Compliant AI Agent Environment for Customer Support Automation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![OpenEnv Compliant](https://img.shields.io/badge/openenv-compliant-success.svg)](https://huggingface.co/open-env)

> **Submission for the Meta PyTorch OpenEnv Hackathon**

</div>

---

## 📌 Overview

An [OpenEnv](https://huggingface.co/open-env)-compliant environment that simulates a Tier-2 support engineer's daily workflow. AI agents receive a live stream of customer tickets spanning bug reports, billing disputes, security incidents, GDPR requests, and accessibility audits. 

The challenge: **classify, prioritize, route, detect duplicates, and draft professional responses** — all while navigating hallucination traps, SLA urgency modifiers, and consistency bonuses that reward coherent decision-making.

✨ **30 hand-crafted synthetic tickets** • Dense multi-dimensional grading • Partial-credit scoring • Advanced mechanics

---

## 💡 The Problem

Every organization with customers faces the same challenge: **ticket queues at scale**. Triage is:
- **Repetitive** — but requires judgment that scales with context
- **High-stakes** — mis-routing a P0 incident costs real money
- **Complex** — spanning multiple dimensions: severity, category, team fit, urgency
- **Error-prone** — duplicate detection, hallucination risks, SLA tracking

This environment captures that complexity with rich observations and dense, multi-dimensional grading.

---

## 🔄 How It Works

The agent operates in an episodic loop, processing one ticket at a time:

```
POST /reset  {"task_id": "task_hard"}     →  First ticket observation
POST /step   {action for ticket 1}        →  Reward + next ticket
POST /step   {action for ticket 2}        →  Reward + next ticket
...
POST /step   {action for ticket N}        →  Reward + done=true
POST /grader {task_id, all_actions}       →  Final episode score (0.0–1.0)
```

**Multi-tenant ready:** Concurrent users are isolated via `X-Session-Id` headers.

---

## 📊 Four Difficulty Levels

Choose your challenge:

- **🟢 task_easy** — 6 straightforward tickets  
  Focus: Classification (40%), Priority (35%), Routing (15%), Labels (10%)

- **🟡 task_medium** — 6 ambiguous tickets with subtlety  
  Focus: Duplicates (25%), Routing (25%), Classification (20%), Priority (15%), Labels (15%)

- **🔴 task_hard** — 6 complex multi-issue tickets  
  Focus: Response drafts (30%), Duplicates (15%), Routing (20%), Classification (15%), Labels (10%), Priority (10%)

- **⚫ task_expert** — 12 mixed tickets, full complexity  
  Focus: Escalations (20%), Response (25%), Duplicates/Labels/Routing (10% each), Classification/Priority (10% each)

---

## 📬 Observation Space

Each step, agents receive a rich ticket observation with 16 fields:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Unique identifier (e.g. `TK-1001`) |
| `subject` | `str` | Ticket subject line |
| `body` | `str` | Full ticket body text |
| `sender_email` | `str` | Customer email address |
| `sender_tier` | `enum` | `free`, `pro`, or `enterprise` |
| `timestamp` | `str` | ISO 8601 timestamp |
| `attachments` | `list[str]` | Filenames attached |
| `ticket_history` | `list[dict]` | Previously triaged tickets (for duplicate detection) |
| `sentiment` | `str` | Customer mood: `angry`, `frustrated`, `neutral`, `polite` |
| `sla_hours_remaining` | `float` | Hours until SLA breach |
| `is_repeat_sender` | `bool` | Sender has filed tickets before |
| `sender_ticket_count` | `int` | Number of prior tickets from sender |
| `knowledge_base` | `list[str]` | Internal docs relevant to ticket |
| `step_number` | `int` | Current step in episode |
| `total_steps` | `int` | Total tickets in this task |
| `task_id` | `str` | Active task identifier |

**Example Observation:**
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

## ✅ Action Space

Agents must respond with these 9 fields per ticket:

| Field | Type | Options |
|-------|------|---------|
| `ticket_id` | `str` | Must match observed ticket ID |
| `classification` | `enum` | `bug_report`, `feature_request`, `account_issue`, `billing`, `general_inquiry`, `performance` |
| `priority` | `enum` | `critical`, `high`, `medium`, `low` |
| `assigned_team` | `enum` | `engineering`, `billing`, `account_management`, `product`, `general_support`, `devops` |
| `labels` | `list[str]` | Free-form tags (scored with semantic synonym matching) |
| `duplicate_of` | `str \| null` | Ticket ID of original, or `null` |
| `response_draft` | `str \| null` | Customer-facing response (*required* on hard/expert) |
| `escalate` | `bool` | Whether to escalate to management (expert task only) |
| `related_to` | `list[str]` | IDs of related (not duplicate) tickets (expert task only) |

**Example Action:**
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

## 🏆 Reward Design

Every action scores **0.0 to 1.0** with **partial credit** at every dimension:

### Core Scoring

| Dimension | Scoring Logic |
|-----------|---------|
| **Classification** | Exact match = 1.0 • Related categories (bug/performance) = 0.2–0.4 |
| **Priority** | Exact = 1.0 • One level off = 0.5 • Two levels off = 0.15 |
| **Routing** | Exact = 1.0 • Related teams (engineering/devops) = 0.2–0.35 |
| **Labels** | Semantic Jaccard similarity across 12 synonym groups |
| **Duplicates** | Correct ID = 1.0 • Wrong ID = 0.25 • False positive = 0.0 |
| **Response** | Keywords (35%) + Tone (20%) + Length (10%) + No profanity (15%) + No hallucination (20%) |
| **Escalation** | Binary: 1.0 if correct, 0.0 if incorrect |

### Advanced Mechanics

- **🚀 SLA Urgency Modifier** — Enterprise + critical priority + tight SLA = **1.15x multiplier** (amplifies both rewards and penalties)
- **⚠️ Hallucination Traps** — Each ticket defines forbidden phrases; using them = up to –20% response score
- **✨ Consistency Bonus** — +0.03 for correct consistent classifications • –0.02 for inconsistent wrong answers
- **📍 Behavior Penalties** — Wrong ticket ID (–0.15) • Empty labels (–0.05) • Missing response on hard (–0.10)
- **⏱️ Max Step Protection** — Episodes auto-terminate at 20 steps with –0.5 penalty

---

## 🚀 Quick Start

### Prerequisites

```
Python 3.11+
Docker (optional, for containerized deployment)
Git
```

### Local Development (Recommended)

1. **Clone and install dependencies:**
   ```bash
   git clone <repo-url>
   cd meta
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python app.py
   # Server runs at http://localhost:7860
   ```

3. **In another terminal, run tests:**
   ```bash
   python -m pytest test_env.py -v  # 52 comprehensive tests
   ```

### Docker Deployment

```bash
# Build the image
docker build -t triage-hub .

# Run the container
docker run -p 7860:7860 triage-hub
```

### Baseline Agent (Inference)

Run a GPT-4o-mini baseline agent against the environment:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
export HF_TOKEN=hf_your_token
export ENV_API_URL=http://localhost:7860

python inference.py
```

### Validation & Testing

```bash
# Start the server, then validate in another terminal:
python validate.py --api-url http://localhost:7860

# Or use the OpenEnv CLI:
openenv validate
```

---

## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description | Input |
|--------|----------|-------------|-------|
| `GET` | `/` | Health check | — |
| `POST` | `/reset` | Start episode | `{"task_id": "task_easy"}` |
| `POST` | `/step` | Submit action | `{full action JSON}` |
| `POST` | `/grader` | Score episode | `{"task_id": "...", "actions": [...]}` |
| `GET` | `/state` | Get current state | — |

### Metadata Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | OpenEnv standard health check |
| `GET` | `/metadata` | Environment metadata (name, version, description) |
| `GET` | `/schema` | JSON schemas (observation, action, state) |
| `GET` | `/tasks` | List all tasks with action schemas |

### Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions/create` | Create isolated session |
| `GET` | `/sessions` | List active sessions |
| `DELETE` | `/sessions/{id}` | Delete session |

### Internal Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/baseline` | Run baseline LLM agent |
| `GET` | `/validate` | Internal validation diagnostic checks |

---

## 📊 Baseline Performance

Reference scores using GPT-4o-mini (temperature=0):

| Task | Score |
|------|-------|
| 🟢 **task_easy** | ~0.82 |
| 🟡 **task_medium** | ~0.68 |
| 🔴 **task_hard** | ~0.55 |

**Perfect agent score:** 1.0. Plenty of room for improvement!

---

## 📁 Project Structure

```
meta/
├── 🚀 app.py                 # FastAPI server (all endpoints)
├── 🔧 environment.py         # Core OpenEnv engine (step, reset, grading, sessions)
├── 📦 models.py              # Pydantic models (Observation, Action, StepResult)
├── 📑 data.py                # 30 synthetic tickets + ground truth + traps
├── 📋 tasks.py               # Task definitions + grader logic
├── 🤖 inference.py           # Baseline inference script (OpenAI/HF client)
├── ✔️  validate.py            # Pre-submission validation (59 checks)
├── 🧪 test_env.py            # 52 unit tests
├── 🎨 ui.py                  # Optional Gradio interactive UI
├── 🐳 Dockerfile             # Container build for HF Spaces
├── ⚙️  pyproject.toml         # Package metadata for openenv multi-mode
├── 📄 openenv.yaml           # OpenEnv specification
├── 📦 requirements.txt        # Python dependencies
├── 📖 README.md              # This file
├── 📋 OVERVIEW.md            # Additional documentation
└── server/
    ├── app.py                # Entry point for openenv serve
    └── Dockerfile            # Alternative container build
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | — | LLM API endpoint (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Yes | — | Model identifier (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) |
| `HF_TOKEN` | Yes | — | HuggingFace API key |
| `ENV_API_URL` | No | `http://localhost:7860` | Environment server URL |

---

## 👥 Contributors

Built with ❤️ by **Naomi Andrea Pereira** and **Pranav M.K.**

---

## 📄 License

This project is part of the Meta PyTorch OpenEnv Hackathon submission.

