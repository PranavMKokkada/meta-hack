# What This Project Does

## The Problem

Every company with customers has a ticket queue. Human agents spend hours reading tickets, deciding if it's a bug or a billing question, figuring out how urgent it is, routing it to the right team, checking if someone already reported the same thing, and writing a reply. It's repetitive, high-stakes, and expensive. Mis-routing a critical outage ticket to the billing team costs real money and real trust.

There was no standardized environment to train or evaluate AI agents on this task — until now.

## What We Built

**Technical Support & Bug Triage Hub** is a simulated support desk that follows the [OpenEnv](https://huggingface.co/open-env) specification. It presents an AI agent with a stream of realistic support tickets and scores how well the agent triages them.

### The Agent's Job

The agent receives one ticket at a time and must decide:

1. **What type is it?** — Bug report, feature request, billing issue, account problem, performance complaint, or general question
2. **How urgent is it?** — Critical, high, medium, or low (enterprise customers with production outages are critical)
3. **Who should handle it?** — Engineering, DevOps, billing, account management, product, or general support
4. **Is it a duplicate?** — Has someone already reported the same underlying issue?
5. **What should we tell the customer?** — Draft a professional, empathetic, factually accurate response that addresses their specific concerns

### The Tickets

18 hand-crafted tickets based on real support patterns:

- An enterprise customer locked out before a client demo (sentiment: frustrated, SLA: 2 hours)
- A double billing charge with bank statement attached (sentiment: frustrated, SLA: 24 hours)
- A GDPR data deletion request from EU legal counsel (sentiment: neutral, SLA: 720 hours)
- A webhook outage breaking a partner's onboarding automation (sentiment: frustrated, SLA: 4 hours)
- A frustrated VP threatening to churn after getting no phone support (sentiment: angry, SLA: 1 hour, repeat sender)
- An accessibility audit with 14 WCAG violations (sentiment: neutral, SLA: 168 hours)
- A security incident where a terminated employee still has access (sentiment: angry, SLA: 1 hour)
- Intermittent 502 errors with request IDs provided (sentiment: frustrated, SLA: 2 hours, repeat sender)
- And 10 more spanning the full spectrum of support complexity

Some tickets are duplicates of each other. Some mix multiple issues (data discrepancy + billing concern). Some are emotionally charged. The hard ones are genuinely ambiguous.

### Three Difficulty Levels

| Level | What Makes It Harder |
|-------|---------------------|
| **Easy** (6 tickets) | Clear-cut. A login issue is obviously an account issue. A double charge is obviously billing. The agent just needs to read carefully. |
| **Medium** (6 tickets) | Ambiguous routing (is a webhook failure an engineering or devops issue?). Some tickets are duplicates of earlier ones — the agent must cross-reference history to catch them. |
| **Hard** (6 tickets) | Multi-issue tickets, emotional customers, legal compliance requests. The agent must also draft a real response — scored on keyword coverage, professional tone, factual accuracy, and absence of hallucinated claims. |

## What the Agent Sees

Each observation gives the agent rich context to make decisions:

- **The ticket itself**: subject, body, sender email, attachments
- **Business context**: sender tier (free/pro/enterprise), SLA hours remaining, whether the sender is a repeat filer
- **Emotional context**: customer sentiment (angry, frustrated, neutral, polite)
- **Episode history**: all previously triaged tickets in this episode (for duplicate detection)
- **Progress**: step number and total steps

This is intentionally richer than a simple text-classification task. A good agent should notice that an angry enterprise customer with 1 hour SLA remaining needs different handling than a polite free-tier user asking a how-to question.

## How Scoring Works

Every action the agent takes gets a score from 0.0 to 1.0. This isn't pass/fail — it's **partial credit everywhere**:

- **Classification**: Exact match = 1.0. Calling a performance issue a "bug" gets 0.4 (they're related). Calling it a "feature request" gets 0.0 (they're not).
- **Priority**: Exact = 1.0. One level off (critical vs high) = 0.5. Two levels off = 0.15. Calling a P0 "low" = 0.0.
- **Routing**: Exact = 1.0. Sending an engineering ticket to devops gets 0.35 (close enough). Sending it to billing gets 0.0.
- **Labels**: Uses semantic similarity with 12 synonym groups. "file-upload" and "upload" are recognized as related. "crash" and "error" are related. Over-labeling is penalized.
- **Duplicates**: Correct duplicate ID = 1.0. Flagged as duplicate but wrong ID = 0.25. False positive = 0.0.
- **Response**: Five components — relevant keywords (35%), professional tone with empathy/ownership/action language (20%), appropriate length (10%), no profanity or dismissive language (15%), and no hallucinated/forbidden claims (20%).

### Hallucination Detection

Each ticket defines phrases that should never appear in a good response — things that would be factually wrong or harmful to say:

- Security incident (TK-1014): "sessions expire automatically", "no data was accessed", "we cannot revoke sessions"
- Dark mode request (TK-1005): "dark mode is available", "already released"
- Webhook outage (TK-1008): "webhooks are working fine", "check your endpoint"
- GDPR request (TK-1016): "we do not delete data", "gdpr does not apply"
- Billing dispute (TK-1002): "no refund", "charge is correct"

If the agent's response contains any of these forbidden phrases, the response score drops by up to 20%. This tests whether the agent invents facts or makes promises it shouldn't.

### Consistency Bonus

If two tickets in the same episode have the same ground-truth classification (e.g., both are bug reports), the agent gets a small bonus (+0.03) for classifying both correctly and consistently. Inconsistent wrong answers are penalized (-0.02). This rewards agents that maintain coherent decision-making across an episode rather than treating each ticket in isolation.

### SLA Urgency Modifier

Enterprise customers with critical tickets have higher stakes. Getting it right earns a 10% bonus. Badly misprioritizing a P0 (e.g., calling it "low") costs a 15% penalty. This rewards agents that understand business context — not all tickets are equal.

### Other Mechanics

- **Behavior Penalties**: Submitting the wrong ticket ID (-0.15), empty labels on harder tasks (-0.05), no response draft on hard task (-0.10).
- **Infinite Loop Protection**: Episode terminates at 20 steps with a -0.5 penalty.
- **Incomplete Submissions**: The grader averages over ALL tickets. Skip one = it scores 0.0.

## How It Runs

The environment is a **FastAPI server** with a simple loop:

```
1. POST /reset  {"task_id": "task_hard"}     → first ticket observation
2. POST /step   {your action for ticket 1}   → reward + next ticket
3. POST /step   {your action for ticket 2}   → reward + next ticket
   ...
6. POST /step   {your action for ticket 6}   → reward + done=true
7. POST /grader {task_id, all_actions}        → final score 0.0–1.0
```

Concurrent users get isolated sessions via `X-Session-Id` header. The server deploys as a Docker container on Hugging Face Spaces (port 7860).

## Baseline Performance

Using `gpt-4o-mini` with temperature=0 and seed=42:

| Task | Score | Interpretation |
|------|-------|----------------|
| Easy | ~0.82 | Gets most classifications and priorities right |
| Medium | ~0.68 | Decent routing, catches some duplicates |
| Hard | ~0.55 | Struggles with nuanced response drafting and hallucination avoidance |

There's clear room for improvement — a perfect agent scores 1.0 on all tasks.

## What's In the Box

```
app.py              → FastAPI server (all endpoints)
environment.py      → Core engine: step/reset/state, grading, sessions
models.py           → Typed Pydantic models
data.py             → 18 tickets with ground truth + hallucination traps
tasks.py            → Task definitions + graders
baseline.py         → OpenAI API baseline script
validate.py         → Pre-submission validator
test_env.py         → 49 unit tests
openenv.yaml        → OpenEnv metadata
Dockerfile          → HF Spaces container
requirements.txt    → Dependencies
.dockerignore       → Docker build exclusions
.gitignore          → Git exclusions
```

## Why It Matters

This environment tests skills that matter for real-world agent deployment:

- **Reading comprehension** — understanding what the customer actually needs
- **Multi-factor reasoning** — weighing urgency, sender tier, SLA countdown, and business impact together
- **Emotional intelligence** — detecting customer sentiment and adapting tone accordingly
- **Memory/context** — checking ticket history for duplicates across the episode
- **Factual accuracy** — not hallucinating claims or making false promises in responses
- **Consistency** — making coherent decisions across similar tickets
- **Communication** — writing responses that are helpful, empathetic, and professional

If an agent can triage a support queue well, it can meaningfully reduce human workload on one of the most common, most tedious tasks in every company.
