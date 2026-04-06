"""Core OpenEnv environment — Technical Support & Bug Triage Hub.

Implements step() / reset() / state() with:
  - Session-based isolation (concurrent users get separate state)
  - Episode-scoped ticket history (not global)
  - Partial-credit grading with semantic similarity
  - SLA urgency modifiers
  - Penalty for undesirable behaviors
  - Max-step protection against infinite loops
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict

from models import (
    Action,
    EnvState,
    Observation,
    RewardBreakdown,
    StepResult,
    TicketHistoryEntry,
)
from data import TicketData, get_tickets_for_task


# ── Semantic similarity maps for partial-credit grading ────────────────────────

# Classification: related categories get partial credit
CLASSIFICATION_SIMILARITY: dict[tuple[str, str], float] = {
    ("bug_report", "performance"): 0.4,
    ("performance", "bug_report"): 0.4,
    ("account_issue", "billing"): 0.3,
    ("billing", "account_issue"): 0.3,
    ("feature_request", "general_inquiry"): 0.3,
    ("general_inquiry", "feature_request"): 0.3,
    ("bug_report", "account_issue"): 0.2,
    ("account_issue", "bug_report"): 0.2,
}

# Label synonym groups — members of the same group partially match
LABEL_SYNONYMS: list[set[str]] = [
    {"crash", "white-screen", "wsod", "error", "failure"},
    {"file-upload", "upload", "attachment", "file"},
    {"performance", "slow", "latency", "speed", "regression"},
    {"login", "authentication", "auth", "sso", "sign-in"},
    {"webhook", "api", "integration", "endpoint"},
    {"billing", "charge", "payment", "invoice", "refund"},
    {"security", "access-control", "permissions", "session"},
    {"duplicate", "dupe", "dup", "duplicate-charge"},
    {"urgent", "blocking", "p0", "critical"},
    {"gdpr", "compliance", "legal", "data-deletion", "privacy"},
    {"accessibility", "a11y", "wcag"},
    {"escalation", "escalate", "churn-risk"},
]

# Profanity / unprofessional terms to penalize in response drafts
UNPROFESSIONAL_PATTERNS = [
    r"\b(stupid|dumb|idiot|wtf|crap|clueless|incompetent)\b",
    r"\b(not my problem|deal with it|figure it out yourself)\b",
    r"\b(lol|lmao|rofl|haha)\b",
]

# Required response structure signals (professional tone indicators)
PROFESSIONAL_INDICATORS = [
    r"(thank|sorry|apolog|understand|appreciate)",    # empathy
    r"(we are|we will|we\'ll|i will|i\'ll|our team)",  # ownership
    r"(investigat|look into|review|resolv|fix|address)", # action
]

MAX_STEPS_PER_EPISODE = 20  # Infinite-loop protection


class TriageEnv:
    """OpenEnv-compliant environment for ticket triage — single session."""

    def __init__(self) -> None:
        self._task_id: str = "task_easy"
        self._tickets: list[TicketData] = []
        self._step: int = 0
        self._total_steps_taken: int = 0  # includes invalid steps
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._reward_history: list[float] = []
        self._results: list[dict] = []
        self._processed_tickets: list[TicketData] = []  # for episode-scoped history

    # ── OpenEnv API ────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_easy") -> Observation:
        """Start (or restart) an episode for the given task."""
        self._task_id = task_id
        self._tickets = get_tickets_for_task(task_id)
        self._step = 0
        self._total_steps_taken = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._results = []
        self._processed_tickets = []
        self._action_history: list[Action] = []  # for consistency bonus
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Process an agent action and advance the episode."""
        if self._done:
            return StepResult(
                observation=None,
                reward=0.0,
                done=True,
                info={"error": "Episode is done. Call reset()."},
            )

        self._total_steps_taken += 1

        # Max-step protection — penalize and force-end the episode
        if self._total_steps_taken > MAX_STEPS_PER_EPISODE:
            self._done = True
            penalty = -0.5
            self._cumulative_reward += penalty
            self._reward_history.append(penalty)
            return StepResult(
                observation=None,
                reward=penalty,
                done=True,
                info={
                    "error": f"Max steps ({MAX_STEPS_PER_EPISODE}) exceeded. Episode terminated.",
                    "episode_reward": self._cumulative_reward,
                    "episode_score": round(max(0.0001, min(0.9999, self._cumulative_reward / len(self._tickets))), 4),
                },
            )

        current_ticket = self._tickets[self._step]

        # Validate action targets the current ticket — penalty for wrong ID
        if action.ticket_id != current_ticket.ticket_id:
            penalty = -0.15
            self._cumulative_reward += penalty
            self._reward_history.append(penalty)
            return StepResult(
                observation=self._make_observation(),
                reward=penalty,
                done=False,
                info={
                    "error": f"Expected ticket {current_ticket.ticket_id}, got {action.ticket_id}",
                    "penalty": "wrong_ticket_id",
                },
            )

        # Score the action
        breakdown = self._grade_action(action, current_ticket)
        self._cumulative_reward += breakdown.total
        self._reward_history.append(breakdown.total)
        self._results.append({
            "ticket_id": current_ticket.ticket_id,
            "reward": breakdown.total,
            "breakdown": breakdown.model_dump(),
        })

        # Track for history and consistency
        self._processed_tickets.append(current_ticket)
        self._action_history.append(action)

        # Advance
        self._step += 1
        if self._step >= len(self._tickets):
            self._done = True
            episode_score = self._cumulative_reward / len(self._tickets)
            return StepResult(
                observation=None,
                reward=breakdown.total,
                reward_breakdown=breakdown,
                done=True,
                info={
                    "episode_reward": round(self._cumulative_reward, 4),
                    "episode_score": round(max(0.0001, min(0.9999, episode_score)), 4),
                    "results": self._results,
                },
            )

        return StepResult(
            observation=self._make_observation(),
            reward=breakdown.total,
            reward_breakdown=breakdown,
            done=False,
            info={"tickets_remaining": len(self._tickets) - self._step},
        )

    def state(self) -> EnvState:
        """Return current environment state."""
        return EnvState(
            task_id=self._task_id,
            current_step=self._step,
            total_steps=len(self._tickets),
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            tickets_processed=len(self._processed_tickets),
            reward_history=[round(r, 4) for r in self._reward_history],
        )

    # ── Internal: Observation building ─────────────────────────────────────

    def _make_observation(self) -> Observation:
        ticket = self._tickets[self._step]
        history = self._build_episode_history()
        return Observation(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject,
            body=ticket.body,
            sender_email=ticket.sender_email,
            sender_tier=ticket.sender_tier,
            timestamp=ticket.timestamp,
            attachments=ticket.attachments,
            step_number=self._step + 1,
            total_steps=len(self._tickets),
            task_id=self._task_id,
            ticket_history=history,
            # Enriched context
            sentiment=ticket.sentiment,
            sla_hours_remaining=ticket.sla_hours_remaining,
            is_repeat_sender=ticket.is_repeat_sender,
            sender_ticket_count=ticket.sender_ticket_count,
            knowledge_base=ticket.knowledge_base,
        )

    def _build_episode_history(self) -> list[TicketHistoryEntry]:
        """Build history from tickets processed IN THIS EPISODE only."""
        return [
            TicketHistoryEntry(
                ticket_id=t.ticket_id,
                subject=t.subject,
                body=t.body,
                category=t.ground_truth.classification,
                resolved=True,
            )
            for t in self._processed_tickets
        ]

    # ── Internal: Grading ──────────────────────────────────────────────────

    def _grade_action(self, action: Action, ticket: TicketData) -> RewardBreakdown:
        """Grade an action against ground truth with partial credit everywhere."""
        gt = ticket.ground_truth
        weights = self._get_weights()

        # 1. Classification (exact = 1.0, related = partial, else 0.0)
        cls_score = self._grade_classification(action.classification, gt.classification)

        # 2. Priority (exact = 1.0, one-off = 0.5, two-off = 0.15, else 0.0)
        pri_score = self._grade_priority(action.priority, gt.priority)

        # 3. Routing / team assignment (exact = 1.0, related = 0.3)
        route_score = self._grade_routing(action.assigned_team, gt.assigned_team)

        # 4. Labels (semantic Jaccard with synonym expansion)
        label_score = self._grade_labels(action.labels, gt.labels)

        # 5. Duplicate detection
        dup_score = self._grade_duplicate(action.duplicate_of, gt.duplicate_of)

        # 6. Response quality (keywords + tone + length + profanity + hallucination + sentiment match)
        resp_score = self._grade_response(
            action.response_draft, gt.response_keywords, gt.response_forbidden,
            ticket.sentiment,
        )

        # 7. Escalation decision
        esc_score = self._grade_escalation(action.escalate, gt.should_escalate)

        # 8. Cross-ticket reasoning (related_to)
        related_bonus = self._grade_related(action.related_to, gt.related_to)

        # Apply SLA urgency modifier: enterprise + critical + tight SLA = higher stakes
        urgency_mult = self._sla_urgency_modifier(ticket, action)

        raw_total = (
            weights["classification"] * cls_score
            + weights["priority"] * pri_score
            + weights["routing"] * route_score
            + weights["labels"] * label_score
            + weights["duplicate"] * dup_score
            + weights["response"] * resp_score
            + weights["escalation"] * esc_score
        )

        # Apply urgency modifier (boosts good, penalizes bad on high-priority)
        total = raw_total * urgency_mult

        # Cross-ticket bonus (additive, not weighted — rewards smart linking)
        total += related_bonus

        # Penalty: empty labels on medium/hard tasks
        if self._task_id != "task_easy" and not action.labels:
            total -= 0.05

        # Penalty: no response draft on hard/expert tasks
        if self._task_id in ("task_hard", "task_expert") and not action.response_draft:
            total -= 0.10

        # Consistency bonus
        consistency = self._consistency_bonus(action, ticket)
        total += consistency

        total = round(max(0.0, min(1.0, total)), 4)

        return RewardBreakdown(
            classification_score=round(cls_score, 4),
            priority_score=round(pri_score, 4),
            routing_score=round(route_score, 4),
            labels_score=round(label_score, 4),
            duplicate_score=round(dup_score, 4),
            response_score=round(resp_score, 4),
            escalation_score=round(esc_score, 4),
            total=total,
        )

    def _get_weights(self) -> dict[str, float]:
        """Reward weights per task difficulty."""
        if self._task_id == "task_easy":
            return {
                "classification": 0.40, "priority": 0.35, "routing": 0.15,
                "labels": 0.10, "duplicate": 0.00, "response": 0.00, "escalation": 0.00,
            }
        elif self._task_id == "task_medium":
            return {
                "classification": 0.20, "priority": 0.15, "routing": 0.25,
                "labels": 0.15, "duplicate": 0.25, "response": 0.00, "escalation": 0.00,
            }
        elif self._task_id == "task_hard":
            return {
                "classification": 0.15, "priority": 0.10, "routing": 0.15,
                "labels": 0.10, "duplicate": 0.15, "response": 0.25, "escalation": 0.10,
            }
        else:  # task_expert
            return {
                "classification": 0.10, "priority": 0.10, "routing": 0.15,
                "labels": 0.10, "duplicate": 0.10, "response": 0.25, "escalation": 0.20,
            }

    # ── Grading: Classification ────────────────────────────────────────────

    @staticmethod
    def _grade_classification(predicted, actual) -> float:
        if predicted == actual:
            return 1.0
        key = (predicted.value, actual.value)
        return CLASSIFICATION_SIMILARITY.get(key, 0.0)

    # ── Grading: Priority ──────────────────────────────────────────────────

    @staticmethod
    def _grade_priority(predicted, actual) -> float:
        order = ["critical", "high", "medium", "low"]
        try:
            diff = abs(order.index(predicted.value) - order.index(actual.value))
        except ValueError:
            return 0.0
        if diff == 0:
            return 1.0
        if diff == 1:
            return 0.5
        if diff == 2:
            return 0.15
        return 0.0

    # ── Grading: Routing ───────────────────────────────────────────────────

    @staticmethod
    def _grade_routing(predicted, actual) -> float:
        if predicted == actual:
            return 1.0
        # Related team pairs get partial credit
        related = {
            frozenset({"engineering", "devops"}): 0.35,
            frozenset({"billing", "account_management"}): 0.30,
            frozenset({"product", "general_support"}): 0.25,
            frozenset({"engineering", "product"}): 0.20,
        }
        pair = frozenset({predicted.value, actual.value})
        return related.get(pair, 0.0)

    # ── Grading: Labels (semantic Jaccard) ─────────────────────────────────

    @staticmethod
    def _grade_labels(predicted: list[str], actual: list[str]) -> float:
        if not actual and not predicted:
            return 1.0
        if not actual:
            return 0.5  # no ground truth labels, agent gave some — meh
        if not predicted:
            return 0.0

        pred_set = {l.lower().strip() for l in predicted if l.strip()}
        actual_set = {l.lower().strip() for l in actual if l.strip()}

        if not pred_set:
            return 0.0

        # Exact Jaccard
        exact_intersection = pred_set & actual_set
        exact_union = pred_set | actual_set

        # Synonym-expanded matching: for each unmatched pred/actual,
        # check if any synonym group connects them
        unmatched_pred = pred_set - exact_intersection
        unmatched_actual = actual_set - exact_intersection

        synonym_hits = 0
        for p in unmatched_pred:
            for a in unmatched_actual:
                if _labels_are_synonyms(p, a):
                    synonym_hits += 1
                    break

        # Score: exact matches worth 1.0, synonym matches worth 0.6
        total_matchable = len(actual_set)
        exact_score = len(exact_intersection) / total_matchable if total_matchable else 0
        synonym_score = (synonym_hits * 0.6) / total_matchable if total_matchable else 0

        # Penalize over-labeling (too many irrelevant labels)
        excess = max(0, len(pred_set) - len(actual_set) - 2)  # 2 extra is fine
        excess_penalty = excess * 0.05

        return max(0.0, min(1.0, exact_score + synonym_score - excess_penalty))

    # ── Grading: Duplicate detection ───────────────────────────────────────

    @staticmethod
    def _grade_duplicate(predicted: str | None, actual: str | None) -> float:
        if actual is None:
            # No duplicate exists
            if predicted is None:
                return 1.0
            return 0.0  # False positive — flagged a non-duplicate

        # Duplicate exists
        if predicted is None:
            return 0.0  # Missed it
        if predicted == actual:
            return 1.0  # Perfect
        # Wrong dup ID but at least detected something is a dup
        return 0.25

    # ── Grading: Escalation ──────────────────────────────────────────────

    @staticmethod
    def _grade_escalation(predicted: bool, actual: bool) -> float:
        """Grade escalation decision. Correct = 1.0, wrong = 0.0."""
        if predicted == actual:
            return 1.0
        return 0.0

    # ── Grading: Cross-ticket reasoning ──────────────────────────────────

    @staticmethod
    def _grade_related(predicted: str | None, actual: str | None) -> float:
        """Bonus for correctly identifying cross-ticket relationships.

        Returns an additive bonus (not weighted), since this is a bonus
        for advanced reasoning, not a core requirement.
        """
        if actual is None:
            # No related ticket — don't penalize for guessing, small penalty for false link
            return 0.0 if predicted is None else -0.02
        # Related ticket exists
        if predicted is None:
            return 0.0  # Missed — no penalty, just no bonus
        if predicted == actual:
            return 0.04  # Correct cross-ticket link — nice bonus
        return 0.01  # Wrong ID but at least tried to link — tiny credit

    # ── Grading: Response quality ──────────────────────────────────────────

    @staticmethod
    def _grade_response(
        draft: str | None,
        keywords: list[str],
        forbidden: list[str] | None = None,
        sentiment: str = "neutral",
    ) -> float:
        if not keywords:
            return 1.0
        if not draft:
            return 0.0

        draft_lower = draft.lower()

        # Component 1: Keyword coverage (30% of response score)
        hits = sum(1 for kw in keywords if kw.lower() in draft_lower)
        keyword_score = hits / len(keywords)

        # Component 2: Professional tone indicators (15% of response score)
        tone_hits = 0
        for pattern in PROFESSIONAL_INDICATORS:
            if re.search(pattern, draft_lower):
                tone_hits += 1
        tone_score = tone_hits / len(PROFESSIONAL_INDICATORS)

        # Component 3: Length adequacy (10% of response score)
        length = len(draft)
        if length < 30:
            length_score = 0.1
        elif length < 80:
            length_score = 0.5
        elif length <= 500:
            length_score = 1.0
        elif length <= 800:
            length_score = 0.85
        else:
            length_score = 0.7  # Rambling

        # Component 4: No unprofessional language (10% of response score)
        professionalism_score = 1.0
        for pattern in UNPROFESSIONAL_PATTERNS:
            if re.search(pattern, draft_lower):
                professionalism_score -= 0.4
        professionalism_score = max(0.0, professionalism_score)

        # Component 5: No hallucinated/forbidden claims (15% of response score)
        hallucination_score = 1.0
        if forbidden:
            for phrase in forbidden:
                if phrase.lower() in draft_lower:
                    hallucination_score -= 0.5
        hallucination_score = max(0.0, hallucination_score)

        # Component 6: Sentiment-appropriate response (20% of response score)
        # Angry/frustrated customers need empathy; polite customers need helpfulness
        sentiment_score = _grade_sentiment_match(draft_lower, sentiment)

        total = (
            0.30 * keyword_score
            + 0.15 * tone_score
            + 0.10 * length_score
            + 0.10 * professionalism_score
            + 0.15 * hallucination_score
            + 0.20 * sentiment_score
        )

        return round(max(0.0, min(1.0, total)), 4)

    # ── SLA Urgency Modifier ───────────────────────────────────────────────

    @staticmethod
    def _sla_urgency_modifier(ticket: TicketData, action: Action) -> float:
        """Urgency modifier based on sender tier, priority, AND SLA deadline.

        Tight SLA + enterprise + critical = highest stakes.
        Good triage on urgent tickets gets a boost.
        Bad triage on urgent tickets gets penalized harder.
        """
        is_enterprise = ticket.sender_tier.value == "enterprise"
        is_critical = ticket.ground_truth.priority.value == "critical"
        sla = ticket.sla_hours_remaining

        # Base modifier from tier + priority
        pri_order = ["critical", "high", "medium", "low"]
        try:
            diff = abs(
                pri_order.index(action.priority.value)
                - pri_order.index(ticket.ground_truth.priority.value)
            )
        except ValueError:
            diff = 3

        modifier = 1.0

        if is_enterprise and is_critical:
            if diff == 0:
                modifier = 1.10  # 10% boost for nailing enterprise+critical
            elif diff >= 2:
                modifier = 0.85  # 15% penalty for badly misprioritizing
        elif is_enterprise:
            modifier = 1.03  # Small boost for enterprise accuracy

        # SLA tightness amplifier: very tight SLA amplifies the modifier
        # SLA <= 2h = tight, SLA <= 4h = moderate, else = normal
        if sla is not None and sla <= 2.0:
            # Tight SLA: amplify both bonus and penalty
            if modifier > 1.0:
                modifier += 0.05  # extra 5% boost for nailing a tight SLA ticket
            elif modifier < 1.0:
                modifier -= 0.05  # extra 5% penalty for blowing a tight SLA ticket
        elif sla is not None and sla <= 4.0:
            if modifier > 1.0:
                modifier += 0.02
            elif modifier < 1.0:
                modifier -= 0.02

        return modifier

    # ── Consistency Bonus ──────────────────────────────────────────────────

    def _consistency_bonus(self, action: Action, ticket: TicketData) -> float:
        """Reward consistent decisions on similar tickets.

        If a previous ticket in this episode had the same ground-truth
        classification, check if the agent classified both the same way.
        Consistent correct = +0.03, consistent wrong = 0, inconsistent = -0.02.
        """
        if not self._action_history:
            return 0.0

        bonus = 0.0
        for prev_action, prev_ticket in zip(self._action_history, self._processed_tickets):
            # Only compare tickets with the same ground-truth classification
            if prev_ticket.ground_truth.classification != ticket.ground_truth.classification:
                continue

            # Did agent classify both the same way?
            same_cls = prev_action.classification == action.classification
            current_correct = action.classification == ticket.ground_truth.classification

            if same_cls and current_correct:
                bonus += 0.03  # Consistent and correct
            elif not same_cls and current_correct:
                pass  # Corrected a previous mistake — neutral
            elif not same_cls and not current_correct:
                bonus -= 0.02  # Inconsistent and wrong

        # Cap the bonus/penalty
        return max(-0.05, min(0.05, bonus))


# ── Session management ─────────────────────────────────────────────────────────

class SessionManager:
    """Manages multiple concurrent environment sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, TriageEnv] = {}
        # Default session for backward compat (single-user / HF Space)
        self._default_id = "default"
        self._sessions[self._default_id] = TriageEnv()

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = TriageEnv()
        return session_id

    def get(self, session_id: str | None = None) -> TriageEnv:
        sid = session_id or self._default_id
        if sid not in self._sessions:
            # Auto-create if missing
            self._sessions[sid] = TriageEnv()
        return self._sessions[sid]

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions and session_id != self._default_id:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# ── Helper ─────────────────────────────────────────────────────────────────────

def _labels_are_synonyms(a: str, b: str) -> bool:
    """Check if two label strings belong to the same synonym group."""
    for group in LABEL_SYNONYMS:
        # Check if either label is a substring of any group member, or vice versa
        a_in = any(a in member or member in a for member in group)
        b_in = any(b in member or member in b for member in group)
        if a_in and b_in:
            return True
    return False


# Sentiment-specific empathy patterns: what a good response should include
# based on the customer's emotional state
SENTIMENT_EMPATHY_PATTERNS: dict[str, list[str]] = {
    "angry": [
        r"(sincerely apolog|deeply sorry|completely understand|take this very seriously)",
        r"(unacceptable|should not have happened|top priority|immediate|right away)",
        r"(escalat|senior|personally ensure|committed to resolving)",
    ],
    "frustrated": [
        r"(apolog|sorry|understand|inconvenien)",
        r"(looking into|investigating|working on|resolv)",
    ],
    "neutral": [
        r"(thank|happy to help|assist|acknowledge|confirm|will ensure)",
    ],
    "polite": [
        r"(thank|great question|happy to|glad to|pleased to)",
    ],
}


def _grade_sentiment_match(draft_lower: str, sentiment: str) -> float:
    """Score how well the response tone matches the customer's emotional state.

    Angry customers need strong empathy + urgency.
    Frustrated customers need acknowledgment + action.
    Neutral/polite customers just need helpfulness.
    """
    patterns = SENTIMENT_EMPATHY_PATTERNS.get(sentiment, SENTIMENT_EMPATHY_PATTERNS["neutral"])
    if not patterns:
        return 1.0

    hits = sum(1 for p in patterns if re.search(p, draft_lower))
    base_score = hits / len(patterns)

    # Angry customers: penalize if NO empathy at all
    if sentiment == "angry" and hits == 0:
        return 0.1  # Very bad — no empathy for angry customer

    # Frustrated customers: penalize if NO acknowledgment
    if sentiment == "frustrated" and hits == 0:
        return 0.3

    return round(base_score, 4)
