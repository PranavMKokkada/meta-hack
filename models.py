"""Typed Pydantic models for the Technical Support & Bug Triage Hub environment."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class TicketCategory(str, Enum):
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    ACCOUNT_ISSUE = "account_issue"
    BILLING = "billing"
    GENERAL_INQUIRY = "general_inquiry"
    PERFORMANCE = "performance"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Team(str, Enum):
    ENGINEERING = "engineering"
    BILLING = "billing"
    ACCOUNT_MANAGEMENT = "account_management"
    PRODUCT = "product"
    GENERAL_SUPPORT = "general_support"
    DEVOPS = "devops"


class SenderTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# ── Observation ────────────────────────────────────────────────────────────────

class TicketHistoryEntry(BaseModel):
    """A previously seen ticket (for duplicate detection / cross-ticket reasoning)."""
    ticket_id: str
    subject: str
    body: str
    category: TicketCategory
    resolved: bool = False


class Observation(BaseModel):
    """What the agent sees each step."""
    ticket_id: str
    subject: str
    body: str
    sender_email: str
    sender_tier: SenderTier
    timestamp: str
    attachments: list[str] = Field(default_factory=list)
    ticket_history: list[TicketHistoryEntry] = Field(default_factory=list)
    step_number: int
    total_steps: int
    task_id: str
    # Enriched context fields
    sentiment: str = "neutral"  # "angry", "frustrated", "neutral", "polite"
    sla_hours_remaining: Optional[float] = None  # hours until SLA breach
    is_repeat_sender: bool = False
    sender_ticket_count: int = 0
    # Knowledge base snippet (internal docs the agent can reference)
    knowledge_base: Optional[str] = None


# ── Action ─────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """What the agent submits each step."""
    ticket_id: str
    classification: TicketCategory
    priority: Priority
    assigned_team: Team
    labels: list[str] = Field(default_factory=list)
    duplicate_of: Optional[str] = None
    response_draft: Optional[str] = None
    # New fields
    escalate: bool = False  # should this ticket be escalated to management?
    related_to: Optional[str] = None  # cross-ticket link (not a dup, but related issue)


# ── Reward ─────────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Detailed per-field reward breakdown."""
    classification_score: float = 0.0
    priority_score: float = 0.0
    routing_score: float = 0.0
    labels_score: float = 0.0
    duplicate_score: float = 0.0
    response_score: float = 0.0
    escalation_score: float = 0.0
    total: float = 0.0


class StepResult(BaseModel):
    """Returned by step()."""
    observation: Optional[Observation] = None
    reward: float = 0.0
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    done: bool = False
    info: dict = Field(default_factory=dict)


# ── State ──────────────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    """Returned by state()."""
    task_id: str
    current_step: int
    total_steps: int
    done: bool
    cumulative_reward: float
    tickets_processed: int
    reward_history: list[float] = Field(default_factory=list)
