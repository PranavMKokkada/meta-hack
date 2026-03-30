"""Synthetic ticket dataset with deterministic ground truth for grading.

Each ticket has a known-correct classification, priority, team, labels,
and (where applicable) a duplicate_of reference.  Response quality is
graded by checking for required keywords / phrases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models import TicketCategory, Priority, Team, SenderTier


@dataclass
class GroundTruth:
    classification: TicketCategory
    priority: Priority
    assigned_team: Team
    labels: list[str]
    duplicate_of: str | None = None
    response_keywords: list[str] = field(default_factory=list)
    response_forbidden: list[str] = field(default_factory=list)
    # Escalation & cross-ticket
    should_escalate: bool = False
    related_to: str | None = None  # cross-ticket link (related but not duplicate)


@dataclass
class TicketData:
    ticket_id: str
    subject: str
    body: str
    sender_email: str
    sender_tier: SenderTier
    timestamp: str
    attachments: list[str] = field(default_factory=list)
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    sentiment: str = "neutral"
    sla_hours_remaining: float | None = None
    is_repeat_sender: bool = False
    sender_ticket_count: int = 0
    knowledge_base: str | None = None  # internal docs snippet for this ticket


# ═══════════════════════════════════════════════════════════════════════════════
#  TICKET POOL — 18 tickets, grouped by difficulty
# ═══════════════════════════════════════════════════════════════════════════════

TICKETS: list[TicketData] = [

    # ── EASY (1-6): Clear-cut classification & priority ────────────────────

    TicketData(
        ticket_id="TK-1001",
        subject="Cannot log in after password reset",
        body=(
            "Hi, I reset my password 30 minutes ago but still can't log in. "
            "I get 'Invalid credentials' every time.  I've cleared cookies and "
            "tried two browsers.  This is blocking my whole team — we have a "
            "client demo in 2 hours.  Enterprise account #4412."
        ),
        sender_email="maria.chen@acmecorp.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T09:15:00Z",
        sentiment="frustrated",
        sla_hours_remaining=2.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="Password resets take up to 15 minutes to propagate. If the issue persists, the account team can force-sync credentials. Enterprise accounts have a dedicated account manager.",
        ground_truth=GroundTruth(
            classification=TicketCategory.ACCOUNT_ISSUE,
            priority=Priority.CRITICAL,
            assigned_team=Team.ACCOUNT_MANAGEMENT,
            labels=["login", "password-reset", "blocking"],
            response_keywords=["password", "reset", "cache", "support"],
            response_forbidden=["your account has been deleted", "we don't support enterprise"],
            should_escalate=True,  # enterprise client demo in 2h
        ),
    ),
    TicketData(
        ticket_id="TK-1002",
        subject="Charge me twice for March subscription",
        body=(
            "I was billed $49.99 twice on March 1st for my Pro plan. "
            "Transaction IDs: TXN-88123 and TXN-88124.  Please refund the "
            "duplicate charge.  Attached my bank statement screenshot."
        ),
        sender_email="javier.ruiz@gmail.com",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T10:02:00Z",
        attachments=["bank_statement.png"],
        sentiment="frustrated",
        sla_hours_remaining=24.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.BILLING,
            priority=Priority.HIGH,
            assigned_team=Team.BILLING,
            labels=["duplicate-charge", "refund"],
            response_keywords=["refund", "transaction", "billing"],
            response_forbidden=["no refund", "charge is correct"],
        ),
    ),
    TicketData(
        ticket_id="TK-1003",
        subject="How do I export data to CSV?",
        body=(
            "I'm new to the platform.  Is there a way to export my dashboard "
            "data to CSV?  I looked in Settings but couldn't find it."
        ),
        sender_email="new.user42@outlook.com",
        sender_tier=SenderTier.FREE,
        timestamp="2026-03-25T10:30:00Z",
        sentiment="polite",
        sla_hours_remaining=None,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.GENERAL_INQUIRY,
            priority=Priority.LOW,
            assigned_team=Team.GENERAL_SUPPORT,
            labels=["export", "csv", "how-to"],
            response_keywords=["export", "csv", "settings", "data"],
            response_forbidden=["this feature is not available", "upgrade to pro"],
        ),
    ),
    TicketData(
        ticket_id="TK-1004",
        subject="App crashes when uploading files > 50 MB",
        body=(
            "Every time I try to upload a file larger than 50 MB the web app "
            "shows a white screen and the console logs 'RangeError: Maximum "
            "call stack size exceeded'.  Reproducible 100% of the time on "
            "Chrome 124 and Firefox 130.  Smaller files work fine."
        ),
        sender_email="dev.ops@startupxyz.io",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T11:00:00Z",
        sentiment="neutral",
        sla_hours_remaining=48.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.HIGH,
            assigned_team=Team.ENGINEERING,
            labels=["crash", "file-upload", "reproducible"],
            response_keywords=["upload", "file", "investigating", "fix"],
            response_forbidden=["works as intended", "not a bug"],
            related_to=None,
        ),
    ),
    TicketData(
        ticket_id="TK-1005",
        subject="Please add dark mode",
        body=(
            "It would be great if you could add a dark mode option. "
            "I work late at night and the bright UI strains my eyes. "
            "Many of your competitors already have this."
        ),
        sender_email="nightowl@protonmail.com",
        sender_tier=SenderTier.FREE,
        timestamp="2026-03-25T11:30:00Z",
        sentiment="polite",
        sla_hours_remaining=None,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.FEATURE_REQUEST,
            priority=Priority.LOW,
            assigned_team=Team.PRODUCT,
            labels=["dark-mode", "ui", "feature-request"],
            response_keywords=["feature", "request", "roadmap", "thank"],
            response_forbidden=["dark mode is available", "already released"],
        ),
    ),
    TicketData(
        ticket_id="TK-1006",
        subject="Dashboard loading very slowly since yesterday",
        body=(
            "Our analytics dashboard used to load in under 2 seconds. "
            "Since yesterday morning it takes 15-20 seconds.  Nothing "
            "changed on our end.  We're on the Enterprise plan and this "
            "is affecting our daily standups."
        ),
        sender_email="ops.lead@bigclient.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T08:45:00Z",
        sentiment="frustrated",
        sla_hours_remaining=4.0,
        is_repeat_sender=True,
        sender_ticket_count=3,
        ground_truth=GroundTruth(
            classification=TicketCategory.PERFORMANCE,
            priority=Priority.CRITICAL,
            assigned_team=Team.DEVOPS,
            labels=["performance", "dashboard", "regression"],
            response_keywords=["performance", "investigating", "dashboard"],
            response_forbidden=["no issues detected", "working as expected"],
        ),
    ),

    # ── MEDIUM (7-12): Routing ambiguity + duplicates ─────────────────────

    TicketData(
        ticket_id="TK-1007",
        subject="File upload broken — white screen of death",
        body=(
            "When I upload a large CSV (about 80 MB) the page goes white. "
            "No error message shown to the user.  I checked the browser "
            "console and saw a stack overflow error.  This started happening "
            "after your last release."
        ),
        sender_email="qa.tester@clientco.com",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T11:45:00Z",
        sentiment="neutral",
        sla_hours_remaining=48.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.HIGH,
            assigned_team=Team.ENGINEERING,
            labels=["crash", "file-upload", "duplicate"],
            duplicate_of="TK-1004",
            response_keywords=["duplicate", "tracking", "upload"],
            response_forbidden=["this is a new issue", "first time seeing this"],
        ),
    ),
    TicketData(
        ticket_id="TK-1008",
        subject="Webhook events not firing for new subscriptions",
        body=(
            "We integrated your webhook API to track new subscriptions but "
            "events stopped arriving 3 days ago.  Our endpoint is healthy "
            "(returns 200).  We confirmed via curl that our server is "
            "reachable.  Webhook ID: WH-5567.  This is breaking our "
            "onboarding automation."
        ),
        sender_email="backend.eng@partnerfirm.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T09:00:00Z",
        sentiment="frustrated",
        sla_hours_remaining=4.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.CRITICAL,
            assigned_team=Team.ENGINEERING,
            labels=["webhook", "api", "integration", "blocking"],
            response_keywords=["webhook", "investigating", "endpoint"],
            response_forbidden=["webhooks are working fine", "check your endpoint"],
            should_escalate=True,  # enterprise, blocking automation
        ),
    ),
    TicketData(
        ticket_id="TK-1009",
        subject="Downgrade from Pro to Free — billing question",
        body=(
            "I want to downgrade to the Free plan.  Will I lose my data? "
            "Also, will I get a prorated refund for the rest of this month? "
            "And will my API keys still work?"
        ),
        sender_email="solodev@hey.com",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T12:00:00Z",
        sentiment="neutral",
        sla_hours_remaining=72.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.BILLING,
            priority=Priority.MEDIUM,
            assigned_team=Team.BILLING,
            labels=["downgrade", "refund", "plan-change"],
            response_keywords=["downgrade", "data", "refund", "api"],
            response_forbidden=["you will lose all data immediately", "no refunds allowed"],
        ),
    ),
    TicketData(
        ticket_id="TK-1010",
        subject="SSO login broken after IdP certificate rotation",
        body=(
            "We rotated our SAML IdP certificate this morning and now none "
            "of our 200+ users can log in via SSO.  We uploaded the new cert "
            "in your admin panel.  Error: 'SAML Response signature validation "
            "failed'.  This is a P0 for us."
        ),
        sender_email="it.admin@megacorp.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T07:30:00Z",
        sentiment="frustrated",
        sla_hours_remaining=2.0,
        is_repeat_sender=True,
        sender_ticket_count=5,
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.CRITICAL,
            assigned_team=Team.ENGINEERING,
            labels=["sso", "saml", "authentication", "blocking"],
            response_keywords=["sso", "certificate", "saml", "investigating"],
            response_forbidden=["use password login instead", "sso is deprecated"],
            should_escalate=True,  # 200+ users locked out
        ),
    ),
    TicketData(
        ticket_id="TK-1011",
        subject="Slow dashboard — is this a known issue?",
        body=(
            "Our dashboard has been loading really slowly since yesterday. "
            "We're an Enterprise customer and we use the analytics dashboard "
            "heavily.  Is this a known issue?  Any ETA on a fix?"
        ),
        sender_email="pm.lead@anothercorp.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T09:30:00Z",
        sentiment="neutral",
        sla_hours_remaining=8.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.PERFORMANCE,
            priority=Priority.HIGH,
            assigned_team=Team.DEVOPS,
            labels=["performance", "dashboard", "duplicate"],
            duplicate_of="TK-1006",
            response_keywords=["known", "issue", "dashboard", "investigating"],
            response_forbidden=["no known issues", "dashboard is performing normally"],
        ),
    ),
    TicketData(
        ticket_id="TK-1012",
        subject="Need API rate limit increase for batch processing",
        body=(
            "We're hitting the 1000 req/min rate limit during our nightly "
            "batch sync.  We need at least 5000 req/min.  We're willing to "
            "pay more if needed.  Enterprise account, contract renewal is "
            "next month."
        ),
        sender_email="cto@growthco.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T14:00:00Z",
        sentiment="neutral",
        sla_hours_remaining=72.0,
        is_repeat_sender=True,
        sender_ticket_count=2,
        ground_truth=GroundTruth(
            classification=TicketCategory.FEATURE_REQUEST,
            priority=Priority.MEDIUM,
            assigned_team=Team.ENGINEERING,
            labels=["rate-limit", "api", "enterprise", "upsell"],
            response_keywords=["rate limit", "enterprise", "account"],
            response_forbidden=["rate limits cannot be changed", "downgrade your usage"],
        ),
    ),

    # ── HARD (13-18): Ambiguous, multi-label, nuanced ─────────────────────

    TicketData(
        ticket_id="TK-1013",
        subject="Data discrepancy between API and dashboard + billing concern",
        body=(
            "We're seeing different numbers in the API response vs the "
            "dashboard for our usage metrics.  The API says we used 8,420 "
            "API calls in February but the dashboard shows 12,100.  This "
            "matters because our bill seems to be based on the higher number. "
            "We're also wondering if this is a bug or if the two data sources "
            "use different counting methods.  Attached comparison spreadsheet."
        ),
        sender_email="finance@techstartup.io",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T10:15:00Z",
        attachments=["usage_comparison.xlsx"],
        sentiment="frustrated",
        sla_hours_remaining=24.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.HIGH,
            assigned_team=Team.ENGINEERING,
            labels=["data-discrepancy", "billing-impact", "api", "dashboard", "metrics"],
            response_keywords=["discrepancy", "usage", "investigating", "billing"],
            response_forbidden=["the dashboard number is correct", "you owe the higher amount"],
        ),
    ),
    TicketData(
        ticket_id="TK-1014",
        subject="Security concern — former employee still has access",
        body=(
            "We terminated an employee last week and removed them from our "
            "IdP, but they appear to still have an active session in your "
            "platform.  They accessed sensitive customer data yesterday "
            "according to our audit logs.  We need their session invalidated "
            "IMMEDIATELY and a full audit trail of their access in the last "
            "7 days."
        ),
        sender_email="ciso@financecorp.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T06:00:00Z",
        sentiment="angry",
        sla_hours_remaining=1.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.ACCOUNT_ISSUE,
            priority=Priority.CRITICAL,
            assigned_team=Team.ACCOUNT_MANAGEMENT,
            labels=["security", "access-control", "urgent", "audit", "session"],
            response_keywords=["session", "invalidat", "audit", "security", "immediate"],
            response_forbidden=["sessions expire automatically", "no data was accessed", "we cannot revoke sessions"],
            should_escalate=True,  # security breach, CISO involved
        ),
    ),
    TicketData(
        ticket_id="TK-1015",
        subject="Intermittent 502 errors on GraphQL endpoint",
        body=(
            "We're seeing sporadic 502 Bad Gateway errors on /graphql — "
            "roughly 5% of requests fail.  No pattern to which queries fail. "
            "Started around 2026-03-24T18:00Z.  We log request IDs; here are "
            "3 failing ones: req-a1b2c, req-d3e4f, req-g5h6i.  This is "
            "degrading our production app.  Also, large file uploads seem "
            "broken (white page) — possibly related?"
        ),
        sender_email="sre@unicornapp.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T08:00:00Z",
        sentiment="frustrated",
        sla_hours_remaining=2.0,
        is_repeat_sender=True,
        sender_ticket_count=7,
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.CRITICAL,
            assigned_team=Team.DEVOPS,
            labels=["502", "graphql", "intermittent", "production", "file-upload"],
            response_keywords=["502", "request", "investigating", "graphql"],
            response_forbidden=["our systems are fully operational", "no 502 errors in our logs"],
            should_escalate=True,  # production degradation
            related_to="TK-1004",  # mentions file uploads broken — related to the upload bug
        ),
    ),
    TicketData(
        ticket_id="TK-1016",
        subject="GDPR data deletion request — user id 77291",
        body=(
            "Under GDPR Article 17, I am requesting the complete deletion "
            "of all personal data associated with user ID 77291 "
            "(john.doe@example.com).  Please confirm deletion within 30 days "
            "and provide written confirmation that all backups have been "
            "purged.  Include the DPO contact if further correspondence is "
            "needed."
        ),
        sender_email="legal@eucompany.de",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T07:00:00Z",
        sentiment="neutral",
        sla_hours_remaining=720.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.ACCOUNT_ISSUE,
            priority=Priority.HIGH,
            assigned_team=Team.ACCOUNT_MANAGEMENT,
            labels=["gdpr", "data-deletion", "legal", "compliance"],
            response_keywords=["gdpr", "deletion", "confirm", "data", "compliance"],
            response_forbidden=["we do not delete data", "gdpr does not apply", "data has already been deleted"],
        ),
    ),
    TicketData(
        ticket_id="TK-1017",
        subject="Webhook failures + can't reach support phone line",
        body=(
            "Our webhooks for subscription events stopped working 3 days "
            "ago (same issue I saw someone else report).  Meanwhile I've "
            "been trying to call your support line and it just rings.  "
            "We're paying Enterprise prices and getting Free-tier support.  "
            "Considering switching to a competitor.  Please escalate."
        ),
        sender_email="vp.eng@loyalclient.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T09:45:00Z",
        sentiment="angry",
        sla_hours_remaining=1.0,
        is_repeat_sender=True,
        sender_ticket_count=4,
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.CRITICAL,
            assigned_team=Team.ENGINEERING,
            labels=["webhook", "churn-risk", "escalation", "duplicate"],
            duplicate_of="TK-1008",
            response_keywords=["webhook", "escalat", "apolog", "investigating"],
            response_forbidden=["our support hours are limited", "webhooks are working", "please call back later"],
            should_escalate=True,  # VP threatening churn
            related_to="TK-1008",  # same webhook issue, but also a churn risk
        ),
    ),
    TicketData(
        ticket_id="TK-1018",
        subject="Accessibility audit findings — WCAG 2.1 AA violations",
        body=(
            "We conducted an accessibility audit and found 14 WCAG 2.1 AA "
            "violations in your platform including: missing alt text on "
            "dashboard charts, insufficient color contrast on the sidebar, "
            "keyboard navigation broken in the settings modal, and screen "
            "reader can't parse the data tables.  Full report attached. "
            "We need these fixed for our own compliance by Q2."
        ),
        sender_email="accessibility@govcontractor.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T11:15:00Z",
        attachments=["wcag_audit_report.pdf"],
        sentiment="neutral",
        sla_hours_remaining=168.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.HIGH,
            assigned_team=Team.ENGINEERING,
            labels=["accessibility", "wcag", "compliance", "ui"],
            response_keywords=["accessibility", "wcag", "audit", "fix"],
            response_forbidden=["we are fully compliant", "no accessibility issues found", "wcag does not apply"],
        ),
    ),

    # ── EXPERT (19-30): Mixed difficulty, escalation, cross-ticket, docs ──

    TicketData(
        ticket_id="TK-1019",
        subject="Payment processing timeout during checkout",
        body=(
            "Our customers are reporting intermittent payment failures during "
            "checkout.  Stripe webhook returns 'timeout' about 10% of the time. "
            "We've verified our Stripe config is correct.  This started after "
            "your platform update on March 23rd.  We're losing revenue."
        ),
        sender_email="payments.team@ecommerce.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T13:00:00Z",
        sentiment="frustrated",
        sla_hours_remaining=4.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="Payment webhook processing was updated in v3.12 (March 23). Known issue: timeout increased from 5s to 30s for large payloads. Rollback available via feature flag PAYMENT_V3_11.",
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.CRITICAL,
            assigned_team=Team.ENGINEERING,
            labels=["payment", "stripe", "timeout", "regression", "revenue-impact"],
            response_keywords=["payment", "investigating", "timeout", "update"],
            response_forbidden=["stripe is down", "we don't integrate with stripe", "payments are working normally"],
            should_escalate=True,
            related_to=None,
        ),
    ),
    TicketData(
        ticket_id="TK-1020",
        subject="Two-factor auth codes not arriving via SMS",
        body=(
            "I've been trying to log in for the past hour but the 2FA SMS "
            "codes never arrive.  I've checked my phone, no blocked numbers. "
            "My number is +1-555-0199.  I can receive other SMS fine.  I need "
            "to access my account urgently to approve a vendor payment."
        ),
        sender_email="accountant@smallbiz.co",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T14:30:00Z",
        sentiment="frustrated",
        sla_hours_remaining=8.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="2FA SMS is sent via Twilio. Known issue: US carrier filtering can delay messages up to 30 minutes. Workaround: users can switch to authenticator app in Settings > Security. Support can temporarily disable 2FA for account recovery.",
        ground_truth=GroundTruth(
            classification=TicketCategory.ACCOUNT_ISSUE,
            priority=Priority.HIGH,
            assigned_team=Team.ACCOUNT_MANAGEMENT,
            labels=["2fa", "sms", "login", "twilio"],
            response_keywords=["2fa", "sms", "authenticator", "account"],
            response_forbidden=["2fa cannot be disabled", "sms is not supported", "use a different phone"],
            should_escalate=False,
        ),
    ),
    TicketData(
        ticket_id="TK-1021",
        subject="Bulk import fails silently — no error, no data",
        body=(
            "I uploaded a CSV with 15,000 rows via the bulk import tool. "
            "The progress bar went to 100% and said 'Import complete' but "
            "zero rows appeared in the database.  No error messages anywhere. "
            "I've tried 3 times with the same result.  CSV validates fine "
            "in the preview step."
        ),
        sender_email="data.ops@midmarket.io",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T15:00:00Z",
        sentiment="frustrated",
        sla_hours_remaining=24.0,
        is_repeat_sender=True,
        sender_ticket_count=2,
        knowledge_base="Bulk import has a hard limit of 10,000 rows per batch. Files exceeding this are silently truncated in v3.10+. This is a known bug (JIRA-4521). Workaround: split into batches of 10K.",
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.HIGH,
            assigned_team=Team.ENGINEERING,
            labels=["bulk-import", "csv", "silent-failure", "data-loss"],
            response_keywords=["import", "rows", "investigating", "batch"],
            response_forbidden=["import is working correctly", "you must have uploaded an empty file"],
            should_escalate=False,
            related_to="TK-1004",  # related to file handling bugs
        ),
    ),
    TicketData(
        ticket_id="TK-1022",
        subject="Request to add SSO support for Okta",
        body=(
            "We're evaluating your platform for enterprise deployment. "
            "A hard requirement is Okta SSO integration via OIDC. "
            "I see you support SAML but not OIDC.  Is Okta OIDC on "
            "your roadmap?  Timeline would help our procurement decision. "
            "This is a 500-seat deal."
        ),
        sender_email="procurement@fortune500.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T10:00:00Z",
        sentiment="neutral",
        sla_hours_remaining=72.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="OIDC support is on the Q3 roadmap (ETA: August 2026). Currently only SAML 2.0 is supported. For large deals, the product team can prioritize. Mention this to the account manager.",
        ground_truth=GroundTruth(
            classification=TicketCategory.FEATURE_REQUEST,
            priority=Priority.MEDIUM,
            assigned_team=Team.PRODUCT,
            labels=["sso", "okta", "oidc", "enterprise", "procurement"],
            response_keywords=["oidc", "roadmap", "saml", "enterprise"],
            response_forbidden=["we already support okta oidc", "oidc is available now", "we have no plans for oidc"],
            should_escalate=True,  # 500-seat deal, needs account manager
            related_to="TK-1010",  # related to SSO infrastructure
        ),
    ),
    TicketData(
        ticket_id="TK-1023",
        subject="API returns stale cached data after writes",
        body=(
            "When we POST a new record and immediately GET the collection, "
            "the new record doesn't appear for 5-30 seconds.  This breaks "
            "our real-time dashboard.  We've confirmed with timestamps that "
            "the POST succeeds (201 returned) but the GET still shows old "
            "data.  Smells like a caching issue."
        ),
        sender_email="lead.dev@realtimeco.com",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T16:00:00Z",
        sentiment="neutral",
        sla_hours_remaining=48.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="API responses are cached for 30 seconds at the CDN layer. Cache-Control headers can be set per-endpoint. For real-time use cases, recommend adding ?_nocache=1 query param or using the WebSocket API instead.",
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.MEDIUM,
            assigned_team=Team.ENGINEERING,
            labels=["api", "caching", "stale-data", "real-time"],
            response_keywords=["cache", "api", "investigating", "data"],
            response_forbidden=["our api does not cache", "this is expected behavior", "data is always fresh"],
            should_escalate=False,
        ),
    ),
    TicketData(
        ticket_id="TK-1024",
        subject="Compliance: need SOC 2 Type II report",
        body=(
            "Our security team requires your SOC 2 Type II audit report "
            "before we can proceed with the enterprise agreement.  Can you "
            "provide the most recent report?  Also need your data processing "
            "addendum (DPA) and a list of sub-processors."
        ),
        sender_email="security.review@bank.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T09:30:00Z",
        sentiment="neutral",
        sla_hours_remaining=120.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="SOC 2 Type II report (2025) is available under NDA. DPA is at docs.example.com/legal/dpa. Sub-processor list at docs.example.com/legal/subprocessors. Route to legal@company.com for NDA execution.",
        ground_truth=GroundTruth(
            classification=TicketCategory.GENERAL_INQUIRY,
            priority=Priority.MEDIUM,
            assigned_team=Team.ACCOUNT_MANAGEMENT,
            labels=["compliance", "soc2", "dpa", "security-review", "enterprise"],
            response_keywords=["soc 2", "report", "dpa", "nda"],
            response_forbidden=["we don't have soc 2", "we are not compliant", "security reports are confidential"],
            should_escalate=False,
        ),
    ),
    TicketData(
        ticket_id="TK-1025",
        subject="Checkout page broken on mobile Safari",
        body=(
            "The checkout button doesn't respond to taps on iPhone Safari "
            "(iOS 18).  Works fine on Chrome mobile.  Multiple customers "
            "have reported this in our support chat.  We're losing mobile "
            "conversions.  Attached a screen recording."
        ),
        sender_email="mobile.pm@directsales.com",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T12:30:00Z",
        attachments=["safari_bug_recording.mp4"],
        sentiment="frustrated",
        sla_hours_remaining=12.0,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="Known Safari WebKit issue with CSS position:sticky + z-index on iOS 18. Fix merged in PR #4891 but not yet deployed. Hotfix branch: hotfix/safari-checkout. ETA: next deploy cycle (Thursday).",
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.HIGH,
            assigned_team=Team.ENGINEERING,
            labels=["mobile", "safari", "checkout", "ios", "css"],
            response_keywords=["safari", "mobile", "investigating", "fix"],
            response_forbidden=["we don't support safari", "use chrome instead", "mobile is not supported"],
            should_escalate=False,
        ),
    ),
    TicketData(
        ticket_id="TK-1026",
        subject="Urgent: production database migration failed midway",
        body=(
            "We ran the v3.12 migration on our production database and it "
            "failed at step 7/12.  The migration tool says 'partial state — "
            "do not retry'.  Some tables have the new schema, others don't. "
            "Our app is throwing 500 errors on every request.  200+ users "
            "affected.  We need immediate help to either rollback or "
            "complete the migration."
        ),
        sender_email="dba@criticalapp.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T05:30:00Z",
        sentiment="angry",
        sla_hours_remaining=0.5,
        is_repeat_sender=True,
        sender_ticket_count=8,
        knowledge_base="v3.12 migration has a known issue with step 7 (index creation on large tables). Workaround: run 'migrate --continue --skip-index' then create indexes manually. Rollback: 'migrate --rollback --to v3.11'. Always backup before migrating.",
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.CRITICAL,
            assigned_team=Team.ENGINEERING,
            labels=["migration", "database", "production", "outage", "500-errors"],
            response_keywords=["migration", "rollback", "immediately", "investigating"],
            response_forbidden=["just retry the migration", "this is user error", "migrations always succeed"],
            should_escalate=True,  # production outage
        ),
    ),
    TicketData(
        ticket_id="TK-1027",
        subject="Feature request: Slack integration for notifications",
        body=(
            "We'd love to get ticket notifications in our Slack channels. "
            "Right now we have to check the dashboard manually.  Even a "
            "simple webhook-to-Slack setup guide would help.  We're on the "
            "Pro plan.  Happy to beta test if you're building this."
        ),
        sender_email="devrel@coolstartup.io",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T16:30:00Z",
        sentiment="polite",
        sla_hours_remaining=None,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="Slack integration is in beta (feature flag SLACK_NOTIFICATIONS). Zapier integration is already available and can bridge to Slack. Docs: docs.example.com/integrations/zapier.",
        ground_truth=GroundTruth(
            classification=TicketCategory.FEATURE_REQUEST,
            priority=Priority.LOW,
            assigned_team=Team.PRODUCT,
            labels=["slack", "integration", "notifications", "feature-request"],
            response_keywords=["slack", "integration", "notification", "thank"],
            response_forbidden=["slack integration is already available", "we will never add slack support"],
            should_escalate=False,
        ),
    ),
    TicketData(
        ticket_id="TK-1028",
        subject="Invoice discrepancy: charged for 50 seats, we only have 32",
        body=(
            "Our March invoice shows charges for 50 seats but we only have "
            "32 active users in our org.  I've counted them in the admin "
            "panel — screenshot attached.  The overage is $1,800.  This has "
            "been happening for 3 months now.  We want a refund for all "
            "3 months of overcharges."
        ),
        sender_email="controller@mediumcorp.com",
        sender_tier=SenderTier.ENTERPRISE,
        timestamp="2026-03-25T11:00:00Z",
        attachments=["user_count_screenshot.png", "invoices_jan_feb_mar.pdf"],
        sentiment="angry",
        sla_hours_remaining=24.0,
        is_repeat_sender=True,
        sender_ticket_count=3,
        knowledge_base="Seat count includes deactivated users who haven't been fully removed. Admin panel > Users > filter 'All' (not just 'Active') shows the full count. To remove deactivated seats from billing, admin must click 'Remove from org' not just 'Deactivate'.",
        ground_truth=GroundTruth(
            classification=TicketCategory.BILLING,
            priority=Priority.HIGH,
            assigned_team=Team.BILLING,
            labels=["invoice", "overcharge", "seats", "refund", "billing-discrepancy"],
            response_keywords=["invoice", "seats", "refund", "investigating"],
            response_forbidden=["the invoice is correct", "no refund is possible", "you have 50 active users"],
            should_escalate=True,  # 3 months of overcharges, angry
            related_to="TK-1013",  # related to data discrepancy pattern
        ),
    ),
    TicketData(
        ticket_id="TK-1029",
        subject="Can't delete test data from sandbox environment",
        body=(
            "I created a bunch of test records in our sandbox but there's "
            "no bulk delete option.  I have to delete them one by one "
            "(there are 2,000+).  Is there a faster way?  Even an API "
            "endpoint for bulk delete would work."
        ),
        sender_email="qa.engineer@devteam.com",
        sender_tier=SenderTier.FREE,
        timestamp="2026-03-25T17:00:00Z",
        sentiment="neutral",
        sla_hours_remaining=None,
        is_repeat_sender=False,
        sender_ticket_count=0,
        knowledge_base="Sandbox environments can be reset via API: DELETE /api/v1/sandbox/reset (requires admin token). Alternatively, Admin Panel > Settings > Sandbox > 'Reset to default' button. Bulk delete API: DELETE /api/v1/records/bulk with body {\"filter\": {\"created_by\": \"user_id\"}}.",
        ground_truth=GroundTruth(
            classification=TicketCategory.GENERAL_INQUIRY,
            priority=Priority.LOW,
            assigned_team=Team.GENERAL_SUPPORT,
            labels=["sandbox", "bulk-delete", "how-to", "api"],
            response_keywords=["sandbox", "reset", "delete", "api"],
            response_forbidden=["data cannot be deleted", "sandbox is permanent", "you must delete records one by one"],
            should_escalate=False,
        ),
    ),
    TicketData(
        ticket_id="TK-1030",
        subject="Concurrent edit conflict — two users lost work",
        body=(
            "Two of our team members were editing the same document "
            "simultaneously.  When the second person saved, it overwrote "
            "the first person's changes completely — no warning, no merge, "
            "no version history.  We lost 3 hours of work.  This is a "
            "critical flaw for a collaboration tool.  We need version "
            "history and conflict resolution ASAP."
        ),
        sender_email="team.lead@collab.co",
        sender_tier=SenderTier.PRO,
        timestamp="2026-03-25T15:30:00Z",
        sentiment="angry",
        sla_hours_remaining=24.0,
        is_repeat_sender=True,
        sender_ticket_count=4,
        knowledge_base="Real-time collaboration (OT/CRDT) is on the H2 roadmap. Currently, last-write-wins. Version history exists but is only accessible via API: GET /api/v1/documents/{id}/versions. Auto-save creates versions every 5 minutes.",
        ground_truth=GroundTruth(
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.HIGH,
            assigned_team=Team.ENGINEERING,
            labels=["collaboration", "conflict", "data-loss", "version-history"],
            response_keywords=["conflict", "version", "history", "investigating"],
            response_forbidden=["collaboration works correctly", "this is expected behavior", "we support real-time editing"],
            should_escalate=True,  # data loss, angry customer
            related_to="TK-1021",  # related to data loss pattern
        ),
    ),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

EASY_TICKET_IDS = [f"TK-{i}" for i in range(1001, 1007)]
MEDIUM_TICKET_IDS = [f"TK-{i}" for i in range(1007, 1013)]
HARD_TICKET_IDS = [f"TK-{i}" for i in range(1013, 1019)]
EXPERT_TICKET_IDS = [f"TK-{i}" for i in range(1019, 1031)]

_TICKET_MAP: dict[str, TicketData] = {t.ticket_id: t for t in TICKETS}


def get_ticket(ticket_id: str) -> TicketData:
    return _TICKET_MAP[ticket_id]


def get_tickets_for_task(task_id: str) -> list[TicketData]:
    """Return tickets for a given task difficulty."""
    if task_id == "task_easy":
        return [_TICKET_MAP[tid] for tid in EASY_TICKET_IDS]
    elif task_id == "task_medium":
        return [_TICKET_MAP[tid] for tid in MEDIUM_TICKET_IDS]
    elif task_id == "task_hard":
        return [_TICKET_MAP[tid] for tid in HARD_TICKET_IDS]
    elif task_id == "task_expert":
        return [_TICKET_MAP[tid] for tid in EXPERT_TICKET_IDS]
    raise ValueError(f"Unknown task_id: {task_id}")


def get_all_tickets_before(ticket_id: str) -> list[TicketData]:
    """Return all tickets with an earlier index (for history context)."""
    idx = next(i for i, t in enumerate(TICKETS) if t.ticket_id == ticket_id)
    return TICKETS[:idx]
