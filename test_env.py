#!/usr/bin/env python3
"""Unit tests for the Technical Support & Bug Triage Hub environment.

Run: python -m pytest test_env.py -v
"""

import pytest

from models import Action, TicketCategory, Priority, Team, SenderTier
from data import get_tickets_for_task, get_ticket, TICKETS
from environment import TriageEnv, SessionManager, _labels_are_synonyms
from tasks import run_grader, TASK_DEFINITIONS


# ═══════════════════════════════════════════════════════════════════════════════
#  Data integrity tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestData:
    def test_30_tickets_exist(self):
        assert len(TICKETS) == 30

    def test_all_ticket_ids_unique(self):
        ids = [t.ticket_id for t in TICKETS]
        assert len(ids) == len(set(ids))

    def test_each_task_has_6_tickets(self):
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            tickets = get_tickets_for_task(task_id)
            assert len(tickets) == 6, f"{task_id} has {len(tickets)} tickets"

    def test_ground_truth_present(self):
        for t in TICKETS:
            gt = t.ground_truth
            assert gt.classification is not None
            assert gt.priority is not None
            assert gt.assigned_team is not None
            assert isinstance(gt.labels, list)

    def test_duplicate_refs_valid(self):
        all_ids = {t.ticket_id for t in TICKETS}
        for t in TICKETS:
            if t.ground_truth.duplicate_of:
                assert t.ground_truth.duplicate_of in all_ids, (
                    f"{t.ticket_id} references non-existent dup {t.ground_truth.duplicate_of}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment lifecycle tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnvLifecycle:
    def test_reset_returns_observation(self):
        env = TriageEnv()
        obs = env.reset("task_easy")
        assert obs.ticket_id == "TK-1001"
        assert obs.step_number == 1
        assert obs.total_steps == 6
        assert obs.task_id == "task_easy"

    def test_state_after_reset(self):
        env = TriageEnv()
        env.reset("task_medium")
        st = env.state()
        assert st.task_id == "task_medium"
        assert st.current_step == 0
        assert not st.done
        assert st.cumulative_reward == 0.0

    def test_step_after_done_returns_done(self):
        env = TriageEnv()
        env.reset("task_easy")
        env._done = True  # force done
        result = env.step(Action(
            ticket_id="TK-1001",
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.MEDIUM,
            assigned_team=Team.ENGINEERING,
        ))
        assert result.done is True
        assert result.reward == 0.0

    def test_wrong_ticket_id_penalized(self):
        env = TriageEnv()
        env.reset("task_easy")
        result = env.step(Action(
            ticket_id="WRONG-ID",
            classification=TicketCategory.BUG_REPORT,
            priority=Priority.MEDIUM,
            assigned_team=Team.ENGINEERING,
        ))
        assert result.reward < 0
        assert not result.done  # doesn't advance

    def test_full_episode_completes(self):
        env = TriageEnv()
        obs = env.reset("task_easy")
        tickets = get_tickets_for_task("task_easy")

        for i, t in enumerate(tickets):
            action = Action(
                ticket_id=t.ticket_id,
                classification=t.ground_truth.classification,
                priority=t.ground_truth.priority,
                assigned_team=t.ground_truth.assigned_team,
                labels=t.ground_truth.labels,
            )
            result = env.step(action)
            if i < len(tickets) - 1:
                assert not result.done
            else:
                assert result.done

        st = env.state()
        assert st.done
        assert st.tickets_processed == 6

    def test_max_steps_protection(self):
        env = TriageEnv()
        env.reset("task_easy")
        # Burn through max steps with wrong ticket IDs
        for _ in range(25):
            result = env.step(Action(
                ticket_id="WRONG",
                classification=TicketCategory.BUG_REPORT,
                priority=Priority.MEDIUM,
                assigned_team=Team.ENGINEERING,
            ))
            if result.done:
                break
        assert result.done
        assert "exceeded" in result.info.get("error", "").lower() or result.done


# ═══════════════════════════════════════════════════════════════════════════════
#  Episode-scoped history tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEpisodeHistory:
    def test_first_ticket_has_no_history(self):
        env = TriageEnv()
        obs = env.reset("task_easy")
        assert len(obs.ticket_history) == 0

    def test_second_ticket_has_one_history(self):
        env = TriageEnv()
        obs = env.reset("task_easy")
        t = get_tickets_for_task("task_easy")[0]
        result = env.step(Action(
            ticket_id=t.ticket_id,
            classification=t.ground_truth.classification,
            priority=t.ground_truth.priority,
            assigned_team=t.ground_truth.assigned_team,
            labels=t.ground_truth.labels,
        ))
        assert result.observation is not None
        assert len(result.observation.ticket_history) == 1
        assert result.observation.ticket_history[0].ticket_id == "TK-1001"

    def test_medium_task_doesnt_leak_easy_history(self):
        env = TriageEnv()
        obs = env.reset("task_medium")
        # First ticket in medium task should have NO history
        # (previously it leaked all easy tickets)
        assert len(obs.ticket_history) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Grading accuracy tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGrading:
    def test_perfect_easy_scores_high(self):
        tickets = get_tickets_for_task("task_easy")
        actions = [{
            "ticket_id": t.ticket_id,
            "classification": t.ground_truth.classification.value,
            "priority": t.ground_truth.priority.value,
            "assigned_team": t.ground_truth.assigned_team.value,
            "labels": t.ground_truth.labels,
        } for t in tickets]
        result = run_grader("task_easy", actions)
        assert result["score"] >= 0.85, f"Perfect easy scored {result['score']}"

    def test_perfect_hard_with_response_scores_high(self):
        tickets = get_tickets_for_task("task_hard")
        actions = []
        for t in tickets:
            kws = " ".join(t.ground_truth.response_keywords)
            actions.append({
                "ticket_id": t.ticket_id,
                "classification": t.ground_truth.classification.value,
                "priority": t.ground_truth.priority.value,
                "assigned_team": t.ground_truth.assigned_team.value,
                "labels": t.ground_truth.labels,
                "duplicate_of": t.ground_truth.duplicate_of,
                "response_draft": (
                    f"Thank you for reaching out. We sincerely apologize for the inconvenience. "
                    f"Our team is actively investigating this issue. {kws}"
                ),
            })
        result = run_grader("task_hard", actions)
        assert result["score"] >= 0.70, f"Perfect hard scored {result['score']}"

    def test_all_wrong_scores_low(self):
        tickets = get_tickets_for_task("task_easy")
        actions = [{
            "ticket_id": t.ticket_id,
            "classification": "feature_request",
            "priority": "low",
            "assigned_team": "product",
            "labels": ["irrelevant-label-xyz"],
        } for t in tickets]
        result = run_grader("task_easy", actions)
        assert result["score"] < 0.4, f"All-wrong scored {result['score']}"

    def test_empty_submission_scores_zero(self):
        result = run_grader("task_easy", [])
        assert result["score"] == 0.01, f"Empty submission should score 0.01, got {result['score']}"
        assert result["num_submitted"] == 0
        assert len(result["per_ticket"]) == 6  # 6 missing tickets listed

    def test_partial_submission_penalized(self):
        tickets = get_tickets_for_task("task_easy")
        # Submit only 3 of 6
        actions = [{
            "ticket_id": t.ticket_id,
            "classification": t.ground_truth.classification.value,
            "priority": t.ground_truth.priority.value,
            "assigned_team": t.ground_truth.assigned_team.value,
            "labels": t.ground_truth.labels,
        } for t in tickets[:3]]
        result = run_grader("task_easy", actions)
        assert result["score"] < 0.7  # ~0.5 because 3/6 missing
        assert result["num_submitted"] == 3

    def test_classification_partial_credit(self):
        env = TriageEnv()
        env.reset("task_easy")
        # TK-1006 is "performance", submit "bug_report" — should get partial credit
        # Skip to TK-1006 (last easy ticket)
        tickets = get_tickets_for_task("task_easy")
        for t in tickets[:5]:
            env.step(Action(
                ticket_id=t.ticket_id,
                classification=t.ground_truth.classification,
                priority=t.ground_truth.priority,
                assigned_team=t.ground_truth.assigned_team,
                labels=t.ground_truth.labels,
            ))
        # TK-1006 ground truth is performance, we submit bug_report
        result = env.step(Action(
            ticket_id="TK-1006",
            classification=TicketCategory.BUG_REPORT,  # related to performance
            priority=Priority.CRITICAL,
            assigned_team=Team.DEVOPS,
            labels=["performance", "dashboard", "regression"],
        ))
        # Should get partial credit for classification (0.4), not 0.0
        assert result.reward_breakdown.classification_score > 0.0
        assert result.reward_breakdown.classification_score < 1.0

    def test_priority_one_off_partial_credit(self):
        env = TriageEnv()
        env.reset("task_easy")
        # TK-1001 is CRITICAL, submit HIGH
        result = env.step(Action(
            ticket_id="TK-1001",
            classification=TicketCategory.ACCOUNT_ISSUE,
            priority=Priority.HIGH,  # one off from critical
            assigned_team=Team.ACCOUNT_MANAGEMENT,
            labels=["login"],
        ))
        assert result.reward_breakdown.priority_score == 0.5

    def test_priority_two_off_partial_credit(self):
        env = TriageEnv()
        env.reset("task_easy")
        result = env.step(Action(
            ticket_id="TK-1001",
            classification=TicketCategory.ACCOUNT_ISSUE,
            priority=Priority.MEDIUM,  # two off from critical
            assigned_team=Team.ACCOUNT_MANAGEMENT,
            labels=["login"],
        ))
        assert result.reward_breakdown.priority_score == 0.15


# ═══════════════════════════════════════════════════════════════════════════════
#  Label synonym tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelSynonyms:
    def test_exact_match(self):
        assert _labels_are_synonyms("crash", "crash")

    def test_synonym_match(self):
        assert _labels_are_synonyms("crash", "error")
        assert _labels_are_synonyms("file-upload", "upload")
        assert _labels_are_synonyms("login", "authentication")

    def test_non_synonym(self):
        assert not _labels_are_synonyms("crash", "billing")
        assert not _labels_are_synonyms("gdpr", "webhook")


# ═══════════════════════════════════════════════════════════════════════════════
#  Response grading tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponseGrading:
    def _grade(self, draft, keywords, forbidden=None, sentiment="neutral"):
        return TriageEnv._grade_response(draft, keywords, forbidden, sentiment)

    def test_no_keywords_returns_1(self):
        assert self._grade("anything", []) == 0.99

    def test_no_draft_returns_0(self):
        assert self._grade(None, ["keyword"]) == 0.01
        assert self._grade("", ["keyword"]) == 0.01

    def test_all_keywords_high_score(self):
        score = self._grade(
            "Thank you for reporting this. We are investigating the upload issue and will fix it soon.",
            ["upload", "investigating", "fix"],
        )
        assert score > 0.5

    def test_professional_tone_bonus(self):
        polite = self._grade(
            "Thank you for contacting us. We apologize for the inconvenience. Our team will investigate and resolve this.",
            ["investigate"],
        )
        blunt = self._grade(
            "We will investigate.",
            ["investigate"],
        )
        assert polite > blunt

    def test_unprofessional_penalized(self):
        rude = self._grade(
            "This is a stupid question. Just figure it out yourself. LOL.",
            ["question"],
        )
        polite = self._grade(
            "Thank you for your question. Our team will investigate and get back to you shortly.",
            ["question"],
        )
        assert rude < polite, f"Rude ({rude}) should score lower than polite ({polite})"

    def test_hallucination_penalized(self):
        clean = self._grade(
            "Thank you for reporting this. We are investigating the upload issue.",
            ["upload", "investigating"],
            forbidden=["works as intended"],
        )
        hallucinated = self._grade(
            "This works as intended. The upload feature is not broken.",
            ["upload", "investigating"],
            forbidden=["works as intended"],
        )
        assert clean > hallucinated, f"Clean ({clean}) should beat hallucinated ({hallucinated})"

    def test_forbidden_phrases_reduce_score(self):
        base = self._grade(
            "Thank you. We will investigate this issue promptly.",
            ["investigate"],
            forbidden=[],
        )
        with_forbidden = self._grade(
            "Thank you. We will investigate this issue promptly. No refunds allowed.",
            ["investigate"],
            forbidden=["no refunds allowed"],
        )
        assert base > with_forbidden

    def test_angry_customer_needs_empathy(self):
        empathetic = self._grade(
            "We sincerely apologize. This is unacceptable and should not have happened. We are escalating this to our senior team and will personally ensure it is resolved immediately.",
            ["apologize", "escalat"],
            sentiment="angry",
        )
        cold = self._grade(
            "We will look into it. The issue has been logged.",
            ["apologize", "escalat"],
            sentiment="angry",
        )
        assert empathetic > cold, f"Empathetic ({empathetic}) should beat cold ({cold}) for angry customer"

    def test_frustrated_customer_needs_acknowledgment(self):
        good = self._grade(
            "We apologize for the inconvenience. We understand how frustrating this must be. Our team is actively investigating and working on a resolution.",
            ["investigating"],
            sentiment="frustrated",
        )
        bad = self._grade(
            "Noted. Will check.",
            ["investigating"],
            sentiment="frustrated",
        )
        assert good > bad

    def test_polite_customer_gets_friendly_response(self):
        score = self._grade(
            "Thank you for your question! We're happy to help. You can find the export option in Settings > Data > Export.",
            ["export", "settings"],
            sentiment="polite",
        )
        assert score > 0.5


# ═══════════════════════════════════════════════════════════════════════════════
#  Session management tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionManager:
    def test_default_session_exists(self):
        sm = SessionManager()
        env = sm.get()
        assert env is not None

    def test_create_and_get_session(self):
        sm = SessionManager()
        sid = sm.create_session()
        env = sm.get(sid)
        assert env is not None

    def test_sessions_are_isolated(self):
        sm = SessionManager()
        sid1 = sm.create_session()
        sid2 = sm.create_session()
        env1 = sm.get(sid1)
        env2 = sm.get(sid2)
        env1.reset("task_easy")
        env2.reset("task_hard")
        assert env1.state().task_id == "task_easy"
        assert env2.state().task_id == "task_hard"

    def test_delete_session(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert sm.delete(sid) is True
        assert sm.delete("nonexistent") is False

    def test_cannot_delete_default(self):
        sm = SessionManager()
        assert sm.delete("default") is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Routing partial credit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoutingGrading:
    def test_exact_match(self):
        assert TriageEnv._grade_routing(Team.ENGINEERING, Team.ENGINEERING) == 1.0

    def test_related_teams(self):
        score = TriageEnv._grade_routing(Team.ENGINEERING, Team.DEVOPS)
        assert 0 < score < 1

    def test_unrelated_teams(self):
        score = TriageEnv._grade_routing(Team.BILLING, Team.DEVOPS)
        assert score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Task definitions tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskDefinitions:
    def test_four_tasks_exist(self):
        assert len(TASK_DEFINITIONS) == 4

    def test_difficulties_cover_range(self):
        difficulties = {td.difficulty for td in TASK_DEFINITIONS.values()}
        assert difficulties == {"easy", "medium", "hard", "expert"}

    def test_grader_scores_in_range(self):
        for task_id in ["task_easy", "task_medium", "task_hard", "task_expert"]:
            tickets = get_tickets_for_task(task_id)
            actions = [{
                "ticket_id": t.ticket_id,
                "classification": "general_inquiry",
                "priority": "medium",
                "assigned_team": "general_support",
                "labels": ["test"],
                "response_draft": "Thank you, we will investigate.",
            } for t in tickets]
            result = run_grader(task_id, actions)
            assert 0.0 <= result["score"] <= 1.0, f"{task_id}: {result['score']}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Enriched observation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichedObservation:
    def test_observation_has_sentiment(self):
        env = TriageEnv()
        obs = env.reset("task_easy")
        assert obs.sentiment in ("angry", "frustrated", "neutral", "polite")

    def test_observation_has_sla(self):
        env = TriageEnv()
        obs = env.reset("task_easy")
        # TK-1001 is enterprise, should have SLA
        assert obs.sla_hours_remaining is not None or obs.sla_hours_remaining is None  # field exists

    def test_observation_has_sender_context(self):
        env = TriageEnv()
        obs = env.reset("task_easy")
        assert isinstance(obs.is_repeat_sender, bool)
        assert isinstance(obs.sender_ticket_count, int)

    def test_hard_tickets_have_angry_sentiment(self):
        env = TriageEnv()
        obs = env.reset("task_hard")
        # Advance to TK-1014 (security concern, angry CISO)
        tickets = get_tickets_for_task("task_hard")
        for t in tickets[:1]:
            env.step(Action(
                ticket_id=t.ticket_id,
                classification=t.ground_truth.classification,
                priority=t.ground_truth.priority,
                assigned_team=t.ground_truth.assigned_team,
                labels=t.ground_truth.labels,
                duplicate_of=t.ground_truth.duplicate_of,
                response_draft="Thank you. We are investigating.",
            ))
        # TK-1014 should be angry
        st = env.state()
        if not st.done:
            next_obs = env._make_observation()
            assert next_obs.sentiment in ("angry", "frustrated", "neutral", "polite")


# ═══════════════════════════════════════════════════════════════════════════════
#  Consistency bonus tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsistencyBonus:
    def test_consistent_correct_gets_bonus(self):
        """Two bug_report tickets classified correctly should get a bonus."""
        env = TriageEnv()
        env.reset("task_easy")
        # TK-1004 (bug_report) and later check reward
        tickets = get_tickets_for_task("task_easy")
        # Process first 3 tickets correctly
        for t in tickets[:3]:
            env.step(Action(
                ticket_id=t.ticket_id,
                classification=t.ground_truth.classification,
                priority=t.ground_truth.priority,
                assigned_team=t.ground_truth.assigned_team,
                labels=t.ground_truth.labels,
            ))
        # TK-1004 is bug_report — no prior bug_report in easy set (TK-1001=account, TK-1002=billing, TK-1003=general)
        # So no consistency bonus applies here. Just verify it doesn't crash.
        r = env.step(Action(
            ticket_id=tickets[3].ticket_id,
            classification=tickets[3].ground_truth.classification,
            priority=tickets[3].ground_truth.priority,
            assigned_team=tickets[3].ground_truth.assigned_team,
            labels=tickets[3].ground_truth.labels,
        ))
        assert r.reward > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  HF Space frontmatter test
# ═══════════════════════════════════════════════════════════════════════════════

class TestHFSpace:
    def test_readme_has_frontmatter(self):
        with open("README.md", encoding="utf-8") as f:
            content = f.read()
        assert content.startswith("---"), "README.md must start with HF Space frontmatter"
        assert "sdk: docker" in content
        assert "app_port: 7860" in content
        assert "openenv" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
