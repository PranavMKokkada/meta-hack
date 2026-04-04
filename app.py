"""FastAPI application — exposes the OpenEnv-compliant HTTP interface.

Endpoints (OpenEnv spec):
    GET  /health             — health check (OpenEnv standard)
    GET  /metadata           — environment metadata (OpenEnv standard)
    GET  /schema             — action/observation/state schemas (OpenEnv standard)
    POST /reset              — reset(task_id) → Observation
    POST /step               — step(action)   → StepResult
    GET  /state              — state()        → EnvState
    GET  /tasks              — list tasks + action schema
    POST /grader             — run grader on submitted actions
    POST /baseline           — trigger baseline inference
    GET  /                   — root health check
    GET  /validate           — internal pre-submission validation
    POST /sessions/create    — create isolated session
    DELETE /sessions/{id}    — delete session
    GET  /sessions           — list active sessions
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import Action, Observation, StepResult, EnvState
from environment import SessionManager
from tasks import TASK_DEFINITIONS, run_grader

app = FastAPI(
    title="Technical Support & Bug Triage Hub",
    description=(
        "OpenEnv environment: triage customer support tickets and bug reports. "
        "Supports session-based isolation for concurrent users."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session manager holds all environment instances
sessions = SessionManager()


# ── Request / Response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class GraderRequest(BaseModel):
    task_id: str
    actions: list[dict]


# ── Helper: resolve session from header ────────────────────────────────────────

def _get_env(session_id: Optional[str] = None):
    return sessions.get(session_id)


# ── OpenEnv Standard Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "environment": "technical-support-bug-triage-hub", "version": "1.0.0"}


@app.get("/health")
def health():
    """OpenEnv standard health check."""
    return {
        "status": "healthy",
        "environment": "technical-support-bug-triage-hub",
        "version": "1.0.0",
    }


@app.get("/metadata")
def metadata():
    """OpenEnv standard metadata endpoint."""
    return {
        "name": "technical-support-bug-triage-hub",
        "description": (
            "An OpenEnv environment that simulates real-world technical support "
            "and bug triage. An AI agent reads incoming tickets (customer support "
            "+ bug reports), classifies them, assigns priority and team, detects "
            "duplicates, and drafts customer responses."
        ),
        "version": "1.0.0",
        "author": "Andrea",
        "tags": ["openenv", "customer-support", "bug-triage", "ticket-routing"],
        "tasks": list(TASK_DEFINITIONS.keys()),
        "num_tasks": len(TASK_DEFINITIONS),
    }


@app.get("/schema")
def schema():
    """OpenEnv standard schema endpoint — returns action, observation, and state schemas."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvState.model_json_schema(),
    }


# ── Core Environment Endpoints ─────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None, task_id: Optional[str] = None, x_session_id: Optional[str] = Header(None)):
    # Accept task_id from either request body or query parameter
    actual_task_id = task_id or (req.task_id if req else None) or "task_easy"
    
    if actual_task_id not in TASK_DEFINITIONS:
        raise HTTPException(400, f"Unknown task_id: {actual_task_id}. Valid: {list(TASK_DEFINITIONS)}")
    env = _get_env(x_session_id)
    obs = env.reset(actual_task_id)
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action, x_session_id: Optional[str] = Header(None)):
    env = _get_env(x_session_id)
    result = env.step(action)
    return result


@app.get("/state", response_model=EnvState)
def state(x_session_id: Optional[str] = Header(None)):
    env = _get_env(x_session_id)
    return env.state()


# ── Task & Grader Endpoints ────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """Return all tasks with their descriptions and required action fields."""
    tasks = []
    for td in TASK_DEFINITIONS.values():
        tasks.append({
            "task_id": td.task_id,
            "name": td.name,
            "description": td.description,
            "difficulty": td.difficulty,
            "num_tickets": td.num_tickets,
            "action_fields_required": td.action_fields_required,
            "scoring_summary": td.scoring_summary,
            "action_schema": Action.model_json_schema(),
        })
    return {"tasks": tasks}


@app.post("/grader")
def grader(req: GraderRequest):
    """Score a completed episode given a list of actions."""
    if req.task_id not in TASK_DEFINITIONS:
        raise HTTPException(400, f"Unknown task_id: {req.task_id}")
    return run_grader(req.task_id, req.actions)


# ── Session Management ─────────────────────────────────────────────────────────

@app.post("/sessions/create")
def create_session():
    """Create a new isolated session and return its ID."""
    session_id = sessions.create_session()
    return {"session_id": session_id}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    ok = sessions.delete(session_id)
    if not ok:
        raise HTTPException(404, "Session not found or is the default session")
    return {"deleted": session_id}


@app.get("/sessions")
def list_sessions():
    return {"sessions": sessions.list_sessions()}


# ── Baseline ───────────────────────────────────────────────────────────────────

@app.post("/baseline")
def baseline():
    """Run the baseline inference script and return scores for all tasks."""
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            500,
            "HF_TOKEN / API_KEY / OPENAI_API_KEY not set. Set one as an environment variable to run the baseline.",
        )
    try:
        env_vars = {
            **os.environ,
            "ENV_API_URL": "http://localhost:7860",
        }
        if not env_vars.get("HF_TOKEN") and not env_vars.get("API_KEY"):
            env_vars["API_KEY"] = api_key

        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env_vars,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Baseline failed: {result.stderr}")
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Baseline timed out after 300s")


# ── Pre-Submission Validation ──────────────────────────────────────────────────

@app.get("/validate")
def validate():
    """Run internal pre-submission validation checks."""
    results = {}
    all_passed = True

    # 1. Check all 3 tasks exist
    try:
        task_ids = list(TASK_DEFINITIONS.keys())
        assert len(task_ids) >= 3, f"Need 3+ tasks, got {len(task_ids)}"
        results["tasks_exist"] = {"passed": True, "tasks": task_ids}
    except Exception as e:
        results["tasks_exist"] = {"passed": False, "error": str(e)}
        all_passed = False

    # 2. Test reset/step/state for each task
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            from environment import TriageEnv
            from data import get_tickets_for_task
            test_env = TriageEnv()
            obs = test_env.reset(task_id)
            assert obs.ticket_id, "Observation missing ticket_id"
            assert obs.task_id == task_id

            st = test_env.state()
            assert not st.done
            assert st.current_step == 0

            tickets = get_tickets_for_task(task_id)
            test_action = Action(
                ticket_id=obs.ticket_id,
                classification=tickets[0].ground_truth.classification,
                priority=tickets[0].ground_truth.priority,
                assigned_team=tickets[0].ground_truth.assigned_team,
                labels=tickets[0].ground_truth.labels,
                duplicate_of=tickets[0].ground_truth.duplicate_of,
                response_draft="Thank you for contacting us. We are investigating this issue and will update you shortly.",
            )
            step_result = test_env.step(test_action)
            assert step_result.reward > 0, f"Perfect action got reward {step_result.reward}"
            assert isinstance(step_result.reward_breakdown.total, float)

            results[f"env_{task_id}"] = {"passed": True, "first_step_reward": step_result.reward}
        except Exception as e:
            results[f"env_{task_id}"] = {"passed": False, "error": str(e), "trace": traceback.format_exc()}
            all_passed = False

    # 3. Test grader returns scores in 0.0-1.0
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            from data import get_tickets_for_task as gtft
            tickets = gtft(task_id)
            perfect_actions = []
            for t in tickets:
                perfect_actions.append({
                    "ticket_id": t.ticket_id,
                    "classification": t.ground_truth.classification.value,
                    "priority": t.ground_truth.priority.value,
                    "assigned_team": t.ground_truth.assigned_team.value,
                    "labels": t.ground_truth.labels,
                    "duplicate_of": t.ground_truth.duplicate_of,
                    "response_draft": (
                        "Thank you for reaching out. We sincerely apologize for the inconvenience. "
                        "Our team is actively investigating this issue and we will provide an update "
                        "shortly. " + " ".join(t.ground_truth.response_keywords)
                    ),
                })
            grader_result = run_grader(task_id, perfect_actions)
            score = grader_result["score"]
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"
            assert score > 0.5, f"Perfect actions scored only {score}"
            results[f"grader_{task_id}"] = {"passed": True, "score": score}
        except Exception as e:
            results[f"grader_{task_id}"] = {"passed": False, "error": str(e)}
            all_passed = False

    # 4. Test grader with empty submission
    try:
        empty_result = run_grader("task_easy", [])
        assert empty_result["score"] == 0.0
        assert empty_result["num_submitted"] == 0
        results["grader_empty"] = {"passed": True}
    except Exception as e:
        results["grader_empty"] = {"passed": False, "error": str(e)}
        all_passed = False

    # 5. Check openenv.yaml exists
    try:
        import yaml
        with open("openenv.yaml") as f:
            config = yaml.safe_load(f)
        assert "name" in config
        results["openenv_yaml"] = {"passed": True}
    except ImportError:
        exists = os.path.exists("openenv.yaml")
        results["openenv_yaml"] = {"passed": exists, "note": "pyyaml not installed, checked existence only"}
        if not exists:
            all_passed = False
    except Exception as e:
        results["openenv_yaml"] = {"passed": False, "error": str(e)}
        all_passed = False

    return {
        "all_passed": all_passed,
        "checks": results,
    }


# ── Gradio UI ──────────────────────────────────────────────────────────────────

try:
    import gradio as gr
    from ui import create_gradio_app
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")
except ImportError:
    pass  # Gradio not installed — API-only mode


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
