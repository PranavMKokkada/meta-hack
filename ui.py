"""Gradio UI for the Technical Support & Bug Triage Hub.

Provides an interactive interface for humans to triage tickets and see scores.
Mounted at /ui on the FastAPI app.
"""

from __future__ import annotations

import gradio as gr

from models import Action, TicketCategory, Priority, Team
from environment import TriageEnv
from data import get_tickets_for_task


def create_gradio_app() -> gr.Blocks:
    env = TriageEnv()

    def reset_env(task_id):
        obs = env.reset(task_id)
        ticket_text = (
            f"**{obs.ticket_id}** — {obs.subject}\n\n"
            f"**From:** {obs.sender_email} ({obs.sender_tier})\n"
            f"**Sentiment:** {obs.sentiment} | "
            f"**SLA:** {obs.sla_hours_remaining}h remaining | "
            f"**Repeat sender:** {obs.is_repeat_sender} ({obs.sender_ticket_count} prior)\n\n"
            f"---\n\n{obs.body}\n\n"
        )
        if obs.attachments:
            ticket_text += f"**Attachments:** {', '.join(obs.attachments)}\n\n"
        if obs.knowledge_base:
            ticket_text += f"**Internal Docs:** {obs.knowledge_base}\n\n"

        history_text = ""
        if obs.ticket_history:
            history_text = "**Previous tickets this episode:**\n"
            for h in obs.ticket_history:
                history_text += f"- [{h.ticket_id}] ({h.category}) {h.subject}\n"

        status = f"Step {obs.step_number}/{obs.total_steps} | Task: {task_id}"
        return ticket_text, history_text, status, "", "0.0", obs.ticket_id

    def submit_action(ticket_id, classification, priority, team, labels_str,
                      duplicate_of, response_draft, escalate):
        labels = [l.strip() for l in labels_str.split(",") if l.strip()] if labels_str else []
        dup = duplicate_of.strip() if duplicate_of and duplicate_of.strip() else None
        try:
            action = Action(
                ticket_id=ticket_id,
                classification=TicketCategory(classification),
                priority=Priority(priority),
                assigned_team=Team(team),
                labels=labels,
                duplicate_of=dup,
                response_draft=response_draft if response_draft else None,
                escalate=escalate,
            )
            result = env.step(action)
        except Exception as e:
            return f"Error: {e}", "", "Error", "", "0.0", ticket_id

        # Format reward breakdown
        b = result.reward_breakdown
        reward_text = (
            f"**Total: {result.reward:.4f}**\n\n"
            f"| Dimension | Score |\n|---|---|\n"
            f"| Classification | {b.classification_score:.2f} |\n"
            f"| Priority | {b.priority_score:.2f} |\n"
            f"| Routing | {b.routing_score:.2f} |\n"
            f"| Labels | {b.labels_score:.2f} |\n"
            f"| Duplicate | {b.duplicate_score:.2f} |\n"
            f"| Response | {b.response_score:.2f} |\n"
            f"| Escalation | {b.escalation_score:.2f} |\n"
        )

        if result.done:
            ep_score = result.info.get("episode_score", 0)
            status = f"EPISODE COMPLETE | Final Score: {ep_score:.4f}"
            ticket_text = f"Episode finished. Final score: **{ep_score:.4f}**\n\nClick Reset to start a new episode."
            history_text = ""
            return ticket_text, history_text, status, reward_text, f"{ep_score:.4f}", ""
        else:
            obs = result.observation
            ticket_text = (
                f"**{obs.ticket_id}** — {obs.subject}\n\n"
                f"**From:** {obs.sender_email} ({obs.sender_tier})\n"
                f"**Sentiment:** {obs.sentiment} | "
                f"**SLA:** {obs.sla_hours_remaining}h | "
                f"**Repeat:** {obs.is_repeat_sender}\n\n"
                f"---\n\n{obs.body}\n\n"
            )
            if obs.knowledge_base:
                ticket_text += f"**Internal Docs:** {obs.knowledge_base}\n\n"

            history_text = ""
            if obs.ticket_history:
                history_text = "**Previous tickets:**\n"
                for h in obs.ticket_history:
                    history_text += f"- [{h.ticket_id}] ({h.category}) {h.subject}\n"

            st = env.state()
            status = f"Step {obs.step_number}/{obs.total_steps} | Cumulative: {st.cumulative_reward:.4f}"
            return ticket_text, history_text, status, reward_text, f"{result.reward:.4f}", obs.ticket_id

    with gr.Blocks(
        title="Triage Hub",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
    ) as demo:
        gr.Markdown("# Technical Support & Bug Triage Hub\nTriage tickets. Get scored. See how good you are.")

        with gr.Row():
            task_dd = gr.Dropdown(
                ["task_easy", "task_medium", "task_hard", "task_expert"],
                value="task_easy", label="Task"
            )
            reset_btn = gr.Button("Reset Episode", variant="primary")

        status_bar = gr.Markdown("Click Reset to start")

        with gr.Row():
            with gr.Column(scale=3):
                ticket_display = gr.Markdown("No ticket loaded", label="Current Ticket")
                history_display = gr.Markdown("", label="Ticket History")
            with gr.Column(scale=2):
                reward_display = gr.Markdown("", label="Last Reward")
                score_display = gr.Textbox(label="Last Step Score", value="0.0", interactive=False)

        gr.Markdown("### Your Triage Decision")

        with gr.Row():
            ticket_id_box = gr.Textbox(label="Ticket ID", interactive=False)
            cls_dd = gr.Dropdown(
                [e.value for e in TicketCategory], label="Classification"
            )
            pri_dd = gr.Dropdown(
                [e.value for e in Priority], label="Priority"
            )

        with gr.Row():
            team_dd = gr.Dropdown(
                [e.value for e in Team], label="Assigned Team"
            )
            labels_box = gr.Textbox(label="Labels (comma-separated)", placeholder="login, password-reset")
            dup_box = gr.Textbox(label="Duplicate Of (ticket ID or empty)", placeholder="TK-1004")

        with gr.Row():
            escalate_cb = gr.Checkbox(label="Escalate to Management")
            related_box = gr.Textbox(label="Related To (ticket ID or empty)", placeholder="TK-1008", visible=True)

        response_box = gr.Textbox(label="Response Draft", lines=3, placeholder="Write your response to the customer...")
        submit_btn = gr.Button("Submit Action", variant="primary")

        reset_btn.click(
            reset_env, inputs=[task_dd],
            outputs=[ticket_display, history_display, status_bar, reward_display, score_display, ticket_id_box]
        )
        submit_btn.click(
            submit_action,
            inputs=[ticket_id_box, cls_dd, pri_dd, team_dd, labels_box, dup_box, response_box, escalate_cb],
            outputs=[ticket_display, history_display, status_bar, reward_display, score_display, ticket_id_box]
        )

    return demo
