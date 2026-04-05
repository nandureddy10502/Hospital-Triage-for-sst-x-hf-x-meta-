"""
Hospital ER Triage — Gradio Dashboard + FastAPI entry point.

The dashboard calls the environment DIRECTLY (in-process) rather than
going through HTTP, which avoids the self-connection deadlock that
occurred when Gradio and the FastAPI server share the same process.
"""

import gradio as gr
from server.app import app as api_app, create_environment
from models import HospitalAction, PatientPresentation
from grader import grade_episode

# ── In-process environment instance (no HTTP round-trip) ─────────────────────
_env = create_environment()


def _to_patient(p) -> PatientPresentation:
    """Safely coerce a raw dict or PatientPresentation object."""
    if isinstance(p, PatientPresentation):
        return p
    if isinstance(p, dict):
        return PatientPresentation(**p)
    raise TypeError(f"Unexpected patient type: {type(p)}")


def format_queue(waiting_room) -> list:
    rows = []
    for raw in waiting_room:
        p = _to_patient(raw)
        rows.append([
            p.patient_id,
            p.status,
            f"{p.health_score:.1f}",
            p.age,
            p.sex,
            p.chief_complaint,
            p.heart_rate,
            p.spo2,
            p.systolic_bp,
        ])
    return rows


# ── Gradio Layout ─────────────────────────────────────────────────────────────

custom_theme = gr.themes.Soft(
    primary_hue="slate",
    neutral_hue="zinc",
).set(
    body_background_fill="*neutral_950",
    body_text_color="white",
    background_fill_primary="*neutral_900",
    block_background_fill="*neutral_800",
    block_label_text_color="white",
)

with gr.Blocks(title="Hospital ER Triage Simulator") as demo:
    gr.Markdown("# 🏥 Hospital ER Triage Dashboard")
    gr.Markdown(
        "Real-time ER simulator. **Health decays every step** — act fast! "
        "Triage puts patients in beds, Diagnose discharges them."
    )

    current_waiting_room = gr.State([])

    with gr.Row():
        # ── Sidebar ──────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### ⚙️ Setup Parameters")
            bed_count     = gr.Slider(1,  50,  value=10,  step=1,   label="Bed Count")
            doctor_count  = gr.Slider(1,  10,  value=3,   step=1,   label="Doctor Capacity")
            critical_rate = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="Criticality Rate")
            gr.Markdown("---")
            reset_btn    = gr.Button("🔄 Reset Hospital",   variant="primary")
            triage_btn   = gr.Button("🚑 Quick Triage",     variant="secondary")
            diagnose_btn = gr.Button("🧪 Diagnose In-Bed",  variant="secondary")

        # ── Main panel ───────────────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Row():
                beds_display  = gr.Textbox(label="Beds Available", value="--", interactive=False)
                score_display = gr.Number(label="Cumulative Reward", value=0.0, interactive=False)
                grade_display = gr.HTML(
                    value="<div style='color:gray;font-size:22px;font-weight:bold;padding-top:8px'>Grade: Pending</div>"
                )
            feedback_display = gr.Textbox(label="Last Action Feedback", interactive=False, lines=2)
            gr.Markdown("### 🛏️ Active Patient Queue")
            headers = ["Patient ID", "Status", "Health", "Age", "Sex",
                       "Chief Complaint", "Heart Rate", "SpO2 (%)", "Systolic BP"]
            queue_table = gr.Dataframe(headers=headers, interactive=False, wrap=True)

    # ── Shared Output Builder ─────────────────────────────────────────────────

    def _build_outputs(obs):
        df        = format_queue(obs.waiting_room)
        beds_txt  = f"{obs.beds_available} / {obs.beds_total}"
        state     = _env.state
        score_num = round(state.total_reward, 2)

        if state.is_done:
            grade = grade_episode({
                "critical_patients_total": state.critical_patients_total,
                "critical_patients_saved_in_time": state.critical_patients_saved_in_time,
            })
            grade_html = (
                "<div style='color:#4ade80;font-size:22px;font-weight:bold'>Grade: 1.0 \u2705</div>"
                if grade >= 1.0 else
                f"<div style='color:#ef4444;font-size:22px;font-weight:bold'>Grade: {grade:.2f} \u274c</div>"
            )
        else:
            grade_html = "<div style='color:gray;font-size:22px;font-weight:bold'>Grade: In Progress…</div>"

        raw_patients = [p.model_dump() for p in obs.waiting_room]
        return df, beds_txt, score_num, obs.message, grade_html, raw_patients

    # ── Handlers ─────────────────────────────────────────────────────────────

    def handle_reset(current_q, b_count, d_count, c_rate):
        try:
            obs = _env.reset(
                episode_id="dashboard_session",
                queue_size=15,
                bed_count=int(b_count),
                doctor_count=int(d_count),
                criticality_rate=float(c_rate),
            )
            return _build_outputs(obs)
        except Exception as e:
            return [], "Error", 0.0, f"Reset failed: {e}", "", []

    def handle_triage(current_q):
        patients = [_to_patient(p) for p in current_q]
        waiting  = [p for p in patients if p.status == "waiting"]

        if not waiting:
            no_msg = "No waiting patients. Try Diagnose to free a bed first."
            state  = _env.state
            return (format_queue([_to_patient(p) for p in current_q]),
                    "--", round(state.total_reward, 2), no_msg,
                    "<div style='color:gray;font-size:22px;font-weight:bold'>Grade: In Progress…</div>",
                    current_q)

        sickest = max(waiting, key=lambda p: p.heart_rate)
        esi     = 1 if sickest.heart_rate > 110 else (2 if sickest.heart_rate > 95 else 3)

        action = HospitalAction(
            action_type="triage",
            assigned_patient_id=sickest.patient_id,
            assigned_doctor_id="Dr. Dashboard",
            esi_level=esi,
            allocate_bed=True,
            notes="Auto-triaged via dashboard heuristic.",
        )
        try:
            obs = _env.step(action)
            return _build_outputs(obs)
        except Exception as e:
            return [], "Error", 0.0, f"Triage failed: {e}", "", current_q

    def handle_diagnose(current_q):
        patients = [_to_patient(p) for p in current_q]
        in_bed   = [p for p in patients if p.status == "in_bed"]

        if not in_bed:
            state  = _env.state
            return (format_queue([_to_patient(p) for p in current_q]),
                    f"{_env._beds_available} / {_env._total_beds}",
                    round(state.total_reward, 2),
                    "No in-bed patients. Triage someone first.",
                    "<div style='color:gray;font-size:22px;font-weight:bold'>Grade: In Progress…</div>",
                    current_q)

        action = HospitalAction(
            action_type="diagnostic",
            assigned_patient_id=in_bed[0].patient_id,
            test_type="labs",
        )
        try:
            obs = _env.step(action)
            return _build_outputs(obs)
        except Exception as e:
            return [], "Error", 0.0, f"Diagnostic failed: {e}", "", current_q

    # ── Wire up ───────────────────────────────────────────────────────────────

    _outputs = [queue_table, beds_display, score_display,
                feedback_display, grade_display, current_waiting_room]

    reset_btn.click(
        fn=handle_reset,
        inputs=[current_waiting_room, bed_count, doctor_count, critical_rate],
        outputs=_outputs,
    )
    triage_btn.click(
        fn=handle_triage,
        inputs=[current_waiting_room],
        outputs=_outputs,
    )
    diagnose_btn.click(
        fn=handle_diagnose,
        inputs=[current_waiting_room],
        outputs=_outputs,
    )
    demo.load(
        fn=handle_reset,
        inputs=[current_waiting_room, bed_count, doctor_count, critical_rate],
        outputs=_outputs,
    )

# Mount onto the OpenEnv FastAPI app
app = gr.mount_gradio_app(api_app, demo, path="/", theme=custom_theme)
