"""
Inference script — Heuristic Doctor agent.

Runs all 3 difficulty tasks against the OpenEnv server and logs
[START], [STEP], and [END] per hackathon rules.
"""

import asyncio
import os
from openai import OpenAI

from client import HospitalERTriageEnv
from models import HospitalAction, PatientPresentation

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Configured OpenAI client for any LLM calls
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-token"
)

# Target: remote HF Space or local server
env_server_url = API_BASE_URL


def _to_patient(p) -> PatientPresentation:
    if isinstance(p, PatientPresentation):
        return p
    if isinstance(p, dict):
        return PatientPresentation(**p)
    raise TypeError(f"Unexpected type: {type(p)}")


def _calc_esi(p: PatientPresentation) -> int:
    """Heuristic ESI estimate from vitals + chief complaint."""
    c = p.chief_complaint.lower()
    if p.heart_rate > 120 or p.spo2 < 90 or any(k in c for k in ("breathing", "stroke", "confusion", "resuscit")):
        return 1
    if p.heart_rate > 100 or p.systolic_bp < 95 or any(k in c for k in ("chest", "abdominal")):
        return 2
    if p.pain_scale > 7 or "fracture" in c:
        return 3
    if "ankle" in c:
        return 4
    return 5


async def run_task(task_name: str, queue_size: int, total_beds: int):
    print(f"[START] Task: {task_name}")
    async with HospitalERTriageEnv(base_url=env_server_url) as env:
        result = await env.reset(
            episode_id=task_name,
            queue_size=queue_size,
            bed_count=total_beds,
        )

        while not result.done:
            obs = result.observation
            patients = [_to_patient(p) for p in obs.waiting_room]

            # Priority 1: discharge in-bed patients to free beds
            in_bed = [p for p in patients if p.status == "in_bed"]
            waiting = [p for p in patients if p.status == "waiting"]

            if in_bed:
                target = in_bed[0]
                action = HospitalAction(
                    action_type="diagnostic",
                    assigned_patient_id=target.patient_id,
                    test_type="labs",
                )
                print(f"[STEP] Diagnose/discharge patient {target.patient_id}")
            elif waiting:
                # Triage the most critical waiting patient
                target = min(waiting, key=_calc_esi)
                esi = _calc_esi(target)
                action = HospitalAction(
                    action_type="triage",
                    assigned_patient_id=target.patient_id,
                    assigned_doctor_id="Dr. Heuristic",
                    esi_level=esi,
                    allocate_bed=True,
                    notes="Heuristic vitals-based triage.",
                )
                print(f"[STEP] Triage patient {target.patient_id} (ESI {esi})")
            else:
                # No actionable patients (all expired or none left)
                break

            result = await env.step(action)
            await asyncio.sleep(0.1)  # small delay so logs are human-readable

        state = await env.state()
        total_critical = state.get("critical_patients_total", 0)
        saved = state.get("critical_patients_saved_in_time", 0)
        success = (total_critical == 0) or (saved == total_critical)
        print(f"[END] Task: {task_name} | Score: {state.get('total_reward', 0.0)} | Success: {str(success).lower()}\n")


async def main():
    tasks = [
        ("Easy",   5,  10),
        ("Medium", 15, 20),
        ("Hard",   30, 10),
    ]
    print(f"Starting Heuristic Inference targeting {env_server_url}...\n")
    for name, q_size, beds in tasks:
        await run_task(name, q_size, beds)


if __name__ == "__main__":
    asyncio.run(main())
