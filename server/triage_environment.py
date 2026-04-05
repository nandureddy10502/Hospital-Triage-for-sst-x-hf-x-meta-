"""
Hospital ER Triage — Environment implementation (server-side).
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from models import (
    PatientPresentation,
    HospitalAction,
    TriageObservation,
    TriageState,
)

# ---------------------------------------------------------------------------
# Patient-generation helpers
# ---------------------------------------------------------------------------

_CHIEF_COMPLAINTS_SEVERE: list[dict[str, Any]] = [
    {"complaint": "Difficulty breathing", "symptoms": ["wheezing", "cyanosis", "accessory muscle use"], "ideal_esi": 1},
    {"complaint": "High fever and confusion", "symptoms": ["fever", "altered mental status", "neck stiffness"], "ideal_esi": 1},
    {"complaint": "Stroke symptoms", "symptoms": ["facial droop", "arm weakness", "slurred speech"], "ideal_esi": 1},
    {"complaint": "Chest pain", "symptoms": ["chest tightness", "shortness of breath", "diaphoresis"], "ideal_esi": 2},
    {"complaint": "Severe abdominal pain", "symptoms": ["nausea", "vomiting", "guarding"], "ideal_esi": 2},
]

_CHIEF_COMPLAINTS_STABLE: list[dict[str, Any]] = [
    {"complaint": "Migraine headache", "symptoms": ["photophobia", "nausea", "throbbing pain"], "ideal_esi": 3},
    {"complaint": "Allergic reaction with hives", "symptoms": ["urticaria", "itching", "mild swelling"], "ideal_esi": 3},
    {"complaint": "Fracture — long bone", "symptoms": ["deformity", "severe pain", "swelling"], "ideal_esi": 3},
    {"complaint": "Twisted ankle", "symptoms": ["swelling", "bruising", "limited range of motion"], "ideal_esi": 4},
    {"complaint": "Minor laceration", "symptoms": ["bleeding", "pain at wound site"], "ideal_esi": 5},
    {"complaint": "Back pain — chronic", "symptoms": ["muscle spasm", "limited mobility"], "ideal_esi": 5},
    {"complaint": "Sore throat and cough", "symptoms": ["mild fever", "sore throat", "non-productive cough"], "ideal_esi": 5},
]


class PatientRecord:
    """Internal tracker for a patient in the waiting room."""
    def __init__(self, presentation: PatientPresentation, ideal_esi: int, spawn_step: int):
        self.presentation = presentation
        self.ideal_esi = ideal_esi
        self.spawn_step = spawn_step


def _generate_patient(rng: random.Random, criticality_rate: float = 0.3) -> tuple[PatientPresentation, int]:
    if rng.random() < criticality_rate:
        template = rng.choice(_CHIEF_COMPLAINTS_SEVERE)
    else:
        template = rng.choice(_CHIEF_COMPLAINTS_STABLE)

    ideal_esi: int = template["ideal_esi"]
    hr_base = {1: 130, 2: 110, 3: 95, 4: 80, 5: 75}[ideal_esi]
    sbp_base = {1: 85, 2: 100, 3: 120, 4: 125, 5: 120}[ideal_esi]

    patient = PatientPresentation(
        patient_id=str(uuid.uuid4())[:8],
        status="waiting",
        is_stable=False,
        health_score=100.0,
        age=rng.randint(1, 95),
        sex=rng.choice(["M", "F"]),
        chief_complaint=template["complaint"],
        symptoms=list(template["symptoms"]),
        heart_rate=max(40, min(250, hr_base + rng.randint(-15, 15))),
        systolic_bp=max(50, min(250, sbp_base + rng.randint(-15, 15))),
        diastolic_bp=max(30, min(150, rng.randint(55, 90))),
        respiratory_rate=max(8, min(60, rng.randint(12, 30))),
        spo2=max(70, min(100, 100 - (5 - ideal_esi) * rng.randint(0, 4))),
        temperature_c=round(rng.uniform(36.0, 40.5), 1),
        pain_scale=rng.randint(0, 10),
        arrival_mode=rng.choice(["ambulance", "walk-in", "transfer"]),
        time_since_onset_min=rng.randint(5, 4320) if rng.random() > 0.3 else None,
    )
    return patient, ideal_esi


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HospitalERTriageEnvironment(Environment[HospitalAction, TriageObservation, TriageState]):
    """Hospital ER Triage simulation environment."""

    def __init__(
        self,
        queue_size: int = 15,
        total_beds: int = 20,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._default_queue_size = queue_size
        self._default_total_beds = total_beds
        self._default_seed = seed

        # Runtime state (initialised in reset)
        self._rng = random.Random(seed)
        self._episode_id: str = ""
        self._step_count: int = 0
        self._waiting_room: list[PatientRecord] = []

        self._queue_size = queue_size
        self._total_beds = total_beds
        self._total_doctors = 3
        self._criticality_rate = 0.3

        self._critical_patients_total: int = 0
        self._critical_patients_saved_in_time: int = 0
        self._patients_triaged: int = 0

        self._beds_available: int = total_beds
        self._doctors_available: int = 3
        self._total_reward: float = 0.0
        self._done: bool = True
        self._start_time: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        effective_seed = seed if seed is not None else self._default_seed
        self._rng = random.Random(effective_seed)
        self._episode_id = episode_id or str(uuid.uuid4())

        # Accept sidebar overrides
        self._queue_size = int(kwargs.get("queue_size", self._default_queue_size))
        self._total_beds = int(kwargs.get("bed_count", self._default_total_beds))
        self._total_doctors = int(kwargs.get("doctor_count", 3))
        self._criticality_rate = float(kwargs.get("criticality_rate", 0.3))

        # Reset counters
        self._step_count = 0
        self._patients_triaged = 0
        self._critical_patients_total = 0
        self._critical_patients_saved_in_time = 0
        self._beds_available = self._total_beds
        self._doctors_available = self._total_doctors
        self._total_reward = 0.0
        self._done = False
        self._start_time = time.time()

        # Populate waiting room
        self._waiting_room = []
        for _ in range(self._queue_size):
            p, ideal_esi = _generate_patient(self._rng, self._criticality_rate)
            self._waiting_room.append(PatientRecord(p, ideal_esi, 0))
            if ideal_esi == 1:
                self._critical_patients_total += 1

        return TriageObservation(
            waiting_room=[r.presentation for r in self._waiting_room],
            waiting_room_count=len(self._waiting_room),
            beds_available=self._beds_available,
            beds_total=self._total_beds,
            elapsed_seconds=0.0,
            message="New shift started. Use 'triage' to assign beds, then 'diagnostic' to discharge.",
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: HospitalAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        if self._done:
            return TriageObservation(
                waiting_room=[],
                waiting_room_count=0,
                beds_available=self._beds_available,
                beds_total=self._total_beds,
                message="Shift over. Reset to start a new episode.",
                done=True,
                reward=0.0,
            )

        self._step_count += 1
        elapsed = time.time() - self._start_time
        step_reward = 0.0
        messages: list[str] = []

        # --- Phase 1: Health Decay (waiting patients only) ---
        expired_indices = []
        for idx, rec in enumerate(self._waiting_room):
            if rec.presentation.status == "waiting":
                rec.presentation.health_score = max(0.0, rec.presentation.health_score - 1.0)
                if rec.presentation.health_score <= 0.0:
                    step_reward -= 50.0
                    messages.append(f"Patient {rec.presentation.patient_id} has expired due to wait time.")
                    expired_indices.append(idx)

        for idx in reversed(expired_indices):
            self._waiting_room.pop(idx)

        # --- Phase 2: Process Action ---
        target_id = action.assigned_patient_id

        if action.action_type == "triage":
            rec = next((r for r in self._waiting_room if r.presentation.patient_id == target_id), None)

            if rec is None:
                step_reward -= 5.0
                messages.append(f"Triage failed: patient '{target_id}' not found in waiting room.")
            elif rec.presentation.status != "waiting":
                step_reward -= 2.0
                messages.append(f"Triage failed: patient '{target_id}' is already in a bed.")
            elif self._doctors_available <= 0:
                step_reward -= 5.0
                messages.append("Triage failed: no doctors available.")
            else:
                self._doctors_available = max(0, self._doctors_available - 1)
                ideal_esi = rec.ideal_esi

                # Base reward by severity
                if ideal_esi == 1:
                    step_reward += 50.0
                elif ideal_esi == 2:
                    step_reward += 20.0
                else:
                    step_reward += 5.0

                # ESI accuracy bonus
                if action.esi_level is not None:
                    esi_diff = abs(action.esi_level - ideal_esi)
                    step_reward += max(0.0, 5.0 - esi_diff * 2.0)
                    if action.esi_level > ideal_esi:
                        step_reward -= 2.0 * esi_diff

                # Early triage bonus: +5 reward if acted within 3 steps of arrival
                # Also counts ESI-1 patients as 'saved in time' only if triaged promptly
                if (self._step_count - rec.spawn_step) <= 3:
                    step_reward += 5.0
                    messages.append(f"Early triage bonus for {target_id} (+5).")
                    if ideal_esi == 1:
                        self._critical_patients_saved_in_time += 1

                # Allocate bed
                if action.allocate_bed and self._beds_available > 0:
                    self._beds_available -= 1
                    rec.presentation.status = "in_bed"
                    step_reward += 1.0
                    messages.append(f"Patient {target_id} triaged → moved to bed.")
                elif action.allocate_bed:
                    step_reward -= 10.0
                    messages.append(f"Resource Gridlock! No beds for {target_id} (-10).")
                else:
                    messages.append(f"Patient {target_id} triaged (no bed allocated).")

        elif action.action_type == "diagnostic":
            idx = next((i for i, r in enumerate(self._waiting_room) if r.presentation.patient_id == target_id), -1)

            if idx == -1:
                step_reward -= 5.0
                messages.append(f"Diagnostic failed: patient '{target_id}' not found.")
            elif self._waiting_room[idx].presentation.status != "in_bed":
                step_reward -= 5.0
                messages.append(f"Diagnostic failed: patient '{target_id}' must be in a bed first.")
            else:
                self._waiting_room[idx].presentation.is_stable = True
                self._waiting_room.pop(idx)
                self._beds_available += 1
                self._doctors_available += 1
                self._patients_triaged += 1
                step_reward += 10.0
                messages.append(f"Patient {target_id} diagnosed, stabilised, and discharged! (+10)")

        else:
            step_reward -= 1.0
            messages.append(f"Unknown action_type '{action.action_type}'.")

        # --- Phase 3: Stochastic Surge (10% probability) ---
        if self._rng.random() < 0.10:
            surge_count = self._rng.randint(3, 5)
            messages.append(f"🚨 SURGE: {surge_count} new patients incoming!")
            for _ in range(surge_count):
                p, esi = _generate_patient(self._rng, self._criticality_rate)
                self._waiting_room.append(PatientRecord(p, esi, self._step_count))
                if esi == 1:
                    self._critical_patients_total += 1
                    if self._beds_available <= 0:
                        step_reward -= 10.0
                        messages.append("Resource Gridlock penalty: critical surge patient, no beds! (-10)")

        self._total_reward += step_reward

        # --- Episode done when all patients are gone ---
        is_done = len(self._waiting_room) == 0
        if is_done:
            self._done = True
            messages.append(f"Shift complete! {self._patients_triaged} patients discharged.")

        return TriageObservation(
            waiting_room=[r.presentation for r in self._waiting_room],
            waiting_room_count=len(self._waiting_room),
            beds_available=self._beds_available,
            beds_total=self._total_beds,
            elapsed_seconds=round(elapsed, 2),
            message=" | ".join(messages) if messages else "Step processed.",
            done=is_done,
            reward=step_reward,
        )

    @property
    def state(self) -> TriageState:
        return TriageState(
            episode_id=self._episode_id or "none",
            step_count=self._step_count,
            patients_triaged=self._patients_triaged,
            patients_remaining=len(self._waiting_room),
            total_reward=round(self._total_reward, 2),
            is_done=self._done,
            critical_patients_total=self._critical_patients_total,
            critical_patients_saved_in_time=self._critical_patients_saved_in_time,
        )
