"""
Typed Pydantic models for the Hospital ER Triage environment.

Shared between client and server for type-safe, serialisable communication.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Domain enumerations
# ---------------------------------------------------------------------------

class ESILevel(IntEnum):
    """Emergency Severity Index (1 = most urgent, 5 = least urgent)."""
    RESUSCITATION = 1
    EMERGENT = 2
    URGENT = 3
    LESS_URGENT = 4
    NON_URGENT = 5


# ---------------------------------------------------------------------------
# Action — single flat action class with action_type discriminator
# ---------------------------------------------------------------------------

class HospitalAction(Action):
    """Master action: covers both Triage (assign doctor) and Diagnostic (discharge)."""

    action_type: Literal["triage", "diagnostic"] = Field(
        default="triage",
        description="'triage' to assign a doctor/bed, 'diagnostic' to run tests and discharge.",
    )
    assigned_patient_id: str = Field(
        ...,
        description="ID of the patient being acted upon.",
    )

    # Triage-only fields (optional for diagnostic actions)
    assigned_doctor_id: Optional[str] = Field(
        default=None,
        description="Doctor name/ID assigned (triage only).",
    )
    esi_level: Optional[int] = Field(
        default=None,
        description="ESI level 1-5 (1=most urgent). Required for triage actions.",
    )
    allocate_bed: bool = Field(
        default=True,
        description="Whether to immediately allocate a bed (triage only).",
    )
    notes: str = Field(
        default="",
        description="Free-text triage notes.",
    )

    # Diagnostic-only fields
    test_type: str = Field(
        default="labs",
        description="Test protocol: 'labs' or 'xray' (diagnostic only).",
    )


# ---------------------------------------------------------------------------
# Patient presentation — nested inside observations (plain BaseModel)
# ---------------------------------------------------------------------------

class PatientPresentation(BaseModel):
    """A single incoming patient's clinical presentation."""

    model_config = ConfigDict(extra="ignore")

    patient_id: str = Field(..., description="Unique patient identifier.")
    status: str = Field(default="waiting", description="Patient location: 'waiting' or 'in_bed'.")
    is_stable: bool = Field(default=False, description="True after diagnostics confirm stability.")
    health_score: float = Field(default=100.0, description="Health score: decreases per step while waiting. 0 = expired.")

    age: int = Field(..., ge=0, le=120, description="Patient age in years.")
    sex: str = Field(..., description="Patient sex (M/F/Other).")
    chief_complaint: str = Field(..., description="Primary complaint on arrival.")
    symptoms: List[str] = Field(default_factory=list, description="List of reported symptoms.")
    heart_rate: int = Field(..., ge=0, le=300, description="Heart rate (bpm).")
    systolic_bp: int = Field(..., ge=0, le=300, description="Systolic blood pressure (mmHg).")
    diastolic_bp: int = Field(..., ge=0, le=200, description="Diastolic blood pressure (mmHg).")
    respiratory_rate: int = Field(..., ge=0, le=80, description="Respiratory rate (breaths/min).")
    spo2: int = Field(..., ge=0, le=100, description="Oxygen saturation (%).")
    temperature_c: float = Field(..., ge=30.0, le=45.0, description="Body temperature (°C).")
    pain_scale: int = Field(..., ge=0, le=10, description="Self-reported pain (0-10).")
    arrival_mode: str = Field(default="walk-in", description="How the patient arrived.")
    time_since_onset_min: Optional[int] = Field(default=None, description="Minutes since symptom onset.")


# ---------------------------------------------------------------------------
# Observation — returned after reset/step
# ---------------------------------------------------------------------------

class TriageObservation(Observation):
    """Observation returned to the agent after each step (or reset)."""

    model_config = ConfigDict(extra="ignore")  # tolerate extra fields on deserialisation

    waiting_room: List[PatientPresentation] = Field(
        default_factory=list,
        description="All patients currently in the hospital (waiting or in-bed).",
    )
    waiting_room_count: int = Field(default=0, description="Number of patients in the list.")
    beds_available: int = Field(default=0, description="Beds currently free.")
    beds_total: int = Field(default=0, description="Total bed capacity.")
    elapsed_seconds: float = Field(default=0.0, description="Time elapsed in episode.")
    message: str = Field(default="", description="Human-readable feedback from last action.")


# ---------------------------------------------------------------------------
# State — ground-truth environment metrics
# ---------------------------------------------------------------------------

class TriageState(State):
    """Ground-truth state of the environment (not shown to the agent)."""

    step_count: int = Field(0, description="Steps taken so far.")
    patients_triaged: int = Field(0, description="Patients fully discharged.")
    patients_remaining: int = Field(0, description="Patients still in hospital.")
    critical_patients_total: int = Field(0, description="Total ESI-1 patients seen.")
    critical_patients_saved_in_time: int = Field(0, description="ESI-1 patients triaged within 3 steps.")
    is_done: bool = Field(False, description="Whether the episode has ended.")
    total_reward: float = Field(0.0, description="Cumulative reward so far.")
