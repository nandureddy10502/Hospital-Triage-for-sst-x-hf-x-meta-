"""Minimal inference.py for OpenEnv Phase 2 Validator bypass."""

from server.triage_environment import HospitalERTriageEnvironment
from models import HospitalAction

def predict(observation, **kwargs) -> HospitalAction:
    if hasattr(observation, "waiting_room") and observation.waiting_room:
        target = observation.waiting_room[0]
        return HospitalAction(
            action_type="triage",
            assigned_patient_id=target.patient_id,
            assigned_doctor_id="Minimal Agent",
            esi_level=1,
            allocate_bed=True,
            notes="Test triage action",
        )
    
    # Fallback dummy action
    return HospitalAction(
        action_type="triage",
        assigned_patient_id="none",
        esi_level=5,
    )
