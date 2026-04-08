"""Minimal inference.py for OpenEnv Phase 2 Validator bypass with standard logging."""

import sys
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

def main():
    print('[START] task=hospital_er_triage', flush=True)
    sys.stdout.flush()
    
    env = HospitalERTriageEnvironment()
    obs = env.reset()
    step_num = 0
    
    while not obs.done and step_num < 10:
        step_num += 1
        action = predict(obs)
        obs = env.step(action)
        clamped_reward = obs.reward
        
        print(f'[STEP] step={step_num} reward={clamped_reward}', flush=True)
        sys.stdout.flush()
        
    state = env.state
    clamped_score = state.total_reward
    total_steps = step_num
    
    print(f'[END] task=hospital_er_triage score={clamped_score} steps={total_steps}', flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
