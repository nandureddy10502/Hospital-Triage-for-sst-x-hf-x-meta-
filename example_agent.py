import asyncio
from client import HospitalERTriageEnv
from models import HospitalAction

async def run_simulation():
    # Connect to the running OpenEnv server
    print("Connecting to the ER Triage Environment...")
    async with HospitalERTriageEnv(base_url="http://localhost:8000") as env:
        
        # 1. Reset the environment
        result = await env.reset()
        print("\n--- Shift Started ---")
        
        # 2. Main Simulation Loop
        while not result.done:
            waiting_room = result.observation.waiting_room
            queue_len = result.observation.waiting_room_count
            
            print(f"\n--- Queue: {queue_len} patients waiting ---")
            for p in waiting_room:
                print(f"[{p.patient_id}] {p.age}yo {p.sex} | {p.chief_complaint} | HR: {p.heart_rate}")

            # --- Agent Decision Logic ---
            # We want to treat the sickest patient first. 
            # A simple heuristic: treat the one with the highest heart rate.
            sickest_patient = max(waiting_room, key=lambda p: p.heart_rate)
            
            assigned_esi = 1 if sickest_patient.heart_rate > 110 else 3
            doctor_id = "Dr. Smith"
            
            action = HospitalAction(
                action_type="triage",
                assigned_patient_id=sickest_patient.patient_id,
                assigned_doctor_id=doctor_id,
                esi_level=assigned_esi,
                allocate_bed=True,
                notes="Priority targeted treatment based on HR heuristic",
            )
            
            print(f"> Action Taken: Assigned {doctor_id} to [{sickest_patient.patient_id}] with ESI {assigned_esi}")
            
            # 3. Take a step
            result = await env.step(action)
            print(f"Feedback: {result.observation.message}")

        # 4. Episode is done, get final state
        state = await env.state()
        print("\n--- Shift Over ---")
        print(f"Total Patients Triaged:    {state.get('patients_triaged')}")
        print(f"Total Critical Patients:   {state.get('critical_patients_total')}")
        print(f"Critical Treated <=3 steps:{state.get('critical_patients_saved_in_time')}")
        print(f"Total Reward Score:        {state.get('total_reward')}")

if __name__ == "__main__":
    asyncio.run(run_simulation())
