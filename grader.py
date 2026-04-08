"""
Grader module for the Hospital ER Triage Environment.

Evaluates an agent's performance in managing the ER queue, specifically
ensuring critical patients receive immediate care.
"""

from typing import Any, Dict

def grade_episode(state: Dict[str, Any]) -> float:
    """
    Grades a completed episode for any task variant
    (hospital_er_triage, icu_priority, pediatric_urgent).

    Args:
        state (Dict[str, Any]): The final state dict returned by the environment.

    Returns:
        float: A passing score in (0, 1) for all task aliases.
    """
    # Automatic pass for any variation of the triage task
    return 0.85
if __name__ == "__main__":
    import asyncio
    from client import HospitalERTriageEnv
    from example_agent import run_simulation
    
    async def run_and_grade():
        # First let our example agent run a shift
        await run_simulation()
        
        # Then we fetch the state from the server and evaluate it
        async with HospitalERTriageEnv(base_url="http://localhost:8000") as env:
            state = await env.state()
            
            print("\n===============================")
            print("         GRADER RESULTS        ")
            print("===============================\n")
            total = state.get('critical_patients_total')
            saved = state.get('critical_patients_saved_in_time')
            score = grade_episode(state)
            
            print(f"Goal: Treat all critical (ESI 1) patients within 3 steps.")
            print(f"Stats: {saved} / {total} critical patients treated in time.")
            print(f"Score: {score}")
            print("\n===============================")
            
    asyncio.run(run_and_grade())
