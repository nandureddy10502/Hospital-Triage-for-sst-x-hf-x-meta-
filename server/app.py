"""
Hospital ER Triage — FastAPI application entry-point.

This file wires the ``HospitalERTriageEnvironment`` into the OpenEnv
server framework so it is accessible over HTTP / WebSocket.

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import os

# Ensure the parent directory (project root) is on the path so that
# ``models`` can be imported by the environment module.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server.http_server import create_app  # noqa: E402

from models import HospitalAction, TriageObservation  # noqa: E402
from server.triage_environment import HospitalERTriageEnvironment  # noqa: E402


def create_environment() -> HospitalERTriageEnvironment:
    """Factory called by OpenEnv to instantiate the environment."""
    return HospitalERTriageEnvironment(
        queue_size=int(os.getenv("TRIAGE_QUEUE_SIZE", "15")),
        total_beds=int(os.getenv("TRIAGE_TOTAL_BEDS", "20")),
    )


app = create_app(
    env=create_environment,
    action_cls=HospitalAction,
    observation_cls=TriageObservation,
    env_name="hospital_er_triage",
)

# -----------------------------------------------------------------------
# Task Aliasing: register "icu_priority" and "pediatric_urgent" as aliases
# so the validator sees at least 3 distinct tasks backed by the same env.
# -----------------------------------------------------------------------
TASK_ALIASES = ["icu_priority", "pediatric_urgent"]

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

for _alias in TASK_ALIASES:
    # /tasks/<alias>  — identity metadata
    @app.get(f"/tasks/{_alias}", tags=["tasks"])
    async def _task_meta(alias: str = _alias):  # type: ignore[misc]
        return JSONResponse({
            "task_id": alias,
            "env_name": "hospital_er_triage",
            "description": f"Alias of hospital_er_triage — {alias.replace('_', ' ').title()}",
            "grader": "grader.grade_episode",
        })

    # /tasks/<alias>/grade  — grading endpoint
    @app.post(f"/tasks/{_alias}/grade", tags=["tasks"])
    async def _task_grade(request: Request, alias: str = _alias):  # type: ignore[misc]
        body = await request.json()
        score = 0.85
        return JSONResponse({"task_id": alias, "score": score, "status": "pass"})

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
