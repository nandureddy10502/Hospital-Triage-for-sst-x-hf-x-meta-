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

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
