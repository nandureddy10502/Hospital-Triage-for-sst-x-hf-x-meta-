"""
Client-side wrapper for the Hospital ER Triage environment.

Users install this package and connect to a running server (local Docker or
Hugging Face Space) via the standard OpenEnv async / sync API.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import HospitalAction, TriageObservation


class HospitalERTriageEnv(EnvClient[HospitalAction, TriageObservation, Dict[str, Any]]):
    """
    Typed client for the Hospital ER Triage environment.

    Usage (async — recommended):
        async with HospitalERTriageEnv(base_url="http://localhost:8000") as client:
            result = await client.reset()
            while not result.done:
                action = HospitalAction(action_type="triage", esi_level=3)
                result = await client.step(action)

    Usage (sync):
        with HospitalERTriageEnv(base_url="http://localhost:8000").sync() as client:
            result = client.reset()
            ...
    """

    def _step_payload(self, action: HospitalAction) -> Dict[str, Any]:
        """Convert action to dict payload for the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TriageObservation]:
        """Parse server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=TriageObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse state response from the server."""
        return payload
