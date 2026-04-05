# Hospital ER Triage Simulator — OpenEnv Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates
an Emergency Room triage workflow.

## Overview

An RL agent receives incoming patient presentations (symptoms, vitals, chief
complaint) and must:

1. **Assign an ESI level** (1–5) — Emergency Severity Index, where 1 is the
   most critical.
2. **Decide resource allocation** — beds, labs, imaging.
3. **Manage patient flow** under time pressure across a shift of ~15 patients.

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Use from Python

```python
import asyncio
from hospital_er_triage_env import HospitalERTriageEnv, TriageAction

async def main():
    async with HospitalERTriageEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        print(result.observation.current_patient)

        while not result.done:
            action = TriageAction(esi_level=3, allocate_bed=True)
            result = await client.step(action)
            print(result.reward, result.observation.message)

asyncio.run(main())
```

### 4. Docker

```bash
docker build -t hospital-er-triage -f server/Dockerfile .
docker run -p 8000:8000 hospital-er-triage
```

## Project Structure

```
Hospital Flow OpenEnv/
├── openenv.yaml                  # OpenEnv manifest
├── pyproject.toml                # Python package config
├── __init__.py                   # Package exports
├── models.py                    # Pydantic Action / Observation / State
├── client.py                    # Typed EnvClient subclass
├── README.md                    # This file
├── .dockerignore
├── outputs/                     # Runtime logs & evals (gitignored)
│   ├── logs/
│   └── evals/
└── server/
    ├── triage_environment.py    # Environment simulation logic
    ├── app.py                   # FastAPI entry-point
    ├── requirements.txt         # Deps for Docker
    └── Dockerfile               # Container image definition
```

## Reward Design

| Outcome                          | Reward        |
|----------------------------------|---------------|
| Exact ESI match                  | +10           |
| Each level off                   | −3 per level  |
| Under-triage (dangerous)        | −2 extra/level |
| Correct bed allocation           | +1            |
| Allocate bed when none available | −2            |

## License

BSD-3-Clause
