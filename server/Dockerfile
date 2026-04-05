FROM python:3.10-slim AS base

# Prevent Python from buffering stdout/stderr (useful for container logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---- Dependencies layer (cached) ----
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# ---- Application code ----
COPY models.py      /app/models.py
COPY __init__.py    /app/__init__.py
COPY client.py      /app/client.py
COPY server/        /app/server/

EXPOSE 7860

# Default env-vars (can be overridden at runtime)
ENV TRIAGE_QUEUE_SIZE=15
ENV TRIAGE_TOTAL_BEDS=20

CMD ["uvicorn", "dashboard:app", "--host", "0.0.0.0", "--port", "7860"]
