# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
# Copy everything; the chroma_db volume will overlay backend/data/chroma_db at runtime
COPY . .

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8000 → FastAPI backend
# 8501 → Streamlit frontend
EXPOSE 8000 8501

# Default entrypoint: FastAPI backend.
# Override CMD in docker-compose for the Streamlit service.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]