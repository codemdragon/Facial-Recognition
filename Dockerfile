# ──────────────────────────────────────────────────────────────────────────────
# CPLC Face Search — Dockerfile
#
# Builds a headless server image (Flask API + PostgreSQL client).
# The PyQt5 desktop app runs on the operator's Windows machine and talks to
# this container via HTTP, OR you can run main_app.py directly on Windows
# without Docker at all.
#
# Usage:
#   docker compose up
#
# To add image folders at a different deployment site, edit the `volumes`
# section in docker-compose.yml — no rebuild needed.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="CPLC Sindh Governor's Secretariat"
LABEL description="CPLC Face Search — DeepFace vector search server"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY face_app/requirements.txt /app/requirements.txt

# Install without PyQt5 (no display in container)
RUN pip install --no-cache-dir \
        tensorflow-cpu>=2.16 \
        tf-keras>=2.16 \
        "opencv-python-headless==4.10.0.84" \
        "faiss-cpu>=1.9" \
        "Flask>=3.0" \
        "Pillow>=10.0" \
        "psycopg2-binary>=2.9"

# ── DeepFace (editable from bundled repo) ─────────────────────────────────────
# If you have a local deepface clone, uncomment and adjust the COPY line.
# Otherwise install from PyPI:
RUN pip install --no-cache-dir deepface

# ── App code ──────────────────────────────────────────────────────────────────
COPY face_app/ /app/

# ── Runtime env ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV FACE_DEVICE=cpu
# PostgreSQL connection — override via docker-compose environment:
ENV CPLC_PG_DSN=postgresql://cplc:cplc@db:5432/cplc_faces

# Image data is mounted at /data — set in docker-compose.yml
ENV CPLC_DATA_DIR=/data

EXPOSE 5000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
