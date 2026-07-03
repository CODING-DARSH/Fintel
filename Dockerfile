# =============================================================================
# Dockerfile — Fintel Phase 1 Pipeline
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): installs all Python deps into a venv
#   Stage 2 (runtime): copies only the venv — keeps final image lean
#
# Why multi-stage:
#   Build tools (gcc, g++) are needed to compile some packages (e.g. numpy)
#   but are not needed at runtime. Multi-stage drops them from the final image,
#   reducing size from ~3GB to ~1.2GB.

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Layer A — heavy base deps (cached unless requirements-base.txt changes)
# This layer almost never invalidates
COPY requirements-base.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-base.txt

# Layer B — lighter frequently changing deps
# Only this layer rebuilds when you add a new package
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Only runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder — no compiler needed
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user — security best practice, required by most banks
RUN useradd --create-home --shell /bin/bash fintel
USER fintel
WORKDIR /app

# Copy project source
COPY --chown=fintel:fintel . .

# Create runtime directories (volumes will mount over data/ but
# these ensure structure exists if volumes aren't mounted)
RUN mkdir -p data/raw/filings data/raw/prices data/raw/macro \
             data/processed/filings data/processed/prices data/processed/macro \
             data/labels logs outputs notebooks

# Default: run the phase 1 pipeline check (override in compose)
CMD ["python", "run_phase1.py", "--step", "check"]
