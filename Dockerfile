# syntax=docker/dockerfile:1

# ===========================================================================
# Stage 1 — builder: install Python deps into an isolated virtualenv
# ===========================================================================
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps only needed at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a self-contained virtualenv we can copy into the final image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /install
COPY requirements.txt .

# Install the CPU-only torch wheels to keep the image small and free-tier friendly.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ===========================================================================
# Stage 2 — runtime: minimal image with just the venv + app code
# ===========================================================================
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_NAME=mobilenet_v3_large \
    TORCH_HOME=/home/appuser/.cache/torch \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Copy the prebuilt virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user for security.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --chown=appuser:appuser . /app
USER appuser

# Warm the weights cache at build time so the first request is fast.
RUN python -c "from app.main import get_model_bundle; get_model_bundle()"

EXPOSE 8000

# Most PaaS platforms (Render, Railway, Koyeb, HF Spaces) inject a $PORT env var.
# Bind to it when present, otherwise fall back to 8000 for local Docker runs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.getenv('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
