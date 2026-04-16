# ============================================
# Stage 1: Builder - install deps, download models
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py

# Show where models actually landed (debug - remove later)
RUN find /root/.cache -maxdepth 3 -type d 2>/dev/null || echo "No cache found at /root/.cache"
RUN find /opt/venv -path "*/fastembed*" -maxdepth 5 -type d 2>/dev/null || echo "No fastembed in venv"
RUN find / -name "bm25" -type d 2>/dev/null | head -20 || echo "bm25 not found"


# ============================================
# Stage 2: Runtime - lean image, no build tools
# ============================================
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy ALL cached models from builder (covers huggingface + fastembed wherever they land)
COPY --from=builder /root/.cache /root/.cache

# Copy application code
COPY api.py app_hosted.py ./
COPY DTO/ DTO/
COPY generation/ generation/
COPY ingestion/ ingestion/
COPY retrieval/ retrieval/

# Copy pre-computed data
COPY qdrant_data/ qdrant_data/
COPY eval/results.json eval/results.json

EXPOSE 7860

CMD ["python", "app_hosted.py"]