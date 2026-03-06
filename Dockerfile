FROM python:3.11-slim

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download ML models (avoids slow cold-start on Spaces) ─────────────────
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-mpnet-base-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; \
               CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p data/raw data/chroma_db data/processed

# ── Startup script ────────────────────────────────────────────────────────────
RUN chmod +x start.sh

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Spaces public port (Streamlit UI)
EXPOSE 7860

CMD ["./start.sh"]
