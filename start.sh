#!/bin/bash
set -e

echo "======================================================"
echo "  Healthcare RAG — Hugging Face Spaces Startup"
echo "======================================================"

# ── 1. Download sample PDFs if data/raw is empty ─────────────────────────────
mkdir -p data/raw data/chroma_db data/processed

PDF_COUNT=$(find data/raw -name "*.pdf" 2>/dev/null | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
    echo "[1/4] Downloading clinical guidelines..."
    curl -L --retry 3 --retry-delay 5 --max-time 120 \
        -o data/raw/who_hypertension.pdf \
        "https://applications.emro.who.int/dsaf/dsa664.pdf" \
        && echo "  ✓ WHO Hypertension guidelines" \
        || echo "  ⚠ Skipped WHO Hypertension (download failed)"

    curl -L --retry 3 --retry-delay 5 --max-time 120 \
        -o data/raw/who_diabetes.pdf \
        "https://applications.emro.who.int/dsaf/dsa509.pdf" \
        && echo "  ✓ WHO Diabetes guidelines" \
        || echo "  ⚠ Skipped WHO Diabetes (download failed)"

    curl -L --retry 3 --retry-delay 5 --max-time 120 \
        -o data/raw/ada_standards_diabetes.pdf \
        "https://diabetes.org/sites/default/files/2023-09/dc22s007.pdf" \
        && echo "  ✓ ADA Standards of Diabetes Care" \
        || echo "  ⚠ Skipped ADA Standards (download failed)"

    # Remove any 0-byte files that result from failed downloads
    find data/raw -name "*.pdf" -empty -delete

    echo "[1/4] Download step complete."
else
    echo "[1/4] PDFs already present (${PDF_COUNT} file(s)), skipping download."
fi

# ── 2. Ingest into ChromaDB ───────────────────────────────────────────────────
PDF_COUNT=$(find data/raw -name "*.pdf" 2>/dev/null | wc -l)
if [ "$PDF_COUNT" -gt 0 ]; then
    echo "[2/4] Ingesting ${PDF_COUNT} PDF(s) into ChromaDB..."
    python scripts/ingest_docs.py --reset
    echo "[2/4] Ingestion complete."
else
    echo "[2/4] ⚠ No PDFs found in data/raw — queries will return 'insufficient context'."
fi

# ── 3. Start FastAPI on port 8000 (internal) ─────────────────────────────────
echo "[3/4] Starting FastAPI backend on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "  Waiting for API to become ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ API ready (attempt ${i})"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ⚠ API did not become ready in 60s — Streamlit will show 'API Offline'"
    fi
    sleep 2
done

# ── 4. Start Streamlit on port 7860 (HF Spaces public port) ──────────────────
echo "[4/4] Starting Streamlit UI on port 7860..."
exec streamlit run src/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    --server.headless true
