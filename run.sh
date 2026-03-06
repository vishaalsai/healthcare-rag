#!/usr/bin/env bash
# ============================================================
# Healthcare RAG — Launcher (Linux / macOS / Git Bash)
# Starts FastAPI backend + Streamlit frontend together.
# Usage:  ./run.sh
# ============================================================
set -euo pipefail

# ── colours ─────────────────────────────────────────────────
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${BLUE}"
echo "  ██╗  ██╗███████╗ █████╗ ██╗  ████████╗██╗  ██╗ ██████╗ █████╗ ██████╗ "
echo "  ██║  ██║██╔════╝██╔══██╗██║  ╚══██╔══╝██║  ██║██╔════╝██╔══██╗██╔══██╗"
echo "  ███████║█████╗  ███████║██║     ██║   ███████║██║     ███████║██████╔╝"
echo "  ██╔══██║██╔══╝  ██╔══██║██║     ██║   ██╔══██║██║     ██╔══██║██╔══██╗"
echo "  ██║  ██║███████╗██║  ██║███████╗██║   ██║  ██║╚██████╗██║  ██║██║  ██║"
echo "  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "${GREEN}  Healthcare RAG — Ask My Docs${NC}"
echo    "  =============================================="
echo

# ── sanity checks ────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo -e "${YELLOW}  WARNING: .env not found. Copy .env.example and set ANTHROPIC_API_KEY.${NC}"
fi

if [ ! -d "data/chroma_db" ] || [ -z "$(ls -A data/chroma_db 2>/dev/null)" ]; then
  echo -e "${YELLOW}  WARNING: ChromaDB is empty. Run ingestion first:${NC}"
  echo    "    python scripts/ingest_docs.py"
  echo
fi

# ── start FastAPI ────────────────────────────────────────────
echo "  Starting FastAPI on http://localhost:8000 ..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload --log-level warning &
FASTAPI_PID=$!
echo "  FastAPI PID: $FASTAPI_PID"

# ── wait for FastAPI to be ready ─────────────────────────────
echo "  Waiting for API to be ready..."
for i in {1..20}; do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓ API ready${NC}"
    break
  fi
  sleep 1
  if [ $i -eq 20 ]; then
    echo "  API did not start in time. Check logs above."
  fi
done

echo
echo -e "  ${GREEN}Services running:${NC}"
echo    "    API  →  http://localhost:8000"
echo    "    Docs →  http://localhost:8000/docs"
echo    "    UI   →  http://localhost:8501"
echo
echo    "  Press Ctrl+C to stop all services."
echo

# ── start Streamlit (foreground) ─────────────────────────────
trap "echo '  Shutting down...'; kill $FASTAPI_PID 2>/dev/null; exit 0" INT TERM

streamlit run src/app.py \
  --server.port 8501 \
  --server.headless false \
  --browser.gatherUsageStats false

# ── cleanup ──────────────────────────────────────────────────
kill $FASTAPI_PID 2>/dev/null || true
echo "  All services stopped."
