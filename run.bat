@echo off
REM ============================================================
REM Healthcare RAG — Launcher (Windows)
REM Starts FastAPI backend and Streamlit frontend.
REM Usage:  run.bat  (double-click or from terminal)
REM ============================================================

title Healthcare RAG Assistant

echo.
echo   ==========================================
echo    Healthcare RAG -- Ask My Docs
echo   ==========================================
echo.

REM ── sanity checks ─────────────────────────────────────────
if not exist ".env" (
  echo   WARNING: .env not found.
  echo   Copy .env.example, rename it to .env, and set ANTHROPIC_API_KEY.
  echo.
)

if not exist "data\chroma_db" (
  echo   WARNING: ChromaDB not found. Run ingestion first:
  echo     python scripts\ingest_docs.py
  echo.
)

REM ── start FastAPI in a separate window ────────────────────
echo   [1/2] Starting FastAPI on http://localhost:8000 ...
start "Healthcare RAG - API" cmd /k "uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload && pause"

REM ── wait 4 seconds for FastAPI to initialise ──────────────
echo   Waiting for API to initialise...
timeout /t 4 /nobreak > nul

REM ── start Streamlit in this window ────────────────────────
echo   [2/2] Starting Streamlit on http://localhost:8501 ...
echo.
echo   Services running:
echo     API   --^>  http://localhost:8000
echo     Docs  --^>  http://localhost:8000/docs
echo     UI    --^>  http://localhost:8501
echo.
echo   Close this window to stop Streamlit.
echo   Close the "API" window to stop FastAPI.
echo.

streamlit run src\app.py --server.port 8501 --browser.gatherUsageStats false

pause
