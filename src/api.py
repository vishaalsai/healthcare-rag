"""
Healthcare RAG — FastAPI Backend
Exposes the full RAG pipeline (hybrid retrieval + reranking + citation
enforcement) over HTTP so any frontend can consume it.

Endpoints:
  POST /query   — run a clinical question through the RAG pipeline
  GET  /health  — service status + documents indexed
  GET  /docs    — interactive Swagger UI (FastAPI built-in)
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

# ── project root on sys.path so we can import src.* and main ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

from main import build_rag_pipeline  # noqa: E402  (after sys.path fix)

# ── settings ──────────────────────────────────────────────────────────────────
_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
with open(_SETTINGS_PATH) as _f:
    _SETTINGS = yaml.safe_load(_f)

APP_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Clinical question to ask the RAG system",
        examples=["What is the first-line treatment for hypertension?"],
    )


class CitationSchema(BaseModel):
    number: int
    source: str
    page: int | str
    label: str
    score: float = 0.0


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationSchema]
    declined: bool
    processing_time_ms: float
    metadata: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str                  # "ok" | "degraded" | "starting"
    pipeline_ready: bool
    documents_indexed: int
    model: str
    embedding_model: str
    version: str


class ErrorResponse(BaseModel):
    detail: str


# ─────────────────────────────────────────────────────────────────────────────
#  Lifespan — load the RAG pipeline once on startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline when the server starts."""
    logger.info("=" * 55)
    logger.info("  Healthcare RAG API — starting up")
    logger.info("=" * 55)

    app.state.pipeline = None
    app.state.doc_count = 0
    app.state.ready = False

    try:
        pipeline = build_rag_pipeline()
        doc_count = pipeline.retriever.vector_store.collection_count()

        app.state.pipeline = pipeline
        app.state.doc_count = doc_count
        app.state.ready = True

        logger.info(f"Pipeline ready — {doc_count} documents indexed")
        logger.info("API docs available at: http://localhost:8000/docs")
        logger.info("=" * 55)

    except Exception as exc:
        logger.error(f"Pipeline initialization FAILED: {exc}")
        logger.warning("API will start but /query will return 503 until fixed")

    yield  # ← server is running here

    logger.info("Healthcare RAG API shutting down")


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Healthcare RAG — Ask My Docs",
    description=(
        "Evidence-based clinical Q&A powered by WHO/CDC/NIH guidelines. "
        "All answers include paragraph-level citations. "
        "Responses are declined when evidence is insufficient."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit (8501) and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",  # future React frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["System"],
)
async def health(req: Request) -> HealthResponse:
    """
    Returns service status and the number of documents indexed in ChromaDB.
    Use this to confirm the server is ready before querying.
    """
    return HealthResponse(
        status="ok" if req.app.state.ready else "degraded",
        pipeline_ready=req.app.state.ready,
        documents_indexed=req.app.state.doc_count,
        model=_SETTINGS["generation"]["model"],
        embedding_model=_SETTINGS["embeddings"]["model"],
        version=APP_VERSION,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Pipeline not ready"},
        500: {"model": ErrorResponse, "description": "Internal processing error"},
        422: {"model": ErrorResponse, "description": "Invalid request"},
    },
    summary="Ask a clinical question",
    tags=["RAG"],
)
async def query(request: QueryRequest, req: Request) -> QueryResponse:
    """
    Run a clinical question through the full RAG pipeline:

    1. Hybrid retrieval (BM25 + vector search, fused via RRF)
    2. Cross-encoder reranking
    3. Claude Opus generation with inline citations
    4. Citation enforcement — declines if evidence is insufficient

    Returns the answer text, structured citations with source PDF and
    page number, and a `declined` flag if the system couldn't find
    sufficient evidence.
    """
    if not req.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG pipeline is not initialized. "
                "Check that documents have been ingested and the server logs."
            ),
        )

    pipeline = req.app.state.pipeline
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    logger.info(f"Query: {question!r}")
    start_ts = time.perf_counter()

    try:
        result = pipeline.answer(question)
    except Exception as exc:
        logger.error(f"Pipeline error for question {question!r}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed. Please try again. (Error: {type(exc).__name__})",
        )

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
    logger.info(
        f"Answered in {elapsed_ms:.0f} ms | "
        f"declined={result.declined} | "
        f"citations={len(result.citations)}"
    )

    return QueryResponse(
        question=question,
        answer=result.answer,
        citations=[
            CitationSchema(
                number=c["number"],
                source=c.get("source", "unknown"),
                page=c.get("page", "?"),
                label=c.get("label", ""),
                score=float(c.get("score", 0.0)),
            )
            for c in result.citations
        ],
        declined=result.declined,
        processing_time_ms=elapsed_ms,
        metadata=result.metadata or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
