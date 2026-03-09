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

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from dotenv import load_dotenv
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
    trace_id: str = ""
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

    trace_id = str(uuid4())
    logger.info(f"Query: {question!r} (trace_id={trace_id})")
    start_ts = time.perf_counter()

    try:
        result = pipeline.answer(question, trace_id=trace_id)
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
        f"citations={len(result.citations)} | "
        f"trace_id={trace_id}"
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
        trace_id=trace_id,
        metadata=result.metadata or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/metrics",
    summary="Aggregated RAG metrics",
    tags=["Observability"],
)
async def metrics(hours: int = 24):
    """
    Returns aggregated metrics over the last `hours` window.
    Pulls data from Langfuse traces. Returns zeroed values when
    Langfuse is not configured.
    """
    from src.observability.metrics import MetricsCollector
    collector = MetricsCollector()
    summary = collector.get_metrics(hours=hours)
    return asdict(summary)


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Observability dashboard",
    tags=["Observability"],
)
async def dashboard(hours: int = 24):
    """
    Dark-themed HTML dashboard showing aggregated metrics and
    the 20 most recent traces. Auto-refreshes every 30 seconds.
    """
    from src.observability.metrics import MetricsCollector
    from src.observability.tracer import get_tracer
    collector = MetricsCollector()
    tracer_configured = get_tracer() is not None
    summary = collector.get_metrics(hours=hours)
    traces = collector.get_recent_traces(limit=20)
    return _build_dashboard_html(summary, traces, tracer_configured)


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard HTML builder
# ─────────────────────────────────────────────────────────────────────────────

_NOT_CONFIGURED_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Healthcare RAG — Dashboard</title>
<style>
body{font-family:sans-serif;background:#0f1117;color:#e0e0e0;
display:flex;align-items:center;justify-content:center;
min-height:100vh;margin:0;}
.box{background:#1e1e2e;padding:40px;border-radius:12px;
text-align:center;max-width:480px;}
h2{color:#fff;margin-bottom:12px;}
p{color:#888;line-height:1.6;}
code{background:#2a2a3e;padding:2px 6px;border-radius:4px;
color:#7c9fff;font-size:0.9rem;}
</style></head>
<body><div class="box">
<h2>🔌 Tracing Not Configured</h2>
<p>Set your Langfuse keys in <code>.env</code> to enable
the observability dashboard.</p>
<p style="margin-top:16px;">
<code>LANGFUSE_PUBLIC_KEY</code><br>
<code>LANGFUSE_SECRET_KEY</code><br>
<code>LANGFUSE_HOST</code>
</p>
</div></body></html>"""

_DASHBOARD_CSS = """\
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0f1117;color:#e0e0e0;
  min-height:100vh;padding:24px;
}
h1{color:#fff;font-size:1.6rem;}
.subtitle{color:#666;font-size:0.9rem;margin:4px 0 28px;}
.cards{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:16px;margin-bottom:32px;
}
.card{
  background:#1e1e2e;border-radius:10px;
  padding:20px;border:1px solid #2a2a3e;
}
.card-value{font-size:2rem;font-weight:700;color:#7c9fff;}
.card-label{font-size:0.8rem;color:#666;margin-top:6px;
  text-transform:uppercase;letter-spacing:.05em;}
h2{color:#ccc;font-size:1rem;margin-bottom:12px;
  text-transform:uppercase;letter-spacing:.06em;}
.tbl-wrap{background:#1e1e2e;border-radius:10px;
  overflow:hidden;border:1px solid #2a2a3e;}
table{width:100%;border-collapse:collapse;}
th{
  background:#16162a;color:#666;font-size:0.75rem;
  text-transform:uppercase;letter-spacing:.06em;
  padding:12px 14px;text-align:left;
}
td{padding:11px 14px;border-top:1px solid #2a2a3e;
  font-size:0.875rem;}
tr:hover td{background:#1a1a2e;}
.badge{
  display:inline-block;padding:3px 10px;
  border-radius:12px;font-size:0.75rem;font-weight:600;
}
.badge-success{background:#0d2e0d;color:#4caf50;}
.badge-warn{background:#2e1e00;color:#ff9800;}
.badge-fail{background:#2e0d0d;color:#f44336;}
.footer{color:#444;font-size:0.78rem;margin-top:20px;}
</style>"""


def _status_badge(status: str) -> str:
    if status == "success":
        return '<span class="badge badge-success">success</span>'
    if status == "insufficient_context":
        return (
            '<span class="badge badge-warn">'
            'insufficient_context</span>'
        )
    return '<span class="badge badge-fail">declined</span>'


def _build_dashboard_html(
    summary,
    traces: list[dict],
    tracer_configured: bool,
) -> str:
    """Render the full observability dashboard as an HTML string."""
    if not tracer_configured:
        return _NOT_CONFIGURED_HTML

    h = summary.period_hours
    cards = [
        ("Total Requests", str(summary.total_requests)),
        ("Success Rate", f"{summary.success_rate * 100:.1f}%"),
        ("P95 Latency", f"{summary.p95_latency_ms:.0f} ms"),
        ("Avg Cost / Request", f"${summary.avg_cost_usd:.5f}"),
        ("Citation Coverage", f"{summary.citation_coverage * 100:.1f}%"),
        (f"Total Cost (last {h}h)", f"${summary.total_cost_usd:.4f}"),
    ]

    cards_html = "".join(
        f'<div class="card">'
        f'<div class="card-value">{val}</div>'
        f'<div class="card-label">{label}</div>'
        f'</div>'
        for label, val in cards
    )

    rows_html = ""
    for t in traces:
        ts = t["timestamp"][:19].replace("T", " ") if t["timestamp"] else "—"
        badge = _status_badge(t["status"])
        rows_html += (
            f"<tr>"
            f"<td>{ts}</td>"
            f"<td>{t['query']}</td>"
            f"<td>{badge}</td>"
            f"<td>{t['latency_ms']:.0f} ms</td>"
            f"<td>${t['cost_usd']:.5f}</td>"
            f"<td>{t['citation_count']}</td>"
            f"</tr>"
        )

    if not rows_html:
        rows_html = (
            '<tr><td colspan="6" style="text-align:center;color:#555;">'
            'No traces found in this window.</td></tr>'
        )

    updated = summary.computed_at[:19].replace("T", " ") + " UTC"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>Healthcare RAG — Dashboard</title>
{_DASHBOARD_CSS}
</head>
<body>
<h1>🏥 Healthcare RAG — Observability Dashboard</h1>
<p class="subtitle">Metrics over the last {h} hours
&nbsp;·&nbsp; auto-refresh every 30 s</p>

<div class="cards">{cards_html}</div>

<h2>Recent Traces (last 20)</h2>
<div class="tbl-wrap">
<table>
<thead><tr>
<th>Timestamp</th><th>Query</th><th>Status</th>
<th>Latency</th><th>Cost</th><th>Citations</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>

<p class="footer">Last updated: {updated}</p>
</body>
</html>"""


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
