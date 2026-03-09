---
title: Healthcare RAG Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
---

# Healthcare RAG — Ask My Docs

A production-grade Retrieval-Augmented Generation system for querying clinical guidelines and public-health documents (WHO, CDC, NIH). Built with **Anthropic Claude**, **ChromaDB**, **BM25 hybrid search**, and **RAGAS evaluation**.

---

## Live Demo

**Hugging Face Spaces:** https://huggingface.co/spaces/vishaalsai29/healthcare-rag

---

## Evaluation Results

Results from the Phase 4 golden-dataset evaluation (20 Q&A pairs, 3 clinical PDFs).

| Category    | N  | Faithfulness | Citation Rate | Declined Correctly |
|-------------|---:|-------------:|--------------:|-------------------:|
| diagnosis   |  5 |        0.487 |         0.600 |              0.600 |
| treatment   |  7 |        0.548 |         0.714 |              0.714 |
| monitoring  |  4 |        0.667 |         1.000 |              1.000 |
| prevention  |  4 |        0.566 |         0.750 |              0.750 |
| **OVERALL** | **20** | **0.560** | **0.750** | **0.750** |

**Threshold:** faithfulness ≥ 0.70 · **Status: requires more PDFs**

> The faithfulness score reflects keyword-overlap between expected key facts and actual answers.
> 5 of 18 answerable questions were incorrectly declined (the pipeline said "insufficient context")
> because those specific clinical details are sparse in the current 3-PDF corpus.
> Expanding the document set to 10–20 PDFs is expected to push faithfulness above 0.70.

### Run evaluation locally

```bash
# 1. Start the FastAPI backend
uvicorn src.api:app --port 8000 --reload

# 2. In another terminal, run the evaluation script
python scripts/run_evaluation.py

# Quick smoke test (5 samples)
python scripts/run_evaluation.py --max-samples 5

# Against a different API endpoint
python scripts/run_evaluation.py --api-url http://localhost:8000
```

Results are saved to `data/eval/results/results.json`.
The CI/CD pipeline (`.github/workflows/ci.yml`) runs all 20 pairs on every push and fails the
build if mean faithfulness drops below 0.70.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │       HybridRetriever           │  Phase 2
              │  ┌─────────────┬─────────────┐  │
              │  │  BM25 (20)  │ Vector (20) │  │
              │  └──────┬──────┴──────┬──────┘  │
              │         └──────┬──────┘          │
              │    Reciprocal Rank Fusion         │
              │         top-10 candidates         │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │    CrossEncoderReranker         │  Phase 2
              │  cross-encoder/ms-marco         │
              │         top-5 chunks            │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │       AnswerGenerator           │  Phase 1
              │                                 │
              │  1. Build [1]…[N] context block │
              │  2. Call Claude (Opus 4.6)       │
              │  3. CitationEnforcer.enforce()  │  Phase 2
              │     ├─ check [N] in range       │
              │     ├─ count uncited sentences  │
              │     └─ detect INSUFFICIENT_CTX  │
              │  4. Return AnswerResult          │
              └─────────────────────────────────┘
```

---

## Project Structure

```
healthcare-rag/
├── .github/
│   └── workflows/
│       └── ci.yml              # 4-job CI/CD pipeline
├── config/
│   ├── prompts.yaml            # Versioned prompt registry
│   └── settings.yaml           # All tuneable parameters
├── data/
│   ├── raw/                    # Place PDFs here (git-ignored)
│   ├── processed/              # Auto-generated chunks.json
│   └── eval/
│       └── golden_dataset.json # 5-sample starter (expand to 50-200)
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py       # Phase 1: PyMuPDF + pdfplumber fallback
│   │   ├── chunker.py          # Phase 1: 500-800 token sliding window
│   │   └── embedder.py         # Phase 1: sentence-transformers wrapper
│   ├── retrieval/
│   │   ├── vector_store.py     # Phase 1: ChromaDB CRUD + query
│   │   ├── bm25_retriever.py   # Phase 2: rank_bm25 sparse retrieval
│   │   ├── hybrid_retriever.py # Phase 2: RRF fusion of BM25 + vector
│   │   └── reranker.py         # Phase 2: cross-encoder reranker
│   ├── generation/
│   │   ├── llm_client.py       # Phase 1: Anthropic SDK wrapper + retry
│   │   └── answer_generator.py # Phase 1+2: full RAG orchestration
│   ├── evaluation/
│   │   ├── evaluator.py        # Phase 3: RAGAS runner
│   │   └── metrics.py          # Phase 3: custom citation/decline metrics
│   └── utils/
│       ├── prompt_manager.py   # Phase 2: versioned YAML prompt loader
│       └── citation_utils.py   # Phase 2: [N] citation enforcement
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── scripts/
│   ├── ingest_docs.py          # One-time ingestion CLI
│   ├── query.py                # Interactive query CLI (+ streaming)
│   └── run_evaluation.py       # Evaluation + CI gate script
├── main.py                     # Programmatic API entry point
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Ingest Documents

Place PDF files (clinical guidelines, WHO/CDC/NIH documents) in `data/raw/`, then:

```bash
python scripts/ingest_docs.py
# Dry run (chunk stats only, no write):
python scripts/ingest_docs.py --dry-run
# Reset and re-index:
python scripts/ingest_docs.py --reset
```

### 4. Query

```bash
# Single question
python scripts/query.py "What is the first-line treatment for hypertension?"

# Streaming response
python scripts/query.py --stream "What are the WHO diabetes diagnosis criteria?"

# Interactive REPL
python scripts/query.py --interactive
```

### 5. Run Evaluation

```bash
python scripts/run_evaluation.py
# Limit sample count for quick test:
python scripts/run_evaluation.py --max-samples 10
```

---

## Architecture Decisions

### Phase 1: Foundation

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PDF parser | PyMuPDF (fitz) + pdfplumber fallback | fitz is fast and handles most PDFs; pdfplumber is more reliable for scanned docs |
| Token counting | tiktoken `cl100k_base` | Same tokenizer as Claude and GPT-4; accurate chunk size control |
| Chunk size | 500–800 tokens, 100-token overlap | Captures full clinical reasoning units; overlap prevents context loss at boundaries |
| Vector store | ChromaDB (persistent) | Embedded, zero-infra, cosine similarity, easy Docker upgrade path |
| Embedding model | `all-mpnet-base-v2` | Strong semantic similarity for medical text; 768-dim; runs on CPU |
| LLM | Anthropic Claude Opus 4.6 | Best reasoning for clinical text; citation-following is reliable |

### Phase 2: Production Quality

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hybrid retrieval | BM25 + vector, fused via RRF | BM25 excels on exact medical terminology (drug names, ICD codes); vector handles paraphrasing. RRF is parameter-free and robust |
| RRF constant | k=60 | Empirically validated default (Cormack et al. 2009); reduces sensitivity to rank differences |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2` | 4× smaller than ELECTRA-base with comparable precision on medical Q&A |
| Citation enforcement | Regex + LLM-signal | Hard rule: if Claude emits `INSUFFICIENT_CONTEXT`, decline immediately. Soft rule: count uncited sentences |
| Prompt versioning | YAML registry | Every prompt change is tracked with a `version` field; reproducible evaluations |

### Phase 3: Evaluation & CI/CD

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Evaluation framework | RAGAS | Standard for RAG evaluation; provides faithfulness (grounding) + relevancy |
| Key metric | Faithfulness ≥ 0.75 | Directly measures hallucination risk — most critical for medical content |
| CI gate | `sys.exit(1)` on threshold failure | Hard failure signal for GitHub Actions; prevents silent regressions |
| Golden dataset | JSON, human-verified | Ground truths must be manually verified against source documents |

---

## Configuration Reference

All tuneable parameters live in `config/settings.yaml`. No code changes needed for most adjustments.

```yaml
ingestion:
  chunk_size_target: 600    # Increase for denser documents
  chunk_overlap: 100        # Increase if retrieval misses context at boundaries

retrieval:
  rrf_k: 60                 # Lower = more aggressive rank fusion
  reranker_top_k: 5         # Increase for better coverage, costs more LLM tokens

generation:
  model: claude-opus-4-6    # Swap to claude-haiku-4-5 for cost reduction
  temperature: 0.1          # Keep low for factual medical responses

evaluation:
  faithfulness_threshold: 0.75  # Raise to tighten quality gate
```

---

## Adding Documents

1. Drop any `.pdf` into `data/raw/`
2. Run `python scripts/ingest_docs.py` (use `--reset` only if re-indexing all)
3. The system incrementally upserts (idempotent by `chunk_id`)

**Recommended sources:**
- WHO Clinical Guidelines: https://www.who.int/publications
- CDC Clinical Practice Guidelines: https://www.cdc.gov
- NIH National Guideline Clearinghouse / AHRQ

---

## Expanding the Golden Dataset

The starter dataset has 5 samples. For reliable evaluation you need **50–200 pairs**.

Each entry in `data/eval/golden_dataset.json` requires:

```json
{
  "id": "q006",
  "question": "...",
  "ground_truth": "...",     ← Must be manually verified from source doc
  "source_document": "*.pdf",
  "category": "...",
  "difficulty": "easy|medium|hard",
  "requires_context": true
}
```

**Do not use LLM-generated ground truths** — RAGAS faithfulness would be artificially inflated.

---

## CI/CD Pipeline

```
push → lint → unit-tests → quality-gate (main only)
```

- **lint**: `ruff` style + import checks
- **unit-tests**: fully mocked; no API key needed; coverage report uploaded
- **quality-gate**: runs on `main` pushes only; requires `ANTHROPIC_API_KEY` secret and a pre-built ChromaDB in Actions cache; fails build if RAGAS metrics regress

Set `ANTHROPIC_API_KEY` in **Settings → Secrets → Actions** to enable the quality gate.

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in **Settings → Secrets → Actions** to enable the observability gate.

---

## Observability Architecture

The system ships a three-layer observability stack that runs alongside every RAG request:

1. **Tracing** (`src/observability/tracer.py`) — Each query creates a Langfuse trace with latency, token counts, cost, citation count, and decline reason attached as metadata.
2. **Metrics aggregation** (`src/observability/metrics.py`) — `MetricsCollector` fetches traces from Langfuse and computes a rolling `MetricsSummary` over a configurable time window.
3. **Regression gate** (`scripts/check_observability.py`) — Reads thresholds from `config/observability_thresholds.yaml` and exits 1 if any hard threshold is breached.

### Metrics & Thresholds

| Metric | Threshold | Kind | Meaning |
|---|---|---|---|
| Citation Coverage | ≥ 0.70 | **Hard** | % of answers containing at least one `[N]` citation |
| Failure Rate | ≤ 0.20 | **Hard** | % of requests declined as `INSUFFICIENT_CONTEXT` |
| P95 Latency | ≤ 10 000 ms | **Hard** | 95th-percentile end-to-end latency |
| Avg Cost / Request | ≤ $0.05 | Soft (warn) | Average Claude API cost per query |

Thresholds are defined in `config/observability_thresholds.yaml` and version-controlled alongside the code.

### Run the observability check locally

```bash
# Requires LANGFUSE_* keys in .env
python scripts/check_observability.py
```

### Access the metrics dashboard

```bash
uvicorn src.api:app --port 8000
# Open http://localhost:8000/dashboard
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Single module
pytest tests/test_retrieval.py -v
```

---

## Swap / Extension Points

| Component | How to swap |
|-----------|-------------|
| Embedding model | Change `embeddings.model` in `settings.yaml` |
| LLM | Change `generation.model` in `settings.yaml` (any Anthropic model) |
| Vector store | Replace `ChromaVectorStore` with a Pinecone/Weaviate adapter implementing the same interface |
| Reranker | Swap model in `retrieval.reranker_model`; or set to `None` in `main.py` to skip |
| Retrieval strategy | Pass a plain `ChromaVectorStore` as `retriever` to `AnswerGenerator` for Phase 1 vector-only mode |
