"""
Shared pytest fixtures for the Healthcare RAG test suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Project root on sys.path ─────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pdf_loader import Document
from src.ingestion.chunker import Chunk, TextChunker
from src.retrieval.vector_store import RetrievedChunk


# ------------------------------------------------------------------ #
#  Sample data fixtures                                                #
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_document() -> Document:
    return Document(
        page_content=(
            "Hypertension is defined as a systolic blood pressure ≥130 mmHg "
            "or diastolic blood pressure ≥80 mmHg according to the 2017 ACC/AHA "
            "guidelines. First-line treatment includes lifestyle modification, "
            "specifically the DASH diet, sodium restriction below 2.3 g/day, "
            "weight loss, and regular aerobic exercise. Pharmacological therapy "
            "with thiazide diuretics, ACE inhibitors, or calcium channel blockers "
            "is indicated when lifestyle changes are insufficient."
        ),
        metadata={
            "source": "hypertension_guidelines.pdf",
            "source_path": "/data/raw/hypertension_guidelines.pdf",
            "page": 3,
            "total_pages": 45,
            "loader": "fitz",
        },
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    base_text = (
        "According to WHO guidelines, Type 2 diabetes is diagnosed when fasting "
        "plasma glucose is ≥7.0 mmol/L. Lifestyle interventions including diet "
        "and physical activity are the cornerstone of management."
    )
    chunks = []
    for i in range(3):
        c = Chunk(
            text=f"{base_text} Section {i+1}.",
            metadata={
                "source": "who_diabetes.pdf",
                "page": i + 1,
                "chunk_index_on_page": 0,
            },
            chunk_id=f"who_diabetes.pdf_p{i+1}_c0",
            token_count=45,
        )
        chunks.append(c)
    return chunks


@pytest.fixture
def sample_retrieved_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="doc1_p1_c0",
            text="Hypertension first-line treatment includes thiazide diuretics. [Evidence Level A]",
            metadata={"source": "guidelines.pdf", "page": 1, "chunk_index_on_page": 0},
            score=0.92,
        ),
        RetrievedChunk(
            chunk_id="doc1_p2_c0",
            text="ACE inhibitors are preferred in patients with diabetes or chronic kidney disease.",
            metadata={"source": "guidelines.pdf", "page": 2, "chunk_index_on_page": 0},
            score=0.84,
        ),
        RetrievedChunk(
            chunk_id="doc2_p5_c1",
            text="Calcium channel blockers are effective for isolated systolic hypertension in elderly patients.",
            metadata={"source": "cardiology.pdf", "page": 5, "chunk_index_on_page": 1},
            score=0.71,
        ),
    ]


@pytest.fixture
def text_chunker() -> TextChunker:
    return TextChunker(
        chunk_size=100,
        chunk_overlap=20,
        chunk_size_min=50,
        chunk_size_max=150,
        min_chunk_chars=50,
    )
