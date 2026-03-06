"""
Tests for Phase 2: BM25, hybrid retrieval, RRF, and reranker.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.bm25_retriever import BM25Retriever, _tokenize
from src.retrieval.hybrid_retriever import HybridRetriever, _reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import RetrievedChunk


# ------------------------------------------------------------------ #
#  _tokenize                                                           #
# ------------------------------------------------------------------ #

def test_tokenize_removes_stopwords():
    tokens = _tokenize("the patient has hypertension and diabetes")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "hypertension" in tokens
    assert "diabetes" in tokens


def test_tokenize_lowercases():
    tokens = _tokenize("Hypertension Treatment")
    assert "hypertension" in tokens
    assert "treatment" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []
    assert _tokenize("the and or") == []  # all stopwords


# ------------------------------------------------------------------ #
#  BM25Retriever                                                       #
# ------------------------------------------------------------------ #

def test_bm25_requires_index(sample_retrieved_chunks):
    bm25 = BM25Retriever()
    with pytest.raises(RuntimeError, match="index"):
        bm25.query("hypertension treatment")


def test_bm25_index_and_query(sample_retrieved_chunks):
    bm25 = BM25Retriever()
    bm25.index(sample_retrieved_chunks)
    assert bm25.is_built
    assert bm25.corpus_size == len(sample_retrieved_chunks)

    results = bm25.query("hypertension treatment", top_k=3)
    assert len(results) <= 3
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.score >= 0


def test_bm25_returns_ranked_results(sample_retrieved_chunks):
    """Results should contain the most relevant chunk at the top."""
    bm25 = BM25Retriever()
    bm25.index(sample_retrieved_chunks)
    results = bm25.query("hypertension thiazide diuretics", top_k=3)
    if len(results) >= 2:
        assert results[0].score >= results[1].score


def test_bm25_top_k_limits_output(sample_retrieved_chunks):
    bm25 = BM25Retriever()
    bm25.index(sample_retrieved_chunks)
    results = bm25.query("treatment", top_k=1)
    assert len(results) <= 1


def test_bm25_empty_corpus():
    bm25 = BM25Retriever()
    with pytest.raises(ValueError, match="empty"):
        bm25.index([])


# ------------------------------------------------------------------ #
#  _reciprocal_rank_fusion                                             #
# ------------------------------------------------------------------ #

def test_rrf_combines_rankings():
    ranking_a = ["doc1", "doc2", "doc3"]
    ranking_b = ["doc2", "doc1", "doc4"]

    scores = _reciprocal_rank_fusion([ranking_a, ranking_b], k=60)

    assert "doc1" in scores
    assert "doc2" in scores
    assert "doc4" in scores
    # doc2 ranks 1st in B and 2nd in A → should outscore doc3 (only in A)
    assert scores["doc2"] > scores["doc3"]


def test_rrf_empty_rankings():
    scores = _reciprocal_rank_fusion([[], []])
    assert scores == {}


def test_rrf_single_ranking():
    ranking = ["a", "b", "c"]
    scores = _reciprocal_rank_fusion([ranking], k=60)
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_k_parameter_effect():
    ranking = ["doc1", "doc2"]
    scores_low_k = _reciprocal_rank_fusion([ranking], k=1)
    scores_high_k = _reciprocal_rank_fusion([ranking], k=1000)
    # With low k, rank differences matter more
    diff_low = scores_low_k["doc1"] - scores_low_k["doc2"]
    diff_high = scores_high_k["doc1"] - scores_high_k["doc2"]
    assert diff_low > diff_high


# ------------------------------------------------------------------ #
#  HybridRetriever                                                     #
# ------------------------------------------------------------------ #

def test_hybrid_retriever_query(sample_retrieved_chunks):
    # Mock vector_store and bm25
    mock_vs = MagicMock()
    mock_vs.get_all_chunks.return_value = sample_retrieved_chunks
    mock_vs.query.return_value = sample_retrieved_chunks[:2]

    bm25 = BM25Retriever()

    hybrid = HybridRetriever(
        vector_store=mock_vs,
        bm25_retriever=bm25,
        bm25_top_k=3,
        vector_top_k=3,
        rrf_k=60,
        final_top_k=3,
    )

    results = hybrid.query("hypertension treatment")
    assert isinstance(results, list)
    assert len(results) <= 3


def test_hybrid_auto_builds_bm25_index(sample_retrieved_chunks):
    mock_vs = MagicMock()
    mock_vs.get_all_chunks.return_value = sample_retrieved_chunks
    mock_vs.query.return_value = sample_retrieved_chunks

    bm25 = BM25Retriever()
    hybrid = HybridRetriever(mock_vs, bm25)

    assert not hybrid._index_built
    hybrid.query("diabetes")
    assert hybrid._index_built


# ------------------------------------------------------------------ #
#  CrossEncoderReranker                                                #
# ------------------------------------------------------------------ #

def test_reranker_reorders_by_score(sample_retrieved_chunks):
    import numpy as np

    reranker = CrossEncoderReranker(top_k=3)
    mock_model = MagicMock()
    # Assign scores in reverse: last chunk gets highest score
    mock_model.predict.return_value = np.array([0.1, 0.5, 0.9])
    reranker._model = mock_model

    results = reranker.rerank("hypertension", sample_retrieved_chunks)
    assert len(results) <= 3
    if len(results) >= 2:
        assert results[0].score >= results[1].score


def test_reranker_respects_top_k(sample_retrieved_chunks):
    import numpy as np

    reranker = CrossEncoderReranker(top_k=2)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.3, 0.7, 0.5])
    reranker._model = mock_model

    results = reranker.rerank("query", sample_retrieved_chunks)
    assert len(results) <= 2


def test_reranker_empty_input():
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", [])
    assert results == []
