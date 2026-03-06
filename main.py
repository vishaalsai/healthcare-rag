#!/usr/bin/env python
"""
Healthcare RAG - Ask My Docs
Entry point: programmatic API for the full RAG pipeline.

For CLI usage see:
  scripts/ingest_docs.py   — one-time document ingestion
  scripts/query.py         — interactive querying
  scripts/run_evaluation.py — RAGAS evaluation
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.ingestion.embedder import EmbeddingModel
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import ChromaVectorStore
from src.generation.answer_generator import AnswerGenerator, AnswerResult
from src.generation.llm_client import AnthropicClient
from src.utils.citation_utils import CitationEnforcer
from src.utils.prompt_manager import PromptManager

load_dotenv()


def load_settings() -> dict:
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rag_pipeline(settings: dict | None = None) -> AnswerGenerator:
    """
    Construct and return a fully wired AnswerGenerator.

    This is the primary entry point for programmatic usage.
    All components are lazily initialised (embedding model / cross-encoder
    are loaded on first call).

    Args:
        settings: Optional dict overriding values from config/settings.yaml.

    Returns:
        A configured AnswerGenerator ready to receive queries.

    Example::

        from main import build_rag_pipeline

        rag = build_rag_pipeline()
        result = rag.answer("What is the treatment for hypertension?")
        print(result.pretty_print())
    """
    cfg = settings or load_settings()

    emb_cfg = cfg["embeddings"]
    chr_cfg = cfg["chromadb"]
    ret_cfg = cfg["retrieval"]
    gen_cfg = cfg["generation"]

    embedding_model = EmbeddingModel(
        model_name=emb_cfg["model"],
        device=emb_cfg["device"],
        batch_size=emb_cfg["batch_size"],
        normalize=emb_cfg["normalize"],
    )

    vector_store = ChromaVectorStore(
        embedding_model=embedding_model,
        persist_directory=chr_cfg["persist_directory"],
        collection_name=chr_cfg["collection_name"],
        distance_metric=chr_cfg["distance_metric"],
    )

    bm25 = BM25Retriever()

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        bm25_top_k=ret_cfg["bm25_top_k"],
        vector_top_k=ret_cfg["vector_top_k"],
        rrf_k=ret_cfg["rrf_k"],
        final_top_k=ret_cfg["final_top_k"],
    )

    reranker = CrossEncoderReranker(
        model_name=ret_cfg["reranker_model"],
        top_k=ret_cfg["reranker_top_k"],
        min_score=ret_cfg.get("citation_min_score"),
    )

    llm_client = AnthropicClient(
        model=gen_cfg["model"],
        max_tokens=gen_cfg["max_tokens"],
        temperature=gen_cfg["temperature"],
    )

    prompt_manager = PromptManager()
    citation_enforcer = CitationEnforcer()

    return AnswerGenerator(
        llm_client=llm_client,
        retriever=hybrid_retriever,
        reranker=reranker,
        prompt_manager=prompt_manager,
        citation_enforcer=citation_enforcer,
    )


# ── Direct execution demo ────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the first-line treatment for hypertension?"
    )

    logger.info("Building RAG pipeline …")
    rag = build_rag_pipeline()

    logger.info(f"Querying: {question!r}")
    result: AnswerResult = rag.answer(question)

    print("\n" + "=" * 60)
    print(result.pretty_print())
    print("=" * 60)
