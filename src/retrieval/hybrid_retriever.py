"""
Phase 2 – Hybrid Retrieval
Hybrid Retriever: combines BM25 sparse scores and vector dense scores
using Reciprocal Rank Fusion (RRF).

RRF formula: score(d) = Σ  1 / (k + rank_i(d))
where k=60 (empirically validated by Cormack et al. 2009).
"""

from __future__ import annotations

from loguru import logger

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import ChromaVectorStore, RetrievedChunk


class HybridRetriever:
    """
    Fuses BM25 and vector search results via Reciprocal Rank Fusion.

    Workflow:
    1. Run BM25 query  → ranked list A (sparse)
    2. Run vector query → ranked list B (dense)
    3. Apply RRF to merge A + B into a single ranking
    4. Return top `final_top_k` chunks

    The BM25 index is built lazily on first query from the full
    ChromaDB collection (no double-storage required).
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25Retriever,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        rrf_k: int = 60,
        final_top_k: int = 10,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self._index_built = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build_bm25_index(self) -> None:
        """Fetch all chunks from ChromaDB and build the BM25 index."""
        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            raise RuntimeError(
                "ChromaDB collection is empty. Ingest documents before building index."
            )
        self.bm25.index(all_chunks)
        self._index_built = True
        logger.info(f"Hybrid retriever ready (corpus size: {self.bm25.corpus_size})")

    def query(self, query_text: str) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval and return fused results.
        Builds the BM25 index on first call if not yet built.
        """
        if not self._index_built:
            logger.info("BM25 index not built; building now …")
            self.build_bm25_index()

        # --- Sparse retrieval ---
        bm25_results = self.bm25.query(query_text, top_k=self.bm25_top_k)
        # --- Dense retrieval ---
        vector_results = self.vector_store.query(query_text, top_k=self.vector_top_k)

        logger.debug(
            f"BM25: {len(bm25_results)} | Vector: {len(vector_results)} candidates"
        )

        # --- Reciprocal Rank Fusion ---
        fused_scores = _reciprocal_rank_fusion(
            rankings=[
                [c.chunk_id for c in bm25_results],
                [c.chunk_id for c in vector_results],
            ],
            k=self.rrf_k,
        )

        # Build a lookup of chunk_id → RetrievedChunk
        chunk_map: dict[str, RetrievedChunk] = {}
        for chunk in bm25_results + vector_results:
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        # Sort by fused score descending
        ranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        top_ids = ranked_ids[: self.final_top_k]

        results: list[RetrievedChunk] = []
        for chunk_id in top_ids:
            chunk = chunk_map[chunk_id]
            # Replace raw score with RRF score for downstream transparency
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    score=fused_scores[chunk_id],
                )
            )

        logger.debug(f"Hybrid retrieval returned {len(results)} fused chunks")
        return results


# ------------------------------------------------------------------ #
#  Reciprocal Rank Fusion                                              #
# ------------------------------------------------------------------ #


def _reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    Combine multiple ranked lists into a single score dict.

    Args:
        rankings: Each inner list is an ordered sequence of document IDs
                  (most relevant first).
        k:        RRF constant. Higher k reduces the influence of rank
                  differences. k=60 is the standard default.

    Returns:
        Dict mapping document_id → RRF score (higher = better).
    """
    scores: dict[str, float] = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
