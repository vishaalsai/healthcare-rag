"""
Phase 2 – Hybrid Retrieval
Cross-Encoder Reranker: re-scores (query, passage) pairs to produce a
more accurate relevance ordering than bi-encoder similarity alone.

Model default: cross-encoder/ms-marco-MiniLM-L-6-v2
  – Fast, ~22M params, state-of-the-art on MS-MARCO
  – Swap for cross-encoder/ms-marco-electra-base for higher quality
"""

from __future__ import annotations

from loguru import logger

from src.retrieval.vector_store import RetrievedChunk


class CrossEncoderReranker:
    """
    Rerank a candidate list of chunks using a cross-encoder model.

    Cross-encoders attend to both query and passage jointly, giving
    significantly better relevance scores than dot-product similarity.
    They are too slow for first-stage retrieval but ideal for reranking
    a small candidate set (10–20 chunks).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        min_score: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score  # optional score floor; None = no filtering
        self._model = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        Score each (query, chunk) pair and return top_k chunks sorted
        by cross-encoder score descending.
        """
        if not chunks:
            return []

        model = self._get_model()
        pairs = [(query, c.text) for c in chunks]

        raw_scores: list[float] = model.predict(pairs).tolist()

        scored: list[tuple[float, RetrievedChunk]] = list(zip(raw_scores, chunks))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievedChunk] = []
        for score, chunk in scored[: self.top_k]:
            if self.min_score is not None and score < self.min_score:
                logger.debug(
                    f"Dropping chunk {chunk.chunk_id!r} (score {score:.3f} < "
                    f"threshold {self.min_score})"
                )
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    score=float(score),
                )
            )

        top_score = f"{results[0].score:.3f}" if results else "N/A"
        logger.debug(
            f"Reranker: {len(chunks)} -> {len(results)} chunks "
            f"(top score: {top_score})"
        )
        return results

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            logger.info(f"Loading cross-encoder model '{self.model_name}' …")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded")
        return self._model
