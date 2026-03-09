"""
Phase 2 – Hybrid Retrieval
BM25 Retriever: sparse keyword-based search over the full document corpus.
Uses rank_bm25 with simple whitespace tokenization plus stopword removal.
"""

from __future__ import annotations

import re

from loguru import logger

from src.retrieval.vector_store import RetrievedChunk


_STOPWORDS = frozenset(
    """a an the and or but in on at to for of with is are was were be been
    being have has had do does did will would could should may might shall
    can this that these those it its from by about as into through during
    before after above below between such no not if so""".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Retriever:
    """
    In-memory BM25 index built from a list of RetrievedChunk objects.

    Build once after initial ingestion (or after collection reset) and
    reuse across queries.  Serialisation support is not included because
    BM25 index build is fast (<1 s for 10 k chunks).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus_chunks: list[RetrievedChunk] = []
        self._bm25 = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def index(self, chunks: list[RetrievedChunk]) -> None:
        """Build the BM25 index from a list of chunks."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 not installed. Run: pip install rank-bm25"
            ) from exc

        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list")

        self._corpus_chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built for {len(chunks)} documents")

    def query(self, query_text: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Return top_k chunks ranked by BM25 score."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call .index() first.")

        tokens = _tokenize(query_text)
        if not tokens:
            logger.warning(f"BM25: query produced no tokens: {query_text!r}")
            return []

        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        ranked = sorted(
            zip(scores, self._corpus_chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for score, chunk in ranked[:top_k]:
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    score=float(score),
                )
            )

        return results

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def corpus_size(self) -> int:
        return len(self._corpus_chunks)
