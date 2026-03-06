"""
Phase 1 – Document Ingestion
Embedding Model: wraps sentence-transformers to produce L2-normalised
dense vectors for ChromaDB ingestion.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from loguru import logger

from .chunker import Chunk


class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.

    Produces normalised embeddings suitable for cosine similarity search
    in ChromaDB (distance_metric=cosine).
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None  # lazy load

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of strings; returns list of float vectors."""
        model = self._get_model()
        logger.info(
            f"Embedding {len(texts)} texts with '{self.model_name}' "
            f"(batch={self.batch_size}, device={self.device})"
        )
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=self.normalize,
            device=self.device,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        model = self._get_model()
        vec = model.encode(
            text,
            normalize_embeddings=self.normalize,
            device=self.device,
        )
        return vec.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> tuple[list[list[float]], list[str]]:
        """
        Embed a list of Chunk objects.
        Returns (embeddings, chunk_ids) in the same order.
        """
        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        embeddings = self.embed_texts(texts)
        return embeddings, chunk_ids

    def dimension(self) -> int:
        """Return vector dimension of the current model."""
        return self._get_model().get_sentence_embedding_dimension()

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            logger.info(f"Loading embedding model '{self.model_name}' on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                f"Model loaded. Vector dimension: {self._model.get_sentence_embedding_dimension()}"
            )
        return self._model
