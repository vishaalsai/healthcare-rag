"""
Phase 1 – Retrieval
ChromaDB vector store wrapper: handles collection management,
document upsert, and similarity search with metadata filtering.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import EmbeddingModel


class RetrievedChunk:
    """A chunk returned by retrieval, augmented with a similarity score."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
        score: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.score = score  # higher = more relevant

    def citation_label(self) -> str:
        src = self.metadata.get("source", "unknown")
        page = self.metadata.get("page", "?")
        idx = self.metadata.get("chunk_index_on_page", "?")
        return f"{src} p.{page} §{idx}"

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(id={self.chunk_id!r}, score={self.score:.3f}, "
            f"chars={len(self.text)})"
        )


class ChromaVectorStore:
    """
    Persistent ChromaDB collection for healthcare document chunks.

    Usage::

        store = ChromaVectorStore(embedding_model, persist_dir="./data/chroma_db")
        store.add_chunks(chunks)
        results = store.query("hypertension treatment", top_k=10)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "healthcare_docs",
        distance_metric: str = "cosine",
    ) -> None:
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> None:
        """Upsert chunks into the collection (idempotent by chunk_id)."""
        if not chunks:
            logger.warning("add_chunks called with empty list")
            return

        collection = self._get_collection()
        logger.info(f"Embedding and indexing {len(chunks)} chunks …")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata for c in batch]
            embeddings = self.embedding_model.embed_texts(texts)

            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")

        logger.info(
            f"Collection '{self.collection_name}' now contains "
            f"{collection.count()} documents"
        )

    def query(self, query_text: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Vector similarity search; returns top_k chunks sorted by relevance."""
        collection = self._get_collection()
        query_embedding = self.embedding_model.embed_query(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            score = 1.0 - distance
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=meta,
                    score=score,
                )
            )

        return chunks

    def get_all_chunks(self) -> list[RetrievedChunk]:
        """Fetch every document from the collection (used for BM25 indexing)."""
        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return []

        results = collection.get(include=["documents", "metadatas"])
        chunks = []
        for chunk_id, text, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            chunks.append(
                RetrievedChunk(chunk_id=chunk_id, text=text, metadata=meta, score=0.0)
            )
        return chunks

    def collection_count(self) -> int:
        return self._get_collection().count()

    def reset_collection(self) -> None:
        """Delete and recreate the collection. Destructive — use with caution."""
        client = self._get_client()
        client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning(f"Collection '{self.collection_name}' has been reset")
        self._get_collection()

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "chromadb not installed. Run: pip install chromadb"
                ) from exc
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(
                f"ChromaDB client initialised (persist_dir={self.persist_directory!r})"
            )
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            import chromadb

            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
            logger.debug(
                f"Using collection '{self.collection_name}' "
                f"({self._collection.count()} docs)"
            )
        return self._collection
