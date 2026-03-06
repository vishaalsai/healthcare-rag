#!/usr/bin/env python
"""
Phase 1 – Ingestion Script
Loads PDFs from data/raw/, chunks them, embeds, and stores in ChromaDB.

Usage:
    python scripts/ingest_docs.py [--source-dir data/raw] [--reset]

Options:
    --source-dir DIR   Directory containing PDFs (default: data/raw)
    --reset            Drop existing collection before re-indexing
    --dry-run          Chunk and show stats without writing to ChromaDB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# ── Allow running from project root ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.prompt_manager import PromptManager

import yaml

load_dotenv()


def load_settings() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(source_dir: str, reset: bool, dry_run: bool) -> None:
    settings = load_settings()
    ing = settings["ingestion"]
    emb = settings["embeddings"]
    chroma = settings["chromadb"]

    # ── 1. Load PDFs ──────────────────────────────────────────────────
    logger.info(f"Loading PDFs from '{source_dir}' …")
    loader = PDFLoader()
    documents = loader.load_directory(source_dir)

    if not documents:
        logger.error("No documents loaded. Exiting.")
        sys.exit(1)

    # ── 2. Chunk ──────────────────────────────────────────────────────
    logger.info("Chunking documents …")
    chunker = TextChunker(
        chunk_size=ing["chunk_size_target"],
        chunk_overlap=ing["chunk_overlap"],
        chunk_size_min=ing["chunk_size_min"],
        chunk_size_max=ing["chunk_size_max"],
        min_chunk_chars=ing["min_chunk_chars"],
        encoding=ing["encoding"],
    )
    chunks = chunker.chunk_documents(documents)
    logger.info(f"Total chunks: {len(chunks)}")

    # Save chunks to disk for inspection
    out_path = Path("data/processed/chunks.json")
    chunker.save_chunks(chunks, out_path)

    if dry_run:
        logger.info("[dry-run] Stopping before ChromaDB write.")
        _print_chunk_stats(chunks, chunker)
        return

    # ── 3. Embed + store ──────────────────────────────────────────────
    logger.info("Loading embedding model …")
    embedding_model = EmbeddingModel(
        model_name=emb["model"],
        device=emb["device"],
        batch_size=emb["batch_size"],
        normalize=emb["normalize"],
    )

    store = ChromaVectorStore(
        embedding_model=embedding_model,
        persist_directory=chroma["persist_directory"],
        collection_name=chroma["collection_name"],
        distance_metric=chroma["distance_metric"],
    )

    if reset:
        logger.warning("--reset flag set: dropping existing collection")
        store.reset_collection()

    store.add_chunks(chunks)
    logger.info(
        f"Ingestion complete. Collection '{chroma['collection_name']}' has "
        f"{store.collection_count()} documents."
    )


def _print_chunk_stats(chunks, chunker) -> None:
    token_counts = [c.token_count for c in chunks]
    print(f"\nChunk statistics:")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Min tokens   : {min(token_counts)}")
    print(f"  Max tokens   : {max(token_counts)}")
    print(f"  Avg tokens   : {sum(token_counts)/len(token_counts):.1f}")
    print(f"\nSample chunk:")
    c = chunks[0]
    print(f"  ID     : {c.chunk_id}")
    print(f"  Tokens : {c.token_count}")
    print(f"  Text   : {c.text[:200]}…")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Healthcare RAG")
    parser.add_argument(
        "--source-dir", default="data/raw", help="Directory with PDFs"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Drop and recreate collection"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Chunk only, skip ChromaDB write"
    )
    args = parser.parse_args()
    main(args.source_dir, args.reset, args.dry_run)
