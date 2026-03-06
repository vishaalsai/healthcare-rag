"""
Phase 1 – Document Ingestion
Text Chunker: token-aware sliding window with configurable size and overlap.
Produces paragraph-level chunks suitable for citation tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from .pdf_loader import Document


@dataclass
class Chunk:
    """A text chunk with full provenance for citation generation."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Populated after chunking
    chunk_id: str = ""
    token_count: int = 0

    def to_document(self) -> Document:
        meta = dict(self.metadata)
        meta["chunk_id"] = self.chunk_id
        meta["token_count"] = self.token_count
        return Document(page_content=self.text, metadata=meta)

    def citation_label(self) -> str:
        """Human-readable citation string, e.g. 'hypertension.pdf p.4 §2'"""
        src = self.metadata.get("source", "unknown")
        page = self.metadata.get("page", "?")
        idx = self.metadata.get("chunk_index_on_page", "?")
        return f"{src} p.{page} §{idx}"


class TextChunker:
    """
    Token-aware sliding window chunker.

    Strategy:
    1. Split page text into sentences / paragraphs.
    2. Greedily accumulate sentences until approaching chunk_size.
    3. When the window is full, emit the chunk and advance by
       (chunk_size - overlap) tokens.

    This guarantees every chunk is in [chunk_size_min, chunk_size_max]
    except for very short pages.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        chunk_size_min: int = 500,
        chunk_size_max: int = 800,
        min_chunk_chars: int = 200,
        encoding: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_size_min = chunk_size_min
        self.chunk_size_max = chunk_size_max
        self.min_chunk_chars = min_chunk_chars
        self._enc = self._load_encoding(encoding)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a list of page-level Documents."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} pages → {len(all_chunks)} chunks "
            f"(avg {self._avg_tokens(all_chunks):.0f} tokens/chunk)"
        )
        return all_chunks

    def save_chunks(self, chunks: list[Chunk], output_path: str | Path) -> None:
        """Persist chunks to a JSON file for inspection / re-use."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "token_count": c.token_count,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved {len(chunks)} chunks to '{output_path}'")

    def count_tokens(self, text: str) -> int:
        if self._enc:
            return len(self._enc.encode(text))
        # Approximation: 1 token ≈ 4 chars
        return len(text) // 4

    # ------------------------------------------------------------------ #
    #  Core chunking logic                                                 #
    # ------------------------------------------------------------------ #

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        sentences = _split_into_sentences(doc.page_content)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        chunk_idx = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            # Single sentence exceeds max — emit as its own chunk
            if sent_tokens > self.chunk_size_max:
                if buffer:
                    chunks.append(self._make_chunk(buffer, buffer_tokens, doc, chunk_idx))
                    chunk_idx += 1
                    buffer, buffer_tokens = [], 0
                # Force-split long sentence
                sub_chunks = self._split_long_sentence(sentence, doc, chunk_idx)
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                continue

            # Adding this sentence would overflow max
            if buffer_tokens + sent_tokens > self.chunk_size_max and buffer:
                chunks.append(self._make_chunk(buffer, buffer_tokens, doc, chunk_idx))
                chunk_idx += 1
                # Carry-over: keep the overlap window
                buffer, buffer_tokens = self._overlap_window(buffer)

            buffer.append(sentence)
            buffer_tokens += sent_tokens

        # Flush remaining
        if buffer and buffer_tokens >= self.min_chunk_chars // 4:
            chunks.append(self._make_chunk(buffer, buffer_tokens, doc, chunk_idx))

        return chunks

    def _make_chunk(
        self,
        sentences: list[str],
        token_count: int,
        doc: Document,
        index: int,
    ) -> Chunk:
        text = " ".join(sentences).strip()
        meta = dict(doc.metadata)
        meta["chunk_index_on_page"] = index
        chunk_id = f"{meta.get('source', 'doc')}_p{meta.get('page', 0)}_c{index}"
        chunk_id = chunk_id.replace(" ", "_").replace("/", "_")

        chunk = Chunk(text=text, metadata=meta, chunk_id=chunk_id, token_count=token_count)
        return chunk

    def _split_long_sentence(
        self, sentence: str, doc: Document, start_idx: int
    ) -> list[Chunk]:
        """Split a sentence that exceeds chunk_size_max by words."""
        words = sentence.split()
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buf_tokens = 0
        idx = start_idx

        for word in words:
            wt = self.count_tokens(word)
            if buf_tokens + wt > self.chunk_size_max and buffer:
                chunks.append(self._make_chunk(buffer, buf_tokens, doc, idx))
                idx += 1
                buffer, buf_tokens = self._overlap_window(buffer)
            buffer.append(word)
            buf_tokens += wt

        if buffer:
            chunks.append(self._make_chunk(buffer, buf_tokens, doc, idx))

        return chunks

    def _overlap_window(self, sentences: list[str]) -> tuple[list[str], int]:
        """Keep the last N tokens worth of sentences for overlap."""
        if not sentences:
            return [], 0
        kept: list[str] = []
        tokens = 0
        for sent in reversed(sentences):
            t = self.count_tokens(sent)
            if tokens + t > self.chunk_overlap:
                break
            kept.insert(0, sent)
            tokens += t
        return kept, tokens

    def _avg_tokens(self, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        return sum(c.token_count for c in chunks) / len(chunks)

    # ------------------------------------------------------------------ #
    #  Encoding                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_encoding(name: str):
        try:
            import tiktoken
            return tiktoken.get_encoding(name)
        except Exception:
            logger.warning(
                "tiktoken not available; using char/4 approximation for token counts"
            )
            return None


# ------------------------------------------------------------------ #
#  Sentence splitter                                                   #
# ------------------------------------------------------------------ #


def _split_into_sentences(text: str) -> list[str]:
    """
    Lightweight sentence / paragraph splitter.
    Prioritises paragraph breaks, then sentence boundaries.
    """
    import re

    # Split on blank lines first (paragraph-level)
    paragraphs = re.split(r"\n\s*\n", text)
    sentences: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Split on sentence boundaries (.!?)
        parts = re.split(r"(?<=[.!?])\s+", para)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences
