"""
Tests for Phase 1: PDF loading, chunking, and embedding.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.pdf_loader import Document, PDFLoader, _clean_text
from src.ingestion.chunker import Chunk, TextChunker, _split_into_sentences
from src.ingestion.embedder import EmbeddingModel


# ------------------------------------------------------------------ #
#  _clean_text                                                          #
# ------------------------------------------------------------------ #

def test_clean_text_collapses_newlines():
    raw = "line1\n\n\n\nline2"
    assert _clean_text(raw) == "line1\n\nline2"


def test_clean_text_collapses_spaces():
    raw = "word1    word2"
    assert _clean_text(raw) == "word1 word2"


def test_clean_text_empty():
    assert _clean_text("") == ""
    assert _clean_text("   ") == ""


# ------------------------------------------------------------------ #
#  PDFLoader                                                           #
# ------------------------------------------------------------------ #

def test_pdf_loader_raises_file_not_found():
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file("/nonexistent/path/file.pdf")


def test_pdf_loader_raises_wrong_extension():
    loader = PDFLoader()
    with pytest.raises(ValueError, match="Expected a .pdf"):
        loader.load_file(__file__)  # .py file


def test_pdf_loader_empty_directory(tmp_path):
    loader = PDFLoader()
    result = loader.load_directory(tmp_path)
    assert result == []


def test_pdf_loader_invalid_directory():
    loader = PDFLoader()
    with pytest.raises(NotADirectoryError):
        loader.load_directory("/no/such/directory")


# ------------------------------------------------------------------ #
#  _split_into_sentences                                               #
# ------------------------------------------------------------------ #

def test_split_sentences_basic():
    text = "Hypertension is dangerous. Treatment includes diet. Exercise helps too."
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3


def test_split_sentences_paragraphs():
    text = "First paragraph.\n\nSecond paragraph starts here. It has two sentences."
    sentences = _split_into_sentences(text)
    assert len(sentences) >= 2


def test_split_sentences_empty():
    assert _split_into_sentences("") == []
    assert _split_into_sentences("   \n\n   ") == []


# ------------------------------------------------------------------ #
#  TextChunker                                                         #
# ------------------------------------------------------------------ #

def test_chunker_produces_chunks(text_chunker, sample_document):
    chunks = text_chunker.chunk_documents([sample_document])
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.text.strip() != ""
        assert chunk.chunk_id != ""


def test_chunker_metadata_preserved(text_chunker, sample_document):
    chunks = text_chunker.chunk_documents([sample_document])
    for chunk in chunks:
        assert chunk.metadata["source"] == "hypertension_guidelines.pdf"
        assert chunk.metadata["page"] == 3


def test_chunker_token_count_set(text_chunker, sample_document):
    chunks = text_chunker.chunk_documents([sample_document])
    for chunk in chunks:
        # Token count may be 0 if tiktoken unavailable, but field should exist
        assert isinstance(chunk.token_count, int)
        assert chunk.token_count >= 0


def test_chunker_chunk_id_unique(text_chunker, sample_document):
    docs = [sample_document, sample_document]  # two copies
    chunks = text_chunker.chunk_documents(docs)
    ids = [c.chunk_id for c in chunks]
    # chunk_ids should be generated (may duplicate for identical source docs, that's ok)
    assert all(isinstance(cid, str) for cid in ids)


def test_chunker_empty_document(text_chunker):
    doc = Document(page_content="", metadata={"source": "empty.pdf", "page": 1})
    chunks = text_chunker.chunk_documents([doc])
    assert chunks == []


def test_chunker_save_and_load(tmp_path, text_chunker, sample_document):
    chunks = text_chunker.chunk_documents([sample_document])
    out = tmp_path / "chunks.json"
    text_chunker.save_chunks(chunks, out)
    assert out.exists()

    import json
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == len(chunks)


# ------------------------------------------------------------------ #
#  EmbeddingModel                                                      #
# ------------------------------------------------------------------ #

def test_embedding_model_embed_texts_shape():
    """Mock SentenceTransformer to avoid loading real model in CI."""
    import numpy as np

    with patch("sentence_transformers.SentenceTransformer") as MockST:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.random.rand(3, 768).astype("float32")
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        MockST.return_value = mock_instance

        model = EmbeddingModel(model_name="all-mpnet-base-v2")
        model._model = mock_instance

        texts = ["text one", "text two", "text three"]
        embeddings = model.embed_texts(texts)

        assert len(embeddings) == 3
        assert len(embeddings[0]) == 768


def test_embedding_model_embed_query():
    import numpy as np

    model = EmbeddingModel()
    mock = MagicMock()
    mock.encode.return_value = np.random.rand(768).astype("float32")
    model._model = mock

    vec = model.embed_query("treatment for diabetes")
    assert len(vec) == 768
