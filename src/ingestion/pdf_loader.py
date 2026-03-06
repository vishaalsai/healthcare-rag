"""
Phase 1 – Document Ingestion
PDF Loader: extracts page-level text from clinical PDFs,
preserves source metadata for paragraph-level citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class Document:
    """Atomic unit of text with provenance metadata."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        src = self.metadata.get("source", "unknown")
        pg = self.metadata.get("page", "?")
        return f"Document(source={src!r}, page={pg}, chars={len(self.page_content)})"


class PDFLoader:
    """
    Load PDF files into page-level Documents using PyMuPDF (fitz).

    Falls back to pdfplumber for scanned/image-heavy PDFs.
    """

    def __init__(self, fallback_to_pdfplumber: bool = True) -> None:
        self.fallback = fallback_to_pdfplumber

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a single PDF; returns one Document per non-empty page."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        docs = self._load_with_fitz(path)

        if not docs and self.fallback:
            logger.warning(
                f"{path.name}: fitz extracted no text, trying pdfplumber fallback"
            )
            docs = self._load_with_pdfplumber(path)

        logger.info(f"Loaded {len(docs)} pages from '{path.name}'")
        return docs

    def load_directory(self, dir_path: str | Path) -> list[Document]:
        """Load all PDFs from a directory (recursive)."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        pdf_files = sorted(dir_path.rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in '{dir_path}'")
            return []

        logger.info(f"Found {len(pdf_files)} PDF(s) in '{dir_path}'")

        all_docs: list[Document] = []
        for pdf in pdf_files:
            try:
                all_docs.extend(self.load_file(pdf))
            except Exception as exc:
                logger.error(f"Skipping '{pdf.name}': {exc}")

        logger.info(f"Total pages loaded: {len(all_docs)}")
        return all_docs

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_with_fitz(self, path: Path) -> list[Document]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install pymupdf")
            return []

        documents: list[Document] = []
        with fitz.open(str(path)) as pdf:
            total = len(pdf)
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                text = _clean_text(text)
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "source_path": str(path.resolve()),
                            "page": page_num,
                            "total_pages": total,
                            "loader": "fitz",
                        },
                    )
                )
        return documents

    def _load_with_pdfplumber(self, path: Path) -> list[Document]:
        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber not installed. Run: pip install pdfplumber")
            return []

        documents: list[Document] = []
        with pdfplumber.open(str(path)) as pdf:
            total = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = _clean_text(text)
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "source_path": str(path.resolve()),
                            "page": page_num,
                            "total_pages": total,
                            "loader": "pdfplumber",
                        },
                    )
                )
        return documents


# ------------------------------------------------------------------ #
#  Utilities                                                           #
# ------------------------------------------------------------------ #


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove common PDF artefacts."""
    if not text:
        return ""
    # Collapse 3+ newlines → paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove form-feed characters
    text = text.replace("\f", "\n")
    return text.strip()
