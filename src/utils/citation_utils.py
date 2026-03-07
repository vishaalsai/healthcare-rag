"""
Phase 2 – Production Quality
Citation Enforcer: parses inline [N] citations from LLM output,
verifies each citation maps to a real retrieved chunk, and
flags or declines responses with unsupported claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.retrieval.vector_store import RetrievedChunk


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_INSUFFICIENT_PREFIX = "INSUFFICIENT_CONTEXT"

# Hedged phrases that indicate the LLM is declining to answer fully
# despite having context — treat these the same as INSUFFICIENT_CONTEXT.
_HEDGED_DECLINE_PHRASES = [
    "the context does not",
    "i can offer a partial",
    "not fully covered",
    "the provided context does not contain",
    "i cannot find",
    "not covered in the",
]


@dataclass
class CitationResult:
    """Outcome of citation validation."""

    is_valid: bool
    answer: str                              # Possibly annotated answer
    citations: list[dict[str, Any]] = field(default_factory=list)
    unsupported_claim_count: int = 0
    missing_citation_numbers: list[int] = field(default_factory=list)
    declined: bool = False                   # True if INSUFFICIENT_CONTEXT detected

    def formatted_references(self) -> str:
        """Generate a references block like: [1] source.pdf p.3 §1"""
        if not self.citations:
            return ""
        lines = []
        for c in self.citations:
            n = c["number"]
            label = c.get("label", "Unknown source")
            lines.append(f"[{n}] {label}")
        return "\n".join(lines)


class CitationEnforcer:
    """
    Validates that every [N] citation in an LLM response corresponds to
    chunk N in the retrieved context, and that no claim is left uncited.

    Policy (configurable):
    - If the LLM signals INSUFFICIENT_CONTEXT → mark as declined.
    - If any cited [N] is out of range  → flag as invalid.
    - If >max_uncited_sentences sentences lack a citation → flag as invalid.
    """

    def __init__(
        self,
        insufficient_token: str = _INSUFFICIENT_PREFIX,
        max_uncited_sentences: int = 3,
    ) -> None:
        self.insufficient_token = insufficient_token
        self.max_uncited_sentences = max_uncited_sentences

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def enforce(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> CitationResult:
        """
        Validate citations in `answer` against the provided `chunks`.

        Chunks are 1-indexed in prompts (chunk 1 = chunks[0]).
        """
        # 1. Check for explicit insufficiency signal
        if answer.strip().startswith(self.insufficient_token):
            logger.info("LLM signalled INSUFFICIENT_CONTEXT — declining response")
            return CitationResult(
                is_valid=False,
                answer=answer,
                declined=True,
            )

        # 1b. Check for hedged decline phrases (LLM hedges instead of answering)
        answer_lower = answer.lower()
        for phrase in _HEDGED_DECLINE_PHRASES:
            if phrase in answer_lower:
                logger.info(
                    f"LLM used hedged decline phrase {phrase!r} — declining response"
                )
                return CitationResult(
                    is_valid=False,
                    answer=answer,
                    declined=True,
                )

        # 2. Parse all [N] citations
        cited_numbers = [
            int(m) for m in _CITATION_PATTERN.findall(answer)
        ]
        unique_cited = sorted(set(cited_numbers))

        # 3. Identify out-of-range citations
        missing_citations: list[int] = [
            n for n in unique_cited if n < 1 or n > len(chunks)
        ]
        if missing_citations:
            logger.warning(
                f"Answer cites {missing_citations} but only {len(chunks)} chunks available"
            )

        # 4. Build citation map  {N → chunk}
        citation_info: list[dict[str, Any]] = []
        for n in unique_cited:
            if 1 <= n <= len(chunks):
                chunk = chunks[n - 1]
                citation_info.append(
                    {
                        "number": n,
                        "chunk_id": chunk.chunk_id,
                        "label": chunk.citation_label(),
                        "source": chunk.metadata.get("source", "unknown"),
                        "page": chunk.metadata.get("page", "?"),
                        "score": chunk.score,
                    }
                )

        # 5. Check for sentences lacking any citation
        uncited = _count_uncited_sentences(answer)
        if uncited > self.max_uncited_sentences:
            logger.warning(
                f"{uncited} sentences lack citations (threshold: {self.max_uncited_sentences})"
            )

        is_valid = (
            len(missing_citations) == 0
            and uncited <= self.max_uncited_sentences
        )

        return CitationResult(
            is_valid=is_valid,
            answer=answer,
            citations=citation_info,
            unsupported_claim_count=uncited,
            missing_citation_numbers=missing_citations,
            declined=False,
        )

    def build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into the numbered context block
        that the LLM prompt references as [1], [2], …
        """
        lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            label = chunk.citation_label()
            lines.append(f"[{i}] ({label})\n{chunk.text}")
        return "\n\n".join(lines)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _count_uncited_sentences(text: str) -> int:
    """
    Count sentences that contain at least one factual claim
    but no inline [N] citation.  Skips structural/formatting lines.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    uncited = 0
    for sentence in sentences:
        stripped = sentence.strip()
        # Skip short lines
        if len(stripped) < 30:
            continue
        # Skip markdown headers (# ## ### ####)
        if re.match(r"^#{1,4}\s", stripped):
            continue
        # Skip markdown table rows and dividers
        if stripped.startswith("|") or re.match(r"^[-|]+$", stripped):
            continue
        # Skip list items (bullet or numbered) that are category labels
        if re.match(r"^[-*•]\s+\*\*[^*]+\*\*\s*$", stripped):
            continue
        # Skip common non-claim patterns
        if re.match(
            r"^(\u26a0|disclaimer|note:|warning:|if you|please consult|i was unable)",
            stripped.lower(),
        ):
            continue
        if not _CITATION_PATTERN.search(stripped):
            uncited += 1
    return uncited
