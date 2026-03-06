"""
Phase 1 + 2 – Generation
Answer Generator: full RAG pipeline that:
  1. Receives a query
  2. Runs hybrid retrieval + reranking
  3. Formats context with numbered citations
  4. Calls Claude for generation
  5. Enforces citation validity
  6. Returns a structured AnswerResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.generation.llm_client import AnthropicClient
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import RetrievedChunk
from src.utils.citation_utils import CitationEnforcer, CitationResult
from src.utils.prompt_manager import PromptManager


@dataclass
class AnswerResult:
    """Complete result from one RAG query."""

    question: str
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[RetrievedChunk]
    citation_result: CitationResult
    declined: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def pretty_print(self) -> str:
        """Format the full answer with references for display."""
        parts: list[str] = [self.answer]
        refs = self.citation_result.formatted_references()
        if refs:
            parts.append("\n\nReferences:")
            parts.append(refs)
        if self.declined:
            parts.insert(0, "[RESPONSE DECLINED: insufficient evidence]\n")
        return "\n".join(parts)


class AnswerGenerator:
    """
    Orchestrates the full RAG pipeline for a single query.

    Injection points:
    - hybrid_retriever: Phase 2 BM25+vector fusion (or swap for a
      simple vector-only retriever in Phase 1)
    - reranker:         Phase 2 cross-encoder (can be None to skip)
    - citation_enforcer: Phase 2 citation checker
    """

    def __init__(
        self,
        llm_client: AnthropicClient,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None,
        prompt_manager: PromptManager,
        citation_enforcer: CitationEnforcer | None = None,
        decline_on_invalid_citations: bool = True,
    ) -> None:
        self.llm = llm_client
        self.retriever = retriever
        self.reranker = reranker
        self.prompts = prompt_manager
        self.enforcer = citation_enforcer or CitationEnforcer()
        self.decline_on_invalid = decline_on_invalid_citations

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def answer(self, question: str) -> AnswerResult:
        """
        End-to-end RAG query.

        Returns:
            AnswerResult with answer text, citations, and provenance.
        """
        logger.info(f"Query: {question!r}")

        # ── Step 1: Hybrid retrieval ─────────────────────────────────
        candidates = self.retriever.query(question)
        logger.info(f"Retrieved {len(candidates)} candidate chunks")

        if not candidates:
            return self._no_context_result(question)

        # ── Step 2: Cross-encoder reranking ──────────────────────────
        if self.reranker is not None:
            chunks = self.reranker.rerank(question, candidates)
            logger.info(f"Reranked to {len(chunks)} final chunks")
        else:
            chunks = candidates

        if not chunks:
            return self._no_context_result(question)

        # ── Step 3: Build numbered context block ─────────────────────
        context_block = self.enforcer.build_context_block(chunks)

        # ── Step 4: Construct prompts ─────────────────────────────────
        system_prompt = self.prompts.get("healthcare_rag_system")
        user_prompt = self.prompts.get(
            "healthcare_rag_user",
            context=context_block,
            question=question,
        )

        # ── Step 5: Claude generation ─────────────────────────────────
        raw_answer = self.llm.complete(system_prompt, user_prompt)
        logger.info(
            f"Generated answer ({len(raw_answer)} chars) using "
            f"prompt v{self.prompts.version('healthcare_rag_system')}"
        )

        # ── Step 6: Citation enforcement ──────────────────────────────
        citation_result = self.enforcer.enforce(raw_answer, chunks)

        if citation_result.declined:
            logger.warning("Response declined — INSUFFICIENT_CONTEXT signalled")
            return AnswerResult(
                question=question,
                answer=self.prompts.get("decline_response"),
                citations=[],
                retrieved_chunks=chunks,
                citation_result=citation_result,
                declined=True,
            )

        if not citation_result.is_valid and self.decline_on_invalid:
            logger.warning(
                "Citation validation failed "
                f"(missing={citation_result.missing_citation_numbers}, "
                f"uncited={citation_result.unsupported_claim_count}). "
                "Declining response."
            )
            return AnswerResult(
                question=question,
                answer=self.prompts.get("decline_response"),
                citations=[],
                retrieved_chunks=chunks,
                citation_result=citation_result,
                declined=True,
            )

        return AnswerResult(
            question=question,
            answer=raw_answer,
            citations=citation_result.citations,
            retrieved_chunks=chunks,
            citation_result=citation_result,
            declined=False,
            metadata={
                "prompt_version": self.prompts.version("healthcare_rag_system"),
                "model": self.llm.model,
                "chunks_retrieved": len(candidates),
                "chunks_after_rerank": len(chunks),
            },
        )

    # ------------------------------------------------------------------ #
    #  Streaming variant                                                   #
    # ------------------------------------------------------------------ #

    def answer_stream(self, question: str):
        """
        Streaming version — yields text deltas, then yields a final
        AnswerResult as the last item.

        Note: Citation enforcement runs on the accumulated text after
        streaming completes.
        """
        candidates = self.retriever.query(question)
        if not candidates:
            yield self._no_context_result(question)
            return

        chunks = (
            self.reranker.rerank(question, candidates)
            if self.reranker
            else candidates
        )

        context_block = self.enforcer.build_context_block(chunks)
        system_prompt = self.prompts.get("healthcare_rag_system")
        user_prompt = self.prompts.get(
            "healthcare_rag_user",
            context=context_block,
            question=question,
        )

        accumulated = ""
        for delta in self.llm.stream(system_prompt, user_prompt):
            accumulated += delta
            yield delta  # stream text delta

        # Final citation check
        citation_result = self.enforcer.enforce(accumulated, chunks)
        yield AnswerResult(
            question=question,
            answer=accumulated,
            citations=citation_result.citations,
            retrieved_chunks=chunks,
            citation_result=citation_result,
            declined=citation_result.declined,
        )

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _no_context_result(self, question: str) -> AnswerResult:
        from src.utils.citation_utils import CitationResult

        logger.warning("No chunks retrieved — returning decline response")
        return AnswerResult(
            question=question,
            answer=self.prompts.get("decline_response"),
            citations=[],
            retrieved_chunks=[],
            citation_result=CitationResult(is_valid=False, answer="", declined=True),
            declined=True,
        )
