"""
Phase 1 + 2 – Generation
Answer Generator: full RAG pipeline that:
  1. Receives a query
  2. Runs hybrid retrieval + reranking
  3. Formats context with numbered citations
  4. Calls Claude for generation
  5. Enforces citation validity
  6. Returns a structured AnswerResult

Phase 3 addition: every call is traced in Langfuse with spans for
retrieval, reranking, prompt_build, and generation.
"""

from __future__ import annotations

import time
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

    def answer(self, question: str, trace_id: str | None = None) -> AnswerResult:
        """
        End-to-end RAG query with Langfuse tracing.

        Args:
            question: The user's clinical question.
            trace_id: Optional UUID to group all spans under one trace in
                      Langfuse (pass from the API layer so HTTP requests
                      map 1-to-1 to traces).

        Returns:
            AnswerResult with answer text, citations, and provenance.
        """
        logger.info(f"Query: {question!r}")

        # ── Initialise top-level Langfuse trace ───────────────────────
        trace = None
        try:
            from src.observability.tracer import calculate_cost, create_trace

            trace = create_trace(
                name="rag-query",
                input=question,
                metadata={"model": self.llm.model},
                trace_id=trace_id,
            )
        except Exception:
            pass  # tracing must never crash the pipeline

        # ── Step 1: Hybrid retrieval ──────────────────────────────────
        _t0 = time.perf_counter()
        candidates = self.retriever.query(question)
        _retrieval_latency = time.perf_counter() - _t0
        logger.info(f"Retrieved {len(candidates)} candidate chunks")

        try:
            if trace is not None:
                trace.span(
                    name="retrieval",
                    input={"query": question},
                    output={
                        "chunks_retrieved": len(candidates),
                        "chunk_ids": [c.chunk_id for c in candidates],
                    },
                    metadata={"latency_s": round(_retrieval_latency, 4)},
                )
        except Exception:
            pass

        if not candidates:
            result = self._no_context_result(question)
            self._end_trace(trace, result)
            return result

        # ── Step 2: Cross-encoder reranking ───────────────────────────
        _t0 = time.perf_counter()
        if self.reranker is not None:
            chunks = self.reranker.rerank(question, candidates)
            logger.info(f"Reranked to {len(chunks)} final chunks")
        else:
            chunks = candidates
        _rerank_latency = time.perf_counter() - _t0

        try:
            if trace is not None:
                trace.span(
                    name="reranking",
                    input={"input_chunk_count": len(candidates)},
                    output={"output_chunk_count": len(chunks)},
                    metadata={"latency_s": round(_rerank_latency, 4)},
                )
        except Exception:
            pass

        if not chunks:
            result = self._no_context_result(question)
            self._end_trace(trace, result)
            return result

        # ── Step 3: Build numbered context block ──────────────────────
        context_block = self.enforcer.build_context_block(chunks)
        system_prompt = self.prompts.get("healthcare_rag_system")
        user_prompt = self.prompts.get(
            "healthcare_rag_user",
            context=context_block,
            question=question,
        )

        try:
            if trace is not None:
                token_estimate = self._estimate_tokens(system_prompt + user_prompt)
                trace.span(
                    name="prompt_build",
                    input={"chunk_count": len(chunks)},
                    output={
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    },
                    metadata={"token_estimate": token_estimate},
                )
        except Exception:
            pass

        # ── Step 4: Claude generation ─────────────────────────────────
        _t0 = time.perf_counter()
        raw_answer, input_tokens, output_tokens = self.llm.complete(
            system_prompt, user_prompt
        )
        _gen_latency = time.perf_counter() - _t0
        logger.info(
            f"Generated answer ({len(raw_answer)} chars) using "
            f"prompt v{self.prompts.version('healthcare_rag_system')}"
        )

        try:
            if trace is not None:
                cost = calculate_cost(input_tokens, output_tokens)
                trace.generation(
                    name="generation",
                    model=self.llm.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    output=raw_answer,
                    usage={"input": input_tokens, "output": output_tokens},
                    metadata={
                        "latency_s": round(_gen_latency, 4),
                        **cost,
                    },
                )
        except Exception:
            pass

        # ── Step 5: Citation enforcement ──────────────────────────────
        citation_result = self.enforcer.enforce(raw_answer, chunks)

        if citation_result.declined:
            logger.warning("Response declined — INSUFFICIENT_CONTEXT signalled")
            result = AnswerResult(
                question=question,
                answer=self.prompts.get("decline_response"),
                citations=[],
                retrieved_chunks=chunks,
                citation_result=citation_result,
                declined=True,
            )
            self._end_trace(
                trace,
                result,
                extra_metadata={
                    "model": self.llm.model,
                    "citation_count": 0,
                    "declined_reason": "INSUFFICIENT_CONTEXT",
                    "insufficient_context": True,
                },
            )
            return result

        if not citation_result.is_valid and self.decline_on_invalid:
            logger.warning(
                "Citation validation failed "
                f"(missing={citation_result.missing_citation_numbers}, "
                f"uncited={citation_result.unsupported_claim_count}). "
                "Declining response."
            )
            result = AnswerResult(
                question=question,
                answer=self.prompts.get("decline_response"),
                citations=[],
                retrieved_chunks=chunks,
                citation_result=citation_result,
                declined=True,
            )
            self._end_trace(
                trace,
                result,
                extra_metadata={
                    "model": self.llm.model,
                    "citation_count": 0,
                    "declined_reason": "invalid_citations",
                    "insufficient_context": False,
                },
            )
            return result

        result = AnswerResult(
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
        self._end_trace(
            trace,
            result,
            extra_metadata={
                "model": self.llm.model,
                "citation_count": len(result.citations),
                "declined": False,
                "insufficient_context": False,
            },
        )
        return result

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

    def _end_trace(
        self,
        trace,
        result: AnswerResult,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update the Langfuse trace with the final answer and metadata."""
        if trace is None:
            return
        try:
            metadata = {"declined": result.declined}
            if extra_metadata:
                metadata.update(extra_metadata)
            trace.update(output=result.answer, metadata=metadata)
        except Exception:
            pass

    @staticmethod
    def _estimate_tokens(text: str) -> int | None:
        """Return cl100k_base token count for text, or None if tiktoken unavailable."""
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return None
