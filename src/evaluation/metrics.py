"""
Phase 3 – Evaluation
Custom evaluation metrics that complement RAGAS.

- citation_coverage: % of answers with at least one valid citation
- decline_rate:      % of queries where system declined due to low confidence
- avg_chunks_used:   Average retrieved chunks per answer
- context_utilization: % of retrieved chunks referenced in the final answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger


@dataclass
class EvaluationMetrics:
    """Lightweight metrics computed without external dependencies."""

    citation_coverage: float       # 0–1
    decline_rate: float            # 0–1
    avg_chunks_used: float         # mean chunks per query
    context_utilization: float     # 0–1: fraction of retrieved chunks cited
    total_queries: int

    def passes_quality_gate(
        self,
        min_citation_coverage: float = 0.90,
        max_decline_rate: float = 0.20,
    ) -> bool:
        return (
            self.citation_coverage >= min_citation_coverage
            and self.decline_rate <= max_decline_rate
        )

    def to_dict(self) -> dict:
        return {
            "citation_coverage": round(self.citation_coverage, 4),
            "decline_rate": round(self.decline_rate, 4),
            "avg_chunks_used": round(self.avg_chunks_used, 2),
            "context_utilization": round(self.context_utilization, 4),
            "total_queries": self.total_queries,
        }


_CITATION_RE = re.compile(r"\[(\d+)\]")


def compute_custom_metrics(answer_results: list) -> EvaluationMetrics:
    """
    Compute custom metrics from a list of AnswerResult objects.

    Args:
        answer_results: List of AnswerResult from AnswerGenerator.answer()

    Returns:
        EvaluationMetrics dataclass.
    """
    if not answer_results:
        raise ValueError("No results to evaluate")

    n = len(answer_results)
    declined = 0
    cited = 0
    total_chunks = 0
    total_cited_refs = 0

    for result in answer_results:
        if result.declined:
            declined += 1
            continue

        cited_numbers = set(int(m) for m in _CITATION_RE.findall(result.answer))
        n_chunks = len(result.retrieved_chunks)
        total_chunks += n_chunks

        if cited_numbers:
            cited += 1
            # How many distinct chunks were actually cited?
            valid_cited = {c for c in cited_numbers if 1 <= c <= n_chunks}
            total_cited_refs += len(valid_cited)

    non_declined = n - declined

    citation_coverage = cited / non_declined if non_declined > 0 else 0.0
    decline_rate = declined / n
    avg_chunks = total_chunks / n
    context_util = total_cited_refs / total_chunks if total_chunks > 0 else 0.0

    metrics = EvaluationMetrics(
        citation_coverage=citation_coverage,
        decline_rate=decline_rate,
        avg_chunks_used=avg_chunks,
        context_utilization=context_util,
        total_queries=n,
    )

    logger.info(
        f"Custom metrics: citation_coverage={citation_coverage:.2%}, "
        f"decline_rate={decline_rate:.2%}, "
        f"context_utilization={context_util:.2%}"
    )
    return metrics
