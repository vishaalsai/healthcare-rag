"""
Phase 3 – Evaluation
RAGAS Evaluator: runs faithfulness, answer_relevancy, and
context_precision metrics against a golden Q&A dataset.

Outputs a JSON results file and an exit code suitable for CI/CD gating.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class EvalSample:
    """A single evaluation data point."""

    question: str
    answer: str          # Generated answer from the RAG system
    contexts: list[str]  # Retrieved chunk texts
    ground_truth: str    # From the golden dataset


@dataclass
class EvalResult:
    """Aggregated metrics from one evaluation run."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    sample_count: int
    timestamp: str
    passed_thresholds: bool
    per_sample: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RAGEvaluator:
    """
    Evaluates the RAG system against a golden dataset using RAGAS.

    Metrics:
    - faithfulness:         Is the answer supported by the retrieved context?
    - answer_relevancy:     Does the answer address the question?
    - context_precision:    Are the retrieved chunks relevant to the question?
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.75,
        answer_relevancy_threshold: float = 0.70,
        context_precision_threshold: float = 0.65,
        results_dir: str = "./data/eval/results",
    ) -> None:
        self.faithfulness_threshold = faithfulness_threshold
        self.answer_relevancy_threshold = answer_relevancy_threshold
        self.context_precision_threshold = context_precision_threshold
        self.results_dir = Path(results_dir)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        """
        Run RAGAS evaluation on a list of EvalSample objects.

        Returns an EvalResult with aggregated metrics and a pass/fail flag.
        """
        if not samples:
            raise ValueError("Cannot evaluate empty sample list")

        logger.info(f"Running RAGAS evaluation on {len(samples)} samples …")

        dataset = self._build_dataset(samples)
        result_df = self._run_ragas(dataset)

        # --- Aggregate metrics ---
        faithfulness = float(result_df["faithfulness"].mean())
        answer_relevancy = float(result_df["answer_relevancy"].mean())
        context_precision = float(result_df["context_precision"].mean())

        passed = (
            faithfulness >= self.faithfulness_threshold
            and answer_relevancy >= self.answer_relevancy_threshold
            and context_precision >= self.context_precision_threshold
        )

        # --- Per-sample detail ---
        per_sample = result_df.to_dict(orient="records")

        result = EvalResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            sample_count=len(samples),
            timestamp=datetime.utcnow().isoformat() + "Z",
            passed_thresholds=passed,
            per_sample=per_sample,
        )

        self._save_results(result)
        self._log_summary(result)

        return result

    def load_golden_dataset(self, path: str | Path) -> list[dict[str, Any]]:
        """Load golden Q&A pairs from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {path}")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        pairs = data.get("qa_pairs", data)  # support both formats
        logger.info(f"Loaded {len(pairs)} golden Q&A pairs from '{path}'")
        return pairs

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _build_dataset(self, samples: list[EvalSample]):
        """Convert EvalSample list to a HuggingFace Dataset for RAGAS."""
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "datasets not installed. Run: pip install datasets"
            ) from exc

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }
        return Dataset.from_dict(data)

    def _run_ragas(self, dataset):
        """Run RAGAS evaluate() and return result as a pandas DataFrame."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
        except ImportError as exc:
            raise ImportError(
                "ragas not installed. Run: pip install ragas"
            ) from exc

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            raise_exceptions=False,
        )
        return result.to_pandas()

    def _save_results(self, result: EvalResult) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"eval_{ts}.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)
        logger.info(f"Evaluation results saved to '{output_path}'")
        # Also write latest.json for easy CI access
        latest_path = self.results_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)

    def _log_summary(self, result: EvalResult) -> None:
        status = "PASSED ✓" if result.passed_thresholds else "FAILED ✗"
        logger.info(
            f"\n{'='*55}\n"
            f"  RAGAS Evaluation Summary — {status}\n"
            f"{'='*55}\n"
            f"  Faithfulness:       {result.faithfulness:.3f}"
            f"  (threshold: {self.faithfulness_threshold})\n"
            f"  Answer Relevancy:   {result.answer_relevancy:.3f}"
            f"  (threshold: {self.answer_relevancy_threshold})\n"
            f"  Context Precision:  {result.context_precision:.3f}"
            f"  (threshold: {self.context_precision_threshold})\n"
            f"  Samples evaluated:  {result.sample_count}\n"
            f"{'='*55}"
        )
