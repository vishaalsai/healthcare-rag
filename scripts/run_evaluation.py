#!/usr/bin/env python
"""
Phase 3 – Evaluation Script
Runs RAGAS + custom metrics against the golden dataset.
Exits with code 1 if any metric is below threshold (for CI/CD gating).

Usage:
    python scripts/run_evaluation.py [--dataset data/eval/golden_dataset.json]
    python scripts/run_evaluation.py --max-samples 20  # quick run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

load_dotenv()


def main(dataset_path: str, max_samples: int | None) -> int:
    """Returns 0 on pass, 1 on failure."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with open(config_path) as f:
        settings = yaml.safe_load(f)

    eval_cfg = settings["evaluation"]

    # ── 1. Load golden dataset ────────────────────────────────────────
    from src.evaluation.evaluator import RAGEvaluator, EvalSample

    evaluator = RAGEvaluator(
        faithfulness_threshold=eval_cfg["faithfulness_threshold"],
        answer_relevancy_threshold=eval_cfg["answer_relevancy_threshold"],
        context_precision_threshold=eval_cfg["context_precision_threshold"],
        results_dir=eval_cfg["results_dir"],
    )

    golden_pairs = evaluator.load_golden_dataset(dataset_path)
    if max_samples:
        golden_pairs = golden_pairs[:max_samples]

    logger.info(f"Evaluating {len(golden_pairs)} Q&A pairs …")

    # ── 2. Build pipeline ─────────────────────────────────────────────
    # Import here to avoid slow startup when --help is used
    from scripts.query import build_pipeline  # reuse pipeline factory

    generator = build_pipeline(settings)

    # ── 3. Generate answers for each golden question ──────────────────
    samples: list[EvalSample] = []
    for pair in golden_pairs:
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        try:
            result = generator.answer(question)
            samples.append(
                EvalSample(
                    question=question,
                    answer=result.answer if not result.declined else "",
                    contexts=[c.text for c in result.retrieved_chunks],
                    ground_truth=ground_truth,
                )
            )
        except Exception as exc:
            logger.error(f"Error on question {question!r}: {exc}")
            samples.append(
                EvalSample(
                    question=question,
                    answer="ERROR",
                    contexts=[],
                    ground_truth=ground_truth,
                )
            )

    # ── 4. Run RAGAS ──────────────────────────────────────────────────
    eval_result = evaluator.evaluate(samples)

    # ── 5. Run custom metrics ─────────────────────────────────────────
    from src.generation.answer_generator import AnswerResult
    from src.evaluation.metrics import compute_custom_metrics

    answer_results = []
    for pair in golden_pairs:
        answer_results.append(generator.answer(pair["question"]))

    custom = compute_custom_metrics(answer_results)

    # ── 6. Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RAGAS Metrics")
    print("=" * 60)
    print(f"  Faithfulness:      {eval_result.faithfulness:.3f}  (>= {eval_cfg['faithfulness_threshold']})")
    print(f"  Answer Relevancy:  {eval_result.answer_relevancy:.3f}  (>= {eval_cfg['answer_relevancy_threshold']})")
    print(f"  Context Precision: {eval_result.context_precision:.3f}  (>= {eval_cfg['context_precision_threshold']})")
    print()
    print("  Custom Metrics")
    print("-" * 60)
    print(f"  Citation Coverage: {custom.citation_coverage:.2%}")
    print(f"  Decline Rate:      {custom.decline_rate:.2%}")
    print(f"  Context Util.:     {custom.context_utilization:.2%}")
    print("=" * 60)

    overall_passed = eval_result.passed_thresholds and custom.passes_quality_gate()
    if overall_passed:
        print("  ✓ All thresholds passed — build APPROVED")
    else:
        print("  ✗ One or more thresholds failed — build REJECTED")
    print("=" * 60 + "\n")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation against golden dataset")
    parser.add_argument(
        "--dataset",
        default="data/eval/golden_dataset.json",
        help="Path to golden Q&A JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (useful for quick testing)",
    )
    args = parser.parse_args()
    exit_code = main(args.dataset, args.max_samples)
    sys.exit(exit_code)
