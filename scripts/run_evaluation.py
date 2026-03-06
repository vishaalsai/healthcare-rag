#!/usr/bin/env python
"""
Phase 4 — Evaluation Script
Runs each Q&A pair from the golden dataset through the live FastAPI /query
endpoint and scores the responses on three metrics:

  faithfulness      (0–1)  : fraction of expected key terms present in answer
  citation_present  (0/1)  : at least one citation was returned
  declined_correctly(0/1)  : system declined iff should_decline is true

Overall result:  PASS  if mean faithfulness >= 0.70
                 FAIL  otherwise

Exits with code 0 (PASS) or 1 (FAIL) — consumed by CI/CD.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --api-url http://localhost:8000
    python scripts/run_evaluation.py --max-samples 5   # quick smoke test
    python scripts/run_evaluation.py --dataset data/eval/golden_dataset.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from loguru import logger

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── constants ─────────────────────────────────────────────────────────────────
FAITHFULNESS_PASS_THRESHOLD = 0.70
REQUEST_TIMEOUT = 120  # seconds per query

_STOP_WORDS = {
    "that", "this", "with", "from", "have", "been", "they", "their",
    "which", "should", "would", "could", "more", "than", "each", "when",
    "will", "also", "both", "into", "such", "very", "most", "some",
    "used", "been", "were", "does", "using", "must", "then", "than",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_faithfulness(expected_answer: str, actual_answer: str) -> float:
    """
    Keyword-overlap faithfulness: fraction of meaningful terms from
    expected_answer that appear in actual_answer.

    Returns 1.0 if expected_answer is 'DECLINE' (not applicable) or empty.
    Returns 0.0 if the system returned an empty answer for a non-decline question.
    """
    if not expected_answer or expected_answer.strip().upper() == "DECLINE":
        return 1.0  # metric not applicable for should_decline=True questions

    if not actual_answer or actual_answer.strip() == "":
        return 0.0  # system failed to produce any answer

    expected_lower = expected_answer.lower()
    actual_lower = actual_answer.lower()

    # Extract meaningful terms (length >= 4, not stop words)
    raw_terms = re.findall(r"\b[a-z][a-z0-9]{3,}\b", expected_lower)
    terms = [t for t in raw_terms if t not in _STOP_WORDS]

    if not terms:
        return 1.0  # nothing meaningful to match against

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_terms = [t for t in terms if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    matches = sum(1 for t in unique_terms if t in actual_lower)
    return round(matches / len(unique_terms), 4)


def compute_citation_present(api_response: dict, should_decline: bool) -> float:
    """1.0 if citations were returned (or question should be declined), else 0.0."""
    if should_decline:
        return 1.0  # not applicable
    return 1.0 if api_response.get("citations") else 0.0


def compute_declined_correctly(api_response: dict, should_decline: bool) -> float:
    """1.0 if declined flag matches should_decline, else 0.0."""
    system_declined = bool(api_response.get("declined", False))
    return 1.0 if system_declined == should_decline else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  API helpers
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_api(api_url: str, timeout: int = 120) -> bool:
    """Poll /health until the pipeline is ready or timeout is reached."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("pipeline_ready"):
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def call_query(api_url: str, question: str) -> dict:
    """POST /query and return the response dict. Raises on HTTP error."""
    r = requests.post(
        f"{api_url}/query",
        json={"question": question},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], category_stats: dict, overall: dict) -> None:
    W = 68
    print()
    print("=" * W)
    print("  Healthcare RAG — Evaluation Report")
    print("=" * W)

    # Per-category table
    cats = ["diagnosis", "treatment", "monitoring", "prevention"]
    print(f"  {'Category':<16} {'N':>3}  {'Faithfulness':>13}  {'Citation':>8}  {'Declined?':>9}")
    print("  " + "-" * (W - 2))
    for cat in cats:
        s = category_stats.get(cat)
        if s is None:
            continue
        n = s["count"]
        faith = f"{s['faithfulness']:.3f}"
        cit   = f"{s['citation_present']:.3f}"
        dec   = f"{s['declined_correctly']:.3f}"
        flag  = " OK" if s["faithfulness"] >= FAITHFULNESS_PASS_THRESHOLD else " !!"
        print(f"  {cat:<16} {n:>3}  {faith:>13}{flag}  {cit:>8}  {dec:>9}")

    print("  " + "-" * (W - 2))

    # Overall row
    n_tot  = overall["total"]
    f_avg  = overall["mean_faithfulness"]
    c_avg  = overall["mean_citation_present"]
    d_avg  = overall["mean_declined_correctly"]
    flag   = " PASS" if overall["passed"] else " FAIL"
    print(f"  {'OVERALL':<16} {n_tot:>3}  {f_avg:.3f}{flag}  {c_avg:.3f}   {d_avg:.3f}")
    print("=" * W)

    thresh_str = f"{FAITHFULNESS_PASS_THRESHOLD:.0%}"
    if overall["passed"]:
        print(f"  [PASS] Mean faithfulness {f_avg:.1%} >= {thresh_str} -- build APPROVED")
    else:
        print(f"  [FAIL] Mean faithfulness {f_avg:.1%} <  {thresh_str} -- build REJECTED")
    print("=" * W)
    print()

    # Per-question detail (compact)
    print(f"  {'ID':<6} {'Cat':<12} {'Faith':>6}  {'Cit':>4}  {'Dec?':>5}  {'Declined':>8}  Question (truncated)")
    print("  " + "-" * (W - 2))
    for r in results:
        q_short = r["question"][:38] + "…" if len(r["question"]) > 38 else r["question"]
        dec_flag = "yes" if r["api_declined"] else "no"
        print(
            f"  {r['id']:<6} {r['category']:<12} "
            f"{r['faithfulness']:>6.3f}  {r['citation_present']:>4.1f}  "
            f"{r['declined_correctly']:>5.1f}  {dec_flag:>8}  {q_short}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(dataset_path: str, api_url: str, max_samples: int | None) -> int:
    """Returns 0 on PASS, 1 on FAIL."""

    # ── 1. Load golden dataset ────────────────────────────────────────────────
    with open(dataset_path) as f:
        dataset = json.load(f)

    qa_pairs: list[dict] = dataset["qa_pairs"]
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from {dataset_path}")

    # ── 2. Verify API is ready ────────────────────────────────────────────────
    logger.info(f"Checking API at {api_url} …")
    if not wait_for_api(api_url, timeout=120):
        logger.error("API did not become ready within 120 s — aborting evaluation")
        return 1
    logger.info("API is ready")

    # ── 3. Run each question through the pipeline ─────────────────────────────
    results: list[dict] = []

    for i, pair in enumerate(qa_pairs, 1):
        qid      = pair["id"]
        question = pair["question"]
        expected = pair["expected_answer"]
        category = pair["category"]
        should_decline = pair.get("should_decline", False)

        logger.info(f"[{i}/{len(qa_pairs)}] {qid}: {question[:60]}…")

        try:
            resp = call_query(api_url, question)
            actual_answer = resp.get("answer", "")

            faith = compute_faithfulness(expected, actual_answer)
            cit   = compute_citation_present(resp, should_decline)
            dec   = compute_declined_correctly(resp, should_decline)

        except requests.HTTPError as exc:
            logger.error(f"  HTTP {exc.response.status_code} — scoring as 0")
            actual_answer = ""
            faith, cit, dec = 0.0, 0.0, 0.0
            resp = {"declined": False, "citations": [], "processing_time_ms": 0}

        except Exception as exc:
            logger.error(f"  Error: {exc} — scoring as 0")
            actual_answer = ""
            faith, cit, dec = 0.0, 0.0, 0.0
            resp = {"declined": False, "citations": [], "processing_time_ms": 0}

        results.append({
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "category": category,
            "should_decline": should_decline,
            "source_document": pair.get("source_document", ""),
            "api_declined": bool(resp.get("declined", False)),
            "actual_answer": actual_answer[:500],   # truncate for storage
            "num_citations": len(resp.get("citations", [])),
            "processing_time_ms": resp.get("processing_time_ms", 0),
            "faithfulness": faith,
            "citation_present": cit,
            "declined_correctly": dec,
        })

    # ── 4. Aggregate per category ─────────────────────────────────────────────
    category_stats: dict[str, Any] = {}
    for r in results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {
                "count": 0,
                "faithfulness": 0.0,
                "citation_present": 0.0,
                "declined_correctly": 0.0,
            }
        s = category_stats[cat]
        s["count"] += 1
        s["faithfulness"]       += r["faithfulness"]
        s["citation_present"]   += r["citation_present"]
        s["declined_correctly"] += r["declined_correctly"]

    for cat, s in category_stats.items():
        n = s["count"]
        s["faithfulness"]       = round(s["faithfulness"] / n, 4)
        s["citation_present"]   = round(s["citation_present"] / n, 4)
        s["declined_correctly"] = round(s["declined_correctly"] / n, 4)

    # ── 5. Overall stats ──────────────────────────────────────────────────────
    n = len(results)
    mean_faith = round(sum(r["faithfulness"]       for r in results) / n, 4)
    mean_cit   = round(sum(r["citation_present"]   for r in results) / n, 4)
    mean_dec   = round(sum(r["declined_correctly"] for r in results) / n, 4)
    passed     = mean_faith >= FAITHFULNESS_PASS_THRESHOLD

    overall = {
        "total": n,
        "mean_faithfulness":       mean_faith,
        "mean_citation_present":   mean_cit,
        "mean_declined_correctly": mean_dec,
        "faithfulness_threshold":  FAITHFULNESS_PASS_THRESHOLD,
        "passed": passed,
    }

    # ── 6. Print summary ──────────────────────────────────────────────────────
    print_summary(results, category_stats, overall)

    # ── 7. Save results ───────────────────────────────────────────────────────
    out_dir = Path("data/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_path,
        "samples_evaluated": n,
        "overall": overall,
        "category_stats": category_stats,
        "per_question": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Healthcare RAG pipeline")
    parser.add_argument(
        "--dataset",
        default="data/eval/golden_dataset.json",
        help="Path to golden Q&A JSON (default: data/eval/golden_dataset.json)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the FastAPI backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for quick smoke tests",
    )
    args = parser.parse_args()
    sys.exit(main(args.dataset, args.api_url, args.max_samples))
