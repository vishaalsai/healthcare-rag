#!/usr/bin/env python
"""
Observability Regression Gate
==============================
Loads metric thresholds from config/observability_thresholds.yaml,
fetches current metrics from Langfuse, and checks each against its
threshold.

Exit codes:
  0 — all hard thresholds pass  (warnings are printed but do not fail)
  0 — Langfuse not configured   (never block CI on missing observability)
  0 — insufficient trace data   (< min_sample_size traces found)
  1 — one or more hard thresholds breached

Usage:
    python scripts/check_observability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# ── project root on sys.path so src.* imports work when run directly ─────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.observability.metrics import (  # noqa: E402
    MetricsCollector,
    MetricsSummary,
)
from src.observability.tracer import get_tracer  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_thresholds(config_path: Path | None = None) -> dict:
    """Load the observability threshold block from YAML config."""
    if config_path is None:
        config_path = _ROOT / "config" / "observability_thresholds.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["observability"]


# ─────────────────────────────────────────────────────────────────────────────
#  Core gate logic  (pure function — easy to unit-test)
# ─────────────────────────────────────────────────────────────────────────────

def gate_check(
    summary: MetricsSummary,
    thresholds: dict,
) -> tuple[int, str]:
    """
    Compare MetricsSummary against thresholds.

    Returns:
        (exit_code, report_text)
        exit_code 0 = all hard checks passed
        exit_code 1 = at least one hard check failed
    """
    hours = thresholds.get("evaluation_window_hours", 24)

    lines: list[str] = [
        "=" * 60,
        "OBSERVABILITY REGRESSION CHECK",
        "=" * 60,
        (
            f"Period: last {hours} hours"
            f" | Total requests: {summary.total_requests}"
        ),
        "",
    ]

    # ── define checks ─────────────────────────────────────────────────────────
    # Each entry: (label, value, threshold, operator, kind)
    # operator: ">="  → pass when value >= threshold
    #           "<="  → pass when value <= threshold
    # kind: "hard" → exit 1 on fail  |  "soft" → warning only
    cc_t = thresholds.get("citation_coverage_threshold", 0.70)
    fr_t = thresholds.get("failure_rate_threshold", 0.20)
    lat_t = thresholds.get("p95_latency_ms_threshold", 10000)
    cost_t = thresholds.get("avg_cost_usd_warning", 0.05)

    checks = [
        ("Citation Coverage", summary.citation_coverage, cc_t, ">=", "hard"),
        ("Failure Rate", summary.failure_rate, fr_t, "<=", "hard"),
        ("P95 Latency", summary.p95_latency_ms, lat_t, "<=", "hard"),
        ("Avg Cost/Request", summary.avg_cost_usd, cost_t, "<=", "soft"),
    ]

    hard_passes = 0
    hard_fails = 0
    warnings = 0

    for label, value, threshold, op, kind in checks:
        passed = (value >= threshold) if op == ">=" else (value <= threshold)

        if kind == "hard":
            if passed:
                tag, hard_passes = "[PASS]", hard_passes + 1
            else:
                tag, hard_fails = "[FAIL]", hard_fails + 1
        else:
            if passed:
                tag = "[OK  ]"
            else:
                tag, warnings = "[WARN]", warnings + 1

        # Format the detail string
        if label == "Citation Coverage":
            detail = f"{value:.2f}  (threshold: {op} {threshold:.2f})"
        elif label == "Failure Rate":
            detail = f"{value:.2f}  (threshold: {op} {threshold:.2f})"
        elif label == "P95 Latency":
            detail = (
                f"{value:.0f}ms"
                f"  (threshold: {op} {threshold:.0f}ms)"
            )
        else:
            detail = f"${value:.3f}  (warning at: ${threshold:.3f})"

        lines.append(f"{tag} {label:<22} {detail}")

    warn_s = "s" if warnings != 1 else ""
    result = "PASS" if hard_fails == 0 else "FAIL"
    lines += [
        "",
        "=" * 60,
        (
            f"RESULT: {result}"
            f" ({hard_passes} checks passed,"
            f" {hard_fails} failed,"
            f" {warnings} warning{warn_s})"
        ),
        "=" * 60,
    ]

    exit_code = 1 if hard_fails > 0 else 0
    return exit_code, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    thresholds = load_thresholds()

    # Never block CI just because observability is not configured
    tracer = get_tracer()
    if tracer is None:
        print(
            "[WARN] Langfuse not configured -- "
            "observability gate skipped (exit 0).\n"
            "       Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env"
            " to enable."
        )
        sys.exit(0)

    hours = thresholds.get("evaluation_window_hours", 24)
    collector = MetricsCollector()
    summary = collector.get_metrics(hours=hours)

    # Skip gate when there is not enough data to make a reliable decision
    min_n = thresholds.get("min_sample_size", 5)
    if summary.total_requests < min_n:
        print(
            f"[WARN] Only {summary.total_requests} trace(s) found "
            f"in the last {hours}h (minimum: {min_n}).\n"
            "       Gate skipped -- re-run after more traffic (exit 0)."
        )
        sys.exit(0)

    exit_code, report = gate_check(summary, thresholds)
    print(report)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
