"""
Unit tests for the observability regression gate.

Tests gate_check() as a pure function, and main() via mocked
MetricsCollector + get_tracer so no Langfuse credentials are needed.

Minimum 7 test functions — all MetricsCollector interactions are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so both src.* and scripts.* resolve
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.check_observability import gate_check, main  # noqa: E402
from src.observability.metrics import MetricsSummary  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_THRESHOLDS = {
    "citation_coverage_threshold": 0.70,
    "failure_rate_threshold": 0.20,
    "p95_latency_ms_threshold": 10000,
    "avg_cost_usd_warning": 0.05,
    "min_sample_size": 5,
    "evaluation_window_hours": 24,
}


def _summary(**overrides) -> MetricsSummary:
    """Return a MetricsSummary that passes all hard thresholds by default."""
    defaults = dict(
        total_requests=20,
        success_rate=0.95,
        failure_rate=0.05,
        citation_coverage=0.85,
        avg_latency_ms=2000.0,
        p50_latency_ms=1800.0,
        p95_latency_ms=3500.0,
        avg_cost_usd=0.03,
        total_cost_usd=0.60,
        avg_input_tokens=800.0,
        avg_output_tokens=300.0,
        period_hours=24,
    )
    defaults.update(overrides)
    return MetricsSummary(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
#  gate_check() — pure function tests (no I/O, no mocking required)
# ─────────────────────────────────────────────────────────────────────────────

def test_all_metrics_pass():
    """All hard thresholds met → exit code 0, PASS result."""
    code, report = gate_check(_summary(), _THRESHOLDS)

    assert code == 0
    assert "RESULT: PASS" in report
    assert "[FAIL]" not in report


def test_citation_coverage_below_threshold_fails():
    """Citation coverage 0.55 < 0.70 → exit code 1, FAIL result."""
    code, report = gate_check(_summary(citation_coverage=0.55), _THRESHOLDS)

    assert code == 1
    assert "RESULT: FAIL" in report
    assert "[FAIL]" in report
    assert "Citation Coverage" in report


def test_failure_rate_above_threshold_fails():
    """Failure rate 0.35 > 0.20 → exit code 1, FAIL result."""
    code, report = gate_check(_summary(failure_rate=0.35), _THRESHOLDS)

    assert code == 1
    assert "RESULT: FAIL" in report
    assert "[FAIL]" in report
    assert "Failure Rate" in report


def test_p95_latency_above_threshold_fails():
    """P95 latency 12 500 ms > 10 000 ms → exit code 1, FAIL result."""
    code, report = gate_check(_summary(p95_latency_ms=12500.0), _THRESHOLDS)

    assert code == 1
    assert "RESULT: FAIL" in report
    assert "[FAIL]" in report
    assert "P95 Latency" in report


def test_cost_above_warning_is_soft_pass():
    """Avg cost $0.08 > warning $0.05 → exit code 0, WARN tag, no FAIL."""
    code, report = gate_check(_summary(avg_cost_usd=0.08), _THRESHOLDS)

    assert code == 0
    assert "RESULT: PASS" in report
    assert "[WARN]" in report
    assert "[FAIL]" not in report
    assert "Avg Cost/Request" in report


# ─────────────────────────────────────────────────────────────────────────────
#  main() — integration tests (MetricsCollector fully mocked)
# ─────────────────────────────────────────────────────────────────────────────

def test_insufficient_sample_size_skips_gate_exit_zero(capsys):
    """
    With only 3 traces (< min_sample_size 5), main() prints a warning and
    exits 0 — prevents false failures on fresh deployments with little data.
    """
    mock_collector = MagicMock()
    mock_collector.get_metrics.return_value = _summary(total_requests=3)

    with (
        patch("scripts.check_observability.get_tracer", return_value=MagicMock()),
        patch(
            "scripts.check_observability.MetricsCollector",
            return_value=mock_collector,
        ),
        pytest.raises(SystemExit) as exc,
    ):
        main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    # The warning message must mention the gate was skipped or the minimum
    assert "Gate skipped" in out or "minimum" in out


def test_langfuse_not_configured_skips_gate_exit_zero(capsys):
    """
    When get_tracer() returns None, main() prints a clear message and exits 0.
    Observability unavailability must never block CI.
    """
    with (
        patch("scripts.check_observability.get_tracer", return_value=None),
        pytest.raises(SystemExit) as exc,
    ):
        main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Langfuse not configured" in out
