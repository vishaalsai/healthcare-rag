"""
Tests for Phase 3 (Phase 2) — MetricsCollector and MetricsSummary.

All Langfuse network calls are mocked; no real credentials required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.observability.metrics import MetricsCollector, MetricsSummary


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_trace(
    trace_id: str = "t-1",
    query: str = "What treats hypertension?",
    latency: float | None = 1.0,
    total_cost: float | None = None,
    metadata: dict | None = None,
    timestamp: datetime | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics a Langfuse Trace object."""
    t = MagicMock()
    t.id = trace_id
    t.input = query
    t.latency = latency
    t.total_cost = total_cost
    t.cost = None
    t.metadata = metadata or {}
    t.timestamp = timestamp or datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return t


def _mock_tracer(traces: list) -> MagicMock:
    """Return a mock Langfuse client whose fetch_traces() returns traces."""
    tracer = MagicMock()
    tracer.fetch_traces.return_value = MagicMock(data=traces)
    return tracer


# ─────────────────────────────────────────────────────────────────────────────
#  MetricsSummary defaults
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_returns_defaults_when_tracer_is_none():
    """MetricsCollector with no Langfuse config returns a zeroed summary."""
    with patch(
        "src.observability.metrics.get_tracer", return_value=None
    ):
        collector = MetricsCollector()
        summary = collector.get_metrics(hours=24)

    assert isinstance(summary, MetricsSummary)
    assert summary.total_requests == 0
    assert summary.success_rate == 0.0
    assert summary.failure_rate == 0.0
    assert summary.avg_cost_usd == 0.0
    assert summary.total_cost_usd == 0.0
    assert summary.period_hours == 24


def test_metrics_returns_defaults_on_empty_traces():
    """Empty trace list returns a zeroed MetricsSummary."""
    tracer = _mock_tracer([])
    with patch(
        "src.observability.metrics.get_tracer", return_value=tracer
    ):
        collector = MetricsCollector()
        summary = collector.get_metrics(hours=12)

    assert summary.total_requests == 0
    assert summary.total_cost_usd == 0.0
    assert summary.period_hours == 12
    assert summary.computed_at  # non-empty ISO timestamp


# ─────────────────────────────────────────────────────────────────────────────
#  Latency percentiles
# ─────────────────────────────────────────────────────────────────────────────

def test_p50_p95_latency_calculation():
    """p50 and p95 match numpy reference values for a known dataset."""
    # latencies in seconds: 0.0, 0.1, 0.2, … 0.9  → ms: 0…900
    traces = [
        _make_trace(trace_id=f"t{i}", latency=i * 0.1)
        for i in range(10)
    ]
    tracer = _mock_tracer(traces)
    with patch(
        "src.observability.metrics.get_tracer", return_value=tracer
    ):
        collector = MetricsCollector()
        summary = collector.get_metrics()

    ms_values = [i * 100.0 for i in range(10)]
    assert summary.p50_latency_ms == pytest.approx(
        np.percentile(ms_values, 50), abs=1.0
    )
    assert summary.p95_latency_ms == pytest.approx(
        np.percentile(ms_values, 95), abs=1.0
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Failure / citation rates
# ─────────────────────────────────────────────────────────────────────────────

def test_insufficient_context_counted_as_failure():
    """Traces flagged insufficient_context raise failure_rate."""
    traces = [
        _make_trace(
            trace_id="a",
            metadata={"insufficient_context": True, "citation_count": 0},
        ),
        _make_trace(
            trace_id="b",
            metadata={"citation_count": 3},
        ),
    ]
    tracer = _mock_tracer(traces)
    with patch(
        "src.observability.metrics.get_tracer", return_value=tracer
    ):
        collector = MetricsCollector()
        summary = collector.get_metrics()

    assert summary.total_requests == 2
    assert summary.failure_rate == pytest.approx(0.5)
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.citation_coverage == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  get_recent_traces
# ─────────────────────────────────────────────────────────────────────────────

def test_get_recent_traces_format():
    """get_recent_traces returns correctly shaped dicts."""
    ts = datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
    trace = _make_trace(
        trace_id="abc-123",
        query="What is first-line treatment for hypertension?",
        latency=2.5,
        total_cost=0.002,
        metadata={"citation_count": 4, "declined": False},
        timestamp=ts,
    )
    tracer = _mock_tracer([trace])
    with patch(
        "src.observability.metrics.get_tracer", return_value=tracer
    ):
        collector = MetricsCollector()
        result = collector.get_recent_traces(limit=5)

    assert len(result) == 1
    row = result[0]
    assert row["trace_id"] == "abc-123"
    assert row["query"] == "What is first-line treatment for hypertension?"
    assert row["latency_ms"] == pytest.approx(2500.0)
    assert row["cost_usd"] == pytest.approx(0.002)
    assert row["citation_count"] == 4
    assert row["status"] == "success"
    assert "2025" in row["timestamp"]


def test_get_recent_traces_truncates_long_query():
    """Queries longer than 80 chars are truncated with an ellipsis."""
    long_q = "A" * 100
    trace = _make_trace(query=long_q)
    tracer = _mock_tracer([trace])
    with patch(
        "src.observability.metrics.get_tracer", return_value=tracer
    ):
        collector = MetricsCollector()
        result = collector.get_recent_traces()

    assert len(result[0]["query"]) == 81  # 80 chars + "…"
    assert result[0]["query"].endswith("…")


def test_get_recent_traces_returns_empty_when_tracer_none():
    """No tracer → get_recent_traces returns an empty list."""
    with patch(
        "src.observability.metrics.get_tracer", return_value=None
    ):
        collector = MetricsCollector()
        result = collector.get_recent_traces()

    assert result == []


# ─────────────────────────────────────────────────────────────────────────────
#  Exception handling
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_handles_fetch_exception_gracefully():
    """fetch_traces() raising an exception returns a zeroed MetricsSummary."""
    tracer = MagicMock()
    tracer.fetch_traces.side_effect = RuntimeError("Langfuse connection error")
    with patch(
        "src.observability.metrics.get_tracer", return_value=tracer
    ):
        collector = MetricsCollector()
        summary = collector.get_metrics()

    assert isinstance(summary, MetricsSummary)
    assert summary.total_requests == 0


def test_recent_traces_handles_fetch_exception_gracefully():
    """fetch_traces() raising an exception returns an empty list."""
    tracer = MagicMock()
    tracer.fetch_traces.side_effect = ConnectionError("timeout")
    with patch(
        "src.observability.metrics.get_tracer", return_value=tracer
    ):
        collector = MetricsCollector()
        result = collector.get_recent_traces()

    assert result == []
