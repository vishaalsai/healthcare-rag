"""
Phase 3 – Observability (Phase 2)
MetricsCollector: fetches Langfuse traces and computes aggregated metrics.

Usage:
    from src.observability.metrics import MetricsCollector, MetricsSummary

    collector = MetricsCollector()
    summary   = collector.get_metrics(hours=24)
    traces    = collector.get_recent_traces(limit=20)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.observability.tracer import get_tracer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsSummary:
    """Aggregated observability metrics over a rolling time window."""

    total_requests: int = 0
    success_rate: float = 0.0        # 1 - failure_rate
    failure_rate: float = 0.0        # % returning INSUFFICIENT_CONTEXT
    citation_coverage: float = 0.0   # % of answers with >= 1 citation
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    period_hours: int = 24
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Collector
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Fetches trace data from Langfuse and computes aggregated metrics.
    Gracefully returns safe defaults when Langfuse is not configured.
    """

    def __init__(self) -> None:
        self._tracer = get_tracer()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_metrics(self, hours: int = 24) -> MetricsSummary:
        """
        Compute a MetricsSummary over the last `hours` window.

        Returns a zero-filled MetricsSummary when:
          - Langfuse is not configured (tracer is None)
          - fetch_traces() raises any exception
          - No traces are found in the window
        """
        base = MetricsSummary(period_hours=hours)

        if self._tracer is None:
            return base

        try:
            from_ts = datetime.now(timezone.utc) - timedelta(hours=hours)
            response = self._tracer.fetch_traces(
                from_timestamp=from_ts,
                limit=100,
            )
            traces = response.data
        except Exception as exc:
            logger.warning(f"Failed to fetch traces from Langfuse: {exc}")
            return base

        if not traces:
            return base

        latencies: list[float] = []
        costs: list[float] = []
        input_tok: list[float] = []
        output_tok: list[float] = []
        citation_counts: list[int] = []
        insufficient_count = 0

        for t in traces:
            meta: dict[str, Any] = getattr(t, "metadata", None) or {}

            # Latency — Langfuse returns seconds; convert to ms
            raw_lat = getattr(t, "latency", None)
            if raw_lat is not None:
                latencies.append(float(raw_lat) * 1000.0)

            # Cost — try trace-level attribute first, then metadata fallback
            cost = (
                getattr(t, "total_cost", None)
                or getattr(t, "cost", None)
                or meta.get("total_cost", 0.0)
            )
            costs.append(float(cost or 0.0))

            # Token counts stored in trace metadata by answer_generator
            input_tok.append(float(meta.get("input_tokens", 0)))
            output_tok.append(float(meta.get("output_tokens", 0)))

            # Citation coverage
            citation_counts.append(int(meta.get("citation_count", 0)))

            # Failure detection
            if (
                meta.get("insufficient_context", False)
                or meta.get("declined_reason") == "INSUFFICIENT_CONTEXT"
            ):
                insufficient_count += 1

        n = len(traces)
        failure_rate = round(insufficient_count / n, 4)
        with_citations = sum(1 for c in citation_counts if c > 0)
        citation_coverage = round(with_citations / n, 4)

        lat_arr = np.array(latencies) if latencies else np.array([0.0])
        cost_arr = np.array(costs)

        return MetricsSummary(
            total_requests=n,
            success_rate=round(1.0 - failure_rate, 4),
            failure_rate=failure_rate,
            citation_coverage=citation_coverage,
            avg_latency_ms=round(float(lat_arr.mean()), 1),
            p50_latency_ms=round(float(np.percentile(lat_arr, 50)), 1),
            p95_latency_ms=round(float(np.percentile(lat_arr, 95)), 1),
            avg_cost_usd=round(float(cost_arr.mean()), 6),
            total_cost_usd=round(float(cost_arr.sum()), 6),
            avg_input_tokens=round(float(np.mean(input_tok)), 1),
            avg_output_tokens=round(float(np.mean(output_tok)), 1),
            period_hours=hours,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Return the `limit` most recent traces as plain dicts for display.

        Each dict contains:
            trace_id, query (truncated to 80 chars), latency_ms, cost_usd,
            citation_count, status (success/insufficient_context/declined),
            timestamp (ISO string).
        """
        if self._tracer is None:
            return []

        try:
            response = self._tracer.fetch_traces(limit=limit)
            traces = response.data
        except Exception as exc:
            logger.warning(f"Failed to fetch recent traces: {exc}")
            return []

        result: list[dict[str, Any]] = []
        for t in traces:
            meta: dict[str, Any] = getattr(t, "metadata", None) or {}

            raw_query = str(getattr(t, "input", "") or "")
            query = (raw_query[:80] + "…") if len(raw_query) > 80 else raw_query

            raw_lat = getattr(t, "latency", None)
            latency_ms = round(float(raw_lat) * 1000.0, 1) if raw_lat else 0.0

            cost = (
                getattr(t, "total_cost", None)
                or getattr(t, "cost", None)
                or meta.get("total_cost", 0.0)
            )
            cost_usd = round(float(cost or 0.0), 6)

            if meta.get("insufficient_context", False):
                status = "insufficient_context"
            elif meta.get("declined", False):
                status = "declined"
            else:
                status = "success"

            ts = getattr(t, "timestamp", None)
            timestamp = ts.isoformat() if ts else ""

            result.append({
                "trace_id": getattr(t, "id", ""),
                "query": query,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "citation_count": int(meta.get("citation_count", 0)),
                "status": status,
                "timestamp": timestamp,
            })

        return result
