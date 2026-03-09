"""
Phase 3 – Observability
Langfuse singleton tracer for end-to-end RAG request tracing.

Usage:
    from src.observability.tracer import get_tracer, create_trace, calculate_cost

    trace = create_trace(name="rag-query", input=question, trace_id="uuid-here")
    # ... add spans to trace ...
    cost = calculate_cost(input_tokens=500, output_tokens=150)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Singleton state ────────────────────────────────────────────────────────────
_langfuse_client = None
_langfuse_initialized = False


def get_tracer():
    """
    Return the singleton Langfuse client.

    Returns None (and logs a warning) when:
      - LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY are missing/empty
      - The langfuse package is not installed
      - Langfuse initialization fails for any other reason

    The RAG pipeline continues normally in all these cases.
    """
    global _langfuse_client, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_client

    _langfuse_initialized = True

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    host = (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com").strip()

    # Treat placeholder values as missing
    if not public_key or not secret_key or public_key.startswith("pk-lf-your"):
        logger.warning(
            "Langfuse keys not configured "
            "(LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Observability tracing is DISABLED. "
            "Set real keys in .env to enable it."
        )
        return None

    try:
        from langfuse import Langfuse  # noqa: PLC0415

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info(f"Langfuse tracer initialised (host={host})")

    except ImportError:
        logger.warning(
            "langfuse package is not installed. "
            "Run: pip install langfuse  — tracing disabled."
        )
    except Exception as exc:
        logger.warning(f"Langfuse initialisation failed: {exc}. Tracing disabled.")

    return _langfuse_client


def create_trace(
    name: str,
    input: Any,  # noqa: A002
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
):
    """
    Create a new Langfuse trace and return it.

    Returns None silently if tracing is disabled or if Langfuse raises.
    """
    tracer = get_tracer()
    if tracer is None:
        return None

    try:
        kwargs: dict[str, Any] = {"name": name, "input": input}
        if metadata:
            kwargs["metadata"] = metadata
        if trace_id:
            kwargs["id"] = trace_id
        return tracer.trace(**kwargs)
    except Exception as exc:
        logger.warning(f"Failed to create Langfuse trace: {exc}")
        return None


def calculate_cost(input_tokens: int, output_tokens: int) -> dict[str, float]:
    """
    Calculate claude-opus-4-6 API cost.

    Pricing:
        Input  tokens: $15.00 / 1M tokens  → $0.000015 per token
        Output tokens: $75.00 / 1M tokens  → $0.000075 per token

    Args:
        input_tokens:  Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.

    Returns:
        {"input_cost": float, "output_cost": float, "total_cost": float}
        All values are in USD.
    """
    input_cost = input_tokens * 0.000015
    output_cost = output_tokens * 0.000075
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
    }
