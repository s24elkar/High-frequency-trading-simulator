"""Utility functions to turn metric logs into human-readable diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .logging import MetricsAggregator


@dataclass(slots=True)
class BacktestSummary:
    fill_ratio: float
    pnl_stats: Dict[str, float]


def summarise(jsonl_path: str) -> BacktestSummary:
    agg = MetricsAggregator.from_jsonl(jsonl_path)
    curve = agg.pnl_curve()
    pnl_stats = {
        "points": len(curve),
        "last_realized": curve[-1]["realized"] if curve else 0.0,
        "last_unrealized": curve[-1]["unrealized"] if curve else 0.0,
    }
    return BacktestSummary(fill_ratio=agg.fill_ratio(), pnl_stats=pnl_stats)


__all__ = ["summarise", "BacktestSummary"]
