"""Utility functions to turn metric logs into human-readable diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .logging import MetricsAggregator, RunSummary


@dataclass(slots=True)
class BacktestSummary:
    fill_ratio: float
    pnl_stats: Dict[str, float]


@dataclass(slots=True)
class SnapshotPoint:
    timestamp_ns: int
    mid: Optional[float]
    imbalance: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]


@dataclass(slots=True)
class OrderEventRecord:
    timestamp_ns: int
    side: Optional[str]
    size: Optional[float]
    latency_ns: Optional[int]


@dataclass(slots=True)
class FillEventRecord:
    timestamp_ns: int
    side: Optional[str]
    size: Optional[float]
    price: Optional[float]


@dataclass(slots=True)
class BacktestRun:
    snapshots: List[SnapshotPoint]
    orders: List[OrderEventRecord]
    fills: List[FillEventRecord]
    summary: Optional[RunSummary]


def summarise(jsonl_path: str) -> BacktestSummary:
    agg = MetricsAggregator.from_jsonl(jsonl_path)
    curve = agg.pnl_curve()
    pnl_stats = {
        "points": len(curve),
        "last_realized": curve[-1]["realized"] if curve else 0.0,
        "last_unrealized": curve[-1]["unrealized"] if curve else 0.0,
    }
    return BacktestSummary(fill_ratio=agg.fill_ratio(), pnl_stats=pnl_stats)


def load_run(jsonl_path: str | Path) -> BacktestRun:
    path = Path(jsonl_path)
    snapshots: List[SnapshotPoint] = []
    orders: List[OrderEventRecord] = []
    fills: List[FillEventRecord] = []
    summary: Optional[RunSummary] = None

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            blob = json.loads(line)
            event_type = blob.get("event_type")
            payload = blob.get("payload", {})
            timestamp_ns = int(blob.get("timestamp_ns", 0))
            if event_type == "snapshot":
                best_bid = payload.get("best_bid")
                best_ask = payload.get("best_ask")
                mid = None
                if best_bid is not None and best_ask is not None:
                    mid = (best_bid + best_ask) / 2.0
                snapshots.append(
                    SnapshotPoint(
                        timestamp_ns=timestamp_ns,
                        mid=mid,
                        imbalance=payload.get("imbalance"),
                        best_bid=best_bid,
                        best_ask=best_ask,
                    )
                )
            elif event_type == "order":
                orders.append(
                    OrderEventRecord(
                        timestamp_ns=timestamp_ns,
                        side=payload.get("side"),
                        size=payload.get("size"),
                        latency_ns=(
                            int(payload["latency_ns"])
                            if "latency_ns" in payload
                            and payload["latency_ns"] is not None
                            else None
                        ),
                    )
                )
            elif event_type == "fill":
                fills.append(
                    FillEventRecord(
                        timestamp_ns=timestamp_ns,
                        side=payload.get("side"),
                        size=payload.get("size"),
                        price=payload.get("price"),
                    )
                )
            elif event_type == "run_summary":
                summary = RunSummary(**payload)

    return BacktestRun(snapshots=snapshots, orders=orders, fills=fills, summary=summary)


__all__ = [
    "summarise",
    "BacktestSummary",
    "BacktestRun",
    "SnapshotPoint",
    "OrderEventRecord",
    "FillEventRecord",
    "load_run",
]
