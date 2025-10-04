"""Structured logging for backtests with JSONL and SQLite sinks."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .backtester import FillEvent, MarketSnapshot, OrderRequest


@dataclass(slots=True)
class RunSummary:
    symbol: str
    realized_pnl: float
    unrealized_pnl: float
    inventory: float
    order_count: int
    fill_count: int
    order_volume: float
    fill_volume: float
    order_to_trade_ratio: Optional[float]
    fill_efficiency: Optional[float]
    avg_latency_ns: Optional[float]
    p95_latency_ns: Optional[int]
    max_latency_ns: Optional[int]
    start_timestamp_ns: Optional[int]
    end_timestamp_ns: Optional[int]
    duration_ns: Optional[int]
    digest: Optional[str] = None


@dataclass(slots=True)
class LogRecord:
    timestamp_ns: int
    event_type: str
    payload: Dict[str, object]


@dataclass(slots=True)
class MetricsSnapshot:
    order_count: int
    fill_count: int
    order_volume: float
    fill_volume: float
    avg_latency_ns: Optional[float]
    p95_latency_ns: Optional[int]
    max_latency_ns: Optional[int]


class MetricsLogger:
    def __init__(
        self,
        json_path: Optional[str | Path] = None,
        sqlite_path: Optional[str | Path] = None,
    ) -> None:
        self.json_path = Path(json_path) if json_path else None
        self.sqlite_path = Path(sqlite_path) if sqlite_path else None
        self._json_handle = None
        if self.json_path:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            self._json_handle = self.json_path.open("w", encoding="utf-8")
        self._conn = None
        if self.sqlite_path:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.sqlite_path)
            self._initialise_sqlite()
        self._order_count = 0
        self._fill_count = 0
        self._order_volume = 0.0
        self._fill_volume = 0.0
        self._latencies: List[int] = []
        self._run_start_ns: Optional[int] = None
        self._run_end_ns: Optional[int] = None
        self._summary_logged = False

    def _initialise_sqlite(self) -> None:
        assert self._conn is not None
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp_ns INTEGER,
                event_type TEXT,
                payload TEXT
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._json_handle:
            self._json_handle.close()
            self._json_handle = None
        if self._conn:
            self._conn.close()
            self._conn = None

    def log_order(self, order: "OrderRequest", latency_ns: Optional[int] = None) -> None:
        payload = asdict(order)
        if latency_ns is not None:
            payload["latency_ns"] = latency_ns
            self._latencies.append(int(latency_ns))
        record = LogRecord(order.timestamp_ns, "order", payload)
        self._write(record)
        self._order_count += 1
        self._order_volume += order.size
        self._mark_run_boundary(order.timestamp_ns)

    def log_cancel(self, order_id: int, timestamp_ns: int) -> None:
        record = LogRecord(timestamp_ns, "cancel", {"order_id": order_id})
        self._write(record)
        self._mark_run_boundary(timestamp_ns)

    def log_fill(self, fill: "FillEvent") -> None:
        record = LogRecord(fill.timestamp_ns, "fill", asdict(fill))
        self._write(record)
        self._fill_count += 1
        self._fill_volume += fill.size
        self._mark_run_boundary(fill.timestamp_ns)

    def log_snapshot(self, snapshot: "MarketSnapshot") -> None:
        record = LogRecord(snapshot.timestamp_ns, "snapshot", asdict(snapshot))
        self._write(record)
        self._mark_run_boundary(snapshot.timestamp_ns)

    def log_run_summary(
        self,
        *,
        symbol: str,
        realized_pnl: float,
        unrealized_pnl: float,
        inventory: float,
        digest: Optional[str] = None,
    ) -> RunSummary:
        summary = self._build_summary(
            symbol=symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            inventory=inventory,
            digest=digest,
        )
        record = LogRecord(
            summary.end_timestamp_ns or 0,
            "run_summary",
            asdict(summary),
        )
        self._write(record)
        self._summary_logged = True
        return summary

    def _write(self, record: LogRecord) -> None:
        payload = {
            "timestamp_ns": record.timestamp_ns,
            "event_type": record.event_type,
            "payload": record.payload,
        }
        if self._json_handle:
            self._json_handle.write(json.dumps(payload) + "\n")
            self._json_handle.flush()
        if self._conn:
            self._conn.execute(
                "INSERT INTO metrics(timestamp_ns, event_type, payload) VALUES(?, ?, ?)",
                (record.timestamp_ns, record.event_type, json.dumps(record.payload)),
            )
            self._conn.commit()

    def snapshot(self) -> MetricsSnapshot:
        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else None
        )
        p95_latency = None
        max_latency = None
        if self._latencies:
            sorted_lat = sorted(self._latencies)
            index = int(round(0.95 * (len(sorted_lat) - 1)))
            p95_latency = sorted_lat[index]
            max_latency = sorted_lat[-1]
        return MetricsSnapshot(
            order_count=self._order_count,
            fill_count=self._fill_count,
            order_volume=self._order_volume,
            fill_volume=self._fill_volume,
            avg_latency_ns=avg_latency,
            p95_latency_ns=p95_latency,
            max_latency_ns=max_latency,
        )

    def _build_summary(
        self,
        *,
        symbol: str,
        realized_pnl: float,
        unrealized_pnl: float,
        inventory: float,
        digest: Optional[str],
    ) -> RunSummary:
        ratio = (
            self._order_count / self._fill_count
            if self._fill_count > 0
            else None
        )
        fill_eff = (
            self._fill_volume / self._order_volume
            if self._order_volume > 0
            else None
        )
        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else None
        )
        p95_latency = None
        max_latency = None
        if self._latencies:
            sorted_lat = sorted(self._latencies)
            index = int(round(0.95 * (len(sorted_lat) - 1)))
            p95_latency = sorted_lat[index]
            max_latency = sorted_lat[-1]
        duration = None
        if self._run_start_ns is not None and self._run_end_ns is not None:
            duration = self._run_end_ns - self._run_start_ns
        return RunSummary(
            symbol=symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            inventory=inventory,
            order_count=self._order_count,
            fill_count=self._fill_count,
            order_volume=self._order_volume,
            fill_volume=self._fill_volume,
            order_to_trade_ratio=ratio,
            fill_efficiency=fill_eff,
            avg_latency_ns=avg_latency,
            p95_latency_ns=p95_latency,
            max_latency_ns=max_latency,
            start_timestamp_ns=self._run_start_ns,
            end_timestamp_ns=self._run_end_ns,
            duration_ns=duration,
            digest=digest,
        )

    def _mark_run_boundary(self, timestamp_ns: int) -> None:
        if self._run_start_ns is None or timestamp_ns < self._run_start_ns:
            self._run_start_ns = timestamp_ns
        if self._run_end_ns is None or timestamp_ns > self._run_end_ns:
            self._run_end_ns = timestamp_ns

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MetricsAggregator:
    def __init__(self, records: Iterable[LogRecord] | None = None) -> None:
        self.records: List[LogRecord] = list(records or [])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "MetricsAggregator":
        records = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                blob = json.loads(line)
                records.append(
                    LogRecord(
                        timestamp_ns=blob["timestamp_ns"],
                        event_type=blob["event_type"],
                        payload=blob["payload"],
                    )
                )
        return cls(records)

    def fill_ratio(self) -> float:
        orders = sum(1 for r in self.records if r.event_type == "order")
        fills = sum(1 for r in self.records if r.event_type == "fill")
        if orders == 0:
            return 0.0
        return fills / orders

    def pnl_curve(self) -> List[Dict[str, float]]:
        realised = 0.0
        unrealised = 0.0
        curve = []
        for record in self.records:
            if record.event_type == "fill":
                realised = float(record.payload.get("liquidity_flag", 0))  # placeholder
            elif record.event_type == "snapshot":
                unrealised = float(
                    record.payload.get("imbalance", 0)
                )  # placeholder until pnl stored
            curve.append(
                {
                    "timestamp_ns": record.timestamp_ns,
                    "realized": realised,
                    "unrealized": unrealised,
                }
            )
        return curve


__all__ = [
    "MetricsLogger",
    "MetricsAggregator",
    "LogRecord",
    "RunSummary",
    "MetricsSnapshot",
]
