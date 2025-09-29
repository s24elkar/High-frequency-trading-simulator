"""Structured logging for backtests with JSONL and SQLite sinks."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .backtester import FillEvent, MarketSnapshot, OrderRequest


@dataclass(slots=True)
class LogRecord:
    timestamp_ns: int
    event_type: str
    payload: Dict[str, object]


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

    def log_order(self, order: "OrderRequest") -> None:
        record = LogRecord(order.timestamp_ns, "order", asdict(order))
        self._write(record)

    def log_cancel(self, order_id: int, timestamp_ns: int) -> None:
        record = LogRecord(timestamp_ns, "cancel", {"order_id": order_id})
        self._write(record)

    def log_fill(self, fill: "FillEvent") -> None:
        record = LogRecord(fill.timestamp_ns, "fill", asdict(fill))
        self._write(record)

    def log_snapshot(self, snapshot: "MarketSnapshot") -> None:
        record = LogRecord(snapshot.timestamp_ns, "snapshot", asdict(snapshot))
        self._write(record)

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


__all__ = ["MetricsLogger", "MetricsAggregator", "LogRecord"]
