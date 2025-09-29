"""Core backtester orchestrating ITCH replays, the limit order book, and strategies.

The design favours C++ for the critical order book operations while exposing a
Python API for orchestration, diagnostics, and visualisation. The default
`limit_book` argument is expected to be a thin wrapper around the C++
`OrderBook` (see `python/backtester/order_book.py` for the Python fallback used
in tests)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Protocol

import numpy as np

from .logging import MetricsLogger
from .risk import RiskEngine

log = logging.getLogger(__name__)


class Strategy(Protocol):
    """Strategy interface invoked by the backtester on each market snapshot."""

    def on_tick(self, snapshot: "MarketSnapshot", backtester: "Backtester") -> None: ...


class LimitOrderBook(Protocol):
    """Protocol the concrete (C++-backed) order book wrapper must honour."""

    def enqueue(self, order: "OrderRequest") -> None: ...

    def cancel(self, order_id: int) -> None: ...

    def apply_event(self, event: "MarketEvent") -> Optional["MarketSnapshot"]: ...

    def snapshot(self, depth: int = 1) -> "MarketSnapshot": ...


@dataclass(slots=True)
class BacktesterConfig:
    symbol: str
    book_depth: int = 5
    record_snapshots: bool = True


@dataclass(slots=True)
class OrderRequest:
    order_id: int
    symbol: str
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    timestamp_ns: int
    metadata: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class FillEvent:
    order_id: int
    symbol: str
    side: str
    price: float
    size: float
    timestamp_ns: int
    liquidity_flag: str = "UNKNOWN"  # e.g. MAKER/TAKER


@dataclass(slots=True)
class MarketEvent:
    timestamp_ns: int
    event_type: str
    payload: Dict[str, float | int | str]


@dataclass(slots=True)
class MarketSnapshot:
    timestamp_ns: int
    best_bid: Optional[float]
    bid_size: Optional[float]
    best_ask: Optional[float]
    ask_size: Optional[float]
    last_trade_price: Optional[float]
    last_trade_size: Optional[float]
    imbalance: Optional[float]
    depth: List[Dict[str, float]] = field(default_factory=list)

    @property
    def midprice(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0


class Backtester:
    """Coordinates event replay, strategy decisions, and bookkeeping."""

    def __init__(
        self,
        config: BacktesterConfig,
        limit_book: LimitOrderBook,
        metrics_logger: MetricsLogger,
        risk_engine: Optional[RiskEngine] = None,
        strategy: Optional[Strategy] = None,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.limit_book = limit_book
        self.metrics_logger = metrics_logger
        self.risk_engine = risk_engine
        self.strategy = strategy
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._id_counter = 1
        self.clock_ns = 0
        self.active_orders: Dict[int, OrderRequest] = {}
        self.pending_cancels: set[int] = set()
        self.strategy_halted = False
        self._digest = hashlib.sha256()

    @property
    def digest(self) -> str:
        """Deterministic digest of the event log for regression tests."""

        return self._digest.hexdigest()

    def submit_order(
        self,
        side: str,
        price: float,
        size: float,
        metadata: Optional[Dict[str, float | int | str]] = None,
    ) -> int:
        order_id = self._id_counter
        self._id_counter += 1
        order = OrderRequest(
            order_id=order_id,
            symbol=self.config.symbol,
            side=side.upper(),
            price=price,
            size=size,
            timestamp_ns=self.clock_ns,
            metadata={} if metadata is None else dict(metadata),
        )
        self.active_orders[order_id] = order
        log.debug("Submitting order %s", order)
        self.limit_book.enqueue(order)
        self.metrics_logger.log_order(order)
        self._update_digest("ORDER", order)
        return order_id

    def cancel_order(self, order_id: int) -> None:
        if order_id not in self.active_orders:
            log.debug("Cancel requested for unknown order %s", order_id)
            return
        self.pending_cancels.add(order_id)
        self.limit_book.cancel(order_id)
        self.metrics_logger.log_cancel(order_id, self.clock_ns)
        self._update_digest("CANCEL", {"order_id": order_id})

    def process_fill(self, fill: FillEvent) -> None:
        log.debug("Processing fill %s", fill)
        resting = self.active_orders.get(fill.order_id)
        if resting:
            remaining = resting.size - fill.size
            if remaining <= 0:
                self.active_orders.pop(fill.order_id, None)
            else:
                resting.size = remaining
        if self.risk_engine is not None:
            self.risk_engine.update_on_fill(fill)
            if self.risk_engine.strategy_halted:
                self.strategy_halted = True
        self.metrics_logger.log_fill(fill)
        self._update_digest("FILL", fill)

    def on_tick(self, snapshot: MarketSnapshot) -> None:
        if self.risk_engine is not None:
            self.risk_engine.update_on_tick(snapshot)
            if self.risk_engine.strategy_halted:
                self.strategy_halted = True
        if self.strategy_halted:
            log.debug("Strategy halted; skipping on_tick")
            return
        if self.strategy is None:
            return
        self.strategy.on_tick(snapshot, self)

    def run(self, replay_session: Iterable[MarketEvent]) -> None:
        for event in replay_session:
            self.clock_ns = event.timestamp_ns
            snapshot = self._dispatch_event(event)
            if snapshot is None:
                continue
            if self.config.record_snapshots:
                self.metrics_logger.log_snapshot(snapshot)
            self._update_digest("SNAPSHOT", snapshot)
            self.on_tick(snapshot)

    def _dispatch_event(self, event: MarketEvent) -> Optional[MarketSnapshot]:
        snapshot = self.limit_book.apply_event(event)
        if snapshot is None:
            # Some events (trading status, imbalance) may not yield a new snapshot
            snapshot = self.limit_book.snapshot(self.config.book_depth)
        return snapshot

    def _update_digest(self, tag: str, payload: object) -> None:
        self._digest.update(tag.encode("ascii"))
        self._digest.update(repr(payload).encode("ascii"))


__all__ = [
    "Backtester",
    "BacktesterConfig",
    "OrderRequest",
    "FillEvent",
    "MarketEvent",
    "MarketSnapshot",
]
