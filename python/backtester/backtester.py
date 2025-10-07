"""Core backtester orchestrating ITCH replays, the limit order book, and strategies.

The design favours C++ for the critical order book operations while exposing a
Python API for orchestration, diagnostics, and visualisation. The default
`limit_book` argument is expected to be a thin wrapper around the C++
`OrderBook` (see `python/backtester/order_book.py` for the Python fallback used
in tests)."""

from __future__ import annotations

import hashlib
import time
import logging
from dataclasses import dataclass, field
import heapq
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    TYPE_CHECKING,
)

import numpy as np

from .logging import MetricsLogger
from .risk import RiskEngine
from .strategy import StrategyCallbacks, StrategySandbox, StrategyError, TimerToken

if TYPE_CHECKING:  # pragma: no cover
    from .dashboard import RiskDashboard

log = logging.getLogger(__name__)


class LimitOrderBook(Protocol):
    """Protocol the concrete (C++-backed) order book wrapper must honour."""

    def enqueue(self, order: "OrderRequest") -> None: ...

    def cancel(self, order_id: int) -> None: ...

    def apply_event(self, event: "MarketEvent") -> "OrderBookUpdate": ...

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
    order_type: str = "LIMIT"
    display_size: Optional[float] = None
    stop_price: Optional[float] = None
    peg_reference: Optional[str] = None
    peg_offset: float = 0.0
    total_size: Optional[float] = None
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


@dataclass(slots=True)
class OrderBookUpdate:
    snapshot: Optional[MarketSnapshot]
    fills: List[FillEvent] = field(default_factory=list)
    latency_ns: Optional[int] = None


Strategy = StrategyCallbacks


class StrategyContext:
    """Narrow context object exposed to strategies during backtests."""

    def __init__(self, backtester: "Backtester") -> None:
        self._backtester = backtester

    @property
    def clock_ns(self) -> int:
        return self._backtester.clock_ns

    @property
    def config(self) -> BacktesterConfig:
        return self._backtester.config

    @property
    def risk_engine(self) -> Optional[RiskEngine]:
        return self._backtester.risk_engine

    def submit_order(
        self,
        side: str,
        price: float,
        size: float,
        *,
        order_type: str = "LIMIT",
        display_size: Optional[float] = None,
        stop_price: Optional[float] = None,
        peg_reference: Optional[str] = None,
        peg_offset: float = 0.0,
        metadata: Optional[Dict[str, float | int | str]] = None,
    ) -> int:
        return self._backtester.submit_order(
            side,
            price,
            size,
            order_type=order_type,
            display_size=display_size,
            stop_price=stop_price,
            peg_reference=peg_reference,
            peg_offset=peg_offset,
            metadata=metadata,
        )

    def cancel_order(self, order_id: int) -> None:
        self._backtester.cancel_order(order_id)

    def active_orders(self) -> Dict[int, OrderRequest]:
        return dict(self._backtester.active_orders)

    def schedule_timer(
        self,
        key: str,
        delay_ns: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TimerToken:
        return self._backtester.schedule_timer(key, delay_ns, metadata)

    def cancel_timer(self, token: TimerToken) -> bool:
        return self._backtester.cancel_timer(token)


class Backtester:
    """Coordinates event replay, strategy decisions, and bookkeeping."""

    def __init__(
        self,
        config: BacktesterConfig,
        limit_book: LimitOrderBook,
        metrics_logger: MetricsLogger,
        risk_engine: Optional[RiskEngine] = None,
        strategy: Optional[Strategy] = None,
        dashboard: Optional["RiskDashboard"] = None,
        seed: int = 0,
        time_source: Optional[Callable[[], int]] = None,
    ) -> None:
        self.config = config
        self.limit_book = limit_book
        self.metrics_logger = metrics_logger
        self.risk_engine = risk_engine
        self.strategy = strategy
        self.strategy_error: Optional[str] = None
        self.dashboard = dashboard
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._id_counter = 1
        self.clock_ns = 0
        self.active_orders: Dict[int, OrderRequest] = {}
        self.pending_cancels: set[int] = set()
        self.strategy_halted = False
        self._digest = hashlib.sha256()
        self._context = StrategyContext(self)
        self._last_snapshot_ns: Optional[int] = None
        self._sandbox = StrategySandbox(strategy) if strategy is not None else None
        self._next_timer_id = 1
        self._timer_heap: List[tuple[int, int, TimerToken]] = []
        self._active_timers: Dict[int, TimerToken] = {}
        self._time_source = time_source or time.perf_counter_ns
        self._rt_last_market_ns: Optional[int] = None
        self._rt_last_decision_ns: Optional[int] = None

        if self.dashboard is not None:
            self.dashboard.bind(
                symbol=self.config.symbol,
                risk_engine=self.risk_engine,
                metrics_logger=self.metrics_logger,
            )

    @property
    def digest(self) -> str:
        """Deterministic digest of the event log for regression tests."""

        return self._digest.hexdigest()

    def submit_order(
        self,
        side: str,
        price: float,
        size: float,
        *,
        order_type: str = "LIMIT",
        display_size: Optional[float] = None,
        stop_price: Optional[float] = None,
        peg_reference: Optional[str] = None,
        peg_offset: float = 0.0,
        metadata: Optional[Dict[str, float | int | str]] = None,
    ) -> int:
        decision_perf = self._time_source()
        self._rt_last_decision_ns = decision_perf
        order_id = self._id_counter
        self._id_counter += 1
        order = OrderRequest(
            order_id=order_id,
            symbol=self.config.symbol,
            side=side.upper(),
            price=price,
            size=size,
            timestamp_ns=self.clock_ns,
            order_type=order_type.upper(),
            display_size=display_size,
            stop_price=stop_price,
            peg_reference=peg_reference.upper() if peg_reference else None,
            peg_offset=peg_offset,
            total_size=size,
            metadata={} if metadata is None else dict(metadata),
        )
        decision_ns = (
            self._last_snapshot_ns
            if self._last_snapshot_ns is not None
            else self.clock_ns
        )
        order.metadata.setdefault("source", "strategy")
        order.metadata.setdefault("decision_ns", decision_ns)
        self.active_orders[order_id] = order
        log.debug("Submitting order %s", order)
        self.limit_book.enqueue(order)
        submit_perf = self._time_source()
        latency_ns = max(order.timestamp_ns - decision_ns, 0)
        self.metrics_logger.log_order(order, latency_ns=latency_ns)
        if self._rt_last_market_ns is not None:
            market_to_decision = max(decision_perf - self._rt_last_market_ns, 0)
            decision_to_submit = max(submit_perf - decision_perf, 0)
            self.metrics_logger.record_latency(market_to_decision, decision_to_submit)
        self._update_digest("ORDER", order)
        self._strategy_call("on_order_accepted", order, self._context)
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
        start_ns = time.perf_counter_ns()
        try:
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
            self._strategy_call("on_fill", fill, self._context)
            self._render_dashboard()
        finally:
            duration = time.perf_counter_ns() - start_ns
            self.metrics_logger.record_timing("pnl_logging", duration)

    def on_market_data(self, snapshot: MarketSnapshot) -> None:
        self._last_snapshot_ns = snapshot.timestamp_ns
        self._rt_last_market_ns = self._time_source()
        self._rt_last_decision_ns = None
        if self.risk_engine is not None:
            self.risk_engine.update_on_tick(snapshot)
            if self.risk_engine.strategy_halted:
                self.strategy_halted = True
        if self.strategy_halted:
            log.debug("Strategy halted; skipping market-data callback")
            self._render_dashboard()
            return
        if self.strategy is None:
            self._render_dashboard()
            self._fire_due_timers(self.clock_ns)
            return
        self._strategy_call("on_market_data", snapshot, self._context)
        self._render_dashboard()
        self._fire_due_timers(self.clock_ns)

    def start_strategy(self) -> None:
        if self.strategy is not None:
            self._strategy_call("on_start", self._context)

    def process_market_event(self, event: MarketEvent) -> None:
        logger = self.metrics_logger
        start_ns = time.perf_counter_ns()
        try:
            self.clock_ns = event.timestamp_ns
            self._fire_due_timers(self.clock_ns)
            update = self._dispatch_event(event)
            for fill in update.fills:
                self.process_fill(fill)
                self._fire_due_timers(self.clock_ns)
            snapshot = update.snapshot
            if snapshot is None:
                self._fire_due_timers(self.clock_ns)
                return
            if self.config.record_snapshots:
                self.metrics_logger.log_snapshot(snapshot)
            self._update_digest("SNAPSHOT", snapshot)
            self.on_market_data(snapshot)
        finally:
            duration = time.perf_counter_ns() - start_ns
            if logger is not None:
                logger.record_timing("message_handling", duration)

    def finalise_run(self) -> None:
        if self.strategy is not None:
            self._strategy_call("on_stop", self._context)
        realized = 0.0
        unrealized = 0.0
        inventory = 0.0
        if self.risk_engine is not None:
            symbol = self.config.symbol
            realized = float(self.risk_engine.realized_pnl.get(symbol, 0.0))
            unrealized = float(self.risk_engine.unrealized_pnl.get(symbol, 0.0))
            inventory = float(self.risk_engine.inventory.get(symbol, 0.0))
        self.metrics_logger.log_run_summary(
            symbol=self.config.symbol,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            inventory=inventory,
            digest=self.digest,
        )
        self._render_dashboard()
        self._fire_due_timers(self.clock_ns)

    def run(self, replay_session: Iterable[MarketEvent]) -> None:
        self.start_strategy()
        for event in replay_session:
            self.process_market_event(event)
        self.finalise_run()

    def _dispatch_event(self, event: MarketEvent) -> OrderBookUpdate:
        match_start = time.perf_counter_ns()
        update = self.limit_book.apply_event(event)
        match_duration = time.perf_counter_ns() - match_start
        if self.metrics_logger is not None:
            self.metrics_logger.record_timing("matching", match_duration)
        if update.snapshot is None:
            # Some events (trading status, imbalance) may not yield a new snapshot
            snapshot_start = time.perf_counter_ns()
            update.snapshot = self.limit_book.snapshot(self.config.book_depth)
            snapshot_duration = time.perf_counter_ns() - snapshot_start
            if self.metrics_logger is not None:
                self.metrics_logger.record_timing("book_snapshot", snapshot_duration)
        return update

    def _update_digest(self, tag: str, payload: object) -> None:
        self._digest.update(tag.encode("ascii"))
        self._digest.update(repr(payload).encode("ascii"))

    def _render_dashboard(self) -> None:
        if self.dashboard is None:
            return
        self.dashboard.update(self.clock_ns)

    def _strategy_call(self, method: str, *args) -> None:
        if self._sandbox is None:
            return
        try:
            self._sandbox.invoke(method, *args)
        except StrategyError as exc:
            self.strategy_halted = True
            self.strategy_error = str(exc)
            log.error("Strategy error triggered halt: %s", exc)

    def schedule_timer(
        self,
        key: str,
        delay_ns: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TimerToken:
        if delay_ns < 0:
            raise ValueError("delay_ns must be non-negative")
        due_ns = self.clock_ns + delay_ns
        token = TimerToken(
            timer_id=self._next_timer_id,
            key=key,
            due_ns=due_ns,
            metadata=None if metadata is None else dict(metadata),
        )
        self._next_timer_id += 1
        heapq.heappush(self._timer_heap, (token.due_ns, token.timer_id, token))
        self._active_timers[token.timer_id] = token
        return token

    def cancel_timer(self, token: TimerToken) -> bool:
        return self._active_timers.pop(token.timer_id, None) is not None

    def _fire_due_timers(self, current_ns: int) -> None:
        if self._sandbox is None or self.strategy_halted:
            return
        while self._timer_heap and self._timer_heap[0][0] <= current_ns:
            _, _, token = heapq.heappop(self._timer_heap)
            active = self._active_timers.pop(token.timer_id, None)
            if active is None:
                continue
            self._strategy_call("on_timer", token, self._context)
            self._render_dashboard()
            if self.strategy_halted:
                break


__all__ = [
    "Backtester",
    "BacktesterConfig",
    "OrderRequest",
    "FillEvent",
    "MarketEvent",
    "MarketSnapshot",
    "OrderBookUpdate",
    "StrategyCallbacks",
    "StrategyContext",
    "TimerToken",
]
