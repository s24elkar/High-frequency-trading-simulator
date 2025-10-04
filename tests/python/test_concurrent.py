from __future__ import annotations

import threading

from python.backtester import (
    Backtester,
    BacktesterConfig,
    ConcurrentBacktester,
    MetricsLogger,
    StrategyCallbacks,
)
from python.backtester.backtester import MarketEvent, MarketSnapshot, StrategyContext
from python.backtester.order_book import PythonOrderBook


def _snapshot_event(order_id: int, side: str, price: float, size: float) -> MarketEvent:
    return MarketEvent(
        timestamp_ns=order_id,
        event_type="add_order",
        payload={
            "order_id": order_id,
            "symbol": "TEST",
            "side": side,
            "price": price,
            "size": size,
        },
    )


class RecordingStrategy(StrategyCallbacks):
    def __init__(self) -> None:
        self.market_threads: list[str] = []
        self.order_threads: list[str] = []

    def on_market_data(self, snapshot: MarketSnapshot, ctx: StrategyContext) -> None:
        self.market_threads.append(threading.current_thread().name)
        price = snapshot.best_bid or snapshot.best_ask or 100.0
        ctx.submit_order("BUY", price, 1.0)

    def on_order_accepted(self, order, ctx: StrategyContext) -> None:
        self.order_threads.append(threading.current_thread().name)


def _build_backtester(strategy: StrategyCallbacks) -> Backtester:
    metrics = MetricsLogger()
    order_book = PythonOrderBook(depth=5)
    config = BacktesterConfig(symbol="TEST")
    return Backtester(
        config=config,
        limit_book=order_book,
        metrics_logger=metrics,
        risk_engine=None,
        strategy=strategy,
    )


def test_concurrent_backtester_runs_in_parallel() -> None:
    strategy = RecordingStrategy()
    backtester = _build_backtester(strategy)
    runner = ConcurrentBacktester(backtester)

    events = [
        _snapshot_event(i, "BUY" if i % 2 == 0 else "SELL", 100.0 + i * 0.01, 1.0)
        for i in range(1, 6)
    ]

    runner.run(events)

    assert strategy.market_threads
    assert strategy.order_threads
    assert any(name == "StrategyThread" for name in strategy.market_threads)
    assert any(name == "OrderThread" for name in strategy.order_threads)
    assert backtester.metrics_logger.snapshot().order_count == len(strategy.order_threads)


def test_concurrent_backtester_matches_serial_results() -> None:
    events = [
        _snapshot_event(i, "BUY" if i % 2 == 0 else "SELL", 100.0 + i * 0.05, 1.5)
        for i in range(1, 10)
    ]

    serial_strategy = RecordingStrategy()
    serial_backtester = _build_backtester(serial_strategy)
    serial_backtester.run(events)

    parallel_strategy = RecordingStrategy()
    parallel_backtester = _build_backtester(parallel_strategy)
    runner = ConcurrentBacktester(parallel_backtester)
    runner.run(events)

    serial_snapshot = serial_backtester.metrics_logger.snapshot()
    parallel_snapshot = parallel_backtester.metrics_logger.snapshot()

    assert serial_snapshot.order_count == parallel_snapshot.order_count
    assert serial_snapshot.fill_count == parallel_snapshot.fill_count
    assert serial_snapshot.order_volume == parallel_snapshot.order_volume
