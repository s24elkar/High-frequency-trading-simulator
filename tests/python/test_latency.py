from __future__ import annotations

from dataclasses import dataclass
from typing import List

from python.backtester import (
    Backtester,
    BacktesterConfig,
    MetricsLogger,
    StrategyCallbacks,
)
from python.backtester.backtester import MarketEvent, MarketSnapshot, StrategyContext
from python.backtester.order_book import PythonOrderBook


def _event(ts: int) -> MarketEvent:
    return MarketEvent(
        timestamp_ns=ts,
        event_type="trade",
        payload={"symbol": "TEST", "side": "BUY", "price": 100.0, "size": 1.0},
    )


def _snapshot_from_event(ts: int) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp_ns=ts,
        best_bid=100.0,
        bid_size=1.0,
        best_ask=100.2,
        ask_size=1.0,
        last_trade_price=100.1,
        last_trade_size=1.0,
        imbalance=0.0,
        depth=[],
    )


class FixedSnapshotBook(PythonOrderBook):
    def __init__(self, snapshot: MarketSnapshot) -> None:
        super().__init__(depth=1)
        self._snapshot = snapshot

    def apply_event(self, event: MarketEvent):  # type: ignore[override]
        return super().apply_event(event)

    def snapshot(self, depth: int = 1) -> MarketSnapshot:  # type: ignore[override]
        return self._snapshot


class ImmediateStrategy(StrategyCallbacks):
    def on_market_data(self, snapshot: MarketSnapshot, ctx: StrategyContext) -> None:
        ctx.submit_order("BUY", 100.0, 1.0)


class DoubleOrderStrategy(StrategyCallbacks):
    def on_market_data(self, snapshot: MarketSnapshot, ctx: StrategyContext) -> None:
        ctx.submit_order("BUY", 100.0, 1.0)
        ctx.submit_order("SELL", 100.2, 1.0)


@dataclass
class FakeClock:
    ticks: List[int]

    def __call__(self) -> int:
        if not self.ticks:
            raise RuntimeError("FakeClock exhausted")
        return self.ticks.pop(0)


def _run_backtester(strategy: StrategyCallbacks, clock: FakeClock) -> MetricsLogger:
    metrics = MetricsLogger()
    snapshot = _snapshot_from_event(10)
    book = FixedSnapshotBook(snapshot)
    backtester = Backtester(
        config=BacktesterConfig(symbol="TEST"),
        limit_book=book,
        metrics_logger=metrics,
        strategy=strategy,
        seed=0,
        time_source=clock,
    )
    event = _event(10)
    backtester.run([event])
    return metrics


def test_latency_measurement_uses_time_source() -> None:
    clock = FakeClock([0, 1_500, 3_000])
    metrics = _run_backtester(ImmediateStrategy(), clock)
    latency = metrics.snapshot().latency_breakdown
    assert latency.last_market_to_decision_us == 1.5
    assert latency.last_decision_to_submit_us == 1.5
    assert latency.last_market_to_submit_us == 3.0
    assert latency.avg_market_to_decision_us == 1.5
    assert latency.avg_market_to_submit_us == 3.0


def test_latency_averages_multiple_orders() -> None:
    clock = FakeClock([0, 1_000, 3_000, 4_000, 5_000])
    metrics = _run_backtester(DoubleOrderStrategy(), clock)
    latency = metrics.snapshot().latency_breakdown
    assert latency.last_market_to_decision_us == 4.0
    assert latency.last_decision_to_submit_us == 1.0
    assert latency.last_market_to_submit_us == 5.0
    assert latency.avg_market_to_decision_us == ((1_000 + 4_000) / 2) / 1_000.0
    assert latency.avg_market_to_submit_us == ((3_000 + 5_000) / 2) / 1_000.0
