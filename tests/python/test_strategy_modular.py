from __future__ import annotations

from dataclasses import dataclass

from python.backtester import (
    Backtester,
    BacktesterConfig,
    MetricsLogger,
    StrategyCallbacks,
    TimerToken,
)
from python.backtester.backtester import MarketEvent, MarketSnapshot, StrategyContext
from python.backtester.order_book import PythonOrderBook


@dataclass
class EchoStrategy(StrategyCallbacks):
    timer_triggered: bool = False

    def on_start(self, ctx: StrategyContext) -> None:
        ctx.schedule_timer("heartbeat", delay_ns=5)

    def on_timer(self, timer: TimerToken, ctx: StrategyContext) -> None:
        if timer.key == "heartbeat":
            self.timer_triggered = True


class FaultyStrategy(StrategyCallbacks):
    def on_market_data(self, snapshot: MarketSnapshot, ctx: StrategyContext) -> None:
        raise RuntimeError("boom")


def test_strategy_timer_invocation() -> None:
    strategy = EchoStrategy()
    logger = MetricsLogger()
    order_book = PythonOrderBook(depth=1)

    # Minimal replay with two events to ensure timer fires before processing snapshots.
    replay = [
        MarketEvent(
            timestamp_ns=10,
            event_type="trade",
            payload={"side": "BUY", "price": 100.0, "size": 1.0, "symbol": "TEST"},
        )
    ]

    backtester = Backtester(
        config=BacktesterConfig(symbol="TEST"),
        limit_book=order_book,
        metrics_logger=logger,
        strategy=strategy,
    )

    backtester.run(replay)
    assert strategy.timer_triggered is True


def test_strategy_sandbox_halts_on_exception() -> None:
    strategy = FaultyStrategy()
    logger = MetricsLogger()
    order_book = PythonOrderBook(depth=1)
    replay = [
        MarketEvent(
            timestamp_ns=10,
            event_type="trade",
            payload={"side": "BUY", "price": 100.0, "size": 1.0, "symbol": "TEST"},
        )
    ]

    backtester = Backtester(
        config=BacktesterConfig(symbol="TEST"),
        limit_book=order_book,
        metrics_logger=logger,
        strategy=strategy,
    )

    backtester.run(replay)
    assert backtester.strategy_halted is True
    assert backtester.strategy_error is not None
