import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from backtester import Backtester, BacktesterConfig, MetricsLogger
from backtester.order_book import PythonOrderBook
from backtester.risk import RiskConfig, RiskEngine
from backtester.backtester import MarketSnapshot
from strategies import MarketMakingConfig, MarketMakingStrategy


def _run_strategy(seed: int):
    snapshot = MarketSnapshot(
        timestamp_ns=1_000,
        best_bid=100.0,
        bid_size=5.0,
        best_ask=100.2,
        ask_size=5.0,
        last_trade_price=100.1,
        last_trade_size=2.0,
        imbalance=0.0,
        depth=[],
    )
    with MetricsLogger() as metrics:
        risk = RiskEngine(RiskConfig(symbol="TEST"))
        strategy = MarketMakingStrategy(
            MarketMakingConfig(spread_ticks=1, quote_size=1.0, tick_size=0.1, update_interval_ns=0),
            risk_engine=risk,
            seed=seed,
        )
        backtester = Backtester(
            config=BacktesterConfig(symbol="TEST"),
            limit_book=PythonOrderBook(depth=5),
            metrics_logger=metrics,
            risk_engine=risk,
            strategy=strategy,
            seed=seed,
        )
        backtester.clock_ns = snapshot.timestamp_ns
        strategy.on_tick(snapshot, backtester)
        return tuple(sorted((order.side, order.price) for order in backtester.active_orders.values()))


def test_strategy_reproducible() -> None:
    first = _run_strategy(seed=123)
    second = _run_strategy(seed=123)
    assert first == second
