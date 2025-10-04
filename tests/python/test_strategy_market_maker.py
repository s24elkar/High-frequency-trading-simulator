import pytest

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
            MarketMakingConfig(
                spread_ticks=1, quote_size=1.0, tick_size=0.1, update_interval_ns=0
            ),
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
        backtester.on_market_data(snapshot)
        return tuple(
            sorted(
                (order.side, order.price) for order in backtester.active_orders.values()
            )
        )


def test_strategy_reproducible() -> None:
    first = _run_strategy(seed=123)
    second = _run_strategy(seed=123)
    assert first == second


def test_strategy_emits_iceberg_orders() -> None:
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
        strategy = MarketMakingStrategy(
            MarketMakingConfig(
                spread_ticks=1,
                quote_size=6.0,
                tick_size=0.1,
                update_interval_ns=0,
                order_type="ICEBERG",
                iceberg_display=2.0,
            )
        )
        backtester = Backtester(
            config=BacktesterConfig(symbol="TEST"),
            limit_book=PythonOrderBook(depth=5),
            metrics_logger=metrics,
            risk_engine=None,
            strategy=strategy,
        )
        backtester.clock_ns = snapshot.timestamp_ns
        backtester.on_market_data(snapshot)
        orders = list(backtester.active_orders.values())
        assert orders
        assert all(order.order_type == "ICEBERG" for order in orders)
        assert all(order.display_size == pytest.approx(2.0) for order in orders)


def test_strategy_pegged_orders_capture_reference() -> None:
    snapshot = MarketSnapshot(
        timestamp_ns=2_000,
        best_bid=50.0,
        bid_size=1.0,
        best_ask=50.2,
        ask_size=1.0,
        last_trade_price=50.1,
        last_trade_size=1.0,
        imbalance=0.0,
        depth=[],
    )
    with MetricsLogger() as metrics:
        strategy = MarketMakingStrategy(
            MarketMakingConfig(
                spread_ticks=1,
                tick_size=0.1,
                quote_size=1.0,
                update_interval_ns=0,
                order_type="PEGGED",
                peg_reference="MID",
                peg_offset=0.0,
            )
        )
        backtester = Backtester(
            config=BacktesterConfig(symbol="TEST"),
            limit_book=PythonOrderBook(depth=5),
            metrics_logger=metrics,
            risk_engine=None,
            strategy=strategy,
        )
        backtester.clock_ns = snapshot.timestamp_ns
        backtester.on_market_data(snapshot)
        orders = list(backtester.active_orders.values())
        assert orders
        assert all(order.order_type == "PEGGED" for order in orders)
        assert all(order.peg_reference == "MID" for order in orders)
