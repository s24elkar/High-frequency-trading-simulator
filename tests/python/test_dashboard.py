import io

from python.backtester import (
    Backtester,
    BacktesterConfig,
    DashboardConfig,
    RiskConfig,
    RiskDashboard,
    RiskEngine,
)
from python.backtester.backtester import FillEvent, MarketSnapshot, OrderRequest
from python.backtester.logging import MetricsLogger
from python.backtester.order_book import PythonOrderBook
from strategies import MarketMakingConfig, MarketMakingStrategy


def _fill_event(order_id: int, side: str, price: float, size: float, ts: int) -> FillEvent:
    return FillEvent(
        order_id=order_id,
        symbol="TEST",
        side=side,
        price=price,
        size=size,
        timestamp_ns=ts,
        liquidity_flag="MAKER",
    )


def _snapshot(ts: int, bid: float, ask: float) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp_ns=ts,
        best_bid=bid,
        bid_size=1.0,
        best_ask=ask,
        ask_size=1.0,
        last_trade_price=None,
        last_trade_size=None,
        imbalance=0.0,
        depth=[],
    )


def test_dashboard_renders_metrics() -> None:
    stream = io.StringIO()
    dashboard = RiskDashboard(
        stream=stream, config=DashboardConfig(symbol="TEST", clear_screen=False)
    )
    logger = MetricsLogger()
    risk_engine = RiskEngine(RiskConfig(symbol="TEST"))
    dashboard.bind(symbol="TEST", risk_engine=risk_engine, metrics_logger=logger)

    snapshot = _snapshot(1_000, 100.0, 100.2)
    risk_engine.update_on_tick(snapshot)

    order = OrderRequest(
        order_id=1,
        symbol="TEST",
        side="BUY",
        price=100.0,
        size=2.0,
        timestamp_ns=1_000,
        metadata={},
    )
    logger.log_order(order, latency_ns=42)

    fill = _fill_event(1, "BUY", price=100.0, size=2.0, ts=1_500)
    risk_engine.update_on_fill(fill)
    logger.log_fill(fill)

    dashboard.update(timestamp_ns=2_000)
    output = stream.getvalue()
    assert "Inventory: 2.00" in output
    assert "PnL (real/unreal/total)" in output
    assert "Latency ns" in output


def test_market_maker_dashboard_alerts() -> None:
    stream = io.StringIO()
    dashboard = RiskDashboard(
        stream=stream, config=DashboardConfig(symbol="TEST", clear_screen=False)
    )
    metrics = MetricsLogger()
    risk_engine = RiskEngine(
        RiskConfig(
            symbol="TEST",
            max_long=5.0,
            max_short=-5.0,
            max_notional_exposure=150.0,
            loss_limit=-5.0,
            warn_fraction=0.6,
        )
    )
    strategy = MarketMakingStrategy(
        MarketMakingConfig(
            spread_ticks=1,
            tick_size=0.1,
            quote_size=1.0,
            update_interval_ns=0,
        ),
        risk_engine=risk_engine,
        seed=1,
    )

    backtester = Backtester(
        config=BacktesterConfig(symbol="TEST"),
        limit_book=PythonOrderBook(depth=5),
        metrics_logger=metrics,
        risk_engine=risk_engine,
        strategy=strategy,
        dashboard=dashboard,
    )

    first_snapshot = _snapshot(10_000, 100.0, 100.2)
    backtester.clock_ns = first_snapshot.timestamp_ns
    backtester.on_market_data(first_snapshot)

    bid_id = strategy.working_orders.get("bid")
    assert bid_id is not None
    bid_order = backtester.active_orders[bid_id]
    fill_buy = _fill_event(bid_id, "BUY", bid_order.price, bid_order.size, ts=11_000)
    backtester.clock_ns = fill_buy.timestamp_ns
    backtester.process_fill(fill_buy)

    rich_snapshot = _snapshot(12_000, 200.0, 200.2)
    backtester.clock_ns = rich_snapshot.timestamp_ns
    backtester.on_market_data(rich_snapshot)

    loss_snapshot = _snapshot(13_000, 94.0, 94.2)
    backtester.clock_ns = loss_snapshot.timestamp_ns
    backtester.on_market_data(loss_snapshot)

    assert backtester.strategy_halted is True
    assert any("Exposure limit breached" in msg for msg in risk_engine.alerts)
    assert any("Loss limit breached" in msg for msg in risk_engine.alerts)

    output = stream.getvalue()
    assert "Strategy halted: YES" in output
    assert "Exposure limit breached" in output
    assert "Loss limit breached" in output
