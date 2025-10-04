"""Unit tests for the risk engine's PnL and inventory management."""

from __future__ import annotations

import pytest

from python.backtester.backtester import FillEvent, MarketSnapshot
from python.backtester.risk import RiskConfig, RiskEngine


def _fill(side: str, price: float, size: float, order_id: int = 0) -> FillEvent:
    return FillEvent(
        order_id=order_id,
        symbol="XYZ",
        side=side,
        price=price,
        size=size,
        timestamp_ns=0,
        liquidity_flag="UNKNOWN",
    )


def _snapshot(best_bid: float, best_ask: float) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp_ns=0,
        best_bid=best_bid,
        bid_size=1.0,
        best_ask=best_ask,
        ask_size=1.0,
        last_trade_price=None,
        last_trade_size=None,
        imbalance=None,
        depth=[],
    )


def test_pnl_engine_tracks_realized_and_unrealized_components() -> None:
    config = RiskConfig(symbol="XYZ", max_long=100.0, max_short=-100.0, warn_fraction=0.8)
    engine = RiskEngine(config)

    engine.update_on_fill(_fill("BUY", price=100.0, size=10.0, order_id=1))
    engine.update_on_tick(_snapshot(101.0, 103.0))  # mid = 102.0
    assert engine.inventory["XYZ"] == pytest.approx(10.0)
    assert engine.realized_pnl["XYZ"] == pytest.approx(0.0)
    assert engine.unrealized_pnl["XYZ"] == pytest.approx(20.0)

    engine.update_on_fill(_fill("SELL", price=103.0, size=4.0, order_id=2))
    engine.update_on_tick(_snapshot(100.0, 102.0))  # mid = 101.0
    assert engine.inventory["XYZ"] == pytest.approx(6.0)
    assert engine.realized_pnl["XYZ"] == pytest.approx(12.0)
    assert engine.unrealized_pnl["XYZ"] == pytest.approx(6.0)

    engine.update_on_fill(_fill("SELL", price=99.0, size=10.0, order_id=3))
    engine.update_on_tick(_snapshot(102.0, 104.0))  # adverse move against the short
    assert engine.inventory["XYZ"] == pytest.approx(-4.0)
    assert engine.realized_pnl["XYZ"] == pytest.approx(6.0)
    assert engine.unrealized_pnl["XYZ"] == pytest.approx(-16.0)

    engine.update_on_tick(_snapshot(94.0, 96.0))  # recovery in favour of the short
    assert engine.unrealized_pnl["XYZ"] == pytest.approx(16.0)

    engine.update_on_fill(_fill("BUY", price=96.0, size=5.0, order_id=4))
    engine.update_on_tick(_snapshot(93.0, 95.0))  # mid = 94.0
    assert engine.inventory["XYZ"] == pytest.approx(1.0)
    assert engine.realized_pnl["XYZ"] == pytest.approx(18.0)
    assert engine.unrealized_pnl["XYZ"] == pytest.approx(-2.0)


def test_inventory_limits_raise_warnings_and_halt() -> None:
    config = RiskConfig(symbol="XYZ", max_long=5.0, max_short=-5.0, warn_fraction=0.5)
    engine = RiskEngine(config)

    engine.update_on_fill(_fill("BUY", price=100.0, size=3.0, order_id=10))
    assert engine.inventory["XYZ"] == pytest.approx(3.0)
    assert engine.strategy_halted is False
    assert engine.warnings[-1].startswith("Inventory warning")

    engine.update_on_fill(_fill("BUY", price=101.0, size=3.0, order_id=11))
    assert engine.strategy_halted is True

    engine_short = RiskEngine(config)
    engine_short.update_on_fill(_fill("SELL", price=100.0, size=3.0, order_id=20))
    assert engine_short.warnings[-1].startswith("Inventory warning")

    engine_short.update_on_fill(_fill("SELL", price=99.0, size=3.0, order_id=21))
    assert engine_short.strategy_halted is True
