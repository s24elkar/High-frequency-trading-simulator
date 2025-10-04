from __future__ import annotations

import pytest

from python.backtester.backtester import OrderRequest, MarketEvent
from python.backtester.order_book import PythonOrderBook


def test_iceberg_replenishes_display_clips() -> None:
    book = PythonOrderBook()
    iceberg = OrderRequest(
        order_id=1,
        symbol="XYZ",
        side="SELL",
        price=101.0,
        size=9.0,
        timestamp_ns=0,
        order_type="ICEBERG",
        display_size=3.0,
        total_size=9.0,
    )
    book.enqueue(iceberg)
    assert pytest.approx(book.asks[101.0][0].size) == 3.0
    fills = book._execute(3.0, "BUY", 0.0, 1)
    assert fills
    assert pytest.approx(book.iceberg_state[1]["remaining"]) == 3.0
    assert pytest.approx(book.asks[101.0][0].size) == 3.0
    book._execute(3.0, "BUY", 0.0, 2)
    assert pytest.approx(book.iceberg_state[1]["remaining"]) == 0.0
    book._execute(3.0, "BUY", 0.0, 3)
    assert 1 not in book.iceberg_state
    assert 101.0 not in book.asks or not book.asks[101.0]


def test_stop_order_triggers_on_trade() -> None:
    book = PythonOrderBook()
    resting_ask = OrderRequest(
        order_id=10,
        symbol="XYZ",
        side="SELL",
        price=101.0,
        size=1.0,
        timestamp_ns=0,
    )
    book.enqueue(resting_ask)
    stop_order = OrderRequest(
        order_id=11,
        symbol="XYZ",
        side="BUY",
        price=101.0,
        size=1.0,
        timestamp_ns=0,
        order_type="STOP",
        stop_price=101.0,
    )
    book.enqueue(stop_order)
    assert 11 in book.stop_orders
    trade_event = MarketEvent(
        timestamp_ns=1,
        event_type="trade",
        payload={"symbol": "XYZ", "price": 101.0, "size": 1.0},
    )
    update = book.apply_event(trade_event)
    assert 11 not in book.stop_orders
    assert any(fill.order_id == 11 for fill in update.fills)
    snap = book.snapshot()
    assert snap.best_ask is None


def test_pegged_order_tracks_best_bid() -> None:
    book = PythonOrderBook()
    anchor = OrderRequest(
        order_id=20,
        symbol="XYZ",
        side="BUY",
        price=99.5,
        size=1.0,
        timestamp_ns=0,
    )
    book.enqueue(anchor)
    pegged = OrderRequest(
        order_id=21,
        symbol="XYZ",
        side="BUY",
        price=0.0,
        size=1.0,
        timestamp_ns=0,
        order_type="PEGGED",
        peg_reference="BID",
        peg_offset=0.0,
    )
    book.enqueue(pegged)
    initial_price = book.bids[max(book.bids.keys())][-1].price
    assert initial_price == pytest.approx(99.5)

    better_bid = OrderRequest(
        order_id=22,
        symbol="XYZ",
        side="BUY",
        price=100.1,
        size=1.0,
        timestamp_ns=1,
    )
    book.enqueue(better_bid)
    book._reprice_pegged_orders()
    assert book.pegged_orders[21].price == pytest.approx(100.1)
