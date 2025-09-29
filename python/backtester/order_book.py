"""Limit order book adapters used by the Python backtester.

The module prefers to load the native C++ implementation for performance; if
that shared library is unavailable the tests fall back to a minimal Python
implementation that preserves price-time priority semantics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .backtester import MarketEvent, MarketSnapshot, OrderRequest

log = logging.getLogger(__name__)


try:
    from . import _order_book_bridge  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - the C++ bridge is optional during tests
    _order_book_bridge = None


@dataclass(slots=True)
class DepthLevel:
    price: float
    size: float
    orders: List[OrderRequest] = field(default_factory=list)


class PythonOrderBook:
    """Simplified order book used for testing and documentation examples."""

    def __init__(self, depth: int = 5) -> None:
        self.depth = depth
        self.bids: Dict[float, List[OrderRequest]] = {}
        self.asks: Dict[float, List[OrderRequest]] = {}
        self.last_trade_price: Optional[float] = None
        self.last_trade_size: Optional[float] = None
        self.last_timestamp_ns: int = 0

    def enqueue(self, order: OrderRequest) -> None:
        book = self.bids if order.side == "BUY" else self.asks
        book.setdefault(order.price, []).append(order)
        log.debug("Enqueued %s", order)

    def cancel(self, order_id: int) -> None:
        for book in (self.bids, self.asks):
            for price, queue in list(book.items()):
                book[price] = [o for o in queue if o.order_id != order_id]
                if not book[price]:
                    book.pop(price)

    def apply_event(self, event: MarketEvent) -> Optional[MarketSnapshot]:
        etype = event.event_type
        side = event.payload.get("side")
        price = float(event.payload.get("price", 0.0))
        size = float(event.payload.get("size", 0.0))
        if etype == "add_order":
            order_id = int(event.payload["order_id"])
            order = OrderRequest(order_id, event.payload["symbol"], side, price, size, event.timestamp_ns)
            self.enqueue(order)
        elif etype == "delete_order":
            self.cancel(int(event.payload["order_id"]))
        elif etype == "execute_order":
            self._execute(size, side, price)
        elif etype == "trade":
            self.last_trade_price = price
            self.last_trade_size = size
        self.last_timestamp_ns = event.timestamp_ns
        return self.snapshot(self.depth)

    def _execute(self, size: float, side: str, price: float) -> None:
        book = self.bids if side == "SELL" else self.asks
        remaining = size
        for level_price in sorted(book.keys(), reverse=(side == "SELL")):
            queue = book[level_price]
            for order in list(queue):
                take = min(order.size, remaining)
                order.size -= take
                remaining -= take
                if order.size <= 1e-9:
                    queue.remove(order)
                if remaining <= 0:
                    break
            if not queue:
                book.pop(level_price, None)
            if remaining <= 0:
                break
        self.last_trade_price = price
        self.last_trade_size = size - remaining

    def snapshot(self, depth: int = 1) -> MarketSnapshot:
        best_bid = max(self.bids.keys(), default=None)
        best_ask = min(self.asks.keys(), default=None)
        bid_size = sum(o.size for o in self.bids.get(best_bid, [])) if best_bid is not None else None
        ask_size = sum(o.size for o in self.asks.get(best_ask, [])) if best_ask is not None else None
        depth_entries: List[Dict[str, float]] = []
        for idx, price in enumerate(sorted(self.bids.keys(), reverse=True)):
            if idx >= depth:
                break
            level_size = sum(o.size for o in self.bids[price])
            depth_entries.append({"side": "BUY", "price": price, "size": level_size})
        for idx, price in enumerate(sorted(self.asks.keys())):
            if idx >= depth:
                break
            level_size = sum(o.size for o in self.asks[price])
            depth_entries.append({"side": "SELL", "price": price, "size": level_size})
        imbalance = None
        if best_bid is not None and best_ask is not None and bid_size and ask_size:
            imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        return MarketSnapshot(
            timestamp_ns=self.last_timestamp_ns,
            best_bid=best_bid,
            bid_size=bid_size,
            best_ask=best_ask,
            ask_size=ask_size,
            last_trade_price=self.last_trade_price,
            last_trade_size=self.last_trade_size,
            imbalance=imbalance,
            depth=depth_entries,
        )


def load_order_book(depth: int = 5) -> PythonOrderBook:
    if _order_book_bridge is None:
        log.info("Using PythonOrderBook fallback (C++ bridge not available)")
        return PythonOrderBook(depth=depth)
    return _order_book_bridge.OrderBook(depth)  # pragma: no cover - requires compiled bridge


__all__ = ["PythonOrderBook", "load_order_book", "DepthLevel"]
