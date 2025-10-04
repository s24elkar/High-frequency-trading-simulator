"""Limit order book adapters used by the Python backtester.

The module prefers to load the native C++ implementation for performance; if
that shared library is unavailable the tests fall back to a minimal Python
implementation that preserves price-time priority semantics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from .backtester import (
    FillEvent,
    MarketEvent,
    MarketSnapshot,
    OrderBookUpdate,
    OrderRequest,
)

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
        self.stop_orders: Dict[int, OrderRequest] = {}
        self.iceberg_state: Dict[int, Dict[str, float]] = {}
        self.pegged_orders: Dict[int, OrderRequest] = {}

    def enqueue(self, order: OrderRequest) -> None:
        order_copy = replace(order)
        order_copy.order_type = order_copy.order_type.upper()
        if order_copy.order_type == "STOP":
            self.stop_orders[order_copy.order_id] = order_copy
            log.debug("Registered stop order %s", order_copy)
            return

        if order_copy.order_type == "ICEBERG":
            total = order_copy.total_size or order_copy.size
            display = order_copy.display_size or total
            visible = min(display, total)
            hidden = max(total - visible, 0.0)
            order_copy.total_size = total
            order_copy.size = visible
            self.iceberg_state[order_copy.order_id] = {
                "remaining": hidden,
                "display": display,
            }

        if order_copy.order_type == "PEGGED":
            order_copy.peg_reference = order_copy.peg_reference or (
                "BID" if order_copy.side == "BUY" else "ASK"
            )
            order_copy.price = self._peg_price(order_copy)
            self.pegged_orders[order_copy.order_id] = order_copy

        book = self.bids if order_copy.side == "BUY" else self.asks
        book.setdefault(order_copy.price, []).append(order_copy)
        log.debug("Enqueued %s", order_copy)

    def cancel(self, order_id: int) -> None:
        self.stop_orders.pop(order_id, None)
        self.iceberg_state.pop(order_id, None)
        pegged = self.pegged_orders.pop(order_id, None)
        if pegged is not None:
            self._remove_from_book(pegged)
            return
        for book in (self.bids, self.asks):
            for price, queue in list(book.items()):
                book[price] = [o for o in queue if o.order_id != order_id]
                if not book[price]:
                    book.pop(price)

    def apply_event(self, event: MarketEvent) -> OrderBookUpdate:
        etype = event.event_type
        side = event.payload.get("side")
        price = float(event.payload.get("price", 0.0))
        size = float(event.payload.get("size", 0.0))
        order_type = str(event.payload.get("order_type", "LIMIT")).upper()
        display_size = event.payload.get("display_size")
        stop_price = event.payload.get("stop_price")
        peg_reference = event.payload.get("peg_reference")
        peg_offset = float(event.payload.get("peg_offset", 0.0))
        fills: List[FillEvent] = []
        if etype == "add_order":
            order_id = int(event.payload["order_id"])
            order = OrderRequest(
                order_id,
                event.payload["symbol"],
                side,
                price,
                size,
                event.timestamp_ns,
                order_type=order_type,
                display_size=float(display_size) if display_size is not None else None,
                stop_price=float(stop_price) if stop_price is not None else None,
                peg_reference=peg_reference,
                peg_offset=peg_offset,
                total_size=size,
            )
            self.enqueue(order)
        elif etype == "delete_order":
            self.cancel(int(event.payload["order_id"]))
        elif etype == "execute_order":
            fills = self._execute(size, side, price, event.timestamp_ns)
        elif etype == "trade":
            self.last_trade_price = price
            self.last_trade_size = size
        self.last_timestamp_ns = event.timestamp_ns
        fills.extend(self._trigger_stop_orders(event.timestamp_ns))
        self._reprice_pegged_orders()
        snapshot = self.snapshot(self.depth)
        return OrderBookUpdate(snapshot=snapshot, fills=fills)

    def _execute(
        self, size: float, side: str, price: float, timestamp_ns: int
    ) -> List[FillEvent]:
        book = self.bids if side == "SELL" else self.asks
        remaining = size
        fills: List[FillEvent] = []
        levels = sorted(book.keys(), reverse=(side == "SELL"))
        for level_price in levels:
            if remaining <= 0:
                break
            queue = book.get(level_price)
            if not queue:
                continue
            for order in list(queue):
                if remaining <= 0:
                    break
                take = min(order.size, remaining)
                if take <= 0:
                    continue
                fill_price = price if price > 0 else level_price
                fills.append(
                    FillEvent(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        price=fill_price,
                        size=take,
                        timestamp_ns=timestamp_ns,
                        liquidity_flag="MAKER",
                    )
                )
                order.size -= take
                remaining -= take
                if order.size <= 1e-9:
                    queue.remove(order)
                    self._on_order_depleted(order, timestamp_ns, queue)
            if not queue:
                book.pop(level_price, None)
        executed_size = size - remaining
        if executed_size > 0:
            self.last_trade_price = (
                price if price > 0 else (levels[0] if levels else price)
            )
            self.last_trade_size = executed_size
        return fills

    def _trigger_stop_orders(self, timestamp_ns: int) -> List[FillEvent]:
        if not self.stop_orders:
            return []
        fills: List[FillEvent] = []
        triggered: List[OrderRequest] = []
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        ref_buy = (
            self.last_trade_price if self.last_trade_price is not None else best_ask
        )
        ref_sell = (
            self.last_trade_price if self.last_trade_price is not None else best_bid
        )
        for order in list(self.stop_orders.values()):
            trigger_price = order.stop_price or order.price
            if order.side == "BUY":
                if ref_buy is not None and ref_buy >= trigger_price:
                    triggered.append(order)
            else:
                if ref_sell is not None and ref_sell <= trigger_price:
                    triggered.append(order)
        for order in triggered:
            self.stop_orders.pop(order.order_id, None)
            executed = self._execute(
                order.total_size or order.size, order.side, 0.0, timestamp_ns
            )
            fills.extend(executed)
            fill_price = self.last_trade_price or order.stop_price or order.price
            fills.append(
                FillEvent(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    price=fill_price,
                    size=order.total_size or order.size,
                    timestamp_ns=timestamp_ns,
                    liquidity_flag="TAKER",
                )
            )
        return fills

    def _peg_price(self, order: OrderRequest) -> float:
        ref = (order.peg_reference or ("BID" if order.side == "BUY" else "ASK")).upper()
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        if ref == "MID" and best_bid is not None and best_ask is not None:
            base = (best_bid + best_ask) / 2.0
        elif ref == "BID":
            base = best_bid
        elif ref == "ASK":
            base = best_ask
        else:
            base = order.price
        if base is None:
            base = order.price
        price = base + order.peg_offset
        if best_ask is not None and order.side == "BUY":
            price = min(price, best_ask)
        if best_bid is not None and order.side == "SELL":
            price = max(price, best_bid)
        return round(price, 8)

    def _reprice_pegged_orders(self) -> None:
        for order_id, order in list(self.pegged_orders.items()):
            new_price = self._peg_price(order)
            if abs(new_price - order.price) <= 1e-9:
                continue
            self._remove_from_book(order)
            order.price = new_price
            book = self.bids if order.side == "BUY" else self.asks
            book.setdefault(order.price, []).append(order)
            log.debug("Repriced pegged order %s to %.4f", order_id, new_price)

    def _remove_from_book(self, order: OrderRequest) -> None:
        book = self.bids if order.side == "BUY" else self.asks
        queue = book.get(order.price)
        if not queue:
            return
        queue[:] = [o for o in queue if o.order_id != order.order_id]
        if not queue:
            book.pop(order.price, None)

    def _best_bid(self) -> Optional[float]:
        return max(self.bids.keys(), default=None)

    def _best_ask(self) -> Optional[float]:
        return min(self.asks.keys(), default=None)

    def _on_order_depleted(
        self, order: OrderRequest, timestamp_ns: int, queue: List[OrderRequest]
    ) -> None:
        iceberg = self.iceberg_state.get(order.order_id)
        if iceberg:
            remaining = iceberg.get("remaining", 0.0)
            display = iceberg.get("display", order.size)
            if remaining > 1e-9:
                clip = min(display, remaining)
                iceberg["remaining"] = remaining - clip
                order.size = clip
                order.timestamp_ns = timestamp_ns
                queue.append(order)
                log.debug(
                    "Replenished iceberg order %s with clip %.4f (remaining %.4f)",
                    order.order_id,
                    clip,
                    iceberg["remaining"],
                )
                return
            self.iceberg_state.pop(order.order_id, None)
        self.pegged_orders.pop(order.order_id, None)

    def snapshot(self, depth: int = 1) -> MarketSnapshot:
        best_bid = max(self.bids.keys(), default=None)
        best_ask = min(self.asks.keys(), default=None)
        bid_size = (
            sum(o.size for o in self.bids.get(best_bid, []))
            if best_bid is not None
            else None
        )
        ask_size = (
            sum(o.size for o in self.asks.get(best_ask, []))
            if best_ask is not None
            else None
        )
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
    return _order_book_bridge.OrderBook(
        depth
    )  # pragma: no cover - requires compiled bridge


__all__ = ["PythonOrderBook", "load_order_book", "DepthLevel"]
