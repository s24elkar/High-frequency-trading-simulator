"""Toy market-making strategy used for regression tests and demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    from backtester.backtester import MarketSnapshot, StrategyCallbacks, StrategyContext
    from backtester.risk import RiskEngine
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from ..backtester.backtester import (  # type: ignore[no-redef]
        MarketSnapshot,
        StrategyCallbacks,
        StrategyContext,
    )
    from ..backtester.risk import RiskEngine  # type: ignore[no-redef]


@dataclass(slots=True)
class MarketMakingConfig:
    spread_ticks: int = 1
    tick_size: float = 0.01
    quote_size: float = 10.0
    inventory_skew: float = 0.0
    update_interval_ns: int = 5_000_000_000  # 5 ms default
    order_type: str = "LIMIT"
    iceberg_display: Optional[float] = None
    stop_price_offset: Optional[float] = None
    peg_reference: Optional[str] = None
    peg_offset: float = 0.0


class MarketMakingStrategy(StrategyCallbacks):
    def __init__(
        self,
        config: MarketMakingConfig,
        risk_engine: Optional[RiskEngine] = None,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.risk_engine = risk_engine
        self.last_quote_ns = 0
        self.working_orders: Dict[str, int] = {}

    def on_market_data(self, snapshot: MarketSnapshot, ctx: StrategyContext) -> None:
        if snapshot.best_bid is None or snapshot.best_ask is None:
            return
        if snapshot.timestamp_ns - self.last_quote_ns < self.config.update_interval_ns:
            return
        inventory_adjust = 0.0
        risk_engine = ctx.risk_engine or self.risk_engine
        if risk_engine is not None:
            inventory = risk_engine.inventory.get(ctx.config.symbol, 0.0)
            inventory_adjust = self.config.inventory_skew * inventory
        mid = snapshot.midprice
        if mid is None:
            return
        spread = self.config.spread_ticks * self.config.tick_size
        bid_price = mid - spread / 2.0 - inventory_adjust * self.config.tick_size
        ask_price = mid + spread / 2.0 - inventory_adjust * self.config.tick_size
        bid_size = self.config.quote_size
        ask_size = self.config.quote_size
        order_type = self.config.order_type.upper()
        common_kwargs: Dict[str, float | str | None] = {"order_type": order_type}
        if order_type == "ICEBERG":
            display = self.config.iceberg_display or max(bid_size / 3.0, 1.0)
            common_kwargs["display_size"] = display
        if order_type == "PEGGED":
            common_kwargs["peg_offset"] = self.config.peg_offset

        bid_kwargs = dict(common_kwargs)
        ask_kwargs = dict(common_kwargs)
        if order_type == "STOP":
            offset = self.config.stop_price_offset or spread
            bid_kwargs["stop_price"] = bid_price + offset
            ask_kwargs["stop_price"] = ask_price - offset
        if order_type == "PEGGED":
            bid_kwargs["peg_reference"] = self.config.peg_reference or "BID"
            ask_kwargs["peg_reference"] = self.config.peg_reference or "ASK"

        bid_id = ctx.submit_order(
            "BUY",
            round(bid_price, 8),
            bid_size,
            order_type=bid_kwargs.get("order_type", "LIMIT"),
            display_size=bid_kwargs.get("display_size"),
            stop_price=bid_kwargs.get("stop_price"),
            peg_reference=bid_kwargs.get("peg_reference"),
            peg_offset=float(bid_kwargs.get("peg_offset", 0.0)),
        )
        ask_id = ctx.submit_order(
            "SELL",
            round(ask_price, 8),
            ask_size,
            order_type=ask_kwargs.get("order_type", "LIMIT"),
            display_size=ask_kwargs.get("display_size"),
            stop_price=ask_kwargs.get("stop_price"),
            peg_reference=ask_kwargs.get("peg_reference"),
            peg_offset=float(ask_kwargs.get("peg_offset", 0.0)),
        )
        self.working_orders["bid"] = bid_id
        self.working_orders["ask"] = ask_id
        self.last_quote_ns = snapshot.timestamp_ns


__all__ = ["MarketMakingStrategy", "MarketMakingConfig"]
