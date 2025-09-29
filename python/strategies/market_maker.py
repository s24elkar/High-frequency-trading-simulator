"""Toy market-making strategy used for regression tests and demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from backtester.backtester import Backtester, MarketSnapshot
from backtester.risk import RiskEngine


@dataclass(slots=True)
class MarketMakingConfig:
    spread_ticks: int = 1
    tick_size: float = 0.01
    quote_size: float = 10.0
    inventory_skew: float = 0.0
    update_interval_ns: int = 5_000_000_000  # 5 ms default


class MarketMakingStrategy:
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

    def on_tick(self, snapshot: MarketSnapshot, backtester: Backtester) -> None:
        if snapshot.best_bid is None or snapshot.best_ask is None:
            return
        if snapshot.timestamp_ns - self.last_quote_ns < self.config.update_interval_ns:
            return
        inventory_adjust = 0.0
        if self.risk_engine is not None:
            inventory = self.risk_engine.inventory.get(backtester.config.symbol, 0.0)
            inventory_adjust = self.config.inventory_skew * inventory
        mid = snapshot.midprice
        if mid is None:
            return
        spread = self.config.spread_ticks * self.config.tick_size
        bid_price = mid - spread / 2.0 - inventory_adjust * self.config.tick_size
        ask_price = mid + spread / 2.0 - inventory_adjust * self.config.tick_size
        bid_size = self.config.quote_size
        ask_size = self.config.quote_size
        bid_id = backtester.submit_order("BUY", round(bid_price, 8), bid_size)
        ask_id = backtester.submit_order("SELL", round(ask_price, 8), ask_size)
        self.working_orders["bid"] = bid_id
        self.working_orders["ask"] = ask_id
        self.last_quote_ns = snapshot.timestamp_ns


__all__ = ["MarketMakingStrategy", "MarketMakingConfig"]
