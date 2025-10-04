"""Risk management, inventory tracking, and PnL accounting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .backtester import FillEvent, MarketSnapshot


@dataclass(slots=True)
class RiskConfig:
    symbol: str
    max_long: float = 500.0
    max_short: float = -500.0
    halt_on_breach: bool = True
    warn_fraction: float = 0.8  # warn when |inventory| > warn_fraction * limit


@dataclass(slots=True)
class PositionLot:
    size: float
    price: float


class RiskEngine:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.inventory: Dict[str, float] = {config.symbol: 0.0}
        self.realized_pnl: Dict[str, float] = {config.symbol: 0.0}
        self.unrealized_pnl: Dict[str, float] = {config.symbol: 0.0}
        self.lots: Dict[str, List[PositionLot]] = {config.symbol: []}
        self.last_mid: Dict[str, float | None] = {config.symbol: None}
        self.strategy_halted = False
        self.warnings: List[str] = []

    def update_on_fill(self, fill: "FillEvent") -> None:
        symbol = fill.symbol
        signed_qty = fill.size if fill.side == "BUY" else -fill.size
        lots = self.lots.setdefault(symbol, [])
        self.inventory.setdefault(symbol, 0.0)
        self.realized_pnl.setdefault(symbol, 0.0)
        pnl = 0.0

        remaining = signed_qty
        while lots and abs(remaining) > 1e-9 and self._has_opposite_sign(remaining, lots[0].size):
            lot = lots[0]
            lot_sign = 1.0 if lot.size > 0 else -1.0
            matched = min(abs(remaining), abs(lot.size))
            pnl += matched * (fill.price - lot.price) * lot_sign
            lot.size -= matched * lot_sign
            remaining += matched * lot_sign
            if abs(lot.size) <= 1e-9:
                lots.pop(0)

        if abs(remaining) > 1e-9:
            lots.append(PositionLot(size=remaining, price=fill.price))

        self.realized_pnl[symbol] += pnl
        self.inventory[symbol] += signed_qty
        if abs(self.inventory[symbol]) <= 1e-9:
            self.inventory[symbol] = 0.0
        self._check_limits(symbol)

    def update_on_tick(self, snapshot: "MarketSnapshot") -> None:
        symbol = self.config.symbol
        mid = snapshot.midprice
        self.last_mid[symbol] = mid
        inventory = self.inventory.get(symbol, 0.0)
        if mid is None:
            self.unrealized_pnl[symbol] = 0.0
            return
        cost_basis = self._average_cost(symbol)
        self.unrealized_pnl[symbol] = inventory * (mid - cost_basis)
        self._check_limits(symbol)

    def _average_cost(self, symbol: str) -> float:
        lots = self.lots.get(symbol, [])
        if not lots:
            return self.last_mid.get(symbol) or 0.0
        total_size = sum(lot.size for lot in lots)
        if abs(total_size) <= 1e-9:
            return self.last_mid.get(symbol) or 0.0
        total_cost = sum(lot.size * lot.price for lot in lots)
        return total_cost / total_size

    def _check_limits(self, symbol: str) -> None:
        inventory = self.inventory.get(symbol, 0.0)
        warn_long = self.config.max_long * self.config.warn_fraction
        warn_short = self.config.max_short * self.config.warn_fraction

        if self.config.max_long > 0 and inventory >= warn_long:
            message = f"Inventory warning: {inventory} units on {symbol}"
            if not self.warnings or self.warnings[-1] != message:
                self.warnings.append(message)
        if self.config.max_short < 0 and inventory <= warn_short:
            message = f"Inventory warning: {inventory} units on {symbol}"
            if not self.warnings or self.warnings[-1] != message:
                self.warnings.append(message)

        if inventory > self.config.max_long or inventory < self.config.max_short:
            if self.config.halt_on_breach:
                self.strategy_halted = True

    @staticmethod
    def _has_opposite_sign(a: float, b: float) -> bool:
        return (a > 0 > b) or (a < 0 < b)


__all__ = ["RiskConfig", "RiskEngine", "PositionLot"]
