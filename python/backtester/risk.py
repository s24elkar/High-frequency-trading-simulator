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
        direction = 1.0 if fill.side == "BUY" else -1.0
        lots = self.lots.setdefault(symbol, [])
        inventory = self.inventory.setdefault(symbol, 0.0)

        if direction > 0:  # buy increases inventory
            lots.append(PositionLot(size=fill.size, price=fill.price))
            self.inventory[symbol] = inventory + fill.size
        else:  # sell reduces inventory and realises pnl
            remaining = fill.size
            pnl = 0.0
            while remaining > 0 and lots:
                lot = lots[0]
                matched = min(lot.size, remaining)
                pnl += matched * (fill.price - lot.price)
                lot.size -= matched
                remaining -= matched
                if lot.size <= 1e-9:
                    lots.pop(0)
            if remaining > 1e-9:
                # short position extends beyond current inventory; treat as new short lot
                lots.insert(0, PositionLot(size=-remaining, price=fill.price))
                self.inventory[symbol] = inventory - fill.size
            else:
                self.inventory[symbol] = inventory - fill.size
            self.realized_pnl[symbol] += pnl
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
        total_size = sum(max(lot.size, 0.0) for lot in lots)
        if total_size <= 1e-9:
            return self.last_mid.get(symbol) or 0.0
        total_cost = sum(max(lot.size, 0.0) * lot.price for lot in lots)
        return total_cost / total_size

    def _check_limits(self, symbol: str) -> None:
        inventory = self.inventory.get(symbol, 0.0)
        warn_level = self.config.warn_fraction * max(
            self.config.max_long, abs(self.config.max_short)
        )
        if abs(inventory) >= warn_level:
            message = f"Inventory warning: {inventory} units on {symbol}"
            if not self.warnings or self.warnings[-1] != message:
                self.warnings.append(message)
        if inventory > self.config.max_long or inventory < self.config.max_short:
            if self.config.halt_on_breach:
                self.strategy_halted = True


__all__ = ["RiskConfig", "RiskEngine", "PositionLot"]
