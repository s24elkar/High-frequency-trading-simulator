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
    max_notional_exposure: float | None = None
    loss_limit: float | None = None
    halt_on_breach: bool = True
    warn_fraction: float = 0.8  # warn when metrics exceed warn_fraction of limit


@dataclass(slots=True)
class PositionLot:
    size: float
    price: float


@dataclass(slots=True)
class RiskSnapshot:
    symbol: str
    timestamp_ns: int
    inventory: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    mid_price: float | None
    notional_exposure: float
    halted: bool
    warnings: List[str]


class RiskEngine:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.inventory: Dict[str, float] = {config.symbol: 0.0}
        self.realized_pnl: Dict[str, float] = {config.symbol: 0.0}
        self.unrealized_pnl: Dict[str, float] = {config.symbol: 0.0}
        self.lots: Dict[str, List[PositionLot]] = {config.symbol: []}
        self.last_mid: Dict[str, float | None] = {config.symbol: None}
        self.notional_exposure: Dict[str, float] = {config.symbol: 0.0}
        self.strategy_halted = False
        self.warnings: List[str] = []
        self.alerts: List[str] = []

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
            self.notional_exposure[symbol] = 0.0
            self._check_limits(symbol)
            return
        cost_basis = self._average_cost(symbol)
        self.unrealized_pnl[symbol] = inventory * (mid - cost_basis)
        self.notional_exposure[symbol] = abs(inventory) * mid
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
        mid = self.last_mid.get(symbol)
        realized = self.realized_pnl.get(symbol, 0.0)
        unrealized = self.unrealized_pnl.get(symbol, 0.0)
        total_pnl = realized + unrealized
        exposure = abs(inventory) * (mid if mid is not None else 0.0)
        self.notional_exposure[symbol] = exposure

        warn_long = self.config.max_long * self.config.warn_fraction
        warn_short = self.config.max_short * self.config.warn_fraction

        if self.config.max_long > 0 and inventory >= warn_long:
            self._emit_warning(f"Inventory warning: {inventory} units on {symbol}")
        if self.config.max_short < 0 and inventory <= warn_short:
            self._emit_warning(f"Inventory warning: {inventory} units on {symbol}")

        if self.config.max_long > 0 and inventory > self.config.max_long:
            self._halt(f"Inventory limit breached: {inventory} > {self.config.max_long}")
        if self.config.max_short < 0 and inventory < self.config.max_short:
            self._halt(f"Inventory limit breached: {inventory} < {self.config.max_short}")

        notional_limit = self.config.max_notional_exposure
        if notional_limit is not None and notional_limit > 0:
            warn_notional = notional_limit * self.config.warn_fraction
            if exposure >= warn_notional:
                self._emit_warning(
                    f"Exposure warning: {exposure:.2f} notional on {symbol}"
                )
            if exposure > notional_limit:
                self._halt(
                    f"Exposure limit breached: {exposure:.2f} > {notional_limit:.2f}"
                )

        loss_limit = self.config.loss_limit
        if loss_limit is not None:
            warn_loss = loss_limit * self.config.warn_fraction
            if loss_limit < 0:
                if total_pnl <= warn_loss:
                    self._emit_warning(
                        f"Loss warning: PnL {total_pnl:.2f} below {warn_loss:.2f}"
                    )
                if total_pnl <= loss_limit:
                    self._halt(
                        f"Loss limit breached: PnL {total_pnl:.2f} <= {loss_limit:.2f}"
                    )
            else:
                if total_pnl >= warn_loss:
                    self._emit_warning(
                        f"Profit cap warning: PnL {total_pnl:.2f} above {warn_loss:.2f}"
                    )
                if total_pnl >= loss_limit:
                    self._halt(
                        f"Profit cap breached: PnL {total_pnl:.2f} >= {loss_limit:.2f}"
                    )

    def snapshot(self, symbol: str, timestamp_ns: int) -> RiskSnapshot:
        inventory = self.inventory.get(symbol, 0.0)
        realized = self.realized_pnl.get(symbol, 0.0)
        unrealized = self.unrealized_pnl.get(symbol, 0.0)
        mid = self.last_mid.get(symbol)
        exposure = self.notional_exposure.get(symbol, 0.0)
        return RiskSnapshot(
            symbol=symbol,
            timestamp_ns=timestamp_ns,
            inventory=inventory,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            total_pnl=realized + unrealized,
            mid_price=mid,
            notional_exposure=exposure,
            halted=self.strategy_halted,
            warnings=list(self.warnings),
        )

    def _emit_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _halt(self, message: str) -> None:
        self.strategy_halted = self.strategy_halted or self.config.halt_on_breach
        if message not in self.alerts:
            self.alerts.append(message)
        self._emit_warning(message)

    @staticmethod
    def _has_opposite_sign(a: float, b: float) -> bool:
        return (a > 0 > b) or (a < 0 < b)


__all__ = ["RiskConfig", "RiskEngine", "PositionLot", "RiskSnapshot"]
