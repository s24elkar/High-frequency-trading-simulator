"""Lightweight risk dashboard for streaming backtests."""

from __future__ import annotations

import datetime as _dt
import sys
from dataclasses import dataclass
from typing import Optional, TextIO

from .logging import MetricsLogger, MetricsSnapshot
from .risk import RiskEngine, RiskSnapshot


@dataclass(slots=True)
class DashboardConfig:
    symbol: str
    render_every: int = 1
    clear_screen: bool = True


class RiskDashboard:
    """Terminal-oriented dashboard printing live risk telemetry."""

    def __init__(
        self,
        stream: TextIO | None = None,
        config: DashboardConfig | None = None,
    ) -> None:
        self.stream: TextIO = stream or sys.stdout
        self.config = config
        self._risk_engine: RiskEngine | None = None
        self._logger: MetricsLogger | None = None
        self._symbol: str | None = None
        self._bound = False
        self._updates = 0

    def bind(
        self,
        *,
        symbol: str,
        risk_engine: RiskEngine | None,
        metrics_logger: MetricsLogger,
    ) -> None:
        if self.config is None:
            self.config = DashboardConfig(symbol=symbol)
        elif self.config.symbol != symbol:
            self.config = DashboardConfig(
                symbol=symbol,
                render_every=self.config.render_every,
                clear_screen=self.config.clear_screen,
            )
        self._symbol = symbol
        self._risk_engine = risk_engine
        self._logger = metrics_logger
        self._bound = True

    def update(self, timestamp_ns: int) -> None:
        if not self._bound or self.config is None:
            raise RuntimeError("RiskDashboard must be bound via bind() before use")
        self._updates += 1
        if self._updates % max(1, self.config.render_every) != 0:
            return
        risk_snapshot: Optional[RiskSnapshot] = None
        if self._risk_engine is not None and self._symbol is not None:
            risk_snapshot = self._risk_engine.snapshot(self._symbol, timestamp_ns)
        assert self._logger is not None  # bind guarantees this
        metrics_snapshot = self._logger.snapshot()
        output = self._format(timestamp_ns, risk_snapshot, metrics_snapshot)
        if self.config.clear_screen:
            self.stream.write("\x1b[2J\x1b[H")
        self.stream.write(output)
        if not output.endswith("\n"):
            self.stream.write("\n")
        self.stream.flush()

    def _format(
        self,
        timestamp_ns: int,
        risk_snapshot: Optional[RiskSnapshot],
        metrics_snapshot: MetricsSnapshot,
    ) -> str:
        symbol = self.config.symbol if self.config else (self._symbol or "-")
        wall_clock = _dt.datetime.utcfromtimestamp(timestamp_ns / 1e9)
        lines = [
            f"Risk Dashboard :: {symbol}",
            f"Timestamp (ns): {timestamp_ns}",
            f"UTC: {wall_clock.isoformat()}Z",
        ]

        if risk_snapshot is None:
            lines.append("Risk metrics unavailable (no risk engine bound)")
        else:
            mid = (
                f"{risk_snapshot.mid_price:.4f}"
                if risk_snapshot.mid_price is not None
                else "N/A"
            )
            lines.extend(
                [
                    f"Inventory: {risk_snapshot.inventory:.2f}",
                    (
                        "PnL (real/unreal/total): "
                        f"{risk_snapshot.realized_pnl:.2f} / "
                        f"{risk_snapshot.unrealized_pnl:.2f} / "
                        f"{risk_snapshot.total_pnl:.2f}"
                    ),
                    f"Mid price: {mid}",
                    f"Exposure: {risk_snapshot.notional_exposure:.2f}",
                    f"Strategy halted: {'YES' if risk_snapshot.halted else 'NO'}",
                ]
            )
            if risk_snapshot.warnings:
                lines.append("Alerts:")
                for message in risk_snapshot.warnings[-3:]:
                    lines.append(f" - {message}")

        latency_line = "Latency ns (avg/p95/max): "
        if metrics_snapshot.avg_latency_ns is None:
            latency_line += "N/A"
        else:
            avg = metrics_snapshot.avg_latency_ns
            p95 = metrics_snapshot.p95_latency_ns
            p95_str = f"{p95}" if p95 is not None else "N/A"
            max_str = (
                f"{metrics_snapshot.max_latency_ns}"
                if metrics_snapshot.max_latency_ns is not None
                else "N/A"
            )
            latency_line += f"{avg:.0f}/{p95_str}/{max_str}"
        lines.extend(
            [
                f"Orders/Fills: {metrics_snapshot.order_count} / {metrics_snapshot.fill_count}",
                latency_line,
            ]
        )

        latency = metrics_snapshot.latency_breakdown
        if latency.last_market_to_submit_us is None:
            lines.append(
                "Latency us (m->d / d->s / total): N/A"
            )
        else:
            lines.append(
                "Latency us (m->d / d->s / total): "
                f"{latency.last_market_to_decision_us:.2f} / "
                f"{latency.last_decision_to_submit_us:.2f} / "
                f"{latency.last_market_to_submit_us:.2f}"
            )
            lines.append(
                "Avg latency us (m->d / d->s / total): "
                f"{latency.avg_market_to_decision_us:.2f} / "
                f"{latency.avg_decision_to_submit_us:.2f} / "
                f"{latency.avg_market_to_submit_us:.2f}"
            )

        return "\n".join(lines) + "\n"


__all__ = ["RiskDashboard", "DashboardConfig"]
