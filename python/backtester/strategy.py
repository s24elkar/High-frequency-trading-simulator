"""Strategy interfaces and sandbox wrappers for modular trading strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .backtester import (
        FillEvent,
        MarketSnapshot,
        OrderRequest,
        StrategyContext,
    )

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TimerToken:
    """Handle representing a scheduled strategy timer."""

    timer_id: int
    key: str
    due_ns: int
    metadata: Dict[str, Any] | None = None


class StrategyCallbacks:
    """Minimal interface strategies must implement to interact with the backtester."""

    def on_start(self, ctx: "StrategyContext") -> None:  # pragma: no cover - default hook
        return None

    def on_stop(self, ctx: "StrategyContext") -> None:  # pragma: no cover - default hook
        return None

    def on_market_data(self, snapshot: "MarketSnapshot", ctx: "StrategyContext") -> None:
        return None

    def on_fill(self, fill: "FillEvent", ctx: "StrategyContext") -> None:
        return None

    def on_order_accepted(self, order: "OrderRequest", ctx: "StrategyContext") -> None:
        return None

    def on_timer(self, timer: TimerToken, ctx: "StrategyContext") -> None:
        return None


class StrategyError(RuntimeError):
    """Raised when a strategy callback throws an exception inside the sandbox."""


class StrategySandbox:
    """Isolates strategy callbacks and captures exceptions."""

    def __init__(self, strategy: StrategyCallbacks) -> None:
        self.strategy = strategy
        self.faulted = False
        self.last_error: Optional[str] = None

    def invoke(self, method: str, *args, **kwargs):
        if self.faulted:
            return None
        fn = getattr(self.strategy, method, None)
        if fn is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive path
            self.faulted = True
            self.last_error = f"{method} raised {exc!r}"
            log.exception("Strategy fault in %s", method)
            raise StrategyError(self.last_error) from exc


Strategy = StrategyCallbacks


__all__ = [
    "Strategy",
    "StrategyCallbacks",
    "StrategyError",
    "StrategySandbox",
    "TimerToken",
]
