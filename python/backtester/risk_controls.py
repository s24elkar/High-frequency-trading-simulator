"""Risk-control primitives such as rate limiters and throttles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass(slots=True)
class RateLimitConfig:
    """Configuration for sliding-window rate limiting."""

    max_actions: int
    interval_ns: int
    name: str = "orders"
    halt_on_violation: bool = True

    def __post_init__(self) -> None:
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if self.interval_ns <= 0:
            raise ValueError("interval_ns must be positive")


@dataclass(slots=True)
class RiskControlViolation:
    """Structured description of a risk-control breach."""

    kind: str
    timestamp_ns: int
    message: str
    limit: RateLimitConfig | None = None
    metadata: dict | None = None


class SlidingWindowRateLimiter:
    """Simple sliding-window limiter for order or cancel actions."""

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._events: Deque[int] = deque()
        self.violation_count = 0

    def allow(self, timestamp_ns: int) -> bool:
        interval = self.config.interval_ns
        while self._events and timestamp_ns - self._events[0] >= interval:
            self._events.popleft()
        if len(self._events) >= self.config.max_actions:
            self.violation_count += 1
            return False
        self._events.append(timestamp_ns)
        return True

    def reset(self) -> None:
        self._events.clear()
        self.violation_count = 0

    def window_size(self) -> int:
        return len(self._events)


__all__ = ["RateLimitConfig", "RiskControlViolation", "SlidingWindowRateLimiter"]
