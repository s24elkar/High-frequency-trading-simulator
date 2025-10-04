"""Market replay utilities for streaming historical events through the simulator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

from .backtester import MarketEvent


@dataclass(slots=True)
class ReplayConfig:
    """Configuration controlling replay speed and behavior."""

    speed: float = 1.0  # 1.0 = real time, >1.0 accelerated, 0 => asap
    real_time: bool = True
    max_events: Optional[int] = None


class ReplayEngine:
    """Replays `MarketEvent` streams at configurable speeds."""

    def __init__(
        self,
        config: ReplayConfig,
        *,
        time_source: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.config = config
        self._time_source = time_source or time.perf_counter
        self._sleep = sleeper or time.sleep

    def stream(self, events: Iterable[MarketEvent]) -> Iterator[MarketEvent]:
        last_ts: Optional[int] = None
        last_wall = self._time_source()
        count = 0
        for event in events:
            if self.config.max_events is not None and count >= self.config.max_events:
                break
            if last_ts is not None and self.config.real_time and self.config.speed > 0:
                delta_ns = event.timestamp_ns - last_ts
                target_seconds = delta_ns / 1e9 / max(self.config.speed, 1e-9)
                elapsed = self._time_source() - last_wall
                to_sleep = target_seconds - elapsed
                if to_sleep > 0:
                    self._sleep(to_sleep)
            yield event
            last_ts = event.timestamp_ns
            last_wall = self._time_source()
            count += 1


def replay(
    events: Iterable[MarketEvent], config: ReplayConfig
) -> Iterator[MarketEvent]:
    """Convenience function producing replayed events."""

    engine = ReplayEngine(config)
    return engine.stream(events)


__all__ = ["ReplayEngine", "ReplayConfig", "replay"]
