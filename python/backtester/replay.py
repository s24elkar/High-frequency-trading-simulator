"""Market replay utilities for streaming historical events through the simulator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .backtester import MarketEvent


@dataclass(slots=True)
class ReplayConfig:
    """Configuration controlling replay speed and behavior."""

    speed: float = 1.0  # 1.0 = real time, >1.0 accelerated, 0 => asap
    real_time: bool = True
    max_events: Optional[int] = None


class ReplayEngine:
    """Replays `MarketEvent` streams at configurable speeds."""

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config

    def stream(self, events: Iterable[MarketEvent]) -> Iterator[MarketEvent]:
        last_ts: Optional[int] = None
        start_wall = time.perf_counter()
        last_wall = start_wall
        count = 0
        for event in events:
            if self.config.max_events is not None and count >= self.config.max_events:
                break
            if last_ts is not None and self.config.real_time and self.config.speed > 0:
                delta_ns = event.timestamp_ns - last_ts
                target_seconds = delta_ns / 1e9 / max(self.config.speed, 1e-9)
                elapsed = time.perf_counter() - last_wall
                to_sleep = target_seconds - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
            yield event
            last_ts = event.timestamp_ns
            last_wall = time.perf_counter()
            count += 1


def replay(events: Iterable[MarketEvent], config: ReplayConfig) -> Iterator[MarketEvent]:
    """Convenience function producing replayed events."""

    engine = ReplayEngine(config)
    return engine.stream(events)


__all__ = ["ReplayEngine", "ReplayConfig", "replay"]
