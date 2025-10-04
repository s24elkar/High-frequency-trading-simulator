from __future__ import annotations

import time

from python.backtester import ReplayConfig, ReplayEngine
from python.backtester.backtester import MarketEvent


def _event(ts: int) -> MarketEvent:
    return MarketEvent(timestamp_ns=ts, event_type="trade", payload={"symbol": "XYZ", "price": 100.0, "size": 1.0})


def test_replay_engine_preserves_order_and_limits_events() -> None:
    events = [_event(i * 1_000_000) for i in range(5)]
    engine = ReplayEngine(ReplayConfig(speed=0.0, real_time=False, max_events=3))
    output = list(engine.stream(events))
    assert len(output) == 3
    assert [evt.timestamp_ns for evt in output] == [0, 1_000_000, 2_000_000]


def test_replay_engine_real_time_delay(monkeypatch) -> None:
    events = [_event(0), _event(1_000_000)]
    timestamps = []

    def fake_sleep(duration: float) -> None:
        timestamps.append(duration)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    engine = ReplayEngine(ReplayConfig(speed=2.0, real_time=True))
    list(engine.stream(events))
    assert timestamps
    expected = 0.0005  # 1e6 ns = 0.001 s / speed 2.0
    assert abs(timestamps[0] - expected) < 0.0004
