from __future__ import annotations

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
    sleep_calls = []

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def time(self) -> float:
            return self.now

        def sleep(self, duration: float) -> None:
            sleep_calls.append(duration)
            self.now += duration

    fake = FakeClock()

    engine = ReplayEngine(
        ReplayConfig(speed=2.0, real_time=True),
        time_source=fake.time,
        sleeper=fake.sleep,
    )
    list(engine.stream(events))
    assert sleep_calls
    expected = 0.0005  # 1e6 ns = 0.001 s / speed 2.0
    assert abs(sleep_calls[0] - expected) < 1e-6
