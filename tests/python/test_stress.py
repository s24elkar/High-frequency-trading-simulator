from __future__ import annotations

from pathlib import Path

from python.backtester import StressConfig, run_order_book_stress


def test_order_book_stress_metrics_smoke(tmp_path: Path) -> None:
    config = StressConfig(message_count=500, seed=42, max_price_jitter=0.5, max_size=5.0)
    profile_path = tmp_path / "profile.txt"
    metrics = run_order_book_stress(config, profiler_output=profile_path)

    assert metrics.message_count == config.message_count
    assert metrics.wall_time_s >= 0.0
    assert metrics.peak_memory_kb >= 0.0
    assert metrics.final_depth >= 0
    assert metrics.hotspots, "Expected at least one hotspot entry"
    assert profile_path.exists()
    assert "ncalls" in profile_path.read_text(encoding="utf-8")


def test_order_book_stress_hotspots_ranked() -> None:
    config = StressConfig(message_count=300, seed=7)
    metrics = run_order_book_stress(config)

    assert metrics.hotspots[0].cumulative_time_s >= metrics.hotspots[-1].cumulative_time_s
    assert all(h.total_calls >= h.primitive_calls for h in metrics.hotspots)
