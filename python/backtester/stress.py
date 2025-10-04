"""Stress-testing utilities for exercising the Python order book under load."""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import random
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .backtester import MarketEvent, MarketSnapshot
from .order_book import PythonOrderBook

log = logging.getLogger(__name__)


@dataclass(slots=True)
class StressConfig:
    symbol: str = "STRESS"
    message_count: int = 10_000
    base_price: float = 100.0
    max_price_jitter: float = 1.0
    max_size: float = 10.0
    depth: int = 10
    seed: int = 0
    cancel_ratio: float = 0.2
    execute_ratio: float = 0.2


@dataclass(slots=True)
class Hotspot:
    location: str
    primitive_calls: int
    total_calls: int
    cumulative_time_s: float


@dataclass(slots=True)
class StressMetrics:
    wall_time_s: float
    peak_memory_kb: float
    message_count: int
    final_depth: int
    hotspots: List[Hotspot]


def _build_event(timestamp_ns: int, event_type: str, payload: dict) -> MarketEvent:
    return MarketEvent(
        timestamp_ns=timestamp_ns, event_type=event_type, payload=payload
    )


def run_order_book_stress(
    config: StressConfig,
    profiler_output: Path | str | None = None,
) -> StressMetrics:
    """Replay synthetic market updates against the Python order book.

    Parameters
    ----------
    config
        Stress test configuration describing the number of messages and price/size ranges.
    profiler_output
        Optional path to write a `pstats` table for offline inspection.
    """

    rng = random.Random(config.seed)
    book = PythonOrderBook(depth=config.depth)
    active_orders: List[tuple[int, str, float, float]] = []
    order_id = 1

    tracemalloc.start()
    start = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        for idx in range(config.message_count):
            action = rng.random()
            timestamp_ns = idx
            if (
                action < 1.0 - (config.cancel_ratio + config.execute_ratio)
                or not active_orders
            ):
                side = rng.choice(["BUY", "SELL"])
                price_offset = rng.uniform(
                    -config.max_price_jitter, config.max_price_jitter
                )
                price = max(0.01, config.base_price + price_offset)
                size = max(0.01, rng.uniform(config.max_size * 0.1, config.max_size))
                payload = {
                    "order_id": order_id,
                    "symbol": config.symbol,
                    "side": side,
                    "price": price,
                    "size": size,
                }
                event = _build_event(timestamp_ns, "add_order", payload)
                book.apply_event(event)
                active_orders.append((order_id, side, price, size))
                order_id += 1
            elif action < 1.0 - config.execute_ratio and active_orders:
                cancel_idx = rng.randrange(len(active_orders))
                cancel_order_id, side, price, size = active_orders.pop(cancel_idx)
                payload = {
                    "order_id": cancel_order_id,
                    "symbol": config.symbol,
                    "side": side,
                    "price": price,
                    "size": size,
                }
                event = _build_event(timestamp_ns, "delete_order", payload)
                book.apply_event(event)
            else:
                exec_idx = rng.randrange(len(active_orders))
                exec_order_id, side, price, size = active_orders[exec_idx]
                take_side = "SELL" if side == "BUY" else "BUY"
                payload = {
                    "order_id": exec_order_id,
                    "symbol": config.symbol,
                    "side": take_side,
                    "price": price,
                    "size": size,
                }
                event = _build_event(timestamp_ns, "execute_order", payload)
                book.apply_event(event)
                if active_orders:
                    active_orders.pop(exec_idx)
    finally:
        profiler.disable()
        wall_time_s = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    hotspots = _collect_hotspots(stats)

    if profiler_output is not None:
        profiler_path = Path(profiler_output)
        profiler_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        stats.stream = buffer
        stats.print_stats(25)
        profiler_path.write_text(buffer.getvalue(), encoding="utf-8")

    snapshot: MarketSnapshot = book.snapshot(config.depth)
    final_depth = len(snapshot.depth)

    metrics = StressMetrics(
        wall_time_s=wall_time_s,
        peak_memory_kb=peak / 1024.0,
        message_count=config.message_count,
        final_depth=final_depth,
        hotspots=hotspots,
    )

    log.info(
        "Order book stress complete: messages=%s wall_time=%.3fs peak_mem=%.1fKB",
        metrics.message_count,
        metrics.wall_time_s,
        metrics.peak_memory_kb,
    )

    return metrics


def _collect_hotspots(stats: pstats.Stats, limit: int = 10) -> List[Hotspot]:
    entries: List[Hotspot] = []
    for func, (primitive_calls, total_calls, _, cumulative_time, _) in sorted(
        stats.stats.items(), key=lambda item: item[1][3], reverse=True
    )[:limit]:
        filename, line, name = func
        location = f"{filename}:{line}:{name}"
        entries.append(
            Hotspot(
                location=location,
                primitive_calls=primitive_calls,
                total_calls=total_calls,
                cumulative_time_s=cumulative_time,
            )
        )
    return entries


__all__ = ["StressConfig", "StressMetrics", "Hotspot", "run_order_book_stress"]
