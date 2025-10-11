"""Stress-testing utilities for exercising the Python order book under load."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import math

try:  # pragma: no cover - allow import when package layout differs
    from ..analysis import Hotspot, profile_capture, stats_to_hotspots
except (ImportError, ValueError):  # pragma: no cover - fallback for flat layout
    from analysis import Hotspot, profile_capture, stats_to_hotspots  # type: ignore[import-not-found]

from .backtester import MarketEvent, MarketSnapshot
from .order_book import PythonOrderBook
from .synthetic import (
    BurstConfig,
    PoissonOrderFlowConfig,
    PoissonOrderFlowGenerator,
    SequenceValidationReport,
    SequenceValidator,
)

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
    poisson: Optional[PoissonOrderFlowConfig] = None
    burst: Optional[BurstConfig] = None
    validate_sequence: bool = False
    record_latency: bool = False


@dataclass(slots=True)
class LatencyHistogramBin:
    upper_ns: Optional[int]
    count: int


@dataclass(slots=True)
class StressMetrics:
    wall_time_s: float
    peak_memory_kb: float
    message_count: int
    final_depth: int
    hotspots: List[Hotspot]
    sequence_report: Optional[SequenceValidationReport] = None
    avg_latency_ns: Optional[float] = None
    p95_latency_ns: Optional[int] = None
    p99_latency_ns: Optional[int] = None
    max_latency_ns: Optional[int] = None
    latency_histogram: Optional[List[LatencyHistogramBin]] = None
    add_order_events: Optional[int] = None
    delete_order_events: Optional[int] = None
    execute_order_events: Optional[int] = None


def _build_event(timestamp_ns: int, event_type: str, payload: dict) -> MarketEvent:
    return MarketEvent(
        timestamp_ns=timestamp_ns, event_type=event_type, payload=payload
    )


def _apply_event_with_latency(
    book: PythonOrderBook, event: MarketEvent, latencies: Optional[List[int]]
) -> None:
    if latencies is None:
        book.apply_event(event)
        return
    start_ns = time.perf_counter_ns()
    book.apply_event(event)
    latencies.append(time.perf_counter_ns() - start_ns)


_LATENCY_BOUNDS_NS: List[int] = [
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
]


def _latency_histogram_from_sorted(
    sorted_latencies: List[int],
) -> List[LatencyHistogramBin]:
    if not sorted_latencies:
        return []

    max_latency = sorted_latencies[-1]
    bounds: List[int] = list(_LATENCY_BOUNDS_NS)
    while bounds[-1] < max_latency:
        bounds.append(bounds[-1] * 2)

    histogram: List[LatencyHistogramBin] = []
    idx = 0
    total = len(sorted_latencies)
    for limit in bounds:
        count = 0
        while idx < total and sorted_latencies[idx] <= limit:
            idx += 1
            count += 1
        histogram.append(LatencyHistogramBin(upper_ns=limit, count=count))
        if idx >= total:
            break
    if idx < total:
        histogram.append(LatencyHistogramBin(upper_ns=None, count=total - idx))
    return histogram


def _percentile(sorted_values: List[int], fraction: float) -> Optional[int]:
    if not sorted_values:
        return None
    index = max(
        0, min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1)
    )
    return sorted_values[index]


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
    processed = 0
    event_type_counts = {
        "add_order": 0,
        "delete_order": 0,
        "execute_order": 0,
    }
    sequence_validator: SequenceValidator | None = (
        SequenceValidator() if config.validate_sequence else None
    )
    latencies_ns: Optional[List[int]] = [] if config.record_latency else None

    with profile_capture(profiler_output, print_limit=25) as profile_result:
        if config.poisson is not None:
            generator = PoissonOrderFlowGenerator(
                config.poisson, burst_config=config.burst
            )
            for event in generator.stream(validator=sequence_validator):
                if event.event_type in event_type_counts:
                    event_type_counts[event.event_type] += 1
                _apply_event_with_latency(book, event, latencies_ns)
                processed += 1
        else:
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
                    size = max(
                        0.01, rng.uniform(config.max_size * 0.1, config.max_size)
                    )
                    payload = {
                        "order_id": order_id,
                        "symbol": config.symbol,
                        "side": side,
                        "price": price,
                        "size": size,
                    }
                    event = _build_event(timestamp_ns, "add_order", payload)
                    if sequence_validator is not None:
                        sequence_validator.observe(event)
                    _apply_event_with_latency(book, event, latencies_ns)
                    event_type_counts["add_order"] += 1
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
                    if sequence_validator is not None:
                        sequence_validator.observe(event)
                    _apply_event_with_latency(book, event, latencies_ns)
                    event_type_counts["delete_order"] += 1
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
                    if sequence_validator is not None:
                        sequence_validator.observe(event)
                    _apply_event_with_latency(book, event, latencies_ns)
                    event_type_counts["execute_order"] += 1
                    if active_orders:
                        active_orders.pop(exec_idx)
                processed += 1

    hotspots = stats_to_hotspots(profile_result.stats, limit=10)

    snapshot: MarketSnapshot = book.snapshot(config.depth)
    final_depth = len(snapshot.depth)

    total_messages = processed
    if total_messages == 0:
        total_messages = (
            config.poisson.message_count
            if config.poisson is not None
            else config.message_count
        )

    avg_latency = None
    p95_latency = None
    p99_latency = None
    max_latency = None
    latency_histogram: Optional[List[LatencyHistogramBin]] = None
    if latencies_ns:
        sorted_latencies = sorted(latencies_ns)
        count = len(sorted_latencies)
        if count:
            avg_latency = sum(sorted_latencies) / count
            p95_latency = _percentile(sorted_latencies, 0.95)
            p99_latency = _percentile(sorted_latencies, 0.99)
            max_latency = sorted_latencies[-1]
            latency_histogram = _latency_histogram_from_sorted(sorted_latencies)

    metrics = StressMetrics(
        wall_time_s=profile_result.wall_time_s,
        peak_memory_kb=profile_result.peak_memory_kb,
        message_count=total_messages,
        final_depth=final_depth,
        hotspots=hotspots,
        sequence_report=(
            sequence_validator.report() if sequence_validator is not None else None
        ),
        avg_latency_ns=avg_latency,
        p95_latency_ns=p95_latency,
        p99_latency_ns=p99_latency,
        max_latency_ns=max_latency,
        latency_histogram=latency_histogram,
        add_order_events=event_type_counts.get("add_order"),
        delete_order_events=event_type_counts.get("delete_order"),
        execute_order_events=event_type_counts.get("execute_order"),
    )

    log.info(
        "Order book stress complete: messages=%s wall_time=%.3fs peak_mem=%.1fKB",
        metrics.message_count,
        metrics.wall_time_s,
        metrics.peak_memory_kb,
    )

    return metrics


__all__ = [
    "StressConfig",
    "StressMetrics",
    "Hotspot",
    "LatencyHistogramBin",
    "run_order_book_stress",
]
