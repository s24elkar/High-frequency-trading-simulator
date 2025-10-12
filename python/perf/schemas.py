"""Tabular schema references for exported performance artefacts."""

from __future__ import annotations

STRESS_SCENARIO_FIELDS: tuple[str, ...] = (
    "multiplier",
    "message_count",
    "wall_time_s",
    "throughput_msgs_per_s",
    "avg_latency_ns",
    "p95_latency_ns",
    "p99_latency_ns",
    "max_latency_ns",
    "add_order_events",
    "delete_order_events",
    "execute_order_events",
    "order_to_trade_ratio",
    "orphan_cancels",
    "orphan_executes",
    "duplicate_order_ids",
)

LATENCY_HISTOGRAM_FIELDS: tuple[str, ...] = ("upper_ns", "count")

THROUGHPUT_SERIES_FIELDS: tuple[str, ...] = (
    "run_id",
    "bucket_idx",
    "bucket_start_ns",
    "bucket_start_ms",
    "messages",
    "throughput_msgs_per_s",
)

PERF_RUN_SUMMARY_FIELDS: tuple[str, ...] = (
    "run_id",
    "symbol",
    "realized_pnl",
    "unrealized_pnl",
    "inventory",
    "order_volume",
    "fill_volume",
    "order_to_trade_ratio",
    "fill_efficiency",
    "avg_latency_ns",
    "p95_latency_ns",
    "p99_latency_ns",
    "max_latency_ns",
    "duration_ns",
    "digest",
)


def normalise_row(
    row: dict[str, object], fields: tuple[str, ...]
) -> dict[str, object | str]:
    """Project a dict onto the requested schema, filling missing values with blanks."""
    return {
        field: row.get(field, "") if row.get(field) is not None else ""
        for field in fields
    }


__all__ = [
    "LATENCY_HISTOGRAM_FIELDS",
    "PERF_RUN_SUMMARY_FIELDS",
    "STRESS_SCENARIO_FIELDS",
    "THROUGHPUT_SERIES_FIELDS",
    "normalise_row",
]
