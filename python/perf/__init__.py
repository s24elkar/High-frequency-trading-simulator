"""Performance benchmarking helpers shared across CLI scripts."""

from __future__ import annotations

from .architecture import (
    ARCHITECTURE_CSV_FIELDS,
    ArchitectureRun,
    rows_for_csv as architecture_rows_for_csv,
    summarise_by_variant,
)
from .benchmarks import (
    BENCHMARK_CSV_FIELDS,
    BenchmarkResult,
    rows_for_csv as benchmark_rows_for_csv,
)
from .schemas import (
    LATENCY_HISTOGRAM_FIELDS,
    PERF_RUN_SUMMARY_FIELDS,
    STRESS_SCENARIO_FIELDS,
    THROUGHPUT_SERIES_FIELDS,
    normalise_row,
)

__all__ = [
    "ArchitectureRun",
    "ARCHITECTURE_CSV_FIELDS",
    "architecture_rows_for_csv",
    "summarise_by_variant",
    "BenchmarkResult",
    "BENCHMARK_CSV_FIELDS",
    "benchmark_rows_for_csv",
    "LATENCY_HISTOGRAM_FIELDS",
    "PERF_RUN_SUMMARY_FIELDS",
    "STRESS_SCENARIO_FIELDS",
    "THROUGHPUT_SERIES_FIELDS",
    "normalise_row",
]
