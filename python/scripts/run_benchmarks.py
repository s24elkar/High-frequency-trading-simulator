"""Batch benchmark runner producing throughput and latency metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from python.backtester import (  # type: ignore[import-not-found]
    Backtester,
    BacktesterConfig,
    MetricsLogger,
    RiskConfig,
    RiskEngine,
    TimingSummary,
)
from python.backtester.order_book import load_order_book  # type: ignore[import-not-found]
from python.backtester.itch import (  # type: ignore[import-not-found]
    load_lobster_csv,
    replay_from_lobster,
)
from python.backtester.order_book import load_order_book  # type: ignore[import-not-found]
from python.strategies.market_maker import (  # type: ignore[import-not-found]
    MarketMakingConfig,
    MarketMakingStrategy,
)
from python.analysis import ArtifactWriter, ReportMetadata, detect_git_commit, plot_metric_bars


@dataclass(slots=True)
class BenchmarkResult:
    label: str
    events: int
    wall_time_s: float
    throughput_msg_s: float
    latency_avg_us: float | None
    latency_p95_us: float | None
    latency_p99_us: float | None
    latency_max_us: float | None
    matching_avg_ns: float | None
    matching_p95_ns: int | None
    matching_p99_ns: int | None
    matching_max_ns: int | None
    message_avg_ns: float | None
    message_p95_ns: int | None
    message_p99_ns: int | None
    message_max_ns: int | None
    digest: str


_CSV_FIELDS: List[str] = [
    "label",
    "events",
    "wall_time_s",
    "throughput_msg_s",
    "latency_avg_us",
    "latency_p95_us",
    "latency_p99_us",
    "latency_max_us",
    "matching_avg_ns",
    "matching_p95_ns",
    "matching_p99_ns",
    "matching_max_ns",
    "message_avg_ns",
    "message_p95_ns",
    "message_p99_ns",
    "message_max_ns",
    "digest",
]


def _latency_us(value_ns: float | int | None) -> float | None:
    if value_ns is None:
        return None
    return float(value_ns) / 1_000.0


def _timing_metrics(summary: Dict[str, object], key: str):
    entry = summary.get(key)
    if not entry:
        return None, None, None, None
    avg_ns = entry.get("avg_ns")
    p95_ns = entry.get("p95_ns")
    p99_ns = entry.get("p99_ns")
    max_ns = entry.get("max_ns")
    return (
        float(avg_ns) if avg_ns is not None else None,
        int(p95_ns) if p95_ns is not None else None,
        int(p99_ns) if p99_ns is not None else None,
        int(max_ns) if max_ns is not None else None,
    )


def run_case(
    *,
    label: str,
    message_count: int,
    data_path: Path,
    symbol: str,
    seed: int,
) -> BenchmarkResult:
    metrics = MetricsLogger()
    config = BacktesterConfig(symbol=symbol, book_depth=10, record_snapshots=False)
    strategy = MarketMakingStrategy(
        MarketMakingConfig(spread_ticks=2, quote_size=5, update_interval_ns=200_000)
    )
    risk_engine = RiskEngine(RiskConfig(symbol=symbol, max_long=100.0, max_short=-100.0))
    order_book = load_order_book(depth=10)
    backtester = Backtester(
        config=config,
        limit_book=order_book,
        metrics_logger=metrics,
        risk_engine=risk_engine,
        strategy=strategy,
        seed=seed,
    )

    messages = load_lobster_csv(data_path, symbol=symbol)
    events = replay_from_lobster(islice(messages, message_count))

    start = time.perf_counter()
    backtester.run(events)
    duration = time.perf_counter() - start

    snapshot = metrics.snapshot()
    timings: Dict[str, Dict[str, float | int | str | None]] = {}
    for key, summary in snapshot.timings.items():
        if isinstance(summary, TimingSummary):
            timings[key] = asdict(summary)
        else:  # pragma: no cover - defensive
            timings[key] = dict(summary)

    metrics.close()

    throughput = message_count / duration if duration > 0 else 0.0
    matching_avg, matching_p95, matching_p99, matching_max = _timing_metrics(
        timings, "matching"
    )
    message_avg, message_p95, message_p99, message_max = _timing_metrics(
        timings, "message_handling"
    )

    return BenchmarkResult(
        label=label,
        events=message_count,
        wall_time_s=duration,
        throughput_msg_s=throughput,
        latency_avg_us=_latency_us(snapshot.avg_latency_ns),
        latency_p95_us=_latency_us(snapshot.p95_latency_ns),
        latency_p99_us=_latency_us(snapshot.p99_latency_ns),
        latency_max_us=_latency_us(snapshot.max_latency_ns),
        matching_avg_ns=matching_avg,
        matching_p95_ns=matching_p95,
        matching_p99_ns=matching_p99,
        matching_max_ns=matching_max,
        message_avg_ns=message_avg,
        message_p95_ns=message_p95,
        message_p99_ns=message_p99,
        message_max_ns=message_max,
        digest=backtester.digest,
    )
def main() -> None:
    parser = argparse.ArgumentParser(description="Run order-book throughput benchmarks")
    parser.add_argument(
        "--baseline",
        type=int,
        default=2_000,
        help="Baseline message count (default: 2,000)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/lobster/LOBSTER_SampleFile_AAPL_2012-06-21_10"
            "/AAPL_2012-06-21_34200000_57600000_message_10.csv"
        ),
        help="LOBSTER message CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/week4/perf/benchmarks.json"),
        help="Where to write the primary JSON results (default: results/week4/perf/benchmarks.json)",
    )
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing artefacts",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("results/week4/plots/benchmark_throughput.png"),
        help="Path to save the throughput summary plot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the throughput summary plot",
    )
    args = parser.parse_args()

    scenarios = {
        "baseline": args.baseline,
        "x10": args.baseline * 10,
        "x100": args.baseline * 100,
    }

    results: List[BenchmarkResult] = []
    for label, count in scenarios.items():
        result = run_case(
            label=label,
            message_count=count,
            data_path=args.data,
            symbol=args.symbol,
            seed=args.seed,
        )
        print(
            f"{label:<8} :: {result.events:>7} events | "
            f"{result.wall_time_s:6.3f}s | {result.throughput_msg_s:9.1f} msg/s"
        )
        results.append(result)

    output_dir = args.output.parent
    metadata = ReportMetadata(
        generator="run_benchmarks",
        git_commit=detect_git_commit(REPO_ROOT),
        seed=args.seed,
        extra={
            "baseline_messages": args.baseline,
            "symbol": args.symbol,
            "data_path": str(args.data),
        },
    )
    writer = ArtifactWriter(output_dir, metadata, overwrite=args.overwrite)
    scenario_definitions = [
        {"label": label, "message_count": count} for label, count in scenarios.items()
    ]
    payload = {
        "run_config": {
            "symbol": args.symbol,
            "seed": args.seed,
            "baseline_messages": args.baseline,
            "data_path": str(args.data),
        },
        "scenarios": scenario_definitions,
        "results": [asdict(result) for result in results],
    }
    writer.write_json(args.output.name, payload)
    csv_rows = [{field: getattr(result, field) for field in _CSV_FIELDS} for result in results]
    writer.write_csv(f"{args.output.stem}.csv", csv_rows, headers=_CSV_FIELDS)
    if not args.no_plot:
        plot_path = args.plot_output
        labels = [result.label for result in results]
        throughputs = [float(result.throughput_msg_s) for result in results]
        plot_metric_bars(
            labels,
            throughputs,
            title="Benchmark throughput",
            ylabel="Messages per second",
            output_path=plot_path,
            overwrite=args.overwrite,
        )
        writer.attach_metadata(plot_path, relative=False)


if __name__ == "__main__":
    main()
