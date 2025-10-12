#!/usr/bin/env python3
"""Compare single-threaded vs concurrent backtester performance."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from python.analysis import (
        ArtifactWriter,
        ReportMetadata,
        detect_git_commit,
        plot_metric_bars,
    )
    from python.backtester import (
        Backtester,
        BacktesterConfig,
        ConcurrentBacktester,
        MetricsLogger,
        RiskConfig,
        RiskEngine,
    )
    from python.backtester.itch import (
        load_lobster_csv,
        replay_from_lobster,
    )
    from python.backtester.logging import MetricsSnapshot
    from python.backtester.order_book import load_order_book
    from python.strategies.market_maker import (
        MarketMakingConfig,
        MarketMakingStrategy,
    )
    from python.perf import (
        ARCHITECTURE_CSV_FIELDS,
        ArchitectureRun,
        architecture_rows_for_csv,
        summarise_by_variant,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for CLI usage
    sys.path.insert(0, str(REPO_ROOT))
    from python.analysis import (  # type: ignore[import-not-found]
        ArtifactWriter,
        ReportMetadata,
        detect_git_commit,
        plot_metric_bars,
    )
    from python.backtester import (  # type: ignore[import-not-found]
        Backtester,
        BacktesterConfig,
        ConcurrentBacktester,
        MetricsLogger,
        RiskConfig,
        RiskEngine,
    )
    from python.backtester.itch import (  # type: ignore[import-not-found]
        load_lobster_csv,
        replay_from_lobster,
    )
    from python.backtester.logging import (  # type: ignore[import-not-found]
        MetricsSnapshot,
    )
    from python.backtester.order_book import (  # type: ignore[import-not-found]
        load_order_book,
    )
    from python.strategies.market_maker import (  # type: ignore[import-not-found]
        MarketMakingConfig,
        MarketMakingStrategy,
    )
    from python.perf import (  # type: ignore[import-not-found]
        ARCHITECTURE_CSV_FIELDS,
        ArchitectureRun,
        architecture_rows_for_csv,
        summarise_by_variant,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-threaded vs concurrent backtester runs"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/lobster/LOBSTER_SampleFile_AAPL_2012-06-21_10"
            "/AAPL_2012-06-21_34200000_57600000_message_10.csv"
        ),
        help="LOBSTER message CSV path",
    )
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--message-count", type=int, default=20_000)
    parser.add_argument("--runs", type=int, default=3, help="Iterations per variant")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/week4/stress/analysis/architecture_comparison.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/week4/stress/analysis/architecture_comparison.csv"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("results/week4/plots/architecture_throughput.png"),
        help="Path to save the architecture throughput comparison plot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the throughput comparison plot",
    )
    return parser.parse_args()


def _latency_fields(
    snapshot: MetricsSnapshot,
) -> Tuple[float | None, float | None, float | None, float | None]:
    return (
        snapshot.avg_latency_ns,
        snapshot.p95_latency_ns,
        snapshot.p99_latency_ns,
        snapshot.max_latency_ns,
    )


def _configure_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_backtester(
    symbol: str, depth: int, seed: int, metrics: MetricsLogger
) -> Backtester:
    config = BacktesterConfig(symbol=symbol, book_depth=depth, record_snapshots=False)
    strategy = MarketMakingStrategy(
        MarketMakingConfig(spread_ticks=2, quote_size=5, update_interval_ns=200_000)
    )
    risk_engine = RiskEngine(
        RiskConfig(symbol=symbol, max_long=100.0, max_short=-100.0)
    )
    order_book = load_order_book(depth=depth)
    return Backtester(
        config=config,
        limit_book=order_book,
        metrics_logger=metrics,
        risk_engine=risk_engine,
        strategy=strategy,
        seed=seed,
    )


def _materialise_events(messages: Iterable[object], message_count: int) -> List[object]:
    sliced = list(islice(messages, message_count))
    return sliced


def run_variant(
    variant: str,
    iteration: int,
    *,
    symbol: str,
    events: List[object],
    message_count: int,
    seed: int,
) -> ArchitectureRun:
    _configure_rngs(seed)
    metrics = MetricsLogger()
    backtester = _build_backtester(symbol, depth=10, seed=seed, metrics=metrics)
    replay = replay_from_lobster(events)

    if variant == "single":
        runner = backtester
    elif variant == "concurrent":
        runner = ConcurrentBacktester(backtester)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown variant {variant}")

    start = time.perf_counter()
    runner.run(replay)  # type: ignore[arg-type]
    wall_time = time.perf_counter() - start

    snapshot = metrics.snapshot()
    timings = snapshot.timings

    metrics.close()

    throughput = message_count / wall_time if wall_time > 0 else 0.0
    avg_lat, p95_lat, p99_lat, max_lat = _latency_fields(snapshot)
    matching_summary = timings.get("matching")
    message_summary = timings.get("message_handling")
    matching_avg = matching_summary.avg_ns if matching_summary else None
    message_avg = message_summary.avg_ns if message_summary else None

    return ArchitectureRun(
        variant=variant,
        iteration=iteration,
        message_count=message_count,
        wall_time_s=wall_time,
        throughput_msg_s=throughput,
        digest=backtester.digest,
        fill_count=snapshot.fill_count,
        fill_volume=snapshot.fill_volume,
        avg_latency_ns=avg_lat,
        p95_latency_ns=p95_lat,
        p99_latency_ns=p99_lat,
        max_latency_ns=max_lat,
        matching_avg_ns=matching_avg,
        message_avg_ns=message_avg,
    )


def main() -> None:
    args = parse_args()
    messages = _materialise_events(
        load_lobster_csv(args.data, symbol=args.symbol), args.message_count
    )

    _configure_rngs(args.seed)

    results: List[ArchitectureRun] = []
    variants = ("single", "concurrent")

    for variant in variants:
        for iteration in range(args.runs):
            seed = args.seed + iteration
            run = run_variant(
                variant,
                iteration,
                symbol=args.symbol,
                events=messages,
                message_count=args.message_count,
                seed=seed,
            )
            results.append(run)
            print(
                f"{variant:<11} iter={iteration} digest={run.digest[:8]} throughput={run.throughput_msg_s:,.1f} msg/s"
            )

    summary = summarise_by_variant(results)

    payload = {
        "parameters": {
            "message_count": args.message_count,
            "runs_per_variant": args.runs,
            "data": str(args.data),
            "symbol": args.symbol,
        },
        "results": [asdict(run) for run in results],
        "summary": summary,
    }
    metadata = ReportMetadata(
        generator="benchmark_architectures",
        git_commit=detect_git_commit(REPO_ROOT),
        seed=args.seed,
        extra={
            "message_count": args.message_count,
            "runs_per_variant": args.runs,
            "data_path": str(args.data),
            "symbol": args.symbol,
        },
    )
    writer = ArtifactWriter(args.output.parent, metadata, overwrite=args.overwrite)
    writer.write_json(args.output.name, payload)

    csv_rows = architecture_rows_for_csv(results)
    csv_headers = list(ARCHITECTURE_CSV_FIELDS)
    if args.output_csv.parent == args.output.parent:
        writer.write_csv(args.output_csv.name, csv_rows, headers=csv_headers)
    else:
        csv_writer = ArtifactWriter(
            args.output_csv.parent, metadata, overwrite=args.overwrite
        )
        csv_writer.write_csv(args.output_csv.name, csv_rows, headers=csv_headers)

    if not args.no_plot:
        labels = [f"{run.variant}#{run.iteration}" for run in results]
        throughputs = [float(run.throughput_msg_s) for run in results]
        plot_path = args.plot_output
        plot_metric_bars(
            labels,
            throughputs,
            title="Architecture throughput per run",
            ylabel="Messages per second",
            output_path=plot_path,
            overwrite=args.overwrite,
        )
        writer.attach_metadata(plot_path, relative=False)


if __name__ == "__main__":
    main()
