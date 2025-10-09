#!/usr/bin/env python3
"""Compare single-threaded vs concurrent backtester performance."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Tuple

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.backtester import Backtester, BacktesterConfig, ConcurrentBacktester, MetricsLogger, RiskConfig, RiskEngine  # type: ignore[import-not-found]
from python.backtester.logging import MetricsSnapshot  # type: ignore[import-not-found]
from python.backtester.itch import (  # type: ignore[import-not-found]
    load_lobster_csv,
    replay_from_lobster,
)
from python.backtester.order_book import load_order_book  # type: ignore[import-not-found]
from python.strategies.market_maker import (  # type: ignore[import-not-found]
    MarketMakingConfig,
    MarketMakingStrategy,
)


@dataclass(slots=True)
class RunMetrics:
    variant: str
    iteration: int
    message_count: int
    wall_time_s: float
    throughput_msg_s: float
    digest: str
    fill_count: int
    fill_volume: float
    avg_latency_ns: float | None
    p95_latency_ns: float | None
    p99_latency_ns: float | None
    max_latency_ns: float | None
    matching_avg_ns: float | None
    message_avg_ns: float | None


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
    return parser.parse_args()


def _latency_fields(snapshot: MetricsSnapshot) -> Tuple[float | None, float | None, float | None, float | None]:
    return (
        snapshot.avg_latency_ns,
        snapshot.p95_latency_ns,
        snapshot.p99_latency_ns,
        snapshot.max_latency_ns,
    )


def _ensure_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_backtester(symbol: str, depth: int, seed: int, metrics: MetricsLogger) -> Backtester:
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
) -> RunMetrics:
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

    return RunMetrics(
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


def summarise(results: List[RunMetrics]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    by_variant: Dict[str, List[RunMetrics]] = {}
    for result in results:
        by_variant.setdefault(result.variant, []).append(result)

    for variant, runs in by_variant.items():
        digests = {run.digest for run in runs}
        summary[variant] = {
            "runs": len(runs),
            "unique_digests": len(digests),
            "avg_wall_time_s": mean(run.wall_time_s for run in runs),
            "avg_throughput_msg_s": mean(run.throughput_msg_s for run in runs),
            "avg_matching_ns": mean(
                run.matching_avg_ns for run in runs if run.matching_avg_ns is not None
            )
            if any(run.matching_avg_ns is not None for run in runs)
            else None,
            "avg_message_ns": mean(
                run.message_avg_ns for run in runs if run.message_avg_ns is not None
            )
            if any(run.message_avg_ns is not None for run in runs)
            else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    _ensure_output(args.output, args.overwrite)
    _ensure_output(args.output_csv, args.overwrite)

    messages = _materialise_events(
        load_lobster_csv(args.data, symbol=args.symbol), args.message_count
    )

    results: List[RunMetrics] = []
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

    summary = summarise(results)

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

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with args.output_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(asdict(results[0]).keys()) if results else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for run in results:
            writer.writerow(asdict(run))


if __name__ == "__main__":
    main()
