from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from python.backtester import (
    BurstConfig,
    PoissonOrderFlowConfig,
    StressConfig,
    run_order_book_stress,
)


def _scenario_config(
    multiplier: int,
    *,
    base_messages: int,
    base_rate_hz: float,
    seed: int,
    symbol: str,
) -> StressConfig:
    poisson = PoissonOrderFlowConfig(
        symbol=symbol,
        message_count=base_messages * multiplier,
        base_rate_hz=base_rate_hz * multiplier,
        seed=seed + multiplier,
        include_metadata=True,
    )
    burst = BurstConfig(
        probability=0.05,
        rate_multiplier=20.0,
        min_duration_us=25_000,
        max_duration_us=150_000,
        cancel_ratio=0.7,
        churn_probability=0.9,
    )
    return StressConfig(
        symbol=symbol,
        depth=10,
        poisson=poisson,
        burst=burst,
        validate_sequence=True,
        record_latency=True,
        seed=seed + multiplier,
    )


def run_suite(
    output_dir: Path,
    *,
    base_messages: int = 10_000,
    base_rate_hz: float = 5_000.0,
    seed: int = 2024,
    symbol: str = "SYN-STRESS",
) -> Dict[str, List[Dict[str, object]]]:
    scenarios: List[Dict[str, object]] = []
    for multiplier in (1, 10, 100):
        config = _scenario_config(
            multiplier,
            base_messages=base_messages,
            base_rate_hz=base_rate_hz,
            seed=seed,
            symbol=symbol,
        )
        metrics = run_order_book_stress(config)
        seq_report = metrics.sequence_report
        throughput = (
            metrics.message_count / metrics.wall_time_s
            if metrics.wall_time_s > 0
            else None
        )
        latency_histogram = (
            [
                {"upper_ns": bin.upper_ns, "count": bin.count}
                for bin in metrics.latency_histogram
            ]
            if metrics.latency_histogram
            else None
        )
        scenario = {
            "multiplier": multiplier,
            "message_count": metrics.message_count,
            "wall_time_s": metrics.wall_time_s,
            "peak_memory_kb": metrics.peak_memory_kb,
            "throughput_msgs_per_s": throughput,
            "avg_latency_ns": metrics.avg_latency_ns,
            "p95_latency_ns": metrics.p95_latency_ns,
            "p99_latency_ns": metrics.p99_latency_ns,
            "max_latency_ns": metrics.max_latency_ns,
            "sequence_ok": seq_report.ok if seq_report else None,
            "orphan_cancels": seq_report.orphan_cancels if seq_report else None,
            "orphan_executes": seq_report.orphan_executes if seq_report else None,
            "duplicate_order_ids": (
                seq_report.duplicate_order_ids if seq_report else None
            ),
            "sequence_errors": (
                [asdict(err) for err in seq_report.errors] if seq_report else []
            ),
            "latency_histogram": latency_histogram,
            "add_order_events": metrics.add_order_events,
            "delete_order_events": metrics.delete_order_events,
            "execute_order_events": metrics.execute_order_events,
            "order_to_trade_ratio": (
                metrics.add_order_events / metrics.execute_order_events
                if metrics.add_order_events is not None
                and metrics.execute_order_events not in (None, 0)
                else None
            ),
        }
        scenarios.append(scenario)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"scenarios": scenarios}
    (output_dir / "stress_suite.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Poisson-based stress suite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write stress suite results.",
    )
    parser.add_argument(
        "--base-messages",
        type=int,
        default=10_000,
        help="Baseline message count before applying multipliers.",
    )
    parser.add_argument(
        "--base-rate-hz",
        type=float,
        default=5_000.0,
        help="Baseline message arrival rate in Hz.",
    )
    parser.add_argument(
        "--seed", type=int, default=2024, help="Base RNG seed for reproducibility."
    )
    parser.add_argument(
        "--symbol", type=str, default="SYN-STRESS", help="Synthetic symbol identifier."
    )
    args = parser.parse_args()
    results = run_suite(
        args.output_dir,
        base_messages=args.base_messages,
        base_rate_hz=args.base_rate_hz,
        seed=args.seed,
        symbol=args.symbol,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
