#!/usr/bin/env python3
"""Aggregate stress test artifacts into structured datasets and charts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from python.analysis import (
    ArtifactWriter,
    ReportMetadata,
    detect_git_commit,
    plot_latency_histogram,
    plot_order_trade_ratio,
    plot_throughput_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate stress-test metrics and produce plots."
    )
    parser.add_argument(
        "--stress-suite",
        type=Path,
        default=Path("results/week4/stress/stress_suite.json"),
        help="Path to stress suite JSON output.",
    )
    parser.add_argument(
        "--log-integrity",
        type=Path,
        default=Path("results/week4/stress/log_integrity.json"),
        help="Path to log integrity summary JSON.",
    )
    parser.add_argument(
        "--perf-runs-dir",
        type=Path,
        default=Path("logs/perf_runs"),
        help="Directory containing performance JSONL runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/week4/stress/analysis"),
        help="Directory to write aggregated CSV/JSON outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory to write figures (defaults to <output-dir>/figures).",
    )
    parser.add_argument(
        "--bucket-ns",
        type=int,
        default=1_000,
        help="Bucket size (ns) for throughput time-series aggregation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def calculate_throughput_series(
    run_path: Path, bucket_ns: int
) -> Tuple[List[dict], Optional[dict]]:
    buckets: Counter[int] = Counter()
    run_summary: Optional[dict] = None
    with run_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            event_type = record.get("event_type")
            if event_type == "run_summary":
                run_summary = record.get("payload") or {}
                continue
            timestamp_ns = int(record.get("timestamp_ns", 0))
            bucket_idx = timestamp_ns // bucket_ns
            buckets[bucket_idx] += 1
    series: List[dict] = []
    for bucket_idx in sorted(buckets):
        count = buckets[bucket_idx]
        bucket_start_ns = bucket_idx * bucket_ns
        throughput = count * 1_000_000_000 / bucket_ns if bucket_ns else 0.0
        series.append(
            {
                "bucket_idx": bucket_idx,
                "bucket_start_ns": bucket_start_ns,
                "bucket_start_ms": bucket_start_ns / 1_000_000.0,
                "messages": count,
                "throughput_msgs_per_s": throughput,
            }
        )
    return series, run_summary


def aggregate() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figures_dir = args.figures_dir or (output_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata = ReportMetadata(
        generator="aggregate_stress_metrics",
        git_commit=detect_git_commit(REPO_ROOT),
        extra={"bucket_ns": args.bucket_ns},
    )
    writer = ArtifactWriter(output_dir, metadata, overwrite=args.overwrite)

    stress = load_json(args.stress_suite)
    data_section = stress.get("data") if isinstance(stress, dict) else None
    scenarios: List[dict]
    if isinstance(data_section, dict):
        scenarios = data_section.get("scenarios", [])
    else:
        scenarios = stress.get("scenarios", []) if isinstance(stress, dict) else []

    # Scenario level dataset
    scenario_rows: List[dict] = []
    for scenario in scenarios:
        scenario_rows.append(
            {
                "multiplier": scenario.get("multiplier"),
                "message_count": scenario.get("message_count"),
                "wall_time_s": scenario.get("wall_time_s"),
                "throughput_msgs_per_s": scenario.get("throughput_msgs_per_s"),
                "avg_latency_ns": scenario.get("avg_latency_ns"),
                "p95_latency_ns": scenario.get("p95_latency_ns"),
                "p99_latency_ns": scenario.get("p99_latency_ns"),
                "max_latency_ns": scenario.get("max_latency_ns"),
                "add_order_events": scenario.get("add_order_events"),
                "delete_order_events": scenario.get("delete_order_events"),
                "execute_order_events": scenario.get("execute_order_events"),
                "order_to_trade_ratio": scenario.get("order_to_trade_ratio"),
                "orphan_cancels": scenario.get("orphan_cancels"),
                "orphan_executes": scenario.get("orphan_executes"),
                "duplicate_order_ids": scenario.get("duplicate_order_ids"),
            }
        )

    if scenario_rows:
        writer.write_csv("scenario_metrics.csv", scenario_rows)

    # Persist latency histograms per scenario
    for scenario in scenarios:
        histogram = scenario.get("latency_histogram") or []
        if not histogram:
            continue
        writer.write_csv(
            f"latency_histogram_x{scenario.get('multiplier')}.csv",
            histogram,
            headers=("upper_ns", "count"),
        )
        figure_path = figures_dir / f"latency_histogram_x{scenario.get('multiplier')}.png"
        plot_latency_histogram(
            histogram,
            multiplier=scenario.get("multiplier"),
            output_path=figure_path,
            overwrite=args.overwrite,
        )
        writer.attach_metadata(figure_path, relative=False)

    # Risk control aggregation
    risk_payload = load_json(args.log_integrity)
    risk_logs = risk_payload.get("logs", [])
    total_orphan_executes = sum(
        entry.get("sequence", {}).get("orphan_executes", 0) for entry in risk_logs
    )
    total_orphan_cancels = sum(
        entry.get("sequence", {}).get("orphan_cancels", 0) for entry in risk_logs
    )
    risk_summary = {
        "total_logs": len(risk_logs),
        "total_orphan_executes": total_orphan_executes,
        "total_orphan_cancels": total_orphan_cancels,
        "files": risk_logs,
    }

    # Performance runs aggregation
    perf_runs: List[dict] = []
    if args.perf_runs_dir.exists():
        for run_file in sorted(args.perf_runs_dir.glob("run_*.jsonl")):
            series, summary = calculate_throughput_series(run_file, args.bucket_ns)
            perf_runs.append(
                {
                    "run_id": run_file.stem,
                    "series": series,
                    "summary": summary,
                }
            )

        throughput_rows: List[dict] = []
        summary_rows: List[dict] = []
        for run in perf_runs:
            run_id = run["run_id"]
            for record in run["series"]:
                throughput_rows.append({"run_id": run_id, **record})
            summary = run.get("summary") or {}
            if summary:
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "symbol": summary.get("symbol"),
                        "realized_pnl": summary.get("realized_pnl"),
                        "unrealized_pnl": summary.get("unrealized_pnl"),
                        "inventory": summary.get("inventory"),
                        "order_volume": summary.get("order_volume"),
                        "fill_volume": summary.get("fill_volume"),
                        "order_to_trade_ratio": summary.get("order_to_trade_ratio"),
                        "fill_efficiency": summary.get("fill_efficiency"),
                        "avg_latency_ns": summary.get("avg_latency_ns"),
                        "p95_latency_ns": summary.get("p95_latency_ns"),
                        "p99_latency_ns": summary.get("p99_latency_ns"),
                        "max_latency_ns": summary.get("max_latency_ns"),
                        "duration_ns": summary.get("duration_ns"),
                        "digest": summary.get("digest"),
                    }
                )
        if throughput_rows:
            writer.write_csv(
                "throughput_timeseries.csv",
                throughput_rows,
                headers=throughput_rows[0].keys(),
            )
            figure_path = figures_dir / "throughput_timeseries.png"
            plot_throughput_series(
                [
                    {
                        "label": run["run_id"],
                        "times": [p["bucket_start_ms"] for p in run["series"]],
                        "values": [p["throughput_msgs_per_s"] for p in run["series"]],
                    }
                    for run in perf_runs
                ],
                output_path=figure_path,
                overwrite=args.overwrite,
            )
            writer.attach_metadata(figure_path, relative=False)
        if summary_rows:
            writer.write_csv("perf_run_summary.csv", summary_rows)

    ratios = []
    labels = []
    for scenario in scenarios:
        ratio = scenario.get("order_to_trade_ratio")
        if ratio:
            ratios.append(float(ratio))
            labels.append(f"×{scenario.get('multiplier')}")
    if ratios:
        figure_path = figures_dir / "order_to_trade_ratio.png"
        plot_order_trade_ratio(
            labels,
            ratios,
            output_path=figure_path,
            overwrite=args.overwrite,
        )
        writer.attach_metadata(figure_path, relative=False)

    aggregated_payload = {
        "scenarios": scenarios,
        "risk_controls": risk_summary,
        "perf_runs": perf_runs,
        "bucket_ns": args.bucket_ns,
    }
    writer.write_json("aggregated_metrics.json", aggregated_payload)


if __name__ == "__main__":
    aggregate()
