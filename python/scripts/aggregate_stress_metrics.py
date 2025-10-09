#!/usr/bin/env python3
"""Aggregate stress test artifacts into structured datasets and charts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def ensure_output(path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


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


def _format_latency_label(upper_ns: Optional[int]) -> str:
    if upper_ns is None:
        return "> max"
    if upper_ns < 1_000:
        return f"≤{upper_ns}ns"
    if upper_ns < 1_000_000:
        return f"≤{upper_ns / 1_000:.0f}µs"
    if upper_ns < 1_000_000_000:
        return f"≤{upper_ns / 1_000_000:.1f}ms"
    return f"≤{upper_ns / 1_000_000_000:.1f}s"


def write_csv(path: Path, headers: Iterable[str], rows: Iterable[dict], overwrite: bool) -> None:
    ensure_output(path, overwrite=overwrite)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: dict, overwrite: bool) -> None:
    ensure_output(path, overwrite=overwrite)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_latency_histogram(
    scenario: dict, output_path: Path, overwrite: bool
) -> None:
    histogram = scenario.get("latency_histogram") or []
    if not histogram:
        return
    ensure_output(output_path, overwrite=overwrite)
    labels = [_format_latency_label(entry.get("upper_ns")) for entry in histogram]
    counts = [entry.get("count", 0) for entry in histogram]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(counts)), counts, color="#4477AA")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Events")
    ax.set_title(f"Latency Histogram (multiplier ×{scenario.get('multiplier')})")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_throughput_series(
    runs: List[dict], output_path: Path, overwrite: bool
) -> None:
    if not runs:
        return
    ensure_output(output_path, overwrite=overwrite)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in runs:
        run_id = run["run_id"]
        points = run["series"]
        if not points:
            continue
        times = [p["bucket_start_ms"] for p in points]
        throughput = [p["throughput_msgs_per_s"] for p in points]
        ax.plot(times, throughput, label=run_id)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Throughput (msg/s)")
    ax.set_title("Throughput During Stress Runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_order_trade_ratio(
    scenarios: List[dict], output_path: Path, overwrite: bool
) -> None:
    ratios = []
    labels = []
    for scenario in scenarios:
        ratio = scenario.get("order_to_trade_ratio")
        if ratio:
            ratios.append(ratio)
            labels.append(f"×{scenario.get('multiplier')}")
    if not ratios:
        return
    ensure_output(output_path, overwrite=overwrite)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, ratios, color="#AA7744")
    ax.set_ylabel("Order-to-Trade Ratio (log scale)")
    ax.set_yscale("log")
    ax.set_title("Order Activity vs Executions")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def aggregate() -> None:
    args = parse_args()
    output_dir = args.output_dir
    figures_dir = args.figures_dir or (output_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stress = load_json(args.stress_suite)
    scenarios: List[dict] = stress.get("scenarios", [])

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

    write_csv(
        output_dir / "scenario_metrics.csv",
        scenario_rows[0].keys() if scenario_rows else [],
        scenario_rows,
        overwrite=args.overwrite,
    )

    # Persist latency histograms per scenario
    for scenario in scenarios:
        histogram = scenario.get("latency_histogram") or []
        if not histogram:
            continue
        write_csv(
            output_dir
            / f"latency_histogram_x{scenario.get('multiplier')}.csv",
            ("upper_ns", "count"),
            histogram,
            overwrite=args.overwrite,
        )
        plot_latency_histogram(
            scenario,
            figures_dir / f"latency_histogram_x{scenario.get('multiplier')}.png",
            overwrite=args.overwrite,
        )

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
        for run in perf_runs:
            run_id = run["run_id"]
            for record in run["series"]:
                throughput_rows.append({"run_id": run_id, **record})
        if throughput_rows:
            write_csv(
                output_dir / "throughput_timeseries.csv",
                throughput_rows[0].keys(),
                throughput_rows,
                overwrite=args.overwrite,
            )
            plot_throughput_series(
                perf_runs,
                figures_dir / "throughput_timeseries.png",
                overwrite=args.overwrite,
            )

    plot_order_trade_ratio(
        scenarios,
        figures_dir / "order_to_trade_ratio.png",
        overwrite=args.overwrite,
    )

    aggregated_payload = {
        "scenarios": scenarios,
        "risk_controls": risk_summary,
        "perf_runs": perf_runs,
        "bucket_ns": args.bucket_ns,
    }
    save_json(
        output_dir / "aggregated_metrics.json",
        aggregated_payload,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    aggregate()
