#!/usr/bin/env python3
"""Aggregate experiment artefacts into a benchmark table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect neural Hawkes benchmark runs")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Directory containing per-run artefacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/summary/benchmarks.csv"),
        help="Destination CSV path",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> Dict[str, object]:
    with path.open() as fh:
        return json.load(fh)


def extract_row(payload: Dict[str, object]) -> Dict[str, object]:
    split = payload.get("split_metrics", {}) if isinstance(payload, dict) else {}
    test = split.get("test", {}) if isinstance(split, dict) else {}
    return {
        "venue": payload.get("venue"),
        "symbol": payload.get("symbol"),
        "backbone": payload.get("backbone"),
        "seed": payload.get("seed"),
        "nll_test": test.get("nll"),
        "mae_time_test": test.get("next_time_mae"),
        "acc_type_test": test.get("next_type_acc"),
        "ks_p": (payload.get("ks", {}) or {}).get("ks_pvalue"),
        "ks_stat": (payload.get("ks", {}) or {}).get("ks_stat"),
        "params_M": payload.get("params_millions"),
        "train_time_s": payload.get("time_sec_train"),
    }


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(args.run_dir.glob("*/metrics.json")):
        payload = load_metrics(metrics_path)
        rows.append(extract_row(payload))

    if not rows:
        print("No runs discovered; nothing to aggregate.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "venue",
        "symbol",
        "backbone",
        "seed",
        "nll_test",
        "mae_time_test",
        "acc_type_test",
        "ks_p",
        "ks_stat",
        "params_M",
        "train_time_s",
    ]
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
