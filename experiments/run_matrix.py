#!/usr/bin/env python3
"""Batch runner for neural Hawkes experiments.

Usage:
    python experiments/run_matrix.py --config experiments/configs/multi_symbol_example.json \
        --results-dir experiments/results

Each experiment entry in the config is passed to `neural_hawkes.run_experiment`.
Results are stored as JSON for downstream aggregation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from neural_hawkes import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matrix of neural Hawkes experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory to store result JSON files",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing result files with same name"
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Optional prefix for result filenames"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as fh:
        config = json.load(fh)

    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments defined in configuration")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        name = exp.get("name", f"experiment_{int(time.time())}")
        filename = f"{args.prefix + '_' if args.prefix else ''}{name}.json"
        result_path = results_dir / filename
        if result_path.exists() and not args.overwrite:
            print(f"Skipping {name} (result exists)")
            continue
        print(f"\n=== Running experiment: {name} ===")
        result = run_experiment(exp)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w") as fh:
            json.dump(result, fh, indent=2)
        print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
