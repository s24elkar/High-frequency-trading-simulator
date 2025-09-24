#!/usr/bin/env python3
"""Aggregate experiment JSON files into Markdown tables."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate neural Hawkes experiment results")
    parser.add_argument("--results-dir", type=str, default="experiments/results", help="Directory with result JSON files")
    parser.add_argument("--output", type=str, default="", help="Optional path to write Markdown summary")
    return parser.parse_args()


def load_results(results_dir: Path):
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open() as fh:
            data = json.load(fh)
        records.append((path.name, data))
    return records


def build_markdown_table(records):
    headers = ["name", "backbone", "test_loss", "test_acc", "test_mae", "ks_stat", "ks_pvalue", "duration"]
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for _, data in records:
        cfg = data.get("config", {})
        training = cfg.get("training", {})
        row = [
            data.get("name", ""),
            training.get("backbone", ""),
            f"{data.get('test_metrics', {}).get('loss', float('nan')):.4f}",
            f"{data.get('test_metrics', {}).get('acc', float('nan')):.4f}",
            f"{data.get('test_metrics', {}).get('mae', float('nan')):.4f}",
            f"{data.get('rescaling', {}).get('ks_statistic', float('nan')):.4f}",
            f"{data.get('rescaling', {}).get('ks_pvalue', float('nan')):.4f}",
            f"{data.get('duration_sec', float('nan')):.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summarize_by_backbone(records):
    groups = defaultdict(list)
    for _, data in records:
        backbone = data.get("config", {}).get("training", {}).get("backbone", "unknown")
        groups[backbone].append(data)
    lines = ["\n## Backbone summary"]
    for backbone, entries in groups.items():
        avg_loss = sum(e.get("test_metrics", {}).get("loss", 0.0) for e in entries) / len(entries)
        avg_acc = sum(e.get("test_metrics", {}).get("acc", 0.0) for e in entries) / len(entries)
        avg_mae = sum(e.get("test_metrics", {}).get("mae", 0.0) for e in entries) / len(entries)
        avg_ks = sum(e.get("rescaling", {}).get("ks_statistic", 0.0) for e in entries) / len(entries)
        lines.append(
            f"- **{backbone}**: loss={avg_loss:.4f}, acc={avg_acc:.4f}, mae={avg_mae:.4f}, ks={avg_ks:.4f} ({len(entries)} runs)"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    records = load_results(results_dir)
    if not records:
        raise ValueError("No result JSON files found")

    markdown = build_markdown_table(records) + summarize_by_backbone(records)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown)
        print(f"Wrote summary to {output_path}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
