#!/usr/bin/env python3
"""Assemble paper-ready diagnostics, tables, and ablations (Day 10 deliverables).

This script collates metrics produced by existing experiment runs, computes
calibration ECE, copies figures into a summary folder, and emits table/CSV
artifacts ready for inclusion in the paper draft.

It assumes the experiment directories under `experiments/runs/*` follow the
structure emitted by `neural_hawkes.run_experiment`.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import os

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "experiments" / "runs"
SUMMARY_DIR = ROOT / "experiments" / "summary"
SUMMARY_FIGS = SUMMARY_DIR / "figs"


RUNS_OF_INTEREST = [
    "binance_gru_long",
    "binance_transformer_long",
    "lobster_gru_long",
    "lobster_transformer_long",
]

ABLATION_PAIRS = {
    "binance": ("binance_gru", "binance_gru_long"),
    "lobster": ("lobster_gru", "lobster_gru_long"),
}


@dataclass
class RunMetrics:
    run_id: str
    venue: str
    symbol: str
    backbone: str
    nll: float
    mae: float
    acc: float
    ks_p: float
    ks_stat: float
    params_m: Optional[float]
    train_time_s: Optional[float]
    ece: Optional[float]


def read_metrics(run: Path) -> RunMetrics:
    data = json.loads((run / "metrics.json").read_text())
    test_metrics = data["split_metrics"]["test"]
    ece = compute_ece(run / "curves" / "calibration_next_time.csv")
    return RunMetrics(
        run_id=run.name,
        venue=data.get("venue", ""),
        symbol=data.get("symbol", ""),
        backbone=data.get("backbone", ""),
        nll=float(test_metrics["nll"]),
        mae=float(test_metrics["next_time_mae"]),
        acc=float(test_metrics["next_type_acc"]),
        ks_p=float(data.get("ks", {}).get("ks_pvalue", math.nan)),
        ks_stat=float(data.get("ks", {}).get("ks_stat", math.nan)),
        params_m=float(data.get("params_millions", math.nan)) if data.get("params_millions") is not None else None,
        train_time_s=float(data.get("time_sec_train", math.nan)) if data.get("time_sec_train") is not None else None,
        ece=ece,
    )


def compute_ece(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    xs: List[float] = []
    ys: List[float] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                pred = float(row["pred_quantile"])
                emp = float(row["empirical_quantile"])
            except (ValueError, KeyError):
                continue
            xs.append(pred)
            ys.append(emp)
    if not xs:
        return None
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    order = np.argsort(xs_arr)
    xs_sorted = xs_arr[order]
    ys_sorted = ys_arr[order]
    ece = np.trapezoid(np.abs(ys_sorted - xs_sorted), xs_sorted)
    return float(ece)


def copy_figures(run: Path) -> None:
    figs_dir = run / "figs"
    if not figs_dir.exists():
        return
    for fig in figs_dir.glob("*.png"):
        destination = SUMMARY_FIGS / f"{run.name}_{fig.name}"
        shutil.copy(fig, destination)


def write_benchmarks_md(rows: List[RunMetrics]) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FIGS.mkdir(parents=True, exist_ok=True)
    header = "| Venue | Backbone | Test NLL ↓ | Next-time MAE ↓ | Next-type Acc ↑ | KS p-value ↑ | ECE ↓ |\n"
    separator = "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
    lines = ["# Benchmark Table\n\n", header, separator]
    for m in rows:
        lines.append(
            f"| {m.venue.title()} {m.symbol} | {m.backbone.upper()} | "
            f"{m.nll:.3f} | {m.mae:.3f} | {m.acc:.3f} | {m.ks_p:.3g} | "
            f"{(m.ece if m.ece is not None else float('nan')):.3f} |\n"
        )
    (SUMMARY_DIR / "benchmarks.md").write_text("".join(lines))

    with (SUMMARY_DIR / "benchmarks.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "venue",
            "symbol",
            "backbone",
            "nll_test",
            "mae_time_test",
            "acc_type_test",
            "ks_p",
            "ks_stat",
            "ece",
            "params_m",
            "train_time_s",
        ])
        for m in rows:
            writer.writerow([
                m.venue,
                m.symbol,
                m.backbone,
                f"{m.nll:.6f}",
                f"{m.mae:.6f}",
                f"{m.acc:.6f}",
                f"{m.ks_p:.6g}",
                f"{m.ks_stat:.6f}",
                f"{m.ece:.6f}" if m.ece is not None else "",
                f"{m.params_m:.6f}" if m.params_m is not None else "",
                f"{m.train_time_s:.6f}" if m.train_time_s is not None else "",
            ])


def write_ablation(rows: Dict[str, Dict[str, RunMetrics]]) -> None:
    output_csv = SUMMARY_DIR / "ablation.csv"
    fields = [
        "venue",
        "context",
        "nll",
        "mae",
        "acc",
        "ks_p",
    ]
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for venue, ctx_runs in rows.items():
            for context, metrics in ctx_runs.items():
                writer.writerow(
                    {
                        "venue": venue,
                        "context": context,
                        "nll": f"{metrics.nll:.4f}",
                        "mae": f"{metrics.mae:.4f}",
                        "acc": f"{metrics.acc:.4f}",
                        "ks_p": f"{metrics.ks_p:.3g}",
                    }
                )

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    contexts = ["short", "long"]
    venues = list(rows.keys())
    x = np.arange(len(venues))
    width = 0.35
    plt.figure(figsize=(6, 4))
    for idx, context in enumerate(contexts):
        nlls = [rows[v][context].nll for v in venues]
        plt.bar(x + (idx - 0.5) * width, nlls, width, label=f"{context} context")
    plt.xticks(x, [v.title() for v in venues])
    plt.ylabel("Test NLL")
    plt.title("Context-length ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SUMMARY_DIR / "ablation.png", dpi=300)
    plt.close()


def main() -> None:
    SUMMARY_FIGS.mkdir(parents=True, exist_ok=True)

    # Collect metrics for main table
    selected = []
    for run_name in RUNS_OF_INTEREST:
        run_path = RUNS_DIR / run_name
        if not (run_path / "metrics.json").exists():
            continue
        metrics = read_metrics(run_path)
        selected.append(metrics)
        copy_figures(run_path)

    selected.sort(key=lambda m: (m.venue, m.backbone))
    write_benchmarks_md(selected)

    # Ablation: short vs long context (GRU model)
    ablation_rows: Dict[str, Dict[str, RunMetrics]] = {}
    for venue, (short_id, long_id) in ABLATION_PAIRS.items():
        short_metrics = read_metrics(RUNS_DIR / short_id)
        long_metrics = read_metrics(RUNS_DIR / long_id)
        ablation_rows[venue] = {"short": short_metrics, "long": long_metrics}
    write_ablation(ablation_rows)


if __name__ == "__main__":
    main()
MPL_CACHE = ROOT / ".matplotlib"
MPL_CACHE.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)
