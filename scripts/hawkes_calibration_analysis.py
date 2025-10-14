#!/usr/bin/env python3

"""Generate calibration diagnostics and plots for Hawkes MLE experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

CACHE_DIR = Path("results/week5/calibration/mpl_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR.resolve()))
cache_home = CACHE_DIR / "xdg_cache"
cache_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_home.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(base_dir / "replicate_summary.csv")
    trajectory = pd.read_csv(base_dir / "trajectories" / "trajectory_0.csv")
    events = pd.read_csv(base_dir / "events" / "replicate_0.csv")
    return summary, trajectory, events


def compute_statistics(summary: pd.DataFrame, true_params: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    converged = summary[summary["converged"] == 1].copy()
    stats: Dict[str, Dict[str, float]] = {}
    for key, label in (("mu_hat", "mu"), ("alpha_hat", "alpha"), ("beta_hat", "beta")):
        series = converged[key]
        stats[label] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "bias": float(series.mean() - true_params[label]),
            "variance": float(series.var(ddof=1)),
            "rmse": float(np.sqrt(((series - true_params[label]) ** 2).mean())),
        }
    stats["branching_ratio"] = {
        "mean": float(converged["branching_ratio"].mean()),
        "std": float(converged["branching_ratio"].std(ddof=1)),
    }
    stats["converged_fraction"] = {
        "value": float(len(converged) / len(summary)) if len(summary) else float("nan"),
    }
    return stats


def log_likelihood_exponential(times: np.ndarray, mu: float, alpha: float, beta: float, horizon: float) -> float:
    if mu <= 0 or alpha < 0 or beta <= 0:
        return float("-inf")
    state = 0.0
    log_l = 0.0
    last_time = 0.0
    for t in times:
        dt = t - last_time
        if dt < -1e-12:
            raise ValueError("event times must be non-decreasing")
        state *= np.exp(-beta * dt)
        intensity = mu + alpha * state
        if intensity <= 0:
            return float("-inf")
        log_l += np.log(intensity)
        state += 1.0
        last_time = t
    tail = np.sum(1.0 - np.exp(-beta * (horizon - times))) / beta
    integral = mu * horizon + alpha * tail
    return log_l - integral


def time_rescaled_intervals(times: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    state = 0.0
    last_time = 0.0
    intervals = []
    for t in times:
        dt = t - last_time
        state *= np.exp(-beta * dt)
        integral = mu * dt + (alpha * state / beta) * (1.0 - np.exp(-beta * dt))
        intervals.append(integral)
        state += 1.0
        last_time = t
    return np.asarray(intervals)


def plot_parameter_distributions(summary: pd.DataFrame, true_params: Dict[str, float], output_path: Path) -> None:
    converged = summary[summary["converged"] == 1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, label in zip(axes, ("mu_hat", "alpha_hat", "beta_hat"), ("mu", "alpha", "beta")):
        ax.hist(converged[key], bins=24, color="C0", alpha=0.75, edgecolor="black")
        ax.axvline(true_params[label], color="C1", linestyle="--", linewidth=1.5, label="true")
        ax.set_title(f"{label} estimates")
        ax.set_xlabel(label)
        ax.set_ylabel("frequency")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_convergence(trajectory: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trajectory["iteration"], trajectory["mu"], label="mu", linewidth=1.6)
    ax.plot(trajectory["iteration"], trajectory["alpha"], label="alpha", linewidth=1.6)
    ax.plot(trajectory["iteration"], trajectory["beta"], label="beta", linewidth=1.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("parameter value")
    ax.set_title("MLE convergence (replicate 0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_log_likelihood_surface(events: pd.DataFrame, true_params: Dict[str, float], horizon: float, output_path: Path) -> None:
    times = events["time"].to_numpy()
    mu_grid = np.linspace(true_params["mu"] * 0.6, true_params["mu"] * 1.4, 60)
    alpha_grid = np.linspace(true_params["alpha"] * 0.4, true_params["alpha"] * 1.6, 60)
    beta = true_params["beta"]
    surface = np.empty((len(alpha_grid), len(mu_grid)))
    for i, alpha in enumerate(alpha_grid):
        for j, mu in enumerate(mu_grid):
            surface[i, j] = log_likelihood_exponential(times, mu, alpha, beta, horizon)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    contour = ax.contourf(mu_grid, alpha_grid, surface, levels=40, cmap="viridis")
    ax.scatter([true_params["mu"]], [true_params["alpha"]], color="red", marker="x", s=60, label="true")
    ax.set_xlabel("mu")
    ax.set_ylabel("alpha")
    ax.set_title("Log-likelihood surface (beta fixed)")
    fig.colorbar(contour, ax=ax, label="log L")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_residual_qq(residuals: np.ndarray, output_path: Path) -> None:
    residuals = np.sort(residuals)
    n = residuals.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = -np.log(1.0 - probs)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(theoretical, residuals, color="C0", s=12, alpha=0.8, label="rescaled gaps")
    max_val = max(theoretical.max(), residuals.max())
    ax.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlabel("Exponential(1) quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_title("Time-rescaled residual Q-Q plot")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base_dir = Path("results/week5/calibration")
    summary, trajectory, events = load_data(base_dir)
    true_params = {"mu": 0.7, "alpha": 0.5, "beta": 1.4}

    stats = compute_statistics(summary, true_params)
    (base_dir / "metrics.json").write_text(json.dumps(stats, indent=2))

    plot_parameter_distributions(summary, true_params, base_dir / "parameter_recovery.png")
    plot_convergence(trajectory, base_dir / "convergence_curve.png")
    plot_log_likelihood_surface(events, true_params, horizon=200.0, output_path=base_dir / "log_likelihood_surface.png")

    converged = summary[summary["converged"] == 1]
    best_idx = int(converged.sort_values("log_likelihood", ascending=False).iloc[0]["replicate"])
    trajectory_path = base_dir / "trajectories" / f"trajectory_{best_idx}.csv"
    if trajectory_path.exists():
        trajectory_best = pd.read_csv(trajectory_path)
        plot_convergence(trajectory_best, base_dir / "convergence_curve_best.png")

    est = converged.iloc[0]
    residuals = time_rescaled_intervals(events["time"].to_numpy(), est["mu_hat"], est["alpha_hat"], est["beta_hat"])
    plot_residual_qq(residuals, base_dir / "qq_plot.png")

    print("Calibration analysis complete. Outputs written to", base_dir)


if __name__ == "__main__":
    main()
