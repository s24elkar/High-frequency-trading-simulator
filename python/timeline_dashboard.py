"""Timeline dashboard helpers for Hawkes simulations.

The utilities here wrap the native simulators and produce matplotlib figures that
combine event timelines, estimated intensities, and binned arrival counts. They
are designed for notebooks or ad-hoc analysis scripts that need a quick
diagnostic view of a simulation run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from .kernels import ExpKernel, PowerLawKernel
    from .simulate import simulate_thinning_exp_fast, simulate_thinning_general
    from .viz import intensity_on_grid, binned_counts
except (ImportError, ValueError):
    import pathlib
    import sys

    _pkg_dir = pathlib.Path(__file__).resolve().parent
    if str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))

    from kernels import ExpKernel, PowerLawKernel
    from simulate import simulate_thinning_exp_fast, simulate_thinning_general
    from viz import intensity_on_grid, binned_counts


MarkSampler = Optional[Callable[[np.random.Generator], float]]


@dataclass
class SimulationTimeline:
    times: np.ndarray
    marks: np.ndarray
    intensity_grid: np.ndarray
    grid_times: np.ndarray
    bin_centres: np.ndarray
    bin_counts: np.ndarray


def _prepare_timeline(times: np.ndarray, marks: np.ndarray, mu: float, kernel, horizon: float, bins: int = 100) -> SimulationTimeline:
    grid = np.linspace(0.0, horizon, max(bins * 2, 200))
    lam = intensity_on_grid(mu, kernel, times, marks, grid)
    centres, counts = binned_counts(times, horizon, horizon / bins)
    return SimulationTimeline(times, marks, lam, grid, centres, counts)


def simulate_exp_timeline(mu: float,
                          kernel: ExpKernel,
                          T: float,
                          mark_sampler: MarkSampler = None,
                          seed: int = 12345,
                          bins: int = 80) -> SimulationTimeline:
    times, marks = simulate_thinning_exp_fast(mu, kernel, mark_sampler, T, seed)
    return _prepare_timeline(times, marks, mu, kernel, T, bins)


def simulate_powerlaw_timeline(mu: float,
                               kernel: PowerLawKernel,
                               T: float,
                               mark_sampler: MarkSampler = None,
                               seed: int = 12345,
                               bins: int = 80) -> SimulationTimeline:
    times, marks = simulate_thinning_general(mu, kernel, mark_sampler, T, seed)
    return _prepare_timeline(times, marks, mu, kernel, T, bins)


def plot_timeline(
    timeline: SimulationTimeline,
    title: str = "Simulation Timeline",
    show_marks: bool = True,
    comparison: SimulationTimeline | None = None,
    labels: Tuple[str, str] = ("Scenario", "Comparison"),
) -> plt.Figure:
    """Render timeline diagnostics with optional comparison overlay."""

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].eventplot(timeline.times, colors="#1f77b4")
    axes[0].set_ylabel("Order arrivals")
    axes[0].set_title(title)

    axes[1].plot(
        timeline.grid_times,
        timeline.intensity_grid,
        color="#ff7f0e",
        label=labels[0],
    )
    axes[1].set_ylabel("λ(t) – intensity")

    bar_width = (
        timeline.bin_centres[1] - timeline.bin_centres[0]
        if timeline.bin_centres.size > 1
        else 0.1
    )
    axes[2].bar(
        timeline.bin_centres,
        timeline.bin_counts,
        width=bar_width,
        color="#2ca02c",
        alpha=0.7,
        label=labels[0],
    )
    axes[2].set_ylabel("Orders per bin")
    axes[2].set_xlabel("Time (s)")

    if show_marks and timeline.marks.size:
        ax_marks = axes[0].twinx()
        ax_marks.plot(
            timeline.times,
            timeline.marks,
            linestyle="none",
            marker="o",
            markersize=4,
            alpha=0.6,
            color="#d62728",
        )
        ax_marks.set_ylabel("Order size (marks)")
        ax_marks.tick_params(axis="y", labelcolor="#d62728")

    if comparison is not None:
        axes[0].eventplot(comparison.times, colors="#17becf")
        axes[1].plot(
            comparison.grid_times,
            comparison.intensity_grid,
            color="#9467bd",
            linestyle="--",
            label=labels[1],
        )
        axes[2].step(
            comparison.bin_centres,
            comparison.bin_counts,
            where="mid",
            color="#8c564b",
            label=labels[1],
        )

    if comparison is not None:
        axes[1].legend(loc="upper right")
        axes[2].legend(loc="upper right")

    fig.tight_layout()
    return fig


__all__ = [
    "SimulationTimeline",
    "simulate_exp_timeline",
    "simulate_powerlaw_timeline",
    "plot_timeline",
]
