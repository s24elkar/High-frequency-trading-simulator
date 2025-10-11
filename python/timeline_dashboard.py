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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _prepare_timeline(
    times: np.ndarray,
    marks: np.ndarray,
    mu: float,
    kernel,
    horizon: float,
    bins: int = 100,
) -> SimulationTimeline:
    grid = np.linspace(0.0, horizon, max(bins * 2, 200))
    lam = intensity_on_grid(mu, kernel, times, marks, grid)
    centres, counts = binned_counts(times, horizon, horizon / bins)
    return SimulationTimeline(times, marks, lam, grid, centres, counts)


def simulate_exp_timeline(
    mu: float,
    kernel: ExpKernel,
    T: float,
    mark_sampler: MarkSampler = None,
    seed: int = 12345,
    bins: int = 80,
) -> SimulationTimeline:
    times, marks = simulate_thinning_exp_fast(mu, kernel, mark_sampler, T, seed)
    return _prepare_timeline(times, marks, mu, kernel, T, bins)


def simulate_powerlaw_timeline(
    mu: float,
    kernel: PowerLawKernel,
    T: float,
    mark_sampler: MarkSampler = None,
    seed: int = 12345,
    bins: int = 80,
) -> SimulationTimeline:
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


def plot_timeline_interactive(
    timeline: SimulationTimeline,
    title: str = "Simulation Timeline",
    show_marks: bool = True,
    comparison: SimulationTimeline | None = None,
    labels: Tuple[str, str] = ("Scenario", "Comparison"),
) -> go.Figure:
    """Render timeline diagnostics as an interactive Plotly figure."""

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        specs=[[{"secondary_y": show_marks}], [{}], [{}]],
    )

    fig.add_trace(
        go.Scatter(
            x=timeline.times,
            y=np.ones_like(timeline.times),
            mode="markers",
            marker=dict(color="#1f77b4", size=6, opacity=0.7),
            name=labels[0],
            hovertemplate=f"t = %{{x:.3f}}<extra>{labels[0]}</extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=timeline.grid_times,
            y=timeline.intensity_grid,
            mode="lines",
            line=dict(color="#ff7f0e"),
            name=f"{labels[0]} intensity",
            hovertemplate=f"λ(t) = %{{y:.3f}}<extra>{labels[0]}</extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=timeline.bin_centres,
            y=timeline.bin_counts,
            name=f"{labels[0]} counts",
            marker=dict(color="#2ca02c"),
            opacity=0.8,
            hovertemplate=f"Orders = %{{y}}<extra>{labels[0]}</extra>",
        ),
        row=3,
        col=1,
    )

    if show_marks and timeline.marks.size:
        fig.add_trace(
            go.Scatter(
                x=timeline.times,
                y=timeline.marks,
                mode="markers",
                marker=dict(color="#d62728", size=6, symbol="circle-open"),
                name=f"{labels[0]} marks",
                hovertemplate=(
                    f"Mark = %{{y:.3f}} at %{{x:.3f}}<extra>{labels[0]}</extra>"
                ),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    if comparison is not None:
        fig.add_trace(
            go.Scatter(
                x=comparison.times,
                y=np.ones_like(comparison.times),
                mode="markers",
                marker=dict(color="#17becf", size=6, opacity=0.6),
                name=labels[1],
                hovertemplate=f"t = %{{x:.3f}}<extra>{labels[1]}</extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison.grid_times,
                y=comparison.intensity_grid,
                mode="lines",
                line=dict(color="#9467bd", dash="dash"),
                name=f"{labels[1]} intensity",
                hovertemplate=f"λ(t) = %{{y:.3f}}<extra>{labels[1]}</extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=comparison.bin_centres,
                y=comparison.bin_counts,
                mode="lines",
                line=dict(color="#8c564b"),
                name=f"{labels[1]} counts",
                hovertemplate=f"Orders = %{{y}}<extra>{labels[1]}</extra>",
            ),
            row=3,
            col=1,
        )

        if show_marks and comparison.marks.size:
            fig.add_trace(
                go.Scatter(
                    x=comparison.times,
                    y=comparison.marks,
                    mode="markers",
                    marker=dict(color="#bcbd22", size=6, symbol="x"),
                    name=f"{labels[1]} marks",
                    hovertemplate=(
                        f"Mark = %{{y:.3f}} at %{{x:.3f}}<extra>{labels[1]}</extra>"
                    ),
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

    fig.update_yaxes(
        title_text="Order arrivals",
        row=1,
        col=1,
        secondary_y=False,
        showticklabels=False,
    )
    if show_marks:
        fig.update_yaxes(title_text="Order size", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="λ(t)", row=2, col=1)
    fig.update_yaxes(title_text="Orders per bin", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)

    fig.update_layout(
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=720,
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="x unified",
    )
    return fig


__all__ = [
    "SimulationTimeline",
    "simulate_exp_timeline",
    "simulate_powerlaw_timeline",
    "plot_timeline",
    "plot_timeline_interactive",
]
