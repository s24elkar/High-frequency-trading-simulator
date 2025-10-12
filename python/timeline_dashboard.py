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


def _rgb_to_hex(color: str) -> str:
    """Normalise Plotly colour strings so Matplotlib can reuse them."""

    color = color.strip()
    if color.startswith("#"):
        return color
    channels = color.strip("rgba() ").split(",")
    parts = [int(float(component.strip())) for component in channels[:3]]
    return f"#{parts[0]:02x}{parts[1]:02x}{parts[2]:02x}"


COLORWAY = [
    "#4af699",  # neon green reminiscent of trading dashboards
    "#f94f6d",  # punchy red for contrast
    "#ffd166",  # warm amber for volume bars
    "#4d9de0",  # calm blue for marks
    "#ff9f1c",  # vibrant orange overlay
    "#9f7aea",  # violet overlay intensity
    "#00c2d1",  # teal overlay counts
    "#f67280",  # soft pink overlay marks
]
PRIMARY_COLOR = COLORWAY[0]
INTENSITY_COLOR = COLORWAY[1]
COUNT_COLOR = COLORWAY[2]
MARKS_COLOR = COLORWAY[3]
OVERLAY_COLOR = COLORWAY[4]
OVERLAY_INTENSITY_COLOR = COLORWAY[5]
OVERLAY_COUNT_COLOR = COLORWAY[6]
OVERLAY_MARKS_COLOR = COLORWAY[7]

PRIMARY_HEX = _rgb_to_hex(PRIMARY_COLOR)
INTENSITY_HEX = _rgb_to_hex(INTENSITY_COLOR)
COUNT_HEX = _rgb_to_hex(COUNT_COLOR)
MARKS_HEX = _rgb_to_hex(MARKS_COLOR)
OVERLAY_HEX = _rgb_to_hex(OVERLAY_COLOR)
OVERLAY_INTENSITY_HEX = _rgb_to_hex(OVERLAY_INTENSITY_COLOR)
OVERLAY_COUNT_HEX = _rgb_to_hex(OVERLAY_COUNT_COLOR)
OVERLAY_MARKS_HEX = _rgb_to_hex(OVERLAY_MARKS_COLOR)
PLOT_BG_HEX = "#0b1220"
PAPER_BG_HEX = "#0b1220"
GRID_HEX = "#1f2a3c"
FONT_HEX = "#e6edf7"

FONT_FAMILY = '"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif'


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

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, facecolor=PAPER_BG_HEX)
    fig.patch.set_facecolor(PAPER_BG_HEX)
    for ax in axes:
        ax.set_facecolor(PLOT_BG_HEX)
        ax.tick_params(colors=FONT_HEX)
        ax.yaxis.label.set_color(FONT_HEX)
        ax.xaxis.label.set_color(FONT_HEX)
        ax.title.set_color(FONT_HEX)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_HEX)
        ax.grid(True, color=GRID_HEX, linewidth=0.6, alpha=0.35)

    axes[0].eventplot(timeline.times, colors=PRIMARY_HEX)
    axes[0].set_ylabel("Order arrivals")
    axes[0].grid(False)
    axes[0].set_title(title, color=FONT_HEX)
    axes[0].yaxis.label.set_color(FONT_HEX)

    axes[1].plot(
        timeline.grid_times,
        timeline.intensity_grid,
        color=INTENSITY_HEX,
        linewidth=2,
        label=labels[0],
    )
    axes[1].set_ylabel("λ(t) – intensity", color=FONT_HEX)

    bar_width = (
        timeline.bin_centres[1] - timeline.bin_centres[0]
        if timeline.bin_centres.size > 1
        else 0.1
    )
    axes[2].bar(
        timeline.bin_centres,
        timeline.bin_counts,
        width=bar_width,
        color=COUNT_HEX,
        alpha=0.7,
        label=labels[0],
    )
    axes[2].set_ylabel("Orders per bin", color=FONT_HEX)
    axes[2].set_xlabel("Time (s)", color=FONT_HEX)

    if show_marks and timeline.marks.size:
        ax_marks = axes[0].twinx()
        ax_marks.plot(
            timeline.times,
            timeline.marks,
            linestyle="none",
            marker="o",
            markersize=4,
            alpha=0.6,
            color=MARKS_HEX,
        )
        ax_marks.set_ylabel("Order size (marks)")
        ax_marks.tick_params(axis="y", labelcolor=MARKS_HEX, colors=MARKS_HEX)
        ax_marks.spines["right"].set_edgecolor(GRID_HEX)
        ax_marks.set_facecolor(PLOT_BG_HEX)

    if comparison is not None:
        axes[0].eventplot(comparison.times, colors=OVERLAY_HEX)
        axes[1].plot(
            comparison.grid_times,
            comparison.intensity_grid,
            color=OVERLAY_INTENSITY_HEX,
            linestyle="--",
            linewidth=2,
            label=labels[1],
        )
        axes[2].step(
            comparison.bin_centres,
            comparison.bin_counts,
            where="mid",
            color=OVERLAY_COUNT_HEX,
            label=labels[1],
        )

    if comparison is not None:
        legend1 = axes[1].legend(loc="upper right")
        if legend1 is not None:
            for text_item in legend1.get_texts():
                text_item.set_color(FONT_HEX)
        legend2 = axes[2].legend(loc="upper right")
        if legend2 is not None:
            for text_item in legend2.get_texts():
                text_item.set_color(FONT_HEX)

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
            marker=dict(color=PRIMARY_COLOR, size=6, opacity=0.75),
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
            line=dict(color=INTENSITY_COLOR, width=2.2),
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
            marker=dict(color=COUNT_COLOR, line=dict(width=0)),
            opacity=0.85,
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
                marker=dict(color=MARKS_COLOR, size=6, symbol="circle-open"),
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
                marker=dict(color=OVERLAY_COLOR, size=6, opacity=0.65),
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
                line=dict(color=OVERLAY_INTENSITY_COLOR, dash="dash", width=2),
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
                line=dict(color=OVERLAY_COUNT_COLOR, width=2),
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
                    marker=dict(color=OVERLAY_MARKS_COLOR, size=6, symbol="x"),
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

    axis_font = dict(color=FONT_HEX, family=FONT_FAMILY)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_HEX,
        zerolinecolor=GRID_HEX,
        linecolor=GRID_HEX,
        tickfont=axis_font,
        title=dict(font=axis_font),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_HEX,
        zerolinecolor=GRID_HEX,
        linecolor=GRID_HEX,
        tickfont=axis_font,
        title=dict(font=axis_font),
    )

    fig.update_layout(
        title=title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(11, 18, 32, 0.85)",
            bordercolor=GRID_HEX,
            borderwidth=1,
            font=dict(color=FONT_HEX, family=FONT_FAMILY),
        ),
        height=720,
        margin=dict(l=60, r=30, t=60, b=40),
        hovermode="x unified",
        template=None,
        colorway=COLORWAY,
        plot_bgcolor=PLOT_BG_HEX,
        paper_bgcolor=PAPER_BG_HEX,
        font=dict(color=FONT_HEX, family=FONT_FAMILY),
        hoverlabel=dict(
            bgcolor="#121c2d",
            font=dict(color=FONT_HEX, family=FONT_FAMILY),
        ),
    )
    return fig


__all__ = [
    "SimulationTimeline",
    "simulate_exp_timeline",
    "simulate_powerlaw_timeline",
    "plot_timeline",
    "plot_timeline_interactive",
]
