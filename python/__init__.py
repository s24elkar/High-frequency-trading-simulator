"""Convenience exports for the analytics toolkit.

Plotly-backed timeline helpers are optional; in minimal environments we expose
lightweight shims that raise a helpful error if the user attempts to call them.
"""

from __future__ import annotations

from typing import Any

from .simulate import (
    simulate_thinning_exp_fast,
    simulate_thinning_general,
)
from .kernels import ExpKernel, PowerLawKernel

try:  # Prefer the real timeline helpers when Plotly is available.
    from .timeline_dashboard import (
        SimulationTimeline,
        simulate_exp_timeline,
        simulate_powerlaw_timeline,
        plot_timeline,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    _plotly_exc = exc

    SimulationTimeline = None  # type: ignore[assignment]

    def _plotly_missing(*_: Any, **__: Any) -> Any:
        raise ImportError(
            "plotly is required for timeline visualisations; install optional "
            "dependency `plotly` to enable these helpers."
        ) from _plotly_exc

    simulate_exp_timeline = _plotly_missing  # type: ignore[assignment]
    simulate_powerlaw_timeline = _plotly_missing  # type: ignore[assignment]
    plot_timeline = _plotly_missing  # type: ignore[assignment]


__all__ = [
    "simulate_thinning_exp_fast",
    "simulate_thinning_general",
    "ExpKernel",
    "PowerLawKernel",
    "SimulationTimeline",
    "simulate_exp_timeline",
    "simulate_powerlaw_timeline",
    "plot_timeline",
]
