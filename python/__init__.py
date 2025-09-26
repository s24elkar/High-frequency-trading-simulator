"""Convenience exports for the analytics toolkit."""

from .simulate import (
    simulate_thinning_exp_fast,
    simulate_thinning_general,
)
from .kernels import ExpKernel, PowerLawKernel
from .timeline_dashboard import (
    SimulationTimeline,
    simulate_exp_timeline,
    simulate_powerlaw_timeline,
    plot_timeline,
)

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
