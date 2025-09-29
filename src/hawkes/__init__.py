"""Bivariate Hawkes calibration toolkit.

Modules provide data loading, likelihood evaluation, model fitting, and
post-fit diagnostics for multivariate Hawkes processes on trade data.
"""

from .io import load_window, WindowConfig, load_config
from .fit import fit_window, window_pipeline
from .diagnostics import compute_residuals, ks_test, qq_points

__all__ = [
    "load_window",
    "WindowConfig",
    "load_config",
    "fit_window",
    "window_pipeline",
    "compute_residuals",
    "ks_test",
    "qq_points",
]
