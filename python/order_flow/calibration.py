"""Log-likelihood utilities and calibration helpers for order-flow models."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, minimize


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("times/marks must be one-dimensional sequences")
    return arr


def log_likelihood_poisson(
    times: Sequence[float] | np.ndarray,
    mu: float,
    horizon: Optional[float] = None,
) -> float:
    """Compute the log-likelihood of a homogeneous Poisson process."""
    if mu <= 0:
        raise ValueError("mu must be positive")

    arr = _as_array(times)
    if horizon is None:
        horizon = float(arr[-1]) if arr.size else 0.0
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if arr.size and np.any(arr < 0):
        raise ValueError("event times must be non-negative")

    return arr.size * np.log(mu) - mu * horizon


def log_likelihood_hawkes_exp(
    times: Sequence[float] | np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
    *,
    marks: Optional[Sequence[float] | np.ndarray] = None,
    horizon: Optional[float] = None,
) -> float:
    """Compute the log-likelihood of an exponential Hawkes process."""
    if mu <= 0 or alpha < 0 or beta <= 0:
        raise ValueError("mu > 0, alpha >= 0, beta > 0 required")

    times_arr = _as_array(times)
    if marks is None:
        marks_arr = np.ones_like(times_arr)
    else:
        marks_arr = _as_array(marks)
        if marks_arr.shape != times_arr.shape:
            raise ValueError("marks must match the shape of times")

    if times_arr.size and np.any(np.diff(times_arr) < 0):
        raise ValueError("event times must be non-decreasing")

    if horizon is None:
        horizon = float(times_arr[-1]) if times_arr.size else 0.0
    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    state = 0.0
    last_time = 0.0
    intensities = np.empty_like(times_arr)

    for idx, (t_i, mark_i) in enumerate(zip(times_arr, marks_arr)):
        dt = t_i - last_time
        if dt < 0:
            raise ValueError("event times must be non-decreasing")
        state *= np.exp(-beta * dt)
        intensities[idx] = mu + alpha * state
        state += mark_i
        last_time = t_i

    if np.any(intensities <= 0):
        return -np.inf

    integral_tail = np.sum(
        marks_arr * (1.0 - np.exp(-beta * (horizon - times_arr)))
    ) / beta
    integral = mu * horizon + alpha * integral_tail
    return float(np.sum(np.log(intensities)) - integral)


def _neg_log_likelihood_hawkes_exp(
    params: np.ndarray,
    times: np.ndarray,
    marks: np.ndarray,
    horizon: float,
) -> float:
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return np.inf
    return -log_likelihood_hawkes_exp(
        times,
        mu,
        alpha,
        beta,
        marks=marks,
        horizon=horizon,
    )


def fit_hawkes_exponential_mle(
    times: Sequence[float] | np.ndarray,
    *,
    marks: Optional[Sequence[float] | np.ndarray] = None,
    horizon: Optional[float] = None,
    initial: Optional[Tuple[float, float, float]] = None,
    bounds: Optional[
        Sequence[Tuple[float | None, float | None]]
    ] = None,
) -> OptimizeResult:
    """Estimate (mu, alpha, beta) by maximising the exponential Hawkes log-likelihood."""
    times_arr = _as_array(times)
    if marks is None:
        marks_arr = np.ones_like(times_arr)
    else:
        marks_arr = _as_array(marks)
        if marks_arr.shape != times_arr.shape:
            raise ValueError("marks must match times")

    if horizon is None:
        horizon = float(times_arr[-1]) if times_arr.size else 0.0

    if initial is None:
        mu0 = max(1e-3, times_arr.size / max(horizon, 1.0))
        alpha0 = 0.5 * mu0
        beta0 = 1.0
        initial = (mu0, alpha0, beta0)

    if bounds is None:
        bounds = (
            (1e-8, None),
            (0.0, None),
            (1e-8, None),
        )

    result = minimize(
        _neg_log_likelihood_hawkes_exp,
        x0=np.asarray(initial, dtype=float),
        args=(times_arr, marks_arr, float(horizon)),
        bounds=bounds,
        method="L-BFGS-B",
    )
    result.log_likelihood_ = (
        -result.fun if np.isfinite(result.fun) else -np.inf
    )
    return result


__all__ = [
    "log_likelihood_poisson",
    "log_likelihood_hawkes_exp",
    "fit_hawkes_exponential_mle",
]
