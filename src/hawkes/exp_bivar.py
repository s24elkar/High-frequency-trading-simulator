"""Exponential-kernel bivariate Hawkes likelihood utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba optional for scaffolding
    def njit(signature=None, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper


@dataclass(slots=True)
class ExponentialParams:
    mu_b: float
    mu_s: float
    alpha_bb: float
    beta_bb: float
    alpha_bs: float
    beta_bs: float
    alpha_sb: float
    beta_sb: float
    alpha_ss: float
    beta_ss: float


def pack_params(params: Dict[str, float]) -> np.ndarray:
    """Pack dict of parameters into optimisation vector."""

    keys = (
        "mu_b",
        "mu_s",
        "alpha_bb",
        "beta_bb",
        "alpha_bs",
        "beta_bs",
        "alpha_sb",
        "beta_sb",
        "alpha_ss",
        "beta_ss",
    )
    return np.array([params[k] for k in keys], dtype=float)


def unpack_params(theta: np.ndarray) -> ExponentialParams:
    return ExponentialParams(*theta.tolist())


@njit(cache=True)
def loglik_exponential(
    theta: np.ndarray,
    tb_hist: np.ndarray,
    ts_hist: np.ndarray,
    tb: np.ndarray,
    ts: np.ndarray,
    T0: float,
    T1: float,
) -> Tuple[float, np.ndarray]:
    """Placeholder for exponential Hawkes log-likelihood.

    Currently returns `NotImplementedError` to highlight remaining work.
    """

    raise NotImplementedError("Exponential Hawkes likelihood not yet implemented")


def omega_matrix(params: ExponentialParams) -> np.ndarray:
    """Compute reproduction matrix Ω = α/β for exponential kernel."""

    return np.array(
        [
            [params.alpha_bb / params.beta_bb, params.alpha_bs / params.beta_bs],
            [params.alpha_sb / params.beta_sb, params.alpha_ss / params.beta_ss],
        ]
    )


__all__ = [
    "ExponentialParams",
    "pack_params",
    "unpack_params",
    "loglik_exponential",
    "omega_matrix",
]
