"""Power-law kernel utilities for bivariate Hawkes models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(signature=None, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper


@dataclass(slots=True)
class PowerLawParams:
    mu_b: float
    mu_s: float
    eta_bb: float
    c_bb: float
    gamma_bb: float
    eta_bs: float
    c_bs: float
    gamma_bs: float
    eta_sb: float
    c_sb: float
    gamma_sb: float
    eta_ss: float
    c_ss: float
    gamma_ss: float


def pack_params(params: Dict[str, float]) -> np.ndarray:
    keys = (
        "mu_b",
        "mu_s",
        "eta_bb",
        "c_bb",
        "gamma_bb",
        "eta_bs",
        "c_bs",
        "gamma_bs",
        "eta_sb",
        "c_sb",
        "gamma_sb",
        "eta_ss",
        "c_ss",
        "gamma_ss",
    )
    return np.array([params[k] for k in keys], dtype=float)


def unpack_params(theta: np.ndarray) -> PowerLawParams:
    return PowerLawParams(*theta.tolist())


@njit(cache=True)
def loglik_powerlaw(
    theta: np.ndarray,
    tb_hist: np.ndarray,
    ts_hist: np.ndarray,
    tb: np.ndarray,
    ts: np.ndarray,
    T0: float,
    T1: float,
    truncation_window: float,
) -> Tuple[float, np.ndarray]:
    raise NotImplementedError("Power-law Hawkes likelihood not yet implemented")


def omega_matrix(params: PowerLawParams) -> np.ndarray:
    return np.array(
        [
            [params.eta_bb / (params.gamma_bb * params.c_bb ** params.gamma_bb), params.eta_bs / (params.gamma_bs * params.c_bs ** params.gamma_bs)],
            [params.eta_sb / (params.gamma_sb * params.c_sb ** params.gamma_sb), params.eta_ss / (params.gamma_ss * params.c_ss ** params.gamma_ss)],
        ]
    )


__all__ = [
    "PowerLawParams",
    "pack_params",
    "unpack_params",
    "loglik_powerlaw",
    "omega_matrix",
]
