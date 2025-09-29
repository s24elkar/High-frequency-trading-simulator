"""Model fitting orchestration for bivariate Hawkes processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize

from . import exp_bivar, pow_bivar
from .io import WindowConfig


@dataclass(slots=True)
class FitResult:
    model: str
    success: bool
    message: str
    theta: np.ndarray
    omega: np.ndarray
    spectral_radius: float
    nll: float
    metadata: Dict[str, float | int | str]


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(x))


def fit_window(
    model: str,
    tb_hist: np.ndarray,
    ts_hist: np.ndarray,
    tb: np.ndarray,
    ts: np.ndarray,
    window: Tuple[float, float],
    cfg: WindowConfig,
) -> FitResult:
    """Placeholder for optimisation routine.

    Raises
    ------
    NotImplementedError
        Until the full optimisation pipeline is implemented.
    """

    raise NotImplementedError("fit_window implementation pending")


def window_pipeline(
    data: Iterable[Tuple[int, float, float, Tuple[np.ndarray, ...]]],
    cfg: WindowConfig,
) -> List[FitResult]:
    """Iterate over windows and call `fit_window` for configured models."""

    results: List[FitResult] = []
    for window_id, start, end, arrays in data:
        for model in cfg.kernel_family:
            try:
                res = fit_window(model, *arrays, (start, end), cfg)
            except NotImplementedError as exc:  # pragma: no cover - scaffolding stage
                res = FitResult(
                    model=model,
                    success=False,
                    message=str(exc),
                    theta=np.array([]),
                    omega=np.zeros((2, 2)),
                    spectral_radius=0.0,
                    nll=float("nan"),
                    metadata={
                        "window_id": window_id,
                        "start": start,
                        "end": end,
                    },
                )
            results.append(res)
    return results


__all__ = ["FitResult", "fit_window", "window_pipeline"]
