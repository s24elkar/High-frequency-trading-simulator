# simulate.py
import numpy as np
from typing import Callable, Tuple

try:  # allow usage both as package and as standalone scripts
    from .kernels import ExpKernel, PowerLawKernel
except (ImportError, ValueError):
    from kernels import ExpKernel, PowerLawKernel

def simulate_thinning_general(mu: float,
                              kernel,
                              mark_sampler: Callable[[np.random.Generator], float],
                              T: float,
                              seed: int = 12345) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ogata/Lewis–Shedler thinning with monotone bound λ* = current λ (valid for decreasing kernels).
    Works for any kernel with decreasing u ↦ φ(u, ·).
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    times, marks = [], []
    # current intensity contribution from past events (at current t)
    S_cur = 0.0

    while t < T:
        lam_star = mu + S_cur
        if lam_star <= 0:
            break
        w = rng.exponential(1.0 / lam_star)  # candidate wait time
        t_cand = t + w
        if t_cand > T:
            break
        # compute exact λ(t_cand)
        if times:
            u = t_cand - np.array(times)
            S_tc = np.sum(kernel.phi(u, np.array(marks)))
        else:
            S_tc = 0.0
        lam_tc = mu + S_tc
        if rng.uniform() <= (lam_tc / lam_star):
            # accept event
            v = mark_sampler(rng)
            times.append(t_cand)
            marks.append(v)
            # intensity right after jump: add φ(0, v)
            S_cur = S_tc + kernel.phi(0.0, v)
            t = t_cand
        else:
            # reject; move time forward, intensity decays accordingly (recompute)
            S_cur = S_tc
            t = t_cand
    return np.array(times), np.array(marks)

def simulate_thinning_exp_fast(mu: float,
                               kernel: ExpKernel,
                               mark_sampler: Callable[[np.random.Generator], float],
                               T: float,
                               seed: int = 12345):
    """
    Specialized Ogata thinning using a scalar state S(t) = sum α v e^{-β(t-T_i)}.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    times, marks = [], []
    S = 0.0  # self-excitation state

    while t < T:
        lam_star = mu + S
        if lam_star <= 0:
            break
        w = rng.exponential(1.0 / lam_star)
        t_cand = t + w
        if t_cand > T:
            break
        # decay state to t_cand
        S_cand = kernel.decay_state(S, w)
        lam_tc = mu + S_cand
        if rng.uniform() <= lam_tc / lam_star:
            # accept
            v = mark_sampler(rng)
            times.append(t_cand)
            marks.append(v)
            S = S_cand + kernel.jump(v)  # add instantaneous jump
            t = t_cand
        else:
            # reject; just move forward with decayed state
            S = S_cand
            t = t_cand
    return np.array(times), np.array(marks)
