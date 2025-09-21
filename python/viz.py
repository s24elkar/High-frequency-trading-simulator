# viz.py
import numpy as np
import matplotlib.pyplot as plt
import inspect

_STEM_SUPPORTS_LINE_COLLECTION = 'use_line_collection' in inspect.signature(plt.stem).parameters

def _stem(x, y):
    kwargs = {'use_line_collection': True} if _STEM_SUPPORTS_LINE_COLLECTION else {}
    return plt.stem(x, y, **kwargs)

def intensity_on_grid(mu, kernel, times, marks, t_grid):
    lam = np.full_like(t_grid, mu, dtype=float)
    for ti, vi in zip(times, marks):
        lam += kernel.phi(t_grid - ti, vi)
    return lam

def binned_counts(times, T, bin_width):
    edges = np.arange(0, T + bin_width, bin_width)
    counts, _ = np.histogram(times, bins=edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return mids, counts

def acf(x, max_lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    ac = np.correlate(x, x, mode='full')
    ac = ac[ac.size // 2:]
    ac = ac[:max_lag+1]
    return ac / ac[0] if ac[0] != 0 else ac

def plot_intensity(t_grid, lam_grid, title="Intensity over time"):
    plt.figure()
    plt.plot(t_grid, lam_grid)
    plt.xlabel("Time")
    plt.ylabel("λ(t)")
    plt.title(title)
    plt.tight_layout()

def plot_counts_acf(times, T, bin_w=0.1, max_lag=50, title="ACF of binned arrivals"):
    mids, counts = binned_counts(times, T, bin_w)
    rho = acf(counts, max_lag)
    plt.figure()
    _stem(np.arange(len(rho)), rho)
    plt.xlabel("Lag (bins)")
    plt.ylabel("Autocorrelation")
    plt.title(title)
    plt.tight_layout()

def plot_mark_acf(marks, max_lag=50, title="ACF of marks (volumes)"):
    rho = acf(marks, max_lag)
    plt.figure()
    _stem(np.arange(len(rho)), rho)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(title)
    plt.tight_layout()
