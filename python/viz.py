# viz.py
import numpy as np
import matplotlib.pyplot as plt
import inspect
from itertools import cycle

_STEM_SUPPORTS_LINE_COLLECTION = (
    "use_line_collection" in inspect.signature(plt.stem).parameters
)

_COLOUR_CYCLE = cycle(
    [
        "#f94144",
        "#f3722c",
        "#f8961e",
        "#f9c74f",
        "#90be6d",
        "#43aa8b",
        "#577590",
    ]
)


def _stem(x, y):
    colour = next(_COLOUR_CYCLE)
    kwargs = {"use_line_collection": True} if _STEM_SUPPORTS_LINE_COLLECTION else {}
    stem = plt.stem(x, y, **kwargs)
    stem.markerline.set_color(colour)
    stem.stemlines.set_color(colour)
    stem.baseline.set_color("#4d4d4d")
    return stem


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
    if x.size == 0:
        raise ValueError("Autocorrelation is undefined for an empty input array.")
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2 :]
    ac = ac[: max_lag + 1]
    return ac / ac[0] if ac[0] != 0 else ac


def plot_intensity(t_grid, lam_grid, title="Intensity over time"):
    plt.figure()
    plt.plot(t_grid, lam_grid, color=next(_COLOUR_CYCLE), linewidth=1.6)
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
    marks = np.asarray(marks, float)
    if marks.size == 0:
        plt.figure()
        plt.title(title)
        plt.text(0.5, 0.5, "No marks available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        return

    rho = acf(marks, max_lag)
    plt.figure()
    _stem(np.arange(len(rho)), rho)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(title)
    plt.tight_layout()
