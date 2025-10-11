"""Utilities for producing reproducible analytics artefacts."""

from __future__ import annotations

from .io import ArtifactWriter, ReportMetadata, detect_git_commit
from .profiling import Hotspot, ProfileResult, profile_capture, stats_to_hotspots
from .plots import (
    ensure_matplotlib_backend,
    plot_latency_histogram,
    plot_metric_bars,
    plot_order_trade_ratio,
    plot_throughput_series,
)

__all__ = [
    "ArtifactWriter",
    "ReportMetadata",
    "detect_git_commit",
    "Hotspot",
    "ProfileResult",
    "profile_capture",
    "stats_to_hotspots",
    "ensure_matplotlib_backend",
    "plot_latency_histogram",
    "plot_metric_bars",
    "plot_order_trade_ratio",
    "plot_throughput_series",
]
