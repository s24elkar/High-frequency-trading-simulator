"""Utilities for locating the hawkes_bridge shared library."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


def _candidate_names() -> Iterable[str]:
    if sys.platform.startswith("darwin"):
        return ("libhawkes_bridge.dylib", "hawkes_bridge.dylib")
    if sys.platform.startswith("win"):
        return ("hawkes_bridge.dll",)
    return ("libhawkes_bridge.so", "hawkes_bridge.so")


def ensure_bridge_path() -> None:
    """Populate HFT_HAWKES_BRIDGE if unset and the build artefact exists."""

    if "HFT_HAWKES_BRIDGE" in os.environ:
        return

    project_root = Path(__file__).resolve().parents[1]
    for build_dir in sorted(project_root.glob("build*")):
        lib_dir = build_dir / "lib"
        if not lib_dir.exists():
            continue
        for name in _candidate_names():
            candidate = lib_dir / name
            if candidate.exists():
                os.environ["HFT_HAWKES_BRIDGE"] = str(candidate)
                return


__all__ = ["ensure_bridge_path"]
