"""Utility functions for persisting simulated event streams."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_csv(path: str | Path, times: Iterable[float], marks: Iterable[float]) -> None:
    _ensure_parent(path)
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "mark"])
        for t, v in zip(times, marks):
            writer.writerow([f"{float(t):.12g}", f"{float(v):.12g}"])


def save_json(
    path: str | Path,
    meta: dict[str, object],
    times: Iterable[float],
    marks: Iterable[float],
) -> None:
    _ensure_parent(path)
    payload = dict(meta)
    payload["events"] = [{"t": float(t), "v": float(v)} for t, v in zip(times, marks)]
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
