"""Ensure the project's Python package directory is on sys.path for tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "python"
SRC_DIR = ROOT / "src"
for candidate in (PACKAGE_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
