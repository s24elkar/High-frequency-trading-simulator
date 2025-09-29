"""CLI for running bivariate Hawkes calibration across windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from hawkes.fit import window_pipeline
from hawkes.io import load_config


def run(cfg_path: str | Path) -> None:
    cfg = load_config(cfg_path)
    windows_dir = Path(cfg.windows_dir)
    if not any(windows_dir.glob("window_*.npz")):
        raise FileNotFoundError(f"no windows found in {windows_dir}; run build_windows.py first")
    raise NotImplementedError("Full calibration pipeline pending implementation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hawkes calibration across windows")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
