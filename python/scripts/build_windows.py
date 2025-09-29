"""Generate rolling Hawkes calibration windows from cleaned trade data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import pandas as pd

from hawkes.io import WindowConfig, iter_windows, load_config


def _load_trades(cfg: WindowConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.dataset)
    required = {"t", "side"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)}")
    return df


def _save_window(
    cfg: WindowConfig,
    window_id: int,
    start: float,
    end: float,
    arrays,
) -> Dict[str, float | int]:
    buy_hist, sell_hist, buy, sell = arrays
    path = Path(cfg.windows_dir) / f"window_{window_id:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "window_id": window_id,
        "start": start,
        "end": end,
        "buy_hist": int(buy_hist.size),
        "sell_hist": int(sell_hist.size),
        "buy": int(buy.size),
        "sell": int(sell.size),
    }
    npz_kwargs = {
        "buy_hist": buy_hist,
        "sell_hist": sell_hist,
        "buy": buy,
        "sell": sell,
    }
    import numpy as np

    np.savez(path, **npz_kwargs)
    log_path = Path(cfg.logs_dir) / "window_metadata.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(metadata) + "\n")
    return metadata


def build_windows(cfg: WindowConfig) -> None:
    df = _load_trades(cfg)
    coverage_threshold = cfg.min_window_coverage
    logs = []
    for idx, start, end, arrays in iter_windows(df, cfg):
        buy = arrays[2]
        window_total = cfg.window_seconds
        observed_span = (buy[-1] - buy[0]) if buy.size > 1 else 0.0
        if coverage_threshold and observed_span < coverage_threshold * window_total:
            logs.append({"window_id": idx, "status": "skipped", "reason": "sparse"})
            continue
        metadata = _save_window(cfg, idx, start, end, arrays)
        metadata["status"] = "saved"
        logs.append(metadata)
    log_path = Path(cfg.logs_dir) / "window_summary.json"
    log_path.write_text(json.dumps(logs, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Hawkes calibration windows")
    parser.add_argument(
        "--config", required=True, help="Path to day14_binance YAML config"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_windows(cfg)


if __name__ == "__main__":
    main()
