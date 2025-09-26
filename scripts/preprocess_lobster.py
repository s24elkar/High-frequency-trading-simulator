#!/usr/bin/env python3
"""Convert LOBSTER message data into neural Hawkes NPZ format.

Example usage:
    python scripts/preprocess_lobster.py \
        --messages data/lobster/AAPL_2012-06-21_34200000_57600000_messages.csv \
        --symbol AAPL --date 2012-06-21 \
        --output data/runs/events/lobster_aapl_2012-06-21.npz

The script produces two artefacts: the NPZ file with `times` and `types` object
arrays (one array per sequence/window) and a companion `.meta.json` capturing
basic metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

LOBSTER_MESSAGE_COLUMNS = [
    "time",
    "type",
    "order_id",
    "size",
    "price",
    "direction",
]

TRADE_TYPES = {4, 5}  # partial / full execution


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack LOBSTER messages into NPZ format")
    parser.add_argument("--messages", type=Path, required=True, help="Path to LOBSTER *_messages.csv")
    parser.add_argument("--output", type=Path, required=True, help="Destination NPZ path")
    parser.add_argument("--symbol", type=str, required=True, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--date", type=str, required=True, help="Trading date (YYYY-MM-DD)")
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=3600,
        help="Window length in seconds for splitting the trading session",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1800,
        help="Stride in seconds between consecutive windows",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=50,
        help="Discard windows with fewer than this many events",
    )
    parser.add_argument(
        "--num-types",
        type=int,
        default=2,
        help="Number of event types (default: buys vs sells)",
    )
    return parser.parse_args(argv)


def load_messages(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=LOBSTER_MESSAGE_COLUMNS)
    return df


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    trades = df[df["type"].isin(TRADE_TYPES)].copy()
    trades.sort_values("time", inplace=True)
    trades.reset_index(drop=True, inplace=True)
    return trades


def build_sequences(
    trades: pd.DataFrame,
    *,
    window_seconds: int,
    window_stride: int,
    min_events: int,
) -> List[Dict[str, np.ndarray]]:
    if trades.empty:
        return []

    t0 = trades["time"].iloc[0]
    rel_time = trades["time"].to_numpy(dtype=np.float64) - float(t0)
    directions = trades["direction"].to_numpy(dtype=np.int64)
    types = np.where(directions > 0, 0, 1)  # 0: buy, 1: sell

    windows: List[Dict[str, np.ndarray]] = []
    start = 0.0
    max_time = rel_time[-1]
    while start < max_time:
        stop = start + window_seconds
        mask = (rel_time >= start) & (rel_time < stop)
        idx = np.nonzero(mask)[0]
        if idx.size >= min_events:
            slice_times = rel_time[idx] - start
            slice_types = types[idx]
            if np.any(np.diff(slice_times) < 0):
                order = np.argsort(slice_times, kind="mergesort")
                slice_times = slice_times[order]
                slice_types = slice_types[order]
            windows.append({"times": slice_times.astype(np.float64), "types": slice_types.astype(np.int64)})
        start += window_stride
    if not windows:
        windows.append({"times": rel_time.astype(np.float64), "types": types.astype(np.int64)})
    return windows


def write_npz(output: Path, sequences: List[Dict[str, np.ndarray]]) -> Tuple[int, int]:
    times = np.array([seq["times"] for seq in sequences], dtype=object)
    types = np.array([seq["types"] for seq in sequences], dtype=object)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, times=times, types=types)
    total_events = int(sum(seq["times"].size for seq in sequences))
    return times.size, total_events


def write_meta(meta_path: Path, payload: Dict[str, object]) -> None:
    with meta_path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    df = load_messages(args.messages)
    trades = extract_trades(df)
    sequences = build_sequences(
        trades,
        window_seconds=args.window_seconds,
        window_stride=args.window_stride,
        min_events=args.min_events,
    )
    if not sequences:
        raise SystemExit("No trade sequences extracted; adjust parameters or verify input")

    seq_count, event_count = write_npz(args.output, sequences)
    meta = {
        "venue": "lobster",
        "symbol": args.symbol,
        "date": args.date,
        "window_seconds": args.window_seconds,
        "window_stride": args.window_stride,
        "min_events": args.min_events,
        "num_sequences": seq_count,
        "num_events": event_count,
        "num_types": args.num_types,
        "created_by": "scripts/preprocess_lobster.py",
        "messages": str(args.messages),
    }
    write_meta(args.output.with_suffix(".meta.json"), meta)
    print(f"Wrote {seq_count} sequences ({event_count} events) to {args.output}")


if __name__ == "__main__":
    main()
