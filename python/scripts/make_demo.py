"""Generate a short GIF showcasing the limit order book depth and trade tape."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import sys

# Use a writable Matplotlib cache when running in sandboxes or CI.
if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path.cwd() / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backtester.backtester import MarketEvent, MarketSnapshot
    from backtester.itch import load_lobster_csv, replay_from_lobster
    from backtester.order_book import PythonOrderBook
else:  # pragma: no cover - intact during package execution
    from ..backtester.backtester import MarketEvent, MarketSnapshot
    from ..backtester.itch import load_lobster_csv, replay_from_lobster
    from ..backtester.order_book import PythonOrderBook


@dataclass(slots=True)
class TradeEntry:
    timestamp_ns: int
    price: float
    size: float
    aggressor: str


@dataclass(slots=True)
class FrameState:
    timestamp_ns: int
    event_type: str
    event_side: str | None
    bids: List[tuple[float, float]]
    asks: List[tuple[float, float]]
    tape: List[TradeEntry]
    best_bid: float | None
    best_ask: float | None


def _extract_levels(
    snapshot: MarketSnapshot, side: str, depth: int
) -> List[tuple[float, float]]:
    raw = [entry for entry in snapshot.depth if entry.get("side") == side]
    if side == "BUY":
        raw.sort(key=lambda item: item.get("price", 0.0), reverse=True)
    else:
        raw.sort(key=lambda item: item.get("price", 0.0))
    return [
        (float(entry.get("price", 0.0)), float(entry.get("size", 0.0)))
        for entry in raw[:depth]
    ]


def _collect_frames(
    events: Sequence[MarketEvent],
    *,
    depth: int,
    tape_length: int,
) -> List[FrameState]:
    book = PythonOrderBook(depth=depth)
    frames: List[FrameState] = []
    tape: List[TradeEntry] = []

    for event in events:
        update = book.apply_event(event)
        snapshot = update.snapshot
        if snapshot is None:
            continue
        for fill in update.fills:
            aggressor = "BUY" if fill.side == "SELL" else "SELL"
            tape.append(
                TradeEntry(
                    timestamp_ns=fill.timestamp_ns,
                    price=fill.price,
                    size=fill.size,
                    aggressor=aggressor,
                )
            )
        if len(tape) > tape_length:
            tape = tape[-tape_length:]
        bids = _extract_levels(snapshot, "BUY", depth)
        asks = _extract_levels(snapshot, "SELL", depth)
        frames.append(
            FrameState(
                timestamp_ns=snapshot.timestamp_ns,
                event_type=event.event_type,
                event_side=event.payload.get("side") if event.payload else None,
                bids=bids,
                asks=asks,
                tape=list(tape),
                best_bid=snapshot.best_bid,
                best_ask=snapshot.best_ask,
            )
        )

    return frames


def _format_time(base_ns: int, current_ns: int) -> str:
    delta = (current_ns - base_ns) / 1e9
    return f"+{delta:0.4f}s"


def _format_tape_entries(entries: Sequence[TradeEntry], base_ns: int) -> str:
    if not entries:
        return "No trades yet"
    lines = ["Trade tape (latest first)"]
    for trade in reversed(entries):
        delta = (trade.timestamp_ns - base_ns) / 1e9
        lines.append(
            f"+{delta:0.4f}s  {trade.aggressor:<4}  {trade.price:7.2f}  x {trade.size:5.2f}"
        )
    return "\n".join(lines)


def _build_animation(
    frames: Sequence[FrameState],
    *,
    output_path: Path,
    depth: int,
    fps: int,
) -> None:
    if not frames:
        raise ValueError("No frames produced; ensure the dataset contains events")

    base_ts = frames[0].timestamp_ns
    max_size = max(
        [max((size for _, size in frame.bids), default=0.0) for frame in frames]
        + [max((size for _, size in frame.asks), default=0.0) for frame in frames]
    )
    max_size = max(max_size, 1.0)

    y_positions = np.arange(depth)
    fig, (ax_book, ax_tape) = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.6),
        gridspec_kw={"width_ratios": [2.3, 1.7]},
    )

    ax_book.set_title("Top 5 depth levels")
    ax_book.set_xlabel("Visible size")
    ax_book.set_yticks(y_positions)
    ax_book.set_yticklabels([f"L{i + 1}" for i in range(depth)])
    ax_book.set_xlim(-max_size * 1.1, max_size * 1.1)
    ax_book.set_ylim(-0.5, depth - 0.5)
    ax_book.axvline(0.0, color="#4d4d4d", linewidth=1.0)

    bid_bars = ax_book.barh(
        y_positions,
        [-1e-9] * depth,
        color="#2ca02c",
        align="center",
    )
    ask_bars = ax_book.barh(
        y_positions,
        [1e-9] * depth,
        color="#d62728",
        align="center",
    )
    bid_texts = [
        ax_book.text(
            -max_size * 0.02,
            y,
            "",
            ha="right",
            va="center",
            color="#1b5e20",
            fontsize=9,
        )
        for y in y_positions
    ]
    ask_texts = [
        ax_book.text(
            max_size * 0.02,
            y,
            "",
            ha="left",
            va="center",
            color="#7f1d1d",
            fontsize=9,
        )
        for y in y_positions
    ]

    ax_tape.set_title("Trade tape")
    ax_tape.axis("off")
    tape_text = ax_tape.text(
        0.0,
        1.0,
        "",
        ha="left",
        va="top",
        fontsize=10,
        transform=ax_tape.transAxes,
    )

    event_text = fig.text(0.5, 0.94, "", ha="center", va="center", fontsize=12)
    touch_text = fig.text(0.5, 0.04, "", ha="center", va="center", fontsize=10)

    def _update(frame: FrameState) -> None:
        bids = frame.bids + [(0.0, 0.0)] * (depth - len(frame.bids))
        asks = frame.asks + [(0.0, 0.0)] * (depth - len(frame.asks))
        for idx, (price, size) in enumerate(bids):
            bid_bars[idx].set_width(-size)
            label = f"{price:7.2f} | {size:5.2f}" if size > 0 else ""
            bid_texts[idx].set_text(label)
        for idx, (price, size) in enumerate(asks):
            ask_bars[idx].set_width(size)
            label = f"{price:7.2f} | {size:5.2f}" if size > 0 else ""
            ask_texts[idx].set_text(label)

        side = frame.event_side or ""
        side_label = f" ({side})" if side else ""
        event_text.set_text(
            f"Event {frame.event_type}{side_label} at {_format_time(base_ts, frame.timestamp_ns)}"
        )
        touch_text.set_text(
            f"Best bid: {frame.best_bid if frame.best_bid is not None else '–'}  |  "
            f"Best ask: {frame.best_ask if frame.best_ask is not None else '–'}"
        )
        tape_text.set_text(_format_tape_entries(frame.tape[-8:], base_ts))

    FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=160,
        repeat=False,
    ).save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _load_events(dataset_path: Path, symbol: str) -> List[MarketEvent]:
    messages = list(load_lobster_csv(dataset_path, symbol=symbol))
    return list(replay_from_lobster(messages))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_dataset = repo_root / "data" / "sample" / "lobster_demo.csv"
    default_output = repo_root / "assets" / "demo.gif"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to the sample LOBSTER-style CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Where to write the demo GIF.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="DEMO",
        help="Symbol tag to associate with the dataset.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Number of price levels to display per side.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for the GIF output.",
    )
    parser.add_argument(
        "--tape-length",
        type=int,
        default=12,
        help="Maximum number of trades to keep in the rolling tape display.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = args.dataset
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    events = _load_events(dataset, args.symbol)
    frames = _collect_frames(events, depth=args.depth, tape_length=args.tape_length)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    _build_animation(frames, output_path=output, depth=args.depth, fps=args.fps)
    print(f"Saved demo GIF to {output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
