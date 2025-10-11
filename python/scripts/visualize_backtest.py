"""Plot key diagnostics from a JSONL backtest log."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from python.analysis import (
    ArtifactWriter,
    ReportMetadata,
    detect_git_commit,
    ensure_matplotlib_backend,
)
from python.backtester.reports import BacktestRun, load_run


def _to_seconds(timestamps: Sequence[int]) -> np.ndarray:
    if not timestamps:
        return np.array([])
    base = timestamps[0]
    return (np.array(timestamps, dtype=float) - float(base)) / 1e9


def _format_latency(ns: Optional[float]) -> str:
    if ns is None:
        return "n/a"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f} µs"
    return f"{ns:.0f} ns"


def _format_duration(ns: Optional[float]) -> str:
    if ns is None:
        return "n/a"
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f} µs"
    return f"{ns:.0f} ns"


def _format_summary_lines(summary: dict[str, float | int | None]) -> list[str]:
    if not summary:
        return ["No run_summary event found"]

    lines = []
    symbol = summary.get("symbol")
    digest = summary.get("digest")
    header_bits = []
    if symbol:
        header_bits.append(f"Symbol {symbol}")
    if digest:
        header_bits.append(f"digest {digest}")
    if header_bits:
        lines.append(" | ".join(header_bits))

    orders = summary.get("orders")
    fills = summary.get("fills")
    otr = summary.get("order_to_trade_ratio")
    fill_eff = summary.get("fill_efficiency")
    activity_line = []
    if orders is not None:
        activity_line.append(f"orders {orders}")
    if fills is not None:
        activity_line.append(f"fills {fills}")
    if otr is not None:
        activity_line.append(f"O/T {otr:.2f}")
    if fill_eff is not None:
        activity_line.append(f"fill eff {fill_eff * 100:.1f}%")
    if activity_line:
        lines.append(" | ".join(activity_line))

    order_vol = summary.get("order_volume")
    fill_vol = summary.get("fill_volume")
    vol_line = []
    if order_vol is not None:
        vol_line.append(f"order vol {order_vol:,.2f}")
    if fill_vol is not None:
        vol_line.append(f"fill vol {fill_vol:,.2f}")
    if vol_line:
        lines.append(" | ".join(vol_line))

    realized = summary.get("PnL_realized")
    unrealized = summary.get("PnL_unrealized")
    inventory = summary.get("inventory")
    pnl_line = []
    if realized is not None:
        pnl_line.append(f"realized PnL {realized:,.2f}")
    if unrealized is not None:
        pnl_line.append(f"unrealized PnL {unrealized:,.2f}")
    if inventory is not None:
        pnl_line.append(f"inventory {inventory:,.2f}")
    if pnl_line:
        lines.append(" | ".join(pnl_line))

    latencies = []
    latencies.append(f"avg {_format_latency(summary.get('avg_latency_ns'))}")
    latencies.append(f"p95 {_format_latency(summary.get('p95_latency_ns'))}")
    latencies.append(f"max {_format_latency(summary.get('max_latency_ns'))}")
    lines.append("latency " + ", ".join(latencies))

    duration = summary.get("duration_ns")
    if duration is not None:
        lines.append(f"duration {_format_duration(duration)}")

    return lines


def _plot_mid_price(ax, run: BacktestRun) -> None:
    timestamps = [snap.timestamp_ns for snap in run.snapshots if snap.mid is not None]
    if not timestamps:
        ax.set_visible(False)
        return
    seconds = _to_seconds(timestamps)
    mids = [snap.mid for snap in run.snapshots if snap.mid is not None]
    ax.plot(seconds, mids, color="#1f77b4", linewidth=1.6)
    ax.set_ylabel("Mid price")
    ax.set_title("Mid-price path")


def _plot_order_activity(ax, run: BacktestRun) -> None:
    if not run.orders and not run.fills:
        ax.set_visible(False)
        return
    order_ts = [o.timestamp_ns for o in run.orders]
    fill_ts = [f.timestamp_ns for f in run.fills]
    xs_orders = _to_seconds(order_ts)
    xs_fills = _to_seconds(fill_ts)
    if xs_orders.size:
        ax.step(
            xs_orders,
            np.arange(1, xs_orders.size + 1),
            where="post",
            label="Orders",
            color="#ff7f0e",
        )
    if xs_fills.size:
        ax.step(
            xs_fills,
            np.arange(1, xs_fills.size + 1),
            where="post",
            label="Fills",
            color="#2ca02c",
        )
    ax.set_ylabel("Count")
    ax.set_title("Order vs fill counts")
    ax.legend(loc="upper left")


def _plot_latency(ax, run: BacktestRun) -> None:
    latencies = [
        o.latency_ns
        for o in run.orders
        if o.latency_ns is not None and o.latency_ns >= 0
    ]
    if not latencies:
        ax.set_visible(False)
        return
    latencies_us = np.array(latencies, dtype=float) / 1_000.0
    bins = min(30, max(5, int(math.sqrt(latencies_us.size))))
    ax.hist(latencies_us, bins=bins, color="#9467bd", alpha=0.75)
    ax.set_xlabel("Latency (µs)")
    ax.set_ylabel("Orders")
    ax.set_title("Order placement latency distribution")


def _summary_dict(run: BacktestRun) -> dict[str, float | int | None]:
    summary = run.summary
    if summary is None:
        return {}
    return {
        "symbol": summary.symbol,
        "orders": summary.order_count,
        "fills": summary.fill_count,
        "order_volume": summary.order_volume,
        "fill_volume": summary.fill_volume,
        "PnL_realized": summary.realized_pnl,
        "PnL_unrealized": summary.unrealized_pnl,
        "inventory": summary.inventory,
        "order_to_trade_ratio": summary.order_to_trade_ratio,
        "fill_efficiency": summary.fill_efficiency,
        "avg_latency_ns": summary.avg_latency_ns,
        "p95_latency_ns": summary.p95_latency_ns,
        "max_latency_ns": summary.max_latency_ns,
        "duration_ns": summary.duration_ns,
        "digest": summary.digest,
    }


def visualise_run(
    run: BacktestRun,
    output: Path | None,
    plt_module,
    writer: ArtifactWriter | None = None,
) -> None:
    fig, axes = plt_module.subplots(3, 1, sharex=True, figsize=(10, 11))
    _plot_mid_price(axes[0], run)
    _plot_order_activity(axes[1], run)
    _plot_latency(axes[2], run)

    summary_payload = _summary_dict(run)
    if summary_payload:
        fig.suptitle("Backtest summary", fontsize=14, fontweight="bold")
        fig.text(
            0.02,
            0.02,
            "\n".join(_format_summary_lines(summary_payload)),
            fontsize=9,
            ha="left",
            va="bottom",
            family="monospace",
        )
    fig.supxlabel("Elapsed time (s)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        if writer is not None:
            writer.attach_metadata(output, relative=False)
        print(f"Saved visualisation to {output}")
    else:
        plt_module.show()
    plt_module.close(fig)

    summary_lines = _format_summary_lines(summary_payload)
    print("Run summary:")
    for line in summary_lines:
        print(f"  - {line}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot diagnostics for a backtest log")
    parser.add_argument("log", type=Path, help="Path to the JSONL metrics log")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure instead of displaying it",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    writer: ArtifactWriter | None = None
    if args.output is not None:
        ensure_matplotlib_backend()
        metadata = ReportMetadata(
            generator="visualize_backtest",
            git_commit=detect_git_commit(REPO_ROOT),
            extra={"log_path": str(args.log)},
        )
        writer = ArtifactWriter(args.output.parent, metadata, overwrite=args.overwrite)
    import matplotlib.pyplot as plt

    run = load_run(args.log)
    if run.summary is None:
        print("Warning: no run_summary event found; plot may be incomplete")
    visualise_run(run, args.output, plt, writer)


if __name__ == "__main__":
    main()
