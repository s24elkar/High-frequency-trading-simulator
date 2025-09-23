#!/usr/bin/env python3
"""Clean Binance trade dumps and derive aggregate bars for simulator inputs.

Typical usage:
    python scripts/preprocess_binance.py data/runs/raw/BTCUSDT-trades-2025-09-19.csv

The script expects Binance trade history CSVs with the seven standard columns:
    trade_id,price,qty,quote_qty,time_ms,is_buyer_maker,is_best_match

Two artefacts are emitted by default:
1. A trade-level CSV with canonical column names, ISO timestamps, and signed volume.
2. A one-second bar file with OHLCV statistics and signed flow totals.

Use ``--skip-trades`` or ``--skip-bars`` to disable either artefact, and ``--bar-seconds``
for coarser aggregations (e.g. 5-second bars).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Iterable, Optional

# Bump the decimal precision to comfortably handle price * size products.
getcontext().prec = 40


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Binance trade CSV dumps")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the raw Binance CSV (supports plain .csv or .csv.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/runs/processed"),
        help="Directory for processed artefacts (created if missing)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Override the trading symbol (inferred from filename when omitted)",
    )
    parser.add_argument(
        "--skip-trades",
        action="store_true",
        help="Disable the trade-level CSV export",
    )
    parser.add_argument(
        "--skip-bars",
        action="store_true",
        help="Disable the bar aggregation export",
    )
    parser.add_argument(
        "--bar-seconds",
        type=int,
        default=1,
        help="Bar aggregation interval in seconds (default: 1 second)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs (warnings still emit to stderr)",
    )
    return parser.parse_args(argv)


def infer_symbol(path: Path) -> Optional[str]:
    stem = path.stem  # strip extension(s)
    if stem.endswith(".csv"):
        stem = stem[:-4]
    marker = "-trades-"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return None


@dataclass
class BarAccumulator:
    start_ts_ms: int
    bar_seconds: int
    open: Decimal = field(default_factory=lambda: Decimal("0"))
    high: Decimal = field(default_factory=lambda: Decimal("0"))
    low: Decimal = field(default_factory=lambda: Decimal("0"))
    close: Decimal = field(default_factory=lambda: Decimal("0"))
    volume: Decimal = field(default_factory=lambda: Decimal("0"))
    quote_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    buy_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    sell_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    buy_quote_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    sell_quote_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    trade_count: int = 0
    price_volume: Decimal = field(default_factory=lambda: Decimal("0"))

    def update(self, *, price: Decimal, qty: Decimal, quote_qty: Decimal, side: str) -> None:
        if self.trade_count == 0:
            self.open = self.high = self.low = self.close = price
        else:
            if price > self.high:
                self.high = price
            if price < self.low:
                self.low = price
            self.close = price
        self.volume += qty
        self.quote_volume += quote_qty
        self.price_volume += price * qty
        if side == "buy":
            self.buy_volume += qty
            self.buy_quote_volume += quote_qty
        else:
            self.sell_volume += qty
            self.sell_quote_volume += quote_qty
        self.trade_count += 1

    @property
    def end_ts_ms(self) -> int:
        return self.start_ts_ms + self.bar_seconds * 1000 - 1

    def start_iso(self) -> str:
        return datetime.fromtimestamp(self.start_ts_ms / 1000.0, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )

    def end_iso(self) -> str:
        return datetime.fromtimestamp(self.end_ts_ms / 1000.0, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )

    def vwap(self) -> Optional[Decimal]:
        if self.volume == 0:
            return None
        return self.price_volume / self.volume


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "t", "yes", "y"}


def decimal_from(value: str, *, line_no: int, column: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid decimal in column '{column}' on line {line_no}: {value!r}") from exc


def fmt_decimal(value: Optional[Decimal], digits: Optional[int] = None) -> str:
    if value is None:
        return ""
    if digits is not None:
        quant = Decimal(1).scaleb(-digits)
        value = value.quantize(quant)
    return format(value, "f")


def normalize_timestamp(raw_value: int) -> tuple[int, datetime]:
    """Return (timestamp_ms, datetime) handling second/milli/micro/nano inputs."""

    if raw_value >= 1_000_000_000_000_000_000:  # nanoseconds
        seconds, remainder = divmod(raw_value, 1_000_000_000)
        dt = EPOCH + timedelta(seconds=seconds, microseconds=remainder // 1000)
        timestamp_ms = raw_value // 1_000_000
    elif raw_value >= 1_000_000_000_000_000:  # microseconds
        seconds, remainder = divmod(raw_value, 1_000_000)
        dt = EPOCH + timedelta(seconds=seconds, microseconds=remainder)
        timestamp_ms = raw_value // 1_000
    elif raw_value >= 1_000_000_000_000:  # milliseconds
        seconds, remainder = divmod(raw_value, 1_000)
        dt = EPOCH + timedelta(seconds=seconds, milliseconds=remainder)
        timestamp_ms = raw_value
    else:  # seconds
        dt = EPOCH + timedelta(seconds=raw_value)
        timestamp_ms = raw_value * 1000
    return timestamp_ms, dt


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", newline="")
    if path.suffixes[-2:] == [".csv", ".gz"]:
        return gzip.open(path, mode="rt", newline="")
    return open(path, mode="r", newline="")


def preprocess(
    input_path: Path,
    *,
    symbol: Optional[str],
    output_dir: Path,
    emit_trades: bool,
    emit_bars: bool,
    bar_seconds: int,
    quiet: bool,
) -> None:
    if bar_seconds <= 0:
        raise ValueError("--bar-seconds must be a positive integer")

    base_name = input_path.stem
    if base_name.endswith(".csv"):
        base_name = base_name[:-4]

    output_dir.mkdir(parents=True, exist_ok=True)

    trades_path = output_dir / f"{base_name}-clean.csv"
    bars_path = output_dir / f"{base_name}-{bar_seconds}s-bars.csv"

    total_rows = 0
    written_trades = 0
    skipped_rows = 0
    bars: list[BarAccumulator] = []
    current_bar: Optional[BarAccumulator] = None
    current_bucket: Optional[int] = None

    trade_writer: Optional[csv.writer] = None
    trade_file = None

    if emit_trades:
        trade_file = open(trades_path, mode="w", newline="")
        trade_writer = csv.writer(trade_file)
        trade_writer.writerow(
            [
                "symbol",
                "trade_id",
                "ts_ms",
                "ts_iso",
                "price",
                "qty",
                "quote_qty",
                "side",
                "is_buyer_maker",
                "is_best_match",
                "signed_qty",
                "signed_quote_qty",
            ]
        )

    try:
        with open_text(input_path) as src:
            reader = csv.reader(src)
            for line_no, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 7:
                    skipped_rows += 1
                    print(
                        f"Skipping line {line_no}: expected 7 columns, found {len(row)}",
                        file=sys.stderr,
                    )
                    continue
                total_rows += 1
                try:
                    trade_id = int(row[0])
                    price = decimal_from(row[1], line_no=line_no, column="price")
                    qty = decimal_from(row[2], line_no=line_no, column="qty")
                    quote_qty = decimal_from(row[3], line_no=line_no, column="quote_qty")
                    timestamp_raw = int(row[4])
                    timestamp_ms, ts_dt = normalize_timestamp(timestamp_raw)
                    is_buyer_maker = parse_bool(row[5])
                    is_best_match = parse_bool(row[6])
                except (ValueError, InvalidOperation) as exc:
                    skipped_rows += 1
                    print(f"Skipping line {line_no}: {exc}", file=sys.stderr)
                    continue

                side = "sell" if is_buyer_maker else "buy"
                signed_qty = qty if side == "buy" else -qty
                signed_quote_qty = quote_qty if side == "buy" else -quote_qty
                ts_iso = ts_dt.isoformat(timespec="microseconds")

                if trade_writer is not None:
                    trade_writer.writerow(
                        [
                            symbol or "",
                            trade_id,
                            timestamp_ms,
                            ts_iso,
                            fmt_decimal(price),
                            fmt_decimal(qty),
                            fmt_decimal(quote_qty),
                            side,
                            str(is_buyer_maker).lower(),
                            str(is_best_match).lower(),
                            fmt_decimal(signed_qty),
                            fmt_decimal(signed_quote_qty),
                        ]
                    )
                    written_trades += 1

                if emit_bars:
                    bucket = (timestamp_ms // 1000) // bar_seconds
                    if current_bucket != bucket:
                        if current_bar is not None:
                            bars.append(current_bar)
                        current_bucket = bucket
                        start_ts_ms = bucket * bar_seconds * 1000
                        current_bar = BarAccumulator(start_ts_ms=start_ts_ms, bar_seconds=bar_seconds)
                    assert current_bar is not None  # for mypy/static type checking
                    current_bar.update(price=price, qty=qty, quote_qty=quote_qty, side=side)

        if current_bar is not None:
            bars.append(current_bar)
    finally:
        if trade_file is not None:
            trade_file.close()

    if emit_bars:
        with open(bars_path, mode="w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "symbol",
                    "start_ts_ms",
                    "end_ts_ms",
                    "start_ts_iso",
                    "end_ts_iso",
                    "bar_seconds",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vwap",
                    "volume",
                    "quote_volume",
                    "buy_volume",
                    "sell_volume",
                    "buy_quote_volume",
                    "sell_quote_volume",
                    "net_volume",
                    "net_quote_volume",
                    "trade_count",
                ]
            )
            for bar in bars:
                vwap = bar.vwap()
                writer.writerow(
                    [
                        symbol or "",
                        bar.start_ts_ms,
                        bar.end_ts_ms,
                        bar.start_iso(),
                        bar.end_iso(),
                        bar.bar_seconds,
                        fmt_decimal(bar.open, digits=8),
                        fmt_decimal(bar.high, digits=8),
                        fmt_decimal(bar.low, digits=8),
                        fmt_decimal(bar.close, digits=8),
                        fmt_decimal(vwap, digits=8) if vwap is not None else "",
                        fmt_decimal(bar.volume, digits=8),
                        fmt_decimal(bar.quote_volume, digits=8),
                        fmt_decimal(bar.buy_volume, digits=8),
                        fmt_decimal(bar.sell_volume, digits=8),
                        fmt_decimal(bar.buy_quote_volume, digits=8),
                        fmt_decimal(bar.sell_quote_volume, digits=8),
                        fmt_decimal(bar.buy_volume - bar.sell_volume, digits=8),
                        fmt_decimal(
                            bar.buy_quote_volume - bar.sell_quote_volume,
                            digits=8,
                        ),
                        bar.trade_count,
                    ]
                )

    if skipped_rows and not quiet:
        print(f"Skipped {skipped_rows} malformed row(s)", file=sys.stderr)

    if not quiet:
        symbol_repr = symbol or "(unknown symbol)"
        print(f"Processed {total_rows} rows for {symbol_repr}")
        if emit_trades:
            print(f"  Trades CSV: {trades_path} ({written_trades} rows)")
        if emit_bars:
            print(f"  Bars CSV:   {bars_path} ({len(bars)} rows @ {bar_seconds}s)")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    symbol = args.symbol or infer_symbol(input_path)

    try:
        preprocess(
            input_path,
            symbol=symbol,
            output_dir=args.output_dir.expanduser(),
            emit_trades=not args.skip_trades,
            emit_bars=not args.skip_bars,
            bar_seconds=args.bar_seconds,
            quiet=args.quiet,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
