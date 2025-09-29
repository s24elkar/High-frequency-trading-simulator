"""Command-line entry point for the backtester."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from . import (
    Backtester,
    BacktesterConfig,
    MetricsLogger,
    RiskConfig,
    RiskEngine,
    load_lobster_csv,
    replay_from_lobster,
)
from .order_book import load_order_book
from strategies import MarketMakingConfig, MarketMakingStrategy


@dataclass(slots=True)
class RunConfig:
    symbol: str
    message_file: str
    time_scale: float = 1e9
    log_jsonl: str = "logs/demo_run.jsonl"
    log_sqlite: Optional[str] = None
    seed: int = 0
    risk: Dict[str, Any] | None = None
    strategy: Dict[str, Any] | None = None


def load_config(path: str | Path) -> RunConfig:
    payload = json.loads(Path(path).read_text())
    return RunConfig(**payload)


def run(config: RunConfig) -> None:
    messages = load_lobster_csv(config.message_file, config.symbol, time_scale=config.time_scale)
    replay = replay_from_lobster(messages)
    order_book = load_order_book(depth=5)
    logger = MetricsLogger(json_path=config.log_jsonl, sqlite_path=config.log_sqlite)
    risk_cfg = config.risk or {}
    risk_engine = RiskEngine(
        RiskConfig(
            symbol=config.symbol,
            max_long=risk_cfg.get("max_long", 500.0),
            max_short=risk_cfg.get("max_short", -500.0),
        )
    )
    strat_cfg = config.strategy or {}
    strategy = MarketMakingStrategy(
        MarketMakingConfig(
            spread_ticks=strat_cfg.get("spread_ticks", 1),
            quote_size=strat_cfg.get("quote_size", 10.0),
            inventory_skew=strat_cfg.get("inventory_skew", 0.0),
            update_interval_ns=int(strat_cfg.get("update_interval_ns", 5_000_000)),
        ),
        risk_engine=risk_engine,
        seed=config.seed,
    )
    backtester = Backtester(
        config=BacktesterConfig(symbol=config.symbol),
        limit_book=order_book,
        metrics_logger=logger,
        risk_engine=risk_engine,
        strategy=strategy,
        seed=config.seed,
    )
    backtester.run(replay)
    logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hawkes simulator backtests")
    parser.add_argument("--config", required=True, help="Path to JSON run configuration")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
