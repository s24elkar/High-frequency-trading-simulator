from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from python.backtester import (
    Backtester,
    BacktesterConfig,
    FillEvent,
    MetricsLogger,
    PythonOrderBook,
    RateLimitConfig,
    RiskConfig,
    RiskEngine,
    StrategyError,
)


def _exercise_order_rate_limit(log_dir: Path) -> Dict[str, object]:
    log_path = log_dir / "order_rate_limit.jsonl"
    metrics = MetricsLogger(json_path=log_path)
    risk_engine = RiskEngine(RiskConfig(symbol="CTRL"))
    rate_limit = RateLimitConfig(max_actions=2, interval_ns=1_000)
    backtester = Backtester(
        config=BacktesterConfig(symbol="CTRL"),
        limit_book=PythonOrderBook(depth=1),
        metrics_logger=metrics,
        risk_engine=risk_engine,
        order_rate_limit=rate_limit,
    )
    error_message = None
    backtester.clock_ns = 0
    backtester.submit_order("BUY", 100.0, 1.0)
    backtester.submit_order("SELL", 101.0, 1.0)
    backtester.clock_ns = 0
    try:
        backtester.submit_order("BUY", 102.0, 1.0)
    except StrategyError as exc:
        error_message = str(exc)
    finally:
        metrics.close()
    return {
        "kind": "order_rate_limit",
        "strategy_halted": backtester.strategy_halted,
        "error": error_message,
        "control_stats": backtester.control_stats,
        "log_file": str(log_path),
    }


def _exercise_cancel_throttle(log_dir: Path) -> Dict[str, object]:
    log_path = log_dir / "cancel_rate_limit.jsonl"
    metrics = MetricsLogger(json_path=log_path)
    risk_engine = RiskEngine(RiskConfig(symbol="CTRL"))
    cancel_limit = RateLimitConfig(
        max_actions=1, interval_ns=1_000, halt_on_violation=False
    )
    backtester = Backtester(
        config=BacktesterConfig(symbol="CTRL"),
        limit_book=PythonOrderBook(depth=1),
        metrics_logger=metrics,
        risk_engine=risk_engine,
        cancel_rate_limit=cancel_limit,
    )
    backtester.clock_ns = 0
    order1 = backtester.submit_order("BUY", 100.0, 1.0)
    order2 = backtester.submit_order("SELL", 101.0, 1.0)
    backtester.clock_ns = 0
    backtester.cancel_order(order1)
    backtester.clock_ns = 0
    backtester.cancel_order(order2)
    metrics.close()
    return {
        "kind": "cancel_rate_limit",
        "strategy_halted": backtester.strategy_halted,
        "control_stats": backtester.control_stats,
        "throttled_order_active": order2 in backtester.active_orders,
        "log_file": str(log_path),
    }


def _exercise_kill_switch(log_dir: Path) -> Dict[str, object]:
    log_path = log_dir / "kill_switch.jsonl"
    metrics = MetricsLogger(json_path=log_path)
    risk_config = RiskConfig(symbol="CTRL", max_long=5.0, halt_on_breach=True)
    risk_engine = RiskEngine(risk_config)
    backtester = Backtester(
        config=BacktesterConfig(symbol="CTRL"),
        limit_book=PythonOrderBook(depth=1),
        metrics_logger=metrics,
        risk_engine=risk_engine,
    )
    fill = FillEvent(
        order_id=999,
        symbol="CTRL",
        side="BUY",
        price=100.0,
        size=10.0,
        timestamp_ns=1,
    )
    backtester.process_fill(fill)
    metrics.close()
    snapshot = risk_engine.snapshot("CTRL", timestamp_ns=1)
    return {
        "kind": "kill_switch",
        "strategy_halted": risk_engine.strategy_halted,
        "inventory": snapshot.inventory,
        "alerts": list(risk_engine.alerts),
        "warnings": list(risk_engine.warnings),
        "log_file": str(log_path),
    }


def run_suite(output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "order_rate_limit": _exercise_order_rate_limit(log_dir),
        "cancel_rate_limit": _exercise_cancel_throttle(log_dir),
        "kill_switch": _exercise_kill_switch(log_dir),
    }
    (output_dir / "risk_controls.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run risk-control validation suite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write risk-control audit results.",
    )
    args = parser.parse_args()
    results = run_suite(args.output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
