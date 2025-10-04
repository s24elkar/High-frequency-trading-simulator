import json
from pathlib import Path

from backtester import (
    Backtester,
    BacktesterConfig,
    MetricsLogger,
    RiskConfig,
    RiskEngine,
    StrategyCallbacks,
    load_lobster_csv,
    replay_from_lobster,
)
from backtester.order_book import PythonOrderBook
from strategies import MarketMakingConfig, MarketMakingStrategy


class CollectingStrategy(StrategyCallbacks):
    def __init__(self) -> None:
        self.snapshots = []

    def on_market_data(
        self, snapshot, ctx
    ) -> None:  # pragma: no cover - simple container
        self.snapshots.append(snapshot)


def test_backtester_digest_deterministic(tmp_path: Path) -> None:
    messages = list(load_lobster_csv("tests/data/itch_sample.csv", symbol="TEST"))
    replay1 = list(replay_from_lobster(messages))
    replay2 = list(replay_from_lobster(messages))

    def run_once(replay):
        logger_path = tmp_path / "run.jsonl"
        with MetricsLogger(json_path=logger_path) as metrics:
            risk = RiskEngine(RiskConfig(symbol="TEST"))
            backtester = Backtester(
                config=BacktesterConfig(symbol="TEST"),
                limit_book=PythonOrderBook(depth=5),
                metrics_logger=metrics,
                risk_engine=risk,
                strategy=CollectingStrategy(),
                seed=42,
            )
            backtester.run(replay)
            digest = backtester.digest
        data = logger_path.read_text().strip().splitlines()
        return digest, data

    digest1, log1 = run_once(replay1)
    digest2, log2 = run_once(replay2)

    assert digest1 == digest2
    assert log1 == log2

    last_record = json.loads(log1[-1])
    assert last_record["event_type"] == "run_summary"
    summary = last_record["payload"]
    assert summary["digest"] == digest1
    assert summary["symbol"] == "TEST"
    assert "order_to_trade_ratio" in summary


def test_market_maker_replay_deterministic(tmp_path: Path) -> None:
    messages = list(load_lobster_csv("tests/data/itch_sample.csv", symbol="TEST"))

    def run_once(seed: int):
        replay = replay_from_lobster(messages)
        log_path = tmp_path / f"run_{seed}.jsonl"
        with MetricsLogger(json_path=log_path) as metrics:
            risk = RiskEngine(RiskConfig(symbol="TEST"))
            strategy = MarketMakingStrategy(
                MarketMakingConfig(
                    spread_ticks=1,
                    quote_size=1.0,
                    tick_size=0.1,
                    update_interval_ns=0,
                ),
                risk_engine=risk,
                seed=seed,
            )
            backtester = Backtester(
                config=BacktesterConfig(symbol="TEST"),
                limit_book=PythonOrderBook(depth=5),
                metrics_logger=metrics,
                risk_engine=risk,
                strategy=strategy,
                seed=seed,
            )
            backtester.run(replay)
            digest = backtester.digest
        summary_record = json.loads(log_path.read_text().strip().splitlines()[-1])
        return digest, summary_record["payload"]

    digest1, summary1 = run_once(seed=99)
    digest2, summary2 = run_once(seed=99)

    assert digest1 == digest2
    assert summary1 == summary2
    assert summary1["order_count"] > 0
    assert summary1["order_volume"] > 0.0


def test_metrics_logger_jsonl(tmp_path: Path) -> None:
    logger_path = tmp_path / "metrics.jsonl"
    with MetricsLogger(json_path=logger_path):
        pass
    assert logger_path.exists()
