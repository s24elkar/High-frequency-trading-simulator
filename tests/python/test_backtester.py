from pathlib import Path

from backtester import (
    Backtester,
    BacktesterConfig,
    MetricsLogger,
    RiskConfig,
    RiskEngine,
    load_lobster_csv,
    replay_from_lobster,
)
from backtester.order_book import PythonOrderBook


class CollectingStrategy:
    def __init__(self) -> None:
        self.snapshots = []

    def on_tick(
        self, snapshot, backtester
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


def test_metrics_logger_jsonl(tmp_path: Path) -> None:
    logger_path = tmp_path / "metrics.jsonl"
    with MetricsLogger(json_path=logger_path):
        pass
    assert logger_path.exists()
