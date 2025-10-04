"""Backtesting toolkit bridging the C++ limit order book with Python analytics."""

from .backtester import (
    Backtester,
    BacktesterConfig,
    OrderRequest,
    FillEvent,
    MarketEvent,
    MarketSnapshot,
    OrderBookUpdate,
    StrategyCallbacks,
    StrategyContext,
)
from .itch import LOBSTERMessage, ITCHEvent, load_lobster_csv, replay_from_lobster
from .risk import RiskConfig, RiskEngine
from .logging import MetricsLogger, MetricsAggregator, RunSummary
from .reports import (
    BacktestRun,
    BacktestSummary,
    FillEventRecord,
    OrderEventRecord,
    SnapshotPoint,
    load_run,
    summarise,
)

__all__ = [
    "Backtester",
    "BacktesterConfig",
    "OrderRequest",
    "FillEvent",
    "MarketEvent",
    "MarketSnapshot",
    "OrderBookUpdate",
    "StrategyCallbacks",
    "StrategyContext",
    "LOBSTERMessage",
    "ITCHEvent",
    "load_lobster_csv",
    "replay_from_lobster",
    "RiskConfig",
    "RiskEngine",
    "MetricsLogger",
    "MetricsAggregator",
    "RunSummary",
    "BacktestRun",
    "BacktestSummary",
    "SnapshotPoint",
    "OrderEventRecord",
    "FillEventRecord",
    "load_run",
    "summarise",
]
