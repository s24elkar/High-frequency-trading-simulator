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
    TimerToken,
)
from .itch import LOBSTERMessage, ITCHEvent, load_lobster_csv, replay_from_lobster
from .risk import RiskConfig, RiskEngine, RiskSnapshot
from .logging import (
    MetricsLogger,
    MetricsAggregator,
    RunSummary,
    MetricsSnapshot,
    LatencyBreakdown,
    TimingSummary,
)
from .dashboard import RiskDashboard, DashboardConfig
from .strategy import StrategySandbox, StrategyError
from .concurrent import ConcurrentBacktester, ConcurrentStrategyContext
from .queues import RingBufferQueue
from .order_book import PythonOrderBook, CppOrderBook, load_order_book
from .stress import StressConfig, StressMetrics, Hotspot, run_order_book_stress
from .replay import ReplayConfig, ReplayEngine, replay
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
    "TimerToken",
    "LOBSTERMessage",
    "ITCHEvent",
    "load_lobster_csv",
    "replay_from_lobster",
    "ReplayConfig",
    "ReplayEngine",
    "replay",
    "RiskConfig",
    "RiskEngine",
    "RiskSnapshot",
    "MetricsLogger",
    "MetricsAggregator",
    "RunSummary",
    "MetricsSnapshot",
    "LatencyBreakdown",
    "TimingSummary",
    "RiskDashboard",
    "DashboardConfig",
    "StrategySandbox",
    "StrategyError",
    "ConcurrentBacktester",
    "ConcurrentStrategyContext",
    "PythonOrderBook",
    "CppOrderBook",
    "load_order_book",
    "RingBufferQueue",
    "StressConfig",
    "StressMetrics",
    "Hotspot",
    "run_order_book_stress",
    "BacktestRun",
    "BacktestSummary",
    "SnapshotPoint",
    "OrderEventRecord",
    "FillEventRecord",
    "load_run",
    "summarise",
]
