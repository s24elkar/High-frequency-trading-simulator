"""Concurrent backtesting utilities enabling threaded execution pipelines."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Callable, Iterable, Optional

from .backtester import Backtester, MarketEvent, StrategyContext


class ConcurrentStrategyContext(StrategyContext):
    """Strategy context that routes order actions through a concurrent runner."""

    def __init__(self, backtester: Backtester, runner: "ConcurrentBacktester") -> None:
        super().__init__(backtester)
        self._runner = runner

    def submit_order(
        self,
        side: str,
        price: float,
        size: float,
        *,
        order_type: str = "LIMIT",
        display_size: Optional[float] = None,
        stop_price: Optional[float] = None,
        peg_reference: Optional[str] = None,
        peg_offset: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> int:
        return self._runner.submit_order(
            side,
            price,
            size,
            order_type=order_type,
            display_size=display_size,
            stop_price=stop_price,
            peg_reference=peg_reference,
            peg_offset=peg_offset,
            metadata=metadata,
        )

    def cancel_order(self, order_id: int) -> None:
        self._runner.cancel_order(order_id)

    def active_orders(self) -> dict[int, object]:  # type: ignore[override]
        return self._runner.active_orders()


class ConcurrentBacktester:
    """Coordinates backtester execution across ingestion, strategy, and order threads."""

    def __init__(
        self,
        backtester: Backtester,
        event_queue_size: int = 10_000,
        queue_factory: Callable[[int], object] | None = None,
    ) -> None:
        self.backtester = backtester
        self._event_queue = self._make_queue(
            queue_factory, event_queue_size
        )  # type: ignore[assignment]
        self._order_queue = self._make_queue(
            queue_factory, event_queue_size
        )  # type: ignore[assignment]
        self._lock = threading.RLock()
        self._threads: list[threading.Thread] = []
        self._exceptions: list[BaseException] = []

        if self.backtester.strategy is not None:
            context = ConcurrentStrategyContext(self.backtester, self)
            self.backtester._context = context

    @staticmethod
    def _make_queue(
        factory: Callable[[int], object] | None, maxsize: int
    ) -> object:
        if factory is None:
            return queue.Queue(maxsize)
        try:
            return factory(maxsize)
        except TypeError:
            return factory()

    def run(self, replay_session: Iterable[MarketEvent]) -> None:
        with self._lock:
            self.backtester.start_strategy()

        ingest_thread = threading.Thread(
            target=self._ingest_loop,
            args=(replay_session,),
            name="IngestThread",
            daemon=True,
        )
        strategy_thread = threading.Thread(
            target=self._strategy_loop,
            name="StrategyThread",
            daemon=True,
        )
        order_thread = threading.Thread(
            target=self._order_loop,
            name="OrderThread",
            daemon=True,
        )
        self._threads = [ingest_thread, strategy_thread, order_thread]

        for thread in self._threads:
            thread.start()
        for thread in self._threads:
            thread.join()

        with self._lock:
            self.backtester.finalise_run()

        if self._exceptions:
            raise self._exceptions[0]

    def submit_order(
        self,
        side: str,
        price: float,
        size: float,
        *,
        order_type: str = "LIMIT",
        display_size: Optional[float] = None,
        stop_price: Optional[float] = None,
        peg_reference: Optional[str] = None,
        peg_offset: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> int:
        future: Future[int] = Future()
        payload = {
            "side": side,
            "price": price,
            "size": size,
            "order_type": order_type,
            "display_size": display_size,
            "stop_price": stop_price,
            "peg_reference": peg_reference,
            "peg_offset": peg_offset,
            "metadata": metadata,
        }
        self._order_queue.put(("submit", payload, future))
        return future.result()

    def cancel_order(self, order_id: int) -> None:
        future: Future[None] = Future()
        self._order_queue.put(("cancel", order_id, future))
        future.result()

    def active_orders(self) -> dict[int, object]:
        with self._lock:
            return dict(self.backtester.active_orders)

    def _ingest_loop(self, replay_session: Iterable[MarketEvent]) -> None:
        try:
            for event in replay_session:
                self._event_queue.put(event)
        except BaseException as exc:  # pragma: no cover - defensive
            self._exceptions.append(exc)
        finally:
            self._event_queue.put(None)

    def _strategy_loop(self) -> None:
        try:
            while True:
                event = self._event_queue.get()
                if event is None:
                    break
                self._handle_event(event)
        except BaseException as exc:  # pragma: no cover - defensive
            self._exceptions.append(exc)
        finally:
            self._order_queue.put(("stop", None, None))

    def _order_loop(self) -> None:
        while True:
            command, payload, future = self._order_queue.get()
            if command == "stop":
                self._order_queue.task_done()
                return
            try:
                if command == "submit":
                    result = self._submit_with_lock(**payload)  # type: ignore[arg-type]
                    if future is not None:
                        future.set_result(result)
                elif command == "cancel":
                    order_id = payload  # type: ignore[assignment]
                    result = self._cancel_with_lock(order_id)
                    if future is not None:
                        future.set_result(result)
                else:  # pragma: no cover - defensive
                    if future is not None:
                        future.set_exception(RuntimeError(f"Unknown command {command}"))
            except BaseException as exc:  # pragma: no cover - defensive
                if future is not None and not future.done():
                    future.set_exception(exc)
                self._exceptions.append(exc)
            finally:
                task_done = getattr(self._order_queue, "task_done", None)
                if callable(task_done):
                    task_done()

    def _submit_with_lock(
        self,
        *,
        side: str,
        price: float,
        size: float,
        order_type: str = "LIMIT",
        display_size: Optional[float] = None,
        stop_price: Optional[float] = None,
        peg_reference: Optional[str] = None,
        peg_offset: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> int:
        with self._lock:
            return self.backtester.submit_order(
                side,
                price,
                size,
                order_type=order_type,
                display_size=display_size,
                stop_price=stop_price,
                peg_reference=peg_reference,
                peg_offset=peg_offset,
                metadata=metadata,
            )

    def _cancel_with_lock(self, order_id: int) -> None:
        with self._lock:
            self.backtester.cancel_order(order_id)

    def _handle_event(self, event: MarketEvent) -> None:
        with self._lock:
            self.backtester.clock_ns = event.timestamp_ns
            self.backtester._fire_due_timers(self.backtester.clock_ns)
            update = self.backtester._dispatch_event(event)
        for fill in update.fills:
            with self._lock:
                self.backtester.process_fill(fill)
        snapshot = update.snapshot
        if snapshot is None:
            with self._lock:
                self.backtester._fire_due_timers(self.backtester.clock_ns)
            return
        with self._lock:
            if self.backtester.config.record_snapshots:
                self.backtester.metrics_logger.log_snapshot(snapshot)
            self.backtester._update_digest("SNAPSHOT", snapshot)
        self.backtester.on_market_data(snapshot)


__all__ = ["ConcurrentBacktester", "ConcurrentStrategyContext"]
