"""Queue primitives optimised for low allocation overhead in threaded pipelines."""

from __future__ import annotations

import threading
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class RingBufferQueue(Generic[T]):
    """Fixed-capacity ring buffer supporting blocking put/get operations.

    The implementation minimises per-element allocations by reusing a circular
    buffer. It is designed for single-process, multi-threaded workloads where
    the producer/consumer counts remain modest. The class intentionally omits
    ``join`` semantics because the backtester does not rely on them.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer: list[Optional[T]] = [None] * capacity
        self._head = 0
        self._tail = 0
        self._size = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def put(self, item: T) -> None:
        with self._not_full:
            while self._size == self._capacity:
                self._not_full.wait()
            self._buffer[self._tail] = item
            self._tail = (self._tail + 1) % self._capacity
            self._size += 1
            self._not_empty.notify()

    def get(self) -> T:
        with self._not_empty:
            while self._size == 0:
                self._not_empty.wait()
            item = self._buffer[self._head]
            self._buffer[self._head] = None
            self._head = (self._head + 1) % self._capacity
            self._size -= 1
            self._not_full.notify()
            return item  # type: ignore[return-value]

    def qsize(self) -> int:
        with self._lock:
            return self._size

    def empty(self) -> bool:
        return self.qsize() == 0

    def full(self) -> bool:
        return self.qsize() == self._capacity

    def task_done(self) -> None:  # API compatibility with queue.Queue
        return None


__all__ = ["RingBufferQueue"]
