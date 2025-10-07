"""ctypes wrapper around the native C++ order book bridge."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

_LIB_NAMES = {
    "darwin": ("liborder_book_bridge.dylib", "order_book_bridge.dylib"),
    "win32": ("order_book_bridge.dll",),
}


def _candidate_names() -> Iterable[str]:
    platform = sys.platform
    if platform.startswith("darwin"):
        return _LIB_NAMES["darwin"]
    if platform.startswith("win"):
        return _LIB_NAMES["win32"]
    return ("liborder_book_bridge.so", "order_book_bridge.so")


def _candidate_paths() -> Iterable[Path]:
    env = os.environ.get("HFT_ORDER_BOOK_BRIDGE")
    if env:
        yield Path(env)
    project_root = Path(__file__).resolve().parents[2]
    for build_dir in sorted(project_root.glob("build*")):
        lib_dir = build_dir / "lib"
        if lib_dir.exists():
            for name in _candidate_names():
                candidate = lib_dir / name
                if candidate.exists():
                    yield candidate
    # Fallback to python/lib directories (for prebuilt artefacts)
    local_lib = project_root / "python" / "lib"
    if local_lib.exists():
        for name in _candidate_names():
            candidate = local_lib / name
            if candidate.exists():
                yield candidate


def _load_library() -> ctypes.CDLL:
    for candidate in _candidate_paths():
        try:
            lib = ctypes.CDLL(str(candidate))
            os.environ.setdefault("HFT_ORDER_BOOK_BRIDGE", str(candidate))
            return lib
        except OSError:
            continue
    raise RuntimeError(
        "order_book_bridge shared library not found. Build the C++ target "
        "`order_book_bridge` (e.g. `cmake --build build-release --target order_book_bridge`)."
    )


_lib = _load_library()


class _Order(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int64),
        ("side", ctypes.c_int),
        ("price", ctypes.c_double),
        ("quantity", ctypes.c_int32),
        ("ts_ns", ctypes.c_int64),
    ]


class _Fill(ctypes.Structure):
    _fields_ = [
        ("order", _Order),
        ("executed_quantity", ctypes.c_int32),
        ("fill_price", ctypes.c_double),
    ]


class _Level(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("side", ctypes.c_int),
        ("total_quantity", ctypes.c_int64),
        ("orders", ctypes.POINTER(_Order)),
        ("order_count", ctypes.c_size_t),
    ]


_lib.ob_create.restype = ctypes.c_void_p
_lib.ob_destroy.argtypes = [ctypes.c_void_p]
_lib.ob_add_order.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int32,
    ctypes.c_int64,
]
_lib.ob_add_order.restype = ctypes.c_int
_lib.ob_cancel_order.argtypes = [ctypes.c_void_p, ctypes.c_int64]
_lib.ob_cancel_order.restype = ctypes.c_int
_lib.ob_best_bid.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Order)]
_lib.ob_best_bid.restype = ctypes.c_int
_lib.ob_best_ask.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Order)]
_lib.ob_best_ask.restype = ctypes.c_int
_lib.ob_execute_order.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.POINTER(_Fill)),
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.ob_execute_order.restype = ctypes.c_int
_lib.ob_snapshot.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(_Level)),
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.ob_snapshot.restype = ctypes.c_int
_lib.ob_free_fills.argtypes = [ctypes.POINTER(_Fill), ctypes.c_size_t]
_lib.ob_free_levels.argtypes = [ctypes.POINTER(_Level), ctypes.c_size_t]


class OrderBook:
    """Thin wrapper around the native order book handle."""

    def __init__(self) -> None:
        handle = _lib.ob_create()
        if not handle:
            raise RuntimeError("Failed to create native OrderBook instance")
        self._handle = ctypes.c_void_p(handle)

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        if getattr(self, "_handle", None):
            _lib.ob_destroy(self._handle)
            self._handle = None

    # Core operations -----------------------------------------------------

    def add_order(
        self,
        order_id: int,
        side: int,
        price: float,
        quantity: int,
        ts_ns: int,
    ) -> None:
        _lib.ob_add_order(
            self._handle,
            ctypes.c_int64(order_id),
            ctypes.c_int(side),
            ctypes.c_double(price),
            ctypes.c_int32(quantity),
            ctypes.c_int64(ts_ns),
        )

    def cancel_order(self, order_id: int) -> bool:
        return bool(
            _lib.ob_cancel_order(self._handle, ctypes.c_int64(order_id))
        )

    def best_bid(self) -> Optional[dict]:
        order = _Order()
        exists = _lib.ob_best_bid(self._handle, ctypes.byref(order))
        return self._order_to_dict(order) if exists else None

    def best_ask(self) -> Optional[dict]:
        order = _Order()
        exists = _lib.ob_best_ask(self._handle, ctypes.byref(order))
        return self._order_to_dict(order) if exists else None

    @staticmethod
    def _order_to_dict(order: _Order) -> dict:
        side = "BUY" if order.side == 0 else "SELL"
        return {
            "id": int(order.id),
            "side": side,
            "price": float(order.price),
            "quantity": int(order.quantity),
            "ts_ns": int(order.ts_ns),
        }

    def execute_order(
        self,
        side: int,
        price: float,
        quantity: int,
    ) -> List[dict]:
        fills_ptr = ctypes.POINTER(_Fill)()
        count = ctypes.c_size_t()
        _lib.ob_execute_order(
            self._handle,
            ctypes.c_int(side),
            ctypes.c_double(price),
            ctypes.c_int32(quantity),
            ctypes.byref(fills_ptr),
            ctypes.byref(count),
        )
        if not bool(fills_ptr):
            return []
        fills: List[dict] = []
        for idx in range(count.value):
            item = fills_ptr[idx]
            fills.append(
                {
                    "order": self._order_to_dict(item.order),
                    "executed_quantity": int(item.executed_quantity),
                    "fill_price": float(item.fill_price),
                }
            )
        _lib.ob_free_fills(fills_ptr, count)
        return fills

    def snapshot(self, depth: int) -> List[dict]:
        levels_ptr = ctypes.POINTER(_Level)()
        count = ctypes.c_size_t()
        success = _lib.ob_snapshot(
            self._handle,
            ctypes.c_size_t(depth),
            ctypes.byref(levels_ptr),
            ctypes.byref(count),
        )
        if not success or not bool(levels_ptr):
            return []
        levels: List[dict] = []
        for idx in range(count.value):
            level = levels_ptr[idx]
            orders: List[dict] = []
            if level.orders:
                for j in range(level.order_count):
                    orders.append(self._order_to_dict(level.orders[j]))
            levels.append(
                {
                    "price": float(level.price),
                    "side": int(level.side),
                    "total_quantity": int(level.total_quantity),
                    "orders": orders,
                }
            )
        _lib.ob_free_levels(levels_ptr, count)
        return levels


__all__ = ["OrderBook"]
