"""Interface to the high-performance Hawkes simulators implemented in C++.

This module now delegates the heavy lifting to the shared library produced by
the CMake target ``hawkes_bridge``.  We keep the public API shape compatible
with the previous pure-Python implementation: ``simulate_thinning_exp_fast`` and
``simulate_thinning_general`` return NumPy arrays of event times and marks, and
accept an optional ``mark_sampler`` callable.
"""

from __future__ import annotations

import ctypes
import inspect
import sys
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

try:  # allow usage both as package and as standalone scripts
    from .kernels import ExpKernel, PowerLawKernel
except (ImportError, ValueError):
    from kernels import ExpKernel, PowerLawKernel


CallbackType = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)
ERROR_SENTINEL = ctypes.c_size_t(-1).value

_LIB: Optional[ctypes.CDLL] = None
_CALLBACK_REGISTRY = []  # keep references alive during native call


def _load_from_env() -> Optional[ctypes.CDLL]:
    override = os.environ.get("HFT_HAWKES_BRIDGE")
    if not override:
        return None
    path = Path(override).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"HFT_HAWKES_BRIDGE points to {path}, but the file does not exist"
        )
    lib = ctypes.CDLL(str(path))
    _configure_signatures(lib)
    return lib


def _bridge_name_candidates() -> Tuple[str, ...]:
    if sys.platform.startswith("darwin"):
        return ("libhawkes_bridge.dylib", "hawkes_bridge.dylib")
    if sys.platform.startswith("win"):
        return ("hawkes_bridge.dll",)
    return ("libhawkes_bridge.so", "hawkes_bridge.so")


def _bridge_search_paths() -> Tuple[Path, ...]:
    here = Path(__file__).resolve().parent
    project_root = here.parent
    candidates = [here, here / "lib", project_root]
    for build_dir in sorted(project_root.glob("build*")):
        candidates.append(build_dir)
        candidates.append(build_dir / "lib")
    # Preserve order but drop duplicates
    seen = []
    for path in candidates:
        if path not in seen:
            seen.append(path)
    return tuple(seen)


def _load_bridge() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    env_lib = _load_from_env()
    if env_lib is not None:
        _LIB = env_lib
        return _LIB

    for directory in _bridge_search_paths():
        for name in _bridge_name_candidates():
            candidate = directory / name
            if candidate.exists():
                lib = ctypes.CDLL(str(candidate))
                _configure_signatures(lib)
                _LIB = lib
                return _LIB

    searched = [str(p) for p in _bridge_search_paths()]
    raise FileNotFoundError(
        "hawkes_bridge shared library not found. "
        "Build the C++ project (e.g. `cmake --build build`) before invoking simulations.\n"
        f"Searched paths: {searched}"
    )


def _configure_signatures(lib: ctypes.CDLL) -> None:
    lib.hawkes_simulate_exp.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ]
    lib.hawkes_simulate_exp.restype = ctypes.c_size_t

    lib.hawkes_simulate_powerlaw.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ]
    lib.hawkes_simulate_powerlaw.restype = ctypes.c_size_t

    lib.hawkes_free.argtypes = [ctypes.c_void_p]
    lib.hawkes_free.restype = None

    lib.hawkes_last_error.argtypes = []
    lib.hawkes_last_error.restype = ctypes.c_char_p


def _normalise_sampler(mark_sampler: Optional[Callable[[np.random.Generator], float]]) -> Callable[[np.random.Generator], float]:
    if mark_sampler is None:
        def _default_sampler(rng: np.random.Generator) -> float:
            return float(rng.exponential(1.0))

        return _default_sampler

    try:
        sig = inspect.signature(mark_sampler)
        arity = len(sig.parameters)
    except (TypeError, ValueError):  # builtins without signature metadata
        arity = 1

    if arity == 0:
        def _wrapper(rng: np.random.Generator) -> float:
            return float(mark_sampler())

        return _wrapper

    def _wrapper(rng: np.random.Generator) -> float:
        return float(mark_sampler(rng))

    return _wrapper


def _make_callback(mark_sampler: Callable[[np.random.Generator], float], seed: int):
    rng = np.random.default_rng(seed)
    sampler_fn = _normalise_sampler(mark_sampler)
    state = (sampler_fn, rng)
    state_holder = ctypes.py_object(state)
    state_ptr = ctypes.pointer(state_holder)

    @CallbackType
    def _callback(raw_ctx):
        py_obj = ctypes.cast(raw_ctx, ctypes.POINTER(ctypes.py_object)).contents.value
        sampler, rng_obj = py_obj
        return float(sampler(rng_obj))

    ctx = ctypes.cast(state_ptr, ctypes.c_void_p)
    registration = (_callback, state_holder, state_ptr)
    _CALLBACK_REGISTRY.append(registration)
    return _callback, ctx, registration


def _release_registration(registration) -> None:
    try:
        _CALLBACK_REGISTRY.remove(registration)
    except ValueError:
        pass


def _run_simulation(func, params, mark_sampler, seed):
    lib = _load_bridge()
    callback = None
    callback_ptr = ctypes.c_void_p()
    ctx = ctypes.c_void_p()
    registration = None
    if mark_sampler is not None:
        callback, ctx, registration = _make_callback(mark_sampler, seed)
        callback_ptr = ctypes.cast(callback, ctypes.c_void_p)

    times_ptr = ctypes.POINTER(ctypes.c_double)()
    marks_ptr = ctypes.POINTER(ctypes.c_double)()

    try:
        count = func(*params, callback_ptr, ctx, ctypes.byref(times_ptr), ctypes.byref(marks_ptr))
    finally:
        if registration is not None:
            _release_registration(registration)

    if count == ERROR_SENTINEL:
        message = lib.hawkes_last_error()
        detail = message.decode("utf-8") if message else "unknown error"
        raise RuntimeError(f"hawkes_bridge simulation failed: {detail}")

    if count == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    times = np.ctypeslib.as_array(times_ptr, shape=(count,)).copy()
    marks = np.ctypeslib.as_array(marks_ptr, shape=(count,)).copy()

    lib.hawkes_free(times_ptr)
    lib.hawkes_free(marks_ptr)
    return times, marks


def simulate_thinning_exp_fast(mu: float,
                               kernel: ExpKernel,
                               mark_sampler: Optional[Callable[[np.random.Generator], float]] = None,
                               T: float = 1.0,
                               seed: int = 12345) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(kernel, ExpKernel):
        raise TypeError("kernel must be an ExpKernel instance")
    params = (float(mu), float(kernel.alpha), float(kernel.beta), float(T), int(seed))
    return _run_simulation(_load_bridge().hawkes_simulate_exp, params, mark_sampler, seed)


def simulate_thinning_general(mu: float,
                              kernel: PowerLawKernel,
                              mark_sampler: Optional[Callable[[np.random.Generator], float]] = None,
                              T: float = 1.0,
                              seed: int = 12345) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(kernel, PowerLawKernel):
        raise TypeError("kernel must be a PowerLawKernel instance")
    params = (
        float(mu),
        float(kernel.alpha),
        float(kernel.c),
        float(kernel.gamma),
        int(seed),
    )
    return _run_simulation(_load_bridge().hawkes_simulate_powerlaw, params, mark_sampler, seed)


__all__ = [
    "simulate_thinning_exp_fast",
    "simulate_thinning_general",
]
