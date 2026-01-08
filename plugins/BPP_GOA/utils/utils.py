from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from math import ceil, log
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def ensure_dir(path: str | os.PathLike) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def stable_json_dumps(obj: Any, *, indent: int = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=indent)


def make_logger(name: str = "bpp", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


@dataclass
class Stopwatch:
    t0: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


def lb_relaxation(items: np.ndarray, capacity: int) -> int:
    s = float(np.sum(items))
    return int(ceil(s / float(capacity)))


def gap_percent(bins_used: int, lb: int) -> float:
    lb = max(int(lb), 1)
    return (float(bins_used) - float(lb)) / float(lb) * 100.0


def summarize_instance(items: np.ndarray, capacity: int) -> Dict[str, Any]:
    items = np.asarray(items)
    return {
        "n_items": int(items.shape[0]),
        "capacity": int(capacity),
        "sum": float(np.sum(items)),
        "mean": float(np.mean(items)),
        "min": float(np.min(items)),
        "max": float(np.max(items)),
        "lb": int(lb_relaxation(items, capacity)),
    }


def compute_tldr_from_trace(
    trace: List[Tuple[float, float, int]],
    *,
    time_limit_s: float,
    delta: float = 1e-9,
) -> float:
    """Compute a tLDR-style score from a monotone best-so-far trace.

    trace format: list of (t_sec, best_gap_percent, best_bins).
    We treat residual as gap_ratio = gap_percent / 100, then compute:
        J(T) = (1/T) * \int_0^T log(max(residual(t), delta)) dt
        tLDR = (2/T) * (log(residual(0)) - J(T))

    If trace is empty, returns 0.0.
    """
    if not trace:
        return 0.0

    T = max(float(time_limit_s), 1e-9)

    # Ensure starting point at t=0
    tr = list(trace)
    if tr[0][0] > 1e-9:
        tr = [(0.0, tr[0][1], tr[0][2])] + tr

    # Clip times to [0, T] and sort
    tr2 = []
    for t, gap, bins_used in tr:
        tt = float(t)
        if tt < 0.0:
            continue
        if tt > T:
            tt = T
        tr2.append((tt, float(gap), int(bins_used)))
    tr2.sort(key=lambda x: x[0])
    if tr2[-1][0] < T:
        tr2.append((T, tr2[-1][1], tr2[-1][2]))

    def resid(gap_pct: float) -> float:
        return max(float(gap_pct) / 100.0, float(delta))

    l0 = log(resid(tr2[0][1]))

    # piecewise-constant integral on intervals [t_i, t_{i+1})
    area = 0.0
    for i in range(len(tr2) - 1):
        t_i, gap_i, _ = tr2[i]
        t_j, _, _ = tr2[i + 1]
        dt = max(0.0, float(t_j) - float(t_i))
        area += dt * log(resid(gap_i))

    J = area / T
    return (2.0 / T) * (l0 - J)
