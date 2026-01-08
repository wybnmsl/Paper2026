from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np


def euclidean_distance_matrix_int(coords: np.ndarray) -> np.ndarray:
    """
    VRPLIB EUC_2D distance: rounded Euclidean distance (integer).
    coords: (N,2) float64
    returns: (N,N) int32
    """
    xy = np.asarray(coords, dtype=np.float64)
    diff = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    d_int = np.rint(d).astype(np.int32)
    np.fill_diagonal(d_int, 0)
    return d_int


def routes_cost(routes: Sequence[Sequence[int]], dist: np.ndarray, depot: int = 0) -> int:
    """
    Sum of route costs, each route starts/ends at depot.
    routes: list of customer lists (excluding depot)
    dist: (N,N) int
    """
    total = 0
    for r in routes:
        if not r:
            continue
        prev = depot
        for v in r:
            total += int(dist[prev, v])
            prev = v
        total += int(dist[prev, depot])
    return int(total)


def edges_from_routes(routes: Sequence[Sequence[int]], depot: int = 0, symmetric: bool = True) -> Iterator[Tuple[int, int]]:
    """
    Yield edges (i,j) from depot->...->depot.
    If symmetric=True, yields undirected edges once (min(i,j), max(i,j)).
    """
    seen = set()
    for r in routes:
        if not r:
            continue
        prev = depot
        for v in r:
            i, j = int(prev), int(v)
            if symmetric:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in seen:
                    seen.add((a, b))
                    yield (a, b)
            else:
                yield (i, j)
            prev = v
        # return to depot
        i, j = int(prev), int(depot)
        if symmetric:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                yield (a, b)
        else:
            yield (i, j)


def route_load(route: Sequence[int], demand: np.ndarray) -> int:
    return int(np.sum(np.asarray(demand, dtype=np.int64)[np.asarray(route, dtype=np.int64)]))


def check_feasible(routes: Sequence[Sequence[int]], demand: np.ndarray, capacity: int) -> bool:
    cap = int(capacity)
    for r in routes:
        if route_load(r, demand) > cap:
            return False
    return True


def gap_percent(cost: float, opt_cost: float) -> float:
    if opt_cost is None or opt_cost <= 0:
        return float("nan")
    return (float(cost) - float(opt_cost)) / float(opt_cost) * 100.0


def summarize_instance(instance) -> dict:
    """Lightweight instance summary for logging/debug and PLR profiling."""
    dist = instance.dist
    demand = instance.demand
    coords = instance.coords
    n = int(dist.shape[0])
    dem = np.asarray(demand[1:], dtype=np.float64) if n > 1 else np.array([], dtype=np.float64)
    out = {
        "name": getattr(instance, "name", ""),
        "n": n,
        "capacity": int(getattr(instance, "capacity", 0)),
        "demand_mean": float(dem.mean()) if dem.size else 0.0,
        "demand_std": float(dem.std()) if dem.size else 0.0,
        "coord_std": float(np.asarray(coords, dtype=np.float64).std()) if coords is not None else 0.0,
    }
    # distance stats (exclude diagonal)
    if dist is not None and n > 1:
        d = np.asarray(dist, dtype=np.float64)
        mask = ~np.eye(n, dtype=bool)
        dv = d[mask]
        out.update({
            "dist_mean": float(dv.mean()),
            "dist_std": float(dv.std()),
        })
    return out


def profile_instance(instance) -> np.ndarray:
    """
    Instance profiling for PLR:
    Returns a numeric feature vector phi(x) suitable for clustering / retrieval.

    Current features (all float):
      [n, capacity, demand_mean, demand_cv, coord_std, dist_mean, dist_cv]
    """
    n = float(getattr(instance, "dimension", getattr(instance, "dist", np.zeros((0,0))).shape[0]))
    cap = float(getattr(instance, "capacity", 0))
    demand = np.asarray(getattr(instance, "demand", np.array([], dtype=np.float64)), dtype=np.float64)
    demand_cust = demand[1:] if demand.size > 1 else demand
    dmean = float(demand_cust.mean()) if demand_cust.size else 0.0
    dstd = float(demand_cust.std()) if demand_cust.size else 0.0
    dcv = float(dstd / (dmean + 1e-9)) if demand_cust.size else 0.0

    coords = np.asarray(getattr(instance, "coords", np.zeros((0,2))), dtype=np.float64)
    cstd = float(coords.std()) if coords.size else 0.0

    dist = np.asarray(getattr(instance, "dist", np.zeros((0,0))), dtype=np.float64)
    if dist.size and dist.shape[0] > 1:
        N = dist.shape[0]
        mask = ~np.eye(N, dtype=bool)
        dv = dist[mask]
        dist_mean = float(dv.mean())
        dist_std = float(dv.std())
        dist_cv = float(dist_std / (dist_mean + 1e-9))
    else:
        dist_mean = 0.0
        dist_cv = 0.0

    return np.array([n, cap, dmean, dcv, cstd, dist_mean, dist_cv], dtype=np.float32)
