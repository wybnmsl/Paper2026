"""Low-level GLS operators for TSP.

This module keeps the numba-accelerated 2-opt / relocate operators, and provides
a pure-Python Or-opt(chain_len=2/3) operator (used as an occasional extra move
in the stronger local search).
"""

from __future__ import annotations

import numpy as np
from numba import jit


# ---------------------------------------------------------------------------
#  2-opt (numba)
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def two_opt(tour, i, j):
    if i == j:
        return tour
    a = tour[i, 0]
    b = tour[j, 0]
    tour[i, 0] = tour[i, 1]
    tour[i, 1] = j
    tour[j, 0] = i
    tour[a, 1] = b
    tour[b, 1] = tour[b, 0]
    tour[b, 0] = a
    c = tour[b, 1]
    while tour[c, 1] != j:
        d = tour[c, 0]
        tour[c, 0] = tour[c, 1]
        tour[c, 1] = d
        c = d
    return tour


@jit(nopython=True, cache=True)
def two_opt_cost(tour, D, i, j):
    if i == j:
        return 0.0
    a = tour[i, 0]
    b = tour[j, 0]
    return D[a, b] + D[i, j] - D[a, i] - D[b, j]


@jit(nopython=True, cache=True)
def two_opt_a2a(tour, D, N, first_improvement=False, set_delta=0.0):
    best_move = None
    best_delta = set_delta
    for i in range(0, len(tour) - 1):
        for j in N[i]:
            if i in tour[j] or j in tour[i]:
                continue
            delta = two_opt_cost(tour, D, i, j)
            if delta < best_delta and not np.isclose(0.0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break
        if first_improvement and best_move is not None:
            break
    if best_move is not None:
        return best_delta, two_opt(tour, best_move[0], best_move[1])
    return 0.0, tour


@jit(nopython=True, cache=True)
def two_opt_o2a(tour, D, i, first_improvement=False):
    assert 0 < i < len(tour) - 1
    best_move = None
    best_delta = 0.0
    for j in range(1, len(tour) - 1):
        if abs(i - j) < 2:
            continue
        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0.0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break
    if best_move is not None:
        return best_delta, two_opt(tour, best_move[0], best_move[1])
    return 0.0, tour


@jit(nopython=True, cache=True)
def two_opt_o2a_all(tour, D, N, i):
    best_delta = 0.0
    for j in N[i]:
        if i in tour[j] or j in tour[i]:
            continue
        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0.0, delta):
            best_delta = delta
            tour = two_opt(tour, i, j)
    return best_delta, tour


# ---------------------------------------------------------------------------
#  Relocate (numba)
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def relocate(tour, i, j):
    a = tour[i, 0]
    c = tour[i, 1]
    tour[a, 1] = c
    tour[c, 0] = a
    d = tour[j, 1]
    tour[d, 0] = i
    tour[i, 0] = j
    tour[i, 1] = d
    tour[j, 1] = i
    return tour


@jit(nopython=True, cache=True)
def relocate_cost(tour, D, i, j):
    if i == j:
        return 0.0
    a = tour[i, 0]
    b = i
    c = tour[i, 1]
    d = j
    e = tour[j, 1]
    return -D[a, b] - D[b, c] + D[a, c] - D[d, e] + D[d, b] + D[b, e]


@jit(nopython=True, cache=True)
def relocate_o2a(tour, D, i, first_improvement=False):
    assert 0 < i < len(tour) - 1
    best_move = None
    best_delta = 0.0
    for j in range(1, len(tour) - 1):
        if i == j:
            continue
        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0.0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break
    if best_move is not None:
        return best_delta, relocate(tour, best_move[0], best_move[1])
    return 0.0, tour


@jit(nopython=True, cache=True)
def relocate_o2a_all(tour, D, N, i):
    best_delta = 0.0
    for j in N[i]:
        if tour[j, 1] == i:
            continue
        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0.0, delta):
            best_delta = delta
            tour = relocate(tour, i, j)
    return best_delta, tour


@jit(nopython=True, cache=True)
def relocate_a2a(tour, D, N, first_improvement=False, set_delta=0.0):
    best_move = None
    best_delta = set_delta
    for i in range(0, len(tour) - 1):
        for j in N[i]:
            if tour[j, 1] == i:
                continue
            delta = relocate_cost(tour, D, i, j)
            if delta < best_delta and not np.isclose(0.0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break
        if first_improvement and best_move is not None:
            break
    if best_move is not None:
        return best_delta, relocate(tour, best_move[0], best_move[1])
    return 0.0, tour


# ---------------------------------------------------------------------------
#  Helpers: 2-end representation <-> 1D tour (pure Python)
# ---------------------------------------------------------------------------

def route2tour_py(route2end: np.ndarray, start: int = 0) -> list[int]:
    """Convert 2-end route representation (n,2) to a 1D tour order."""
    n = route2end.shape[0]
    tour = [int(start)]
    cur = int(route2end[start, 1])
    while cur != start:
        tour.append(cur)
        cur = int(route2end[cur, 1])
        if len(tour) > n + 1:
            raise RuntimeError("Invalid 2-end route: cycle longer than n.")
    if len(tour) != n:
        raise RuntimeError(f"Invalid 2-end route: visited {len(tour)} of {n}.")
    return tour


def tour2route_py(tour: list[int]) -> np.ndarray:
    """Convert a 1D tour order to 2-end representation (n,2)."""
    n = len(tour)
    route = np.empty((n, 2), dtype=np.int64)
    for idx, u in enumerate(tour):
        prev_u = int(tour[idx - 1])
        next_u = int(tour[(idx + 1) % n])
        route[int(u), 0] = prev_u
        route[int(u), 1] = next_u
    return route


# ---------------------------------------------------------------------------
#  Or-opt(chain_len=2/3) on 2-end route (pure Python; occasional use)
# ---------------------------------------------------------------------------

def or_opt_chain(route2end: np.ndarray, D: np.ndarray, chain_len: int = 2, first_improvement: bool = True):
    """Apply a single Or-opt(chain_len) move on a 2-end route.

    This is a pure-Python operator (O(n^2)), intended to be called only
    occasionally (e.g., in a stronger local search when stuck).

    Returns:
        (delta, new_route2end)
        - delta < 0 indicates an improving move under distance matrix D.
    """
    if chain_len not in (2, 3):
        raise ValueError("chain_len must be 2 or 3.")
    n = int(route2end.shape[0])
    if n <= chain_len + 2:
        return 0.0, route2end

    tour = route2tour_py(route2end, start=0)

    best_delta = 0.0
    best_move = None  # (s, e, j) in tour indices, segment [s,e) inserted after j

    # choose a contiguous segment [s, e) of length chain_len
    for s in range(n):
        e = s + chain_len
        if e > n:
            # wrap-around segments are skipped for simplicity/stability
            continue

        pre = tour[(s - 1) % n]
        head = tour[s]
        tail = tour[e - 1]
        post = tour[e % n]

        # insert after position j (edge tour[j] -> tour[j+1])
        for j in range(n):
            if j >= s - 1 and j < e:
                # insertion edge touches the segment (no-op / invalid)
                continue
            a = tour[j]
            b = tour[(j + 1) % n]

            # delta: remove (pre,head),(tail,post),(a,b) add (pre,post),(a,head),(tail,b)
            delta = (
                D[pre, post] + D[a, head] + D[tail, b]
                - (D[pre, head] + D[tail, post] + D[a, b])
            )
            if delta < best_delta - 1e-12:
                best_delta = float(delta)
                best_move = (s, e, j)
                if first_improvement:
                    break
        if first_improvement and best_move is not None:
            break

    if best_move is None:
        return 0.0, route2end

    s, e, j = best_move
    seg = tour[s:e]
    base = tour[:s] + tour[e:]

    # compute insertion index in base
    seg_len = e - s
    if j < s:
        ins = j + 1
    else:
        ins = j - seg_len + 1
    new_tour = base[:ins] + seg + base[ins:]
    new_route2end = tour2route_py(new_tour)
    return best_delta, new_route2end
