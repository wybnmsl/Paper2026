from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from utils.utils import routes_cost, route_load, check_feasible
except ImportError:  # pragma: no cover
    from ..utils.utils import routes_cost, route_load, check_feasible


# ----------------------- Init: Clarke-Wright Savings -----------------------

def clarke_wright_savings_init(dist: np.ndarray, demand: np.ndarray, capacity: int, rng: random.Random) -> List[List[int]]:
    """
    Parallel Clarke-Wright savings heuristic.
    - Start with one route per customer: [i]
    - Merge routes based on savings s(i,j)=d(0,i)+d(0,j)-d(i,j) when feasible.
    """
    N = int(dist.shape[0])
    cap = int(capacity)

    # routes represented by list of customers (no depot)
    routes: List[List[int]] = [[i] for i in range(1, N)]
    # map customer -> route index
    which: Dict[int, int] = {i: i - 1 for i in range(1, N)}

    def load_of(r: List[int]) -> int:
        return route_load(r, demand)

    # compute savings list
    sav = []
    for i in range(1, N):
        for j in range(i + 1, N):
            s = int(dist[0, i] + dist[0, j] - dist[i, j])
            sav.append((s, i, j))
    # sort high to low (shuffle tie groups lightly)
    sav.sort(key=lambda x: x[0], reverse=True)

    for _s, i, j in sav:
        ri = which.get(i, None)
        rj = which.get(j, None)
        if ri is None or rj is None or ri == rj:
            continue

        a = routes[ri]
        b = routes[rj]
        if not a or not b:
            continue

        # can merge only if i is at an end of a and j is at an end of b (or reversed)
        cand_merges = []
        if a[0] == i and b[-1] == j:
            cand_merges.append(b + a)
        if a[-1] == i and b[0] == j:
            cand_merges.append(a + b)
        if a[0] == i and b[0] == j:
            cand_merges.append(list(reversed(b)) + a)
        if a[-1] == i and b[-1] == j:
            cand_merges.append(a + list(reversed(b)))

        if not cand_merges:
            continue

        rng.shuffle(cand_merges)
        merged = None
        for m in cand_merges:
            if load_of(m) <= cap:
                merged = m
                break
        if merged is None:
            continue

        # perform merge: keep ri, invalidate rj
        routes[ri] = merged
        routes[rj] = []
        for v in merged:
            which[v] = ri

    routes = [r for r in routes if r]
    # safety: if any customer missing, fall back to singleton routes
    if sum(len(r) for r in routes) != (N - 1):
        routes = [[i] for i in range(1, N)]
    return routes


# ----------------------- Destroy operators -----------------------

def destroy_random(routes: List[List[int]], k_remove: int, rng: random.Random) -> Tuple[List[List[int]], List[int]]:
    flat = [v for r in routes for v in r]
    k = min(max(1, int(k_remove)), len(flat))
    removed = rng.sample(flat, k)
    removed_set = set(removed)
    kept = [[v for v in r if v not in removed_set] for r in routes]
    kept = [r for r in kept if r]
    return kept, removed


def destroy_route(routes: List[List[int]], k_remove: int, rng: random.Random) -> Tuple[List[List[int]], List[int]]:
    if not routes:
        return [], []
    ridx = rng.randrange(len(routes))
    removed = routes[ridx][:]
    kept = [r[:] for i, r in enumerate(routes) if i != ridx and r]
    # if still need remove more, do random
    flat = [v for r in kept for v in r]
    need = max(0, min(int(k_remove) - len(removed), len(flat)))
    if need > 0:
        extra = rng.sample(flat, need)
        removed.extend(extra)
        removed_set = set(extra)
        kept = [[v for v in r if v not in removed_set] for r in kept]
        kept = [r for r in kept if r]
    return kept, removed


def destroy_worst(routes: List[List[int]], dist: np.ndarray, k_remove: int, rng: random.Random) -> Tuple[List[List[int]], List[int]]:
    # score each customer by its incident edge costs in its route
    scores = []
    for r in routes:
        if not r:
            continue
        prev = 0
        for idx, v in enumerate(r):
            nxt = r[idx + 1] if idx + 1 < len(r) else 0
            contrib = int(dist[prev, v] + dist[v, nxt])
            scores.append((contrib, v))
            prev = v
    if not scores:
        return [r[:] for r in routes if r], []
    scores.sort(reverse=True)  # high contrib first
    k = min(max(1, int(k_remove)), len(scores))
    # add slight randomness: sample from top-2k
    head = min(len(scores), 2 * k)
    cand = [v for _, v in scores[:head]]
    removed = rng.sample(cand, k)
    removed_set = set(removed)
    kept = [[v for v in r if v not in removed_set] for r in routes]
    kept = [r for r in kept if r]
    return kept, removed


def destroy_shaw(
    routes: List[List[int]],
    dist: np.ndarray,
    demand: np.ndarray,
    k_remove: int,
    rng: random.Random,
    relatedness_w_dist: float = 1.0,
    relatedness_w_dem: float = 0.3,
) -> Tuple[List[List[int]], List[int]]:
    """
    Shaw removal: pick a seed customer, then remove related customers.
    relatedness(i,j) = w1 * d(i,j) + w2 * |dem_i - dem_j|
    """
    customers = [v for r in routes for v in r]
    if not customers:
        return [r[:] for r in routes if r], []
    k = min(max(1, int(k_remove)), len(customers))
    seed = rng.choice(customers)
    removed = [int(seed)]

    cand = [v for v in customers if v != seed]
    # precompute relatedness to seed (and expand greedily)
    while len(removed) < k and cand:
        last = removed[-1]
        rel = []
        for v in cand:
            rel_score = relatedness_w_dist * float(dist[last, v]) + relatedness_w_dem * abs(float(demand[last] - demand[v]))
            rel.append((rel_score, v))
        rel.sort(key=lambda x: x[0])  # most related (small) first
        # choose from top band to diversify a bit
        band = max(1, min(len(rel), 10))
        pick = rel[rng.randrange(band)][1]
        removed.append(int(pick))
        cand.remove(pick)

    removed_set = set(removed)
    kept = [[v for v in r if v not in removed_set] for r in routes]
    kept = [r for r in kept if r]
    return kept, removed


# ----------------------- Repair (Regret-k insertion) -----------------------

def _best_insertion(route: List[int], cust: int, dist: np.ndarray) -> Tuple[int, int]:
    """
    Returns (delta_cost, position) to insert cust into route (excluding depot).
    position is index in [0..len(route)].
    """
    if not route:
        delta = int(dist[0, cust] + dist[cust, 0])
        return delta, 0

    best_delta = 10**18
    best_pos = 0

    # insert between prev and nxt in depot-extended tour
    prev = 0
    for pos in range(len(route) + 1):
        nxt = route[pos] if pos < len(route) else 0
        delta = int(dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt])
        if delta < best_delta:
            best_delta = delta
            best_pos = pos
        if pos < len(route):
            prev = route[pos]
    return int(best_delta), int(best_pos)


def repair_regret_k(
    routes_partial: List[List[int]],
    removed: List[int],
    dist: np.ndarray,
    demand: np.ndarray,
    capacity: int,
    regret_k: int,
    rng: random.Random,
) -> List[List[int]]:
    routes = [r[:] for r in routes_partial if r]
    cap = int(capacity)
    rem_list = list(dict.fromkeys(int(v) for v in removed))  # unique, preserve order

    # maintain route loads
    loads = [route_load(r, demand) for r in routes]

    def feasible_route_insert(ridx: int, cust: int) -> bool:
        return loads[ridx] + int(demand[cust]) <= cap

    while rem_list:
        best_choice = None  # (regret, best_delta, cust, ridx, pos)
        # evaluate candidates
        for cust in rem_list:
            options = []
            for ridx in range(len(routes)):
                if not feasible_route_insert(ridx, cust):
                    continue
                delta, pos = _best_insertion(routes[ridx], cust, dist)
                options.append((delta, ridx, pos))
            # option to create new route
            new_delta = int(dist[0, cust] + dist[cust, 0])
            options.append((new_delta, -1, 0))

            options.sort(key=lambda x: x[0])
            # regret = k-th best - best
            k_eff = min(max(2, int(regret_k)), len(options))
            regret = float(options[k_eff - 1][0] - options[0][0])

            delta0, ridx0, pos0 = options[0]
            key = (regret, -float(delta0), cust, ridx0, pos0)  # tie-break: smaller delta is better
            if best_choice is None or key > best_choice[0]:
                best_choice = (key, regret, delta0, cust, ridx0, pos0)

        assert best_choice is not None
        _, _reg, _delta0, cust, ridx0, pos0 = best_choice

        # execute insertion
        if ridx0 == -1:
            routes.append([cust])
            loads.append(int(demand[cust]))
        else:
            routes[ridx0].insert(int(pos0), cust)
            loads[ridx0] += int(demand[cust])

        rem_list.remove(cust)

        # clean empty (shouldn't happen)
        routes = [r for r in routes if r]
        loads = [route_load(r, demand) for r in routes]

    return routes


# ----------------------- Local Search (VND) -----------------------

def _two_opt_intra(route: List[int], dist: np.ndarray) -> Tuple[bool, List[int]]:
    n = len(route)
    if n < 4:
        return False, route
    best = route
    best_cost = None
    improved = False
    # compute current cost with depot
    def cost(r):
        c = int(dist[0, r[0]]) + int(dist[r[-1], 0])
        for i in range(len(r)-1):
            c += int(dist[r[i], r[i+1]])
        return c
    best_cost = cost(route)

    for i in range(n - 2):
        a = 0 if i == 0 else route[i - 1]
        b = route[i]
        for k in range(i + 1, n - 1):
            c = route[k]
            d = 0 if k == n - 1 else route[k + 1]
            # delta of reversing [i..k]
            delta = int(dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d])
            if delta < 0:
                new = route[:i] + list(reversed(route[i : k + 1])) + route[k + 1 :]
                new_cost = best_cost + delta
                best = new
                best_cost = new_cost
                improved = True
                return True, best  # first improvement
    return improved, best


def _relocate1(routes: List[List[int]], demand: np.ndarray, capacity: int, dist: np.ndarray) -> Tuple[bool, List[List[int]]]:
    cap = int(capacity)
    best_routes = routes
    best_cost = routes_cost(routes, dist, depot=0)

    for r_from in range(len(routes)):
        for idx in range(len(routes[r_from])):
            cust = routes[r_from][idx]
            dem = int(demand[cust])

            # remove
            r_from_list = routes[r_from]
            new_from = r_from_list[:idx] + r_from_list[idx + 1 :]

            for r_to in range(len(routes)):
                if r_to == r_from and len(new_from) == 0:
                    continue

                if r_to == r_from:
                    cand_routes = [r[:] for r in routes]
                    cand_routes[r_from] = new_from
                else:
                    load_to = route_load(routes[r_to], demand)
                    if load_to + dem > cap:
                        continue
                    cand_routes = [r[:] for r in routes]
                    cand_routes[r_from] = new_from
                # if from becomes empty, drop later

                # insert in best position
                base_to = cand_routes[r_to] if r_to != r_from else cand_routes[r_from]
                best_delta = 10**18
                best_pos = 0
                prev = 0
                for pos in range(len(base_to) + 1):
                    nxt = base_to[pos] if pos < len(base_to) else 0
                    delta = int(dist[prev, cust] + dist[cust, nxt] - dist[prev, nxt])
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                    if pos < len(base_to):
                        prev = base_to[pos]

                base_to2 = base_to[:]
                base_to2.insert(best_pos, cust)
                if r_to == r_from:
                    cand_routes[r_from] = base_to2
                else:
                    cand_routes[r_to] = base_to2

                cand_routes = [r for r in cand_routes if r]
                if not check_feasible(cand_routes, demand, cap):
                    continue

                c = routes_cost(cand_routes, dist, depot=0)
                if c < best_cost:
                    return True, cand_routes
    return False, best_routes


def _swap1(routes: List[List[int]], demand: np.ndarray, capacity: int, dist: np.ndarray) -> Tuple[bool, List[List[int]]]:
    cap = int(capacity)
    best_cost = routes_cost(routes, dist, depot=0)
    for r1 in range(len(routes)):
        for i in range(len(routes[r1])):
            a = routes[r1][i]
            for r2 in range(r1, len(routes)):
                j_start = i + 1 if r2 == r1 else 0
                for j in range(j_start, len(routes[r2])):
                    b = routes[r2][j]
                    if r1 != r2:
                        load1 = route_load(routes[r1], demand) - int(demand[a]) + int(demand[b])
                        load2 = route_load(routes[r2], demand) - int(demand[b]) + int(demand[a])
                        if load1 > cap or load2 > cap:
                            continue
                    cand = [r[:] for r in routes]
                    cand[r1][i] = b
                    cand[r2][j] = a
                    cand = [r for r in cand if r]
                    c = routes_cost(cand, dist, depot=0)
                    if c < best_cost:
                        return True, cand
    return False, routes


def _two_opt_star(routes: List[List[int]], demand: np.ndarray, capacity: int, dist: np.ndarray) -> Tuple[bool, List[List[int]]]:
    cap = int(capacity)
    best_cost = routes_cost(routes, dist, depot=0)

    for r1 in range(len(routes)):
        for r2 in range(r1 + 1, len(routes)):
            A = routes[r1]
            B = routes[r2]
            if not A or not B:
                continue
            for i in range(1, len(A)):
                for j in range(1, len(B)):
                    newA = A[:i] + B[j:]
                    newB = B[:j] + A[i:]
                    if route_load(newA, demand) > cap or route_load(newB, demand) > cap:
                        continue
                    cand = [r[:] for r in routes]
                    cand[r1] = newA
                    cand[r2] = newB
                    cand = [r for r in cand if r]
                    c = routes_cost(cand, dist, depot=0)
                    if c < best_cost:
                        return True, cand
    return False, routes


def vnd_local_search(
    routes0: List[List[int]],
    dist: np.ndarray,
    demand: np.ndarray,
    capacity: int,
    neigh: Sequence[str],
    max_moves: int,
    end_time: float,
    first_improve: bool,
    rng: random.Random,
    cand: Optional[List[np.ndarray]] = None,
    granular_k: int = 24,
) -> List[List[int]]:
    routes = [r[:] for r in routes0 if r]
    moves = 0

    while time.time() < end_time and moves < int(max_moves):
        improved_any = False

        for op in neigh:
            if time.time() >= end_time:
                break

            if op == "2opt_intra":
                # apply to each route
                for ridx in range(len(routes)):
                    ok, nr = _two_opt_intra(routes[ridx], dist)
                    if ok:
                        routes[ridx] = nr
                        improved_any = True
                        moves += 1
                        if first_improve:
                            break
                if improved_any and first_improve:
                    break

            elif op == "relocate1":
                ok, routes2 = _relocate1(routes, demand, capacity, dist)
                if ok:
                    routes = routes2
                    improved_any = True
                    moves += 1
                    if first_improve:
                        break

            elif op == "swap1":
                ok, routes2 = _swap1(routes, demand, capacity, dist)
                if ok:
                    routes = routes2
                    improved_any = True
                    moves += 1
                    if first_improve:
                        break

            elif op == "2opt_star":
                ok, routes2 = _two_opt_star(routes, demand, capacity, dist)
                if ok:
                    routes = routes2
                    improved_any = True
                    moves += 1
                    if first_improve:
                        break

            else:
                # ignore unknown op names
                continue

            if moves >= int(max_moves):
                break

        if not improved_any:
            break

    return [r for r in routes if r]
