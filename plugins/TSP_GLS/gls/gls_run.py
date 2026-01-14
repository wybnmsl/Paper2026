# plugins/TSP_GLS/gls/gls_run.py

from __future__ import annotations

import time
import math
from typing import Dict, Tuple, Any, Optional, List
import numpy as np

from ..utils import utils
from . import gls_evol
from .spec import GLSSpec, default_gls_spec, from_json
from . import gls_operators


_NUMBA_PREHEATED = False
_CAND_CACHE: Dict[Tuple[int, int, int, bool], np.ndarray] = {}


def _compute_tldr_traj_from_events(
    incumbent_events: List[Tuple[float, float]],
    opt_cost: float,
    T_used: float,
    delta: float = 1e-6,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute trajectory-aware tLDR(T) from incumbent improvement events.

    We approximate ell(t)=log(V_best(t)) as piecewise-constant between events:
      V_best(t) = max(gap%(t), delta),
      gap%(t) = (best_cost(t) - opt_cost) / opt_cost * 100.

    J(T) = (1/T) * ∫_0^T ell(t) dt
    tLDR_traj(T) = (2/T) * ( ell(0) - J(T) )
    """
    aux: Dict[str, Any] = {}
    if opt_cost <= 0 or T_used <= 1e-12:
        aux["reason"] = "invalid_opt_or_T"
        return None, aux
    if not incumbent_events:
        aux["reason"] = "empty_events"
        return None, aux

    ev = sorted((float(t), float(c)) for t, c in incumbent_events)
    ev = [(max(0.0, min(t, float(T_used))), c) for t, c in ev]

    cleaned: List[Tuple[float, float]] = []
    best_c = float("inf")
    for t, c in ev:
        if c < best_c - 1e-12:
            best_c = c
            cleaned.append((t, c))
    if not cleaned:
        cleaned = [ev[0]]

    def cost_to_ell(cost: float) -> float:
        gap = (float(cost) - float(opt_cost)) / float(opt_cost) * 100.0
        V = max(float(gap), float(delta))
        return math.log(V)

    times = [t for t, _ in cleaned]
    ells = [cost_to_ell(c) for _, c in cleaned]

    if times[0] > 0.0:
        times = [0.0] + times
        ells = [ells[0]] + ells

    times.append(float(T_used))
    J_num = 0.0
    for k in range(len(ells)):
        dt = times[k + 1] - times[k]
        if dt > 0.0:
            J_num += float(ells[k]) * float(dt)

    J = J_num / float(T_used)
    ell0 = float(ells[0])
    tldr = (2.0 / float(T_used)) * (ell0 - float(J))

    aux.update({
        "ell0": float(ell0),
        "J": float(J),
        "T_used": float(T_used),
        "n_events": int(len(cleaned)),
    })
    return float(tldr), aux


def _preheat_numba_once():
    """A tiny warm-up to trigger numba JIT compilation once per process."""
    global _NUMBA_PREHEATED
    if _NUMBA_PREHEATED:
        return

    try:
        N = 6
        rng = np.random.RandomState(0)
        D = rng.rand(N, N).astype(np.float64)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        route = -1 * np.ones((N, 2), dtype=np.int64)
        for i in range(N):
            route[i, 0] = (i - 1) % N
            route[i, 1] = (i + 1) % N

        knn = 3
        NNs = np.argsort(D, axis=1)[:, 1:knn + 1].astype(np.int64)

        gls_operators.two_opt_a2a(route, D, NNs, True)
        gls_operators.relocate_a2a(route, D, NNs, True, 0.0)
    except Exception:
        pass

    _NUMBA_PREHEATED = True


def _prim_mst_edges(D: np.ndarray):
    """Build MST edges on a complete graph using Prim's algorithm."""
    n = D.shape[0]
    in_mst = np.zeros(n, dtype=bool)
    parent = np.zeros(n, dtype=np.int64)

    in_mst[0] = True
    min_dist = D[0].copy()

    edges = []
    for _ in range(n - 1):
        best_v = -1
        best_w = np.inf
        for v in range(n):
            if not in_mst[v] and min_dist[v] < best_w:
                best_w = min_dist[v]
                best_v = v
        if best_v == -1:
            break
        u = int(parent[best_v])
        edges.append((u, best_v))
        in_mst[best_v] = True

        for v in range(n):
            if not in_mst[v] and D[best_v, v] < min_dist[v]:
                min_dist[v] = D[best_v, v]
                parent[v] = best_v

    return edges


def _build_candidate_indices(dis_matrix: np.ndarray, k: int, use_mst: bool = True) -> np.ndarray:
    """Build candidate sets: base kNN + optional MST-edge augmentation."""
    n_nodes = dis_matrix.shape[0]
    k = max(1, min(int(k), n_nodes - 1))

    base = np.argsort(dis_matrix, axis=1)[:, 1:k + 1].astype(np.int64)

    if not use_mst or n_nodes <= 2:
        return base

    cand_lists = [list(base[i]) for i in range(n_nodes)]

    mst_edges = _prim_mst_edges(dis_matrix)
    for u, v in mst_edges:
        if v not in cand_lists[u]:
            cand_lists[u].append(v)
        if u not in cand_lists[v]:
            cand_lists[v].append(u)

    max_len = max(len(lst) for lst in cand_lists)
    argsort_full = np.argsort(dis_matrix, axis=1)

    cand_arr = np.zeros((n_nodes, max_len), dtype=np.int64)
    for i in range(n_nodes):
        seen = set()
        row = []
        for j in cand_lists[i]:
            if j == i:
                continue
            if j not in seen:
                seen.add(j)
                row.append(j)
        if len(row) < max_len:
            for j in argsort_full[i]:
                if j == i:
                    continue
                if j not in seen:
                    seen.add(j)
                    row.append(int(j))
                if len(row) >= max_len:
                    break
        cand_arr[i, :] = np.asarray(row[:max_len], dtype=np.int64)

    return cand_arr


def _get_candidate_indices(dis_matrix: np.ndarray, k: int, use_mst: bool = True) -> np.ndarray:
    """Candidate-set construction with caching per matrix identity and size."""
    key = (id(dis_matrix), dis_matrix.shape[0], int(k), bool(use_mst))
    if key in _CAND_CACHE:
        return _CAND_CACHE[key]
    cand = _build_candidate_indices(dis_matrix, k, use_mst)
    _CAND_CACHE[key] = cand
    return cand


def solve_instance(n, opt_cost, dis_matrix, coord, time_limit, ite_max, perturbation_moves, heuristic):
    """Legacy interface (kept for MDL/MCL); internally uses default_gls_spec()."""
    spec = default_gls_spec()
    try:
        gap = solve_instance_with_spec(
            n, opt_cost, dis_matrix, coord,
            time_limit, ite_max, perturbation_moves,
            heuristic, spec
        )
    except Exception:
        gap = 1E10
    return gap


def solve_instance_with_spec(
    n, opt_cost, dis_matrix, coord,
    time_limit, ite_max, perturbation_moves,
    heuristic, spec: GLSSpec | dict | None, return_meta: bool = False
):
    """Run GLS under the unified backbone according to GLSSpec."""
    t0_perf = time.perf_counter()
    instance_meta: Dict[str, Any] = {}

    if isinstance(spec, GLSSpec):
        gls_spec = spec
    elif isinstance(spec, dict):
        gls_spec = from_json(spec)
    else:
        gls_spec = default_gls_spec()

    _preheat_numba_once()

    n_nodes = dis_matrix.shape[0]

    cand_type = str(gls_spec.candset.get("type", "kNN"))
    k = int(gls_spec.candset.get("k", 60))
    use_mst = (cand_type == "kNN")
    nearest_indices = _get_candidate_indices(dis_matrix, k, use_mst)

    start_node = int(gls_spec.init.get("start", 0))
    multi_start = int(gls_spec.init.get("multi_start", 1))
    multi_start = max(1, multi_start)

    best_cost = np.inf
    best_route = None
    best_events: Optional[List[Tuple[float, float]]] = None

    t_end = time.time() + float(time_limit)
    ite_max = int(ite_max)

    guide_algorithm = heuristic

    for s in range(multi_start):
        if time.time() >= t_end:
            break

        depot = (start_node + s) % n_nodes
        init_route = gls_evol.nearest_neighbor_2End(dis_matrix, depot)
        init_cost = utils.tour_cost_2End(dis_matrix, init_route)

        remain = max(0.1, t_end - time.time())
        t_lim = time.time() + remain

        events: List[Tuple[float, float]] = []
        route, cost, _ = gls_evol.guided_local_search(
            coord, dis_matrix, nearest_indices,
            init_route, init_cost,
            t_lim, ite_max, perturbation_moves,
            first_improvement=True,
            guide_algorithm=guide_algorithm,
            spec=gls_spec,
            trace_events=events,
            trace_t0=t0_perf,
        )

        if cost < best_cost:
            best_cost = cost
            best_route = route
            best_events = events

    if not np.isfinite(best_cost) or best_route is None or opt_cost <= 0:
        if return_meta:
            return 1e10, {"reason": "invalid_solution"}
        return 1e10

    gap = (best_cost - float(opt_cost)) / float(opt_cost) * 100.0
    gap = float(gap)

    T_used = float(time.perf_counter() - t0_perf)
    tldr_traj, aux = _compute_tldr_traj_from_events(
        incumbent_events=best_events or [],
        opt_cost=float(opt_cost),
        T_used=T_used,
        delta=1e-6,
    )
    instance_meta.update({
        "gap": gap,
        "T_used": T_used,
        "tLDR_traj": tldr_traj,
        "n_events": aux.get("n_events") if isinstance(aux, dict) else None,
        "ell0": aux.get("ell0") if isinstance(aux, dict) else None,
        "J": aux.get("J") if isinstance(aux, dict) else None,
        "incumbent_events": best_events,
    })

    if return_meta:
        return gap, instance_meta
    return gap
