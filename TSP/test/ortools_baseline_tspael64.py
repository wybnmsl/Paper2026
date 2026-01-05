#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OR-Tools baseline for TSPAEL64.pkl
- Loads dataset
- Solves each instance as symmetric TSP with OR-Tools Routing Solver
- Records per-instance stats into CSV with same columns as your GLS export
- Shows tqdm progress

Usage:
  python zTSP/test/ortools_baseline_tspael64.py \
    --dataset zTSP/TrainingData/TSPAEL64.pkl \
    --csv zTSP/evaluation/tspael64_ortools_baseline.csv \
    --time_limit 10 \
    --limit 64 \
    --first_solution PATH_CHEAPEST_ARC \
    --metaheuristic GUIDED_LOCAL_SEARCH

Notes:
- Distances in the pkl are floats; OR-Tools requires integer arc costs.
  We use a safe integer scaling for the search, but the reported solution
  cost is recomputed on the original float matrix to get an accurate gap%.
"""

import os
import sys
import time
import math
import numpy as np
import pickle
import argparse
from tqdm import tqdm

# Try to import ortools
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception as e:
    raise ImportError(
        "Failed to import OR-Tools. Please install with: `pip install ortools`"
    ) from e


# -------------------------
# Data loading
# -------------------------
def load_tspael64(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    coords = data.get("coordinate")
    dists  = data.get("distance_matrix")
    costs  = data.get("cost")  # optimal tour length per instance (float)
    if coords is None or dists is None or costs is None:
        raise ValueError("TSPAEL64.pkl missing keys: coordinate / distance_matrix / cost")
    coords = [np.asarray(c) for c in coords]
    dists  = [np.asarray(D) for D in dists]
    costs  = [float(x) for x in costs]
    if not (len(coords) == len(dists) == len(costs)):
        raise ValueError("coordinate / distance_matrix / cost length mismatch")
    return coords, dists, costs


# -------------------------
# OR-Tools TSP Solver
# -------------------------
def _build_routing_from_distance_matrix_int(dist_int: np.ndarray,
                                            time_limit_s: float,
                                            first_solution: str,
                                            metaheuristic: str):
    """
    Build and solve TSP with OR-Tools using an integer distance matrix.

    Returns: (solution, routing, manager)
    """
    n = int(dist_int.shape[0])
    if n <= 1:
        return None, None, None

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 vehicle, depot=0
    routing = pywrapcp.RoutingModel(manager)

    # Cost callback
    def distance_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(dist_int[i, j])

    transit_callback_index = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()

    # First solution strategy
    fs_map = {
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        "ALL_UNPERFORMED": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        "BEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
        "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        "GLOBAL_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
        "LOCAL_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
        "SEQUENTIAL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION,
    }
    params.first_solution_strategy = fs_map.get(first_solution.upper(),
                                                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Local search metaheuristic
    mh_map = {
        "AUTOMATIC": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        "GREEDY_DESCENT": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    }
    params.local_search_metaheuristic = mh_map.get(metaheuristic.upper(),
                                                   routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    # Time limit
    if time_limit_s is None or time_limit_s <= 0:
        time_limit_s = 10.0
    params.time_limit.FromSeconds(int(math.ceil(float(time_limit_s))))

    # Optional: silence logs
    params.log_search = False

    solution = routing.SolveWithParameters(params)
    return solution, routing, manager


def solve_tsp_route_ortools(D_float: np.ndarray,
                            time_limit_s: float = 10.0,
                            first_solution: str = "PATH_CHEAPEST_ARC",
                            metaheuristic: str = "GUIDED_LOCAL_SEARCH"):
    """
    Solve symmetric TSP on D_float (NxN float) using OR-Tools.
    Returns: (route_list, status_str)

    - We scale D_float to int for the solver, but compute final cost on D_float.
    """
    n = int(D_float.shape[0])
    if n <= 1:
        return list(range(n)), "TRIVIAL"

    # Safety: ensure non-negative and finite
    D = np.array(D_float, dtype=np.float64)
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distance matrix must be non-negative and finite.")

    # Scale to int for OR-Tools (preserve symmetry & zero diag)
    # Choose factor to keep integers within ~1e9 comfortably
    max_val = float(np.max(D))
    if max_val == 0.0:
        scale = 1.0
    else:
        # aim for ~6 decimal places if numbers are <= ~2
        scale = 1_000_000.0 if max_val < 100.0 else max(1.0, 1_000.0 / max_val)
    D_int = np.rint(D * scale).astype(np.int64)
    np.fill_diagonal(D_int, 0)

    solution, routing, manager = _build_routing_from_distance_matrix_int(
        D_int, time_limit_s, first_solution, metaheuristic
    )

    if solution is None:
        return None, "NO_SOLUTION"

    # Extract route
    index = routing.Start(0)
    order = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        order.append(node)
        index = solution.Value(routing.NextVar(index))

    # OR-Tools TSP returns route ending at depot; ensure we have N nodes
    # If it returns [0, a, b, ..., z], it might miss last return to 0 in the node list
    # but we compute cost as cycle anyway.
    if len(order) < n:
        # attempt to fill missing nodes (rare)
        seen = set(order)
        for v in range(n):
            if v not in seen:
                order.append(v)

    # Deduplicate while keeping order (rare defensive step)
    seen = set()
    order_unique = []
    for v in order:
        if v not in seen:
            seen.add(v)
            order_unique.append(int(v))
    if len(order_unique) != n:
        # fallback to 0..n-1 if something odd happened
        order_unique = list(range(n))

    return order_unique, "SUCCESS"


def cycle_cost_from_route(D: np.ndarray, order: list[int]) -> float:
    n = len(order)
    s = 0.0
    for i in range(n):
        a = order[i]
        b = order[(i + 1) % n]
        s += float(D[a, b])
    return float(s)


# -------------------------
# Batch evaluation & CSV
# -------------------------
def evaluate_tspael64_ortools_to_csv(pkl_path: str,
                                     csv_out: str,
                                     time_limit: float = 10.0,
                                     limit: int | None = None,
                                     first_solution: str = "PATH_CHEAPEST_ARC",
                                     metaheuristic: str = "GUIDED_LOCAL_SEARCH",
                                     tqdm_disable: bool = False):
    """
    Solve all/first-N instances with OR-Tools and write CSV:
    Columns matched to your GLS export:
      idx,N,opt_cost,sol_cost_est,gap_percent,time_s,time_limit_s,loop_max,max_no_improve,k,top_k,perturb,perturb_interval,error
    (GLS-specific fields are left empty for baseline.)
    """
    try:
        import pandas as pd
        use_pandas = True
    except Exception:
        import csv as _csv
        use_pandas = False

    coords, dists, costs = load_tspael64(pkl_path)
    total = len(dists)
    n = min(total, int(limit)) if limit is not None else total

    rows = []
    it = tqdm(range(n), disable=tqdm_disable, desc="OR-Tools baseline", unit="inst")

    for i in it:
        D = dists[i]
        opt = float(costs[i])
        N  = int(D.shape[0])

        t0 = time.time()
        err = ""
        try:
            route, status = solve_tsp_route_ortools(
                D, time_limit_s=time_limit,
                first_solution=first_solution, metaheuristic=metaheuristic
            )
            if route is None or status != "SUCCESS":
                err = f"status={status}"
                sol_cost = float("nan")
                gap = float("nan")
            else:
                sol_cost = cycle_cost_from_route(D, route)
                gap = (sol_cost - opt) / opt * 100.0
                if abs(gap) < 1e-9:
                    gap = 0.0
        except Exception as e:
            err = str(e)
            sol_cost = float("nan")
            gap = float("nan")

        dt = time.time() - t0

        rows.append({
            "idx": i,
            "N": N,
            "opt_cost": opt,
            "sol_cost_est": sol_cost,        # keep same column name for comparability
            "gap_percent": gap,
            "time_s": dt,
            "time_limit_s": float(time_limit),
            "loop_max": "",                  # not applicable for OR-Tools
            "max_no_improve": "",
            "k": "",
            "top_k": "",
            "perturb": "",
            "perturb_interval": "",
            "error": err,
        })

    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)

    if use_pandas:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(csv_out, index=False)
    else:
        import csv as _csv
        keys = list(rows[0].keys()) if rows else [
            "idx","N","opt_cost","sol_cost_est","gap_percent","time_s","time_limit_s",
            "loop_max","max_no_improve","k","top_k","perturb","perturb_interval","error"
        ]
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # brief summary
    valid = [r for r in rows if isinstance(r["gap_percent"], (int, float)) and math.isfinite(r["gap_percent"])]
    avg_gap = float(np.mean([r["gap_percent"] for r in valid])) if valid else float("nan")
    avg_time = float(np.mean([r["time_s"] for r in rows])) if rows else float("nan")
    print(f"[baseline-summary] instances={n}/{total} | avg_gap%={avg_gap:.6f} | avg_time_s={avg_time:.3f} | csv -> {csv_out}")

    return rows


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="OR-Tools baseline on TSPAEL64.pkl with CSV output (tqdm).")
    parser.add_argument("--dataset", type=str, default="zTSP/TrainingData/TSPAEL64.pkl",
                        help="Path to TSPAEL64.pkl (default: zTSP/TrainingData/TSPAEL64.pkl)")
    parser.add_argument("--csv", type=str, default="zTSP/evaluation/tspael64_ortools_baseline.csv",
                        help="Output CSV path (default: zTSP/evaluation/tspael64_ortools_baseline.csv)")
    parser.add_argument("--time_limit", type=float, default=10.0, help="Time limit per instance in seconds (default: 10)")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N instances (default: all)")
    parser.add_argument("--first_solution", type=str, default="PATH_CHEAPEST_ARC",
                        help="First solution strategy (e.g., PATH_CHEAPEST_ARC, PARALLEL_CHEAPEST_INSERTION, CHRISTOFIDES, ...)")
    parser.add_argument("--metaheuristic", type=str, default="GUIDED_LOCAL_SEARCH",
                        help="Local search metaheuristic (e.g., GUIDED_LOCAL_SEARCH, GREEDY_DESCENT, SIMULATED_ANNEALING, TABU_SEARCH)")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    evaluate_tspael64_ortools_to_csv(
        pkl_path=args.dataset,
        csv_out=args.csv,
        time_limit=args.time_limit,
        limit=args.limit,
        first_solution=args.first_solution,
        metaheuristic=args.metaheuristic,
        tqdm_disable=args.no_tqdm
    )


if __name__ == "__main__":
    main()
