#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
import re
import time
import threading
from array import array
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ===========================
# Per-instance budgets = LDR Time(s) from your LaTeX table
# Only these cases will be evaluated.
# ===========================
LDR_BUDGET_SEC: Dict[str, float] = {
    "burma14": 0.160,
    "ulysses16": 0.114,
    "gr17": 0.099,
    "gr21": 0.198,
    "ulysses22": 0.210,
    "gr24": 0.304,
    "fri26": 0.285,
    "bayg29": 0.200,
    "bays29": 0.114,
    "dantzig42": 0.128,
    "swiss42": 0.044,
    "att48": 0.098,
    "gr48": 0.053,
    "hk48": 0.326,
    "eil51": 0.308,
    "berlin52": 0.032,
    "brazil58": 0.028,
    "st70": 0.067,
    "eil76": 0.054,
    "pr76": 0.130,
    "gr96": 0.108,
    "rat99": 0.325,
    "kroA100": 0.153,
    "kroB100": 0.164,
    "kroC100": 0.158,
    "kroD100": 0.178,
    "kroE100": 0.066,
    "rd100": 0.114,
    "eil101": 0.305,
    "lin105": 0.240,
    "pr107": 0.787,
    "gr120": 0.785,
    "pr124": 0.272,
    "bier127": 0.334,
    "ch130": 0.241,
    "pr136": 0.090,
    "gr137": 0.122,
    "pr144": 0.131,
    "ch150": 0.193,
    "kroA150": 0.397,
    "kroB150": 0.303,
    "pr152": 0.115,
    "u159": 0.365,
    "si175": 0.211,
    "brg180": 1.945,
    "rat195": 0.193,
    "d198": 0.227,
    "kroA200": 0.996,
    "kroB200": 0.424,
    "gr202": 1.144,
    "ts225": 0.989,
    "tsp225": 4.202,
    "pr226": 0.627,
    "gr229": 0.129,
    "gil262": 1.440,
    "pr264": 0.445,
    "a280": 2.053,
    "pr299": 1.228,
    "lin318": 0.928,
    "rd400": 3.110,
    "fl417": 6.143,
    "gr431": 1.437,
    "pr439": 1.330,
    "pcb442": 5.701,
    "d493": 3.694,
    "att532": 5.692,
    "ali535": 3.511,
    "si535": 9.588,
    "pa561": 4.956,
    "u574": 4.970,
    "rat575": 5.111,
    "p654": 3.302,
    "d657": 7.911,
    "gr666": 16.217,
    "u724": 17.717,
    "rat783": 8.462,
}


# ---------------------------
# solutions file (optimum)
# ---------------------------
def read_opt_map(solutions_path: Path) -> dict[str, int]:
    opt = {}
    for line in solutions_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        name, val = line.split(":", 1)
        name = name.strip()
        m = re.search(r"(\d+)", val)
        if m:
            opt[name] = int(m.group(1))
    return opt


# ---------------------------
# TSPLIB distances for coord-based
# ---------------------------
def nint(x: float) -> int:
    return int(math.floor(x + 0.5))

def geo_to_radians(x: float) -> float:
    deg = int(x)
    minute = x - deg
    return math.pi * (deg + 5.0 * minute / 3.0) / 180.0

def dist_euc_2d(a, b) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return nint(math.sqrt(dx * dx + dy * dy))

def dist_ceil_2d(a, b) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(math.ceil(math.sqrt(dx * dx + dy * dy)))

def dist_att(a, b) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    rij = math.sqrt((dx * dx + dy * dy) / 10.0)
    tij = nint(rij)
    return tij + 1 if tij < rij else tij

def dist_geo(a, b) -> int:
    RRR = 6378.388
    lat_i = geo_to_radians(a[0])
    lon_i = geo_to_radians(a[1])
    lat_j = geo_to_radians(b[0])
    lon_j = geo_to_radians(b[1])

    q1 = math.cos(lon_i - lon_j)
    q2 = math.cos(lat_i - lat_j)
    q3 = math.cos(lat_i + lat_j)
    dij = RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
    return int(dij)

def build_distance_matrix_from_coords(coords, ew_type: str):
    n = len(coords)
    if ew_type == "EUC_2D":
        f = dist_euc_2d
    elif ew_type == "CEIL_2D":
        f = dist_ceil_2d
    elif ew_type == "ATT":
        f = dist_att
    elif ew_type == "GEO":
        f = dist_geo
    else:
        raise ValueError(f"Unsupported coord EDGE_WEIGHT_TYPE={ew_type}")

    rows = []
    for i in range(n):
        row = array("I", [0]) * n
        ai = coords[i]
        for j in range(n):
            row[j] = 0 if i == j else f(ai, coords[j])
        rows.append(row)
    return rows


# ---------------------------
# TSPLIB parsing (supports your repo formats)
# ---------------------------
def parse_tsplib_problem(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    name = path.stem
    typ = None
    dim = None
    ew_type = None
    ew_format = None

    section = None
    coords = []
    weights_tokens = []

    def set_header(k: str, v: str):
        nonlocal name, typ, dim, ew_type, ew_format
        k = k.strip().upper()
        v = v.strip()
        if k == "NAME":
            name = v
        elif k == "TYPE":
            typ = v.upper()
        elif k == "DIMENSION":
            dim = int(re.search(r"\d+", v).group(0))
        elif k == "EDGE_WEIGHT_TYPE":
            ew_type = v.upper()
        elif k == "EDGE_WEIGHT_FORMAT":
            ew_format = v.upper()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        up = line.upper()
        if up == "NODE_COORD_SECTION":
            section = "COORD"
            i += 1
            break
        if up == "EDGE_WEIGHT_SECTION":
            section = "WEIGHT"
            i += 1
            break
        if up == "EOF":
            break
        if ":" in line:
            k, v = line.split(":", 1)
            set_header(k, v)
        i += 1

    if typ is None:
        typ = "TSP"
    typ0 = typ.split()[0] if typ else "TSP"  # "TSP (M.~HOFMEISTER)" -> "TSP"

    if dim is None or ew_type is None:
        raise ValueError(f"Missing DIMENSION/EDGE_WEIGHT_TYPE in {path}")

    if section == "COORD":
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            up = line.upper()
            if up == "EOF" or up.startswith("DISPLAY_DATA_SECTION") or up.startswith("EDGE_WEIGHT_SECTION"):
                break
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))

    elif section == "WEIGHT":
        stop_markers = {
            "EOF",
            "DISPLAY_DATA_SECTION",
            "NODE_COORD_SECTION",
            "EDGE_DATA_SECTION",
            "TOUR_SECTION",
            "DEMAND_SECTION",
            "FIXED_EDGES_SECTION",
        }
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            up = line.upper()
            if up in stop_markers or up.startswith("DISPLAY_DATA_SECTION") or up.startswith("NODE_COORD_SECTION") or up.startswith("EDGE_DATA_SECTION"):
                break
            if line == "-1":
                break
            for tok in line.split():
                if tok == "-1":
                    break
                try:
                    weights_tokens.append(int(float(tok)))
                except ValueError:
                    break
    else:
        raise ValueError(f"No NODE_COORD_SECTION or EDGE_WEIGHT_SECTION found in {path}")

    return {
        "name": name,
        "type": typ0,
        "dimension": dim,
        "edge_weight_type": ew_type,
        "edge_weight_format": ew_format,
        "coords": coords,
        "weights_tokens": weights_tokens,
    }

def build_matrix_from_explicit(tokens: list[int], n: int, fmt: str, typ: str):
    fmt = (fmt or "").upper()
    typ = (typ or "TSP").upper()

    mat = [array("I", [0]) * n for _ in range(n)]

    def set_ij(i: int, j: int, v: int):
        mat[i][j] = v

    idx = 0
    if fmt == "FULL_MATRIX":
        need = n * n
        if len(tokens) < need:
            raise ValueError(f"FULL_MATRIX needs {need} numbers but got {len(tokens)}")
        for i in range(n):
            for j in range(n):
                set_ij(i, j, int(tokens[idx])); idx += 1
        return mat

    if typ == "ATSP":
        raise ValueError(f"ATSP with EDGE_WEIGHT_FORMAT={fmt} not supported (expect FULL_MATRIX)")

    if fmt == "LOWER_DIAG_ROW":
        for i in range(n):
            for j in range(i + 1):
                v = int(tokens[idx]); idx += 1
                set_ij(i, j, v)
                set_ij(j, i, v)
        return mat

    if fmt == "UPPER_DIAG_ROW":
        for i in range(n):
            for j in range(i, n):
                v = int(tokens[idx]); idx += 1
                set_ij(i, j, v)
                set_ij(j, i, v)
        return mat

    if fmt == "LOWER_ROW":
        for i in range(n):
            for j in range(i):
                v = int(tokens[idx]); idx += 1
                set_ij(i, j, v)
                set_ij(j, i, v)
        return mat

    if fmt == "UPPER_ROW":
        for i in range(n):
            for j in range(i + 1, n):
                v = int(tokens[idx]); idx += 1
                set_ij(i, j, v)
                set_ij(j, i, v)
        return mat

    if fmt == "LOWER_DIAG_COL":
        for j in range(n):
            for i in range(j, n):
                v = int(tokens[idx]); idx += 1
                set_ij(i, j, v)
                set_ij(j, i, v)
        return mat

    if fmt == "UPPER_DIAG_COL":
        for j in range(n):
            for i in range(0, j + 1):
                v = int(tokens[idx]); idx += 1
                set_ij(i, j, v)
                set_ij(j, i, v)
        return mat

    if fmt == "LOWER_COL":
        for j in range(n):
            for i in range(j + 1, n):
                v = int(tokens[idx]); idx += 1
                set_ij(i, j, v)
                set_ij(j, i, v)
        return mat

    if fmt == "UPPER_COL":
        for j in range(n):
            for i in range(0, j):
                v = int(tokens[idx]); idx += 1
                set_ij(i, j, v)
                set_ij(j, i, v)
        return mat

    raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT={fmt}")

def build_distance_matrix_tsplib(problem: dict):
    n = problem["dimension"]
    typ = problem["type"]
    ew_type = problem["edge_weight_type"]

    if ew_type in {"EUC_2D", "CEIL_2D", "ATT", "GEO"}:
        coords = problem["coords"]
        if len(coords) != n:
            coords = coords[:n]
        if not coords:
            raise ValueError("No coords for coord-based instance")
        return build_distance_matrix_from_coords(coords, ew_type)

    if ew_type == "EXPLICIT":
        tokens = problem["weights_tokens"]
        if not tokens:
            raise ValueError("No EDGE_WEIGHT_SECTION tokens for EXPLICIT")
        return build_matrix_from_explicit(tokens, n, problem["edge_weight_format"], typ)

    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE={ew_type}")


# ---------------------------
# OR-Tools solver
# ---------------------------
def _set_duration_ms(duration_obj, ms: int) -> None:
    """Compatible millisecond time limit setter across OR-Tools builds."""
    if ms < 1:
        ms = 1
    sec = ms // 1000
    nanos = (ms % 1000) * 1_000_000
    # Duration is google.protobuf.duration_pb2.Duration
    duration_obj.seconds = int(sec)
    duration_obj.nanos = int(nanos)

def solve_tsp_ortools(distance_matrix, time_limit_ms: int, meta: str, seed: int) -> Tuple[Optional[int], Optional[list[int]]]:
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(distance_matrix[i][j])

    transit_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    meta = meta.upper()
    if meta == "NONE":
        pass
    elif meta == "GLS":
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    elif meta == "TABU":
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    elif meta == "SA":
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    else:
        raise ValueError("meta must be one of NONE/GLS/TABU/SA")

    # random seed (best-effort compatibility)
    try:
        if hasattr(search_params, "random_seed"):
            search_params.random_seed = int(seed)
    except Exception:
        pass

    # millisecond budget (NO FromMilliseconds; set seconds/nanos)
    _set_duration_ms(search_params.time_limit, int(time_limit_ms))

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        return None, None

    best_cost = int(solution.ObjectiveValue())

    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    return best_cost, route

def write_tsplib_tour(out_path: Path, name: str, route_nodes_0based):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : TOUR\n")
        f.write(f"DIMENSION : {len(route_nodes_0based)}\n")
        f.write("TOUR_SECTION\n")
        for v in route_nodes_0based:
            f.write(f"{v + 1}\n")
        f.write("-1\nEOF\n")


# ---------------------------
# Worker
# ---------------------------
def run_one(
    tsp: Path,
    opt_map: dict[str, int],
    out_dir: Path,
    meta: str,
    save_tour: bool,
    seed: int,
    enforce_total_budget: bool,
) -> Dict[str, Any]:
    stem = tsp.stem
    budget_sec = LDR_BUDGET_SEC[stem]

    try:
        t_total0 = time.perf_counter()

        prob = parse_tsplib_problem(tsp)
        n = prob["dimension"]

        t_build0 = time.perf_counter()
        dist_mat = build_distance_matrix_tsplib(prob)
        t_build1 = time.perf_counter()
        build_s = t_build1 - t_build0

        # remaining solve budget
        remain_sec = budget_sec - build_s if enforce_total_budget else budget_sec
        if remain_sec <= 0:
            t_total1 = time.perf_counter()
            return {
                "name": stem,
                "n": n,
                "type": prob["type"],
                "edge_weight_type": prob["edge_weight_type"],
                "edge_weight_format": prob["edge_weight_format"] or "",
                "budget_sec": f"{budget_sec:.6f}",
                "opt": opt_map.get(stem, ""),
                "best": "",
                "gap_percent": "",
                "time_build_sec": f"{build_s:.6f}",
                "time_solve_sec": f"{0.0:.6f}",
                "time_total_sec": f"{(t_total1 - t_total0):.6f}",
                "status": "BUDGET_TOO_SMALL",
            }

        solve_ms = int(math.ceil(remain_sec * 1000.0))
        t0 = time.perf_counter()
        best, route = solve_tsp_ortools(dist_mat, solve_ms, meta, seed)
        t1 = time.perf_counter()
        solve_s = t1 - t0

        t_total1 = time.perf_counter()
        total_s = t_total1 - t_total0

        optimum = opt_map.get(stem) or opt_map.get(prob["name"])

        if best is None:
            return {
                "name": stem,
                "n": n,
                "type": prob["type"],
                "edge_weight_type": prob["edge_weight_type"],
                "edge_weight_format": prob["edge_weight_format"] or "",
                "budget_sec": f"{budget_sec:.6f}",
                "opt": optimum if optimum is not None else "",
                "best": "",
                "gap_percent": "",
                "time_build_sec": f"{build_s:.6f}",
                "time_solve_sec": f"{solve_s:.6f}",
                "time_total_sec": f"{total_s:.6f}",
                "status": "NO_SOLUTION",
            }

        gap = ""
        if optimum is not None and optimum > 0:
            gap = f"{((best - optimum) / float(optimum) * 100.0):.6f}"

        if save_tour and route is not None:
            tour_path = out_dir / stem / f"{stem}.ortools.tour"
            write_tsplib_tour(tour_path, stem, route)

        return {
            "name": stem,
            "n": n,
            "type": prob["type"],
            "edge_weight_type": prob["edge_weight_type"],
            "edge_weight_format": prob["edge_weight_format"] or "",
            "budget_sec": f"{budget_sec:.6f}",
            "opt": optimum if optimum is not None else "",
            "best": best,
            "gap_percent": gap,
            "time_build_sec": f"{build_s:.6f}",
            "time_solve_sec": f"{solve_s:.6f}",
            "time_total_sec": f"{total_s:.6f}",
            "status": "OK",
        }

    except Exception as e:
        return {
            "name": stem,
            "n": "",
            "type": "",
            "edge_weight_type": "",
            "edge_weight_format": "",
            "budget_sec": f"{budget_sec:.6f}",
            "opt": "",
            "best": "",
            "gap_percent": "",
            "time_build_sec": "",
            "time_solve_sec": "",
            "time_total_sec": "",
            "status": f"EXCEPTION:{type(e).__name__}",
        }


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsplib_root", type=str, required=True)
    ap.add_argument("--solutions", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="ortools_out_budgeted")
    ap.add_argument("--csv", type=str, default="ortools_budgeted_by_ldr.csv")
    ap.add_argument("--meta", type=str, default="GLS", help="NONE / GLS / TABU / SA")
    ap.add_argument("--save_tour", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="0 => min(8, cpu_count)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--enforce_total_budget", action="store_true",
                    help="If set: (build_time + solve_time) <= budget approx; solve budget = budget - build_time.")
    args = ap.parse_args()

    tsplib_root = Path(os.path.expanduser(args.tsplib_root)).resolve()
    sol_path = Path(os.path.expanduser(args.solutions)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    csv_path = Path(os.path.expanduser(args.csv)).resolve()

    if not tsplib_root.exists():
        raise FileNotFoundError(tsplib_root)
    if not sol_path.exists():
        raise FileNotFoundError(sol_path)

    opt_map = read_opt_map(sol_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ONLY run cases in the provided table
    selected = []
    missing = []
    for name in sorted(LDR_BUDGET_SEC.keys()):
        p = tsplib_root / f"{name}.tsp"
        if p.exists():
            selected.append(p)
        else:
            hits = list(tsplib_root.rglob(f"{name}.tsp"))
            if hits:
                selected.append(hits[0])
            else:
                missing.append(name)

    cpu = os.cpu_count() or 4
    workers = args.workers if args.workers and args.workers > 0 else min(8, cpu)

    print(f"Budgeted cases in table: {len(LDR_BUDGET_SEC)}")
    print(f"Found files to run: {len(selected)}")
    if missing:
        print(f"WARNING: missing .tsp files for: {missing}")
    print(f"Workers: {workers} | meta={args.meta} | seed={args.seed} | enforce_total_budget={args.enforce_total_budget}")

    rows = []
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(run_one, tsp, opt_map, out_dir, args.meta, args.save_tour, args.seed, args.enforce_total_budget)
            for tsp in selected
        ]
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            with print_lock:
                if row["status"] == "OK":
                    print(f"[{row['name']}] n={row['n']} budget={float(row['budget_sec']):.3f}s "
                          f"best={row['best']} opt={row['opt']} gap={row['gap_percent']}% "
                          f"build={float(row['time_build_sec']):.3f}s solve={float(row['time_solve_sec']):.3f}s "
                          f"total={float(row['time_total_sec']):.3f}s")
                else:
                    print(f"[{row['name']}] status={row['status']} budget={float(row['budget_sec']):.3f}s")

    rows.sort(key=lambda r: r["name"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name", "n", "type",
                "edge_weight_type", "edge_weight_format",
                "budget_sec",
                "opt", "best", "gap_percent",
                "time_build_sec", "time_solve_sec", "time_total_sec",
                "status",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. CSV saved to: {csv_path}")
    print(f"Tours (optional) under: {out_dir}")


if __name__ == "__main__":
    main()
