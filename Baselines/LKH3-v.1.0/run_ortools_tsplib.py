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
from typing import Optional, Dict, Any, Tuple

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ---------------------------
# fast DIMENSION parse for prefilter
# ---------------------------
RE_DIM = re.compile(r"^\s*DIMENSION\s*:?\s*(\d+)\s*$", re.IGNORECASE)

def parse_dimension_fast(tsp_path: Path, max_lines: int = 500) -> Optional[int]:
    """快速从头部解析 DIMENSION，用于先过滤 >1002 的实例，避免浪费解析/建矩阵时间。"""
    try:
        with tsp_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                m = RE_DIM.match(line.strip())
                if m:
                    return int(m.group(1))
                up = line.upper()
                if up.startswith("NODE_COORD_SECTION") or up.startswith("EDGE_WEIGHT_SECTION"):
                    break
    except Exception:
        return None
    return None


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
    # TSPLIB GEO uses DDD.MM format
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
    # TSPLIB ATT (pseudo-Euclidean)
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    rij = math.sqrt((dx * dx + dy * dy) / 10.0)
    tij = nint(rij)
    return tij + 1 if tij < rij else tij

def dist_geo(a, b) -> int:
    # TSPLIB GEO
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
            if i == j:
                row[j] = 0
            else:
                row[j] = f(ai, coords[j])
        rows.append(row)
    return rows


# ---------------------------
# TSPLIB parser that supports your repo:
#  - coord-based: EUC_2D / CEIL_2D / ATT / GEO
#  - explicit matrices: FULL_MATRIX / LOWER_DIAG_ROW / UPPER_ROW / UPPER_DIAG_ROW etc
#  - handles DISPLAY_DATA_SECTION after EDGE_WEIGHT_SECTION (bayg29/bays29/dantzig42)
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

    # Normalize TYPE like "TSP (M.~HOFMEISTER)" -> "TSP"
    # (so your audit里那3个也能稳定处理)
    typ0 = typ.split()[0] if typ else "TSP"

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
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))

    elif section == "WEIGHT":
        # IMPORTANT FIX:
        # stop when meet DISPLAY_DATA_SECTION / NODE_COORD_SECTION / etc,
        # not only EOF / -1
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
                    # treat non-numeric token as section start / malformed line
                    break

    else:
        raise ValueError(f"No NODE_COORD_SECTION or EDGE_WEIGHT_SECTION found in {path}")

    return {
        "name": name,
        "type_raw": typ,
        "type": typ0,  # normalized
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

    symmetric = (typ == "TSP")

    idx = 0

    if fmt == "FULL_MATRIX":
        need = n * n
        if len(tokens) < need:
            raise ValueError(f"FULL_MATRIX needs {need} numbers but got {len(tokens)}")
        for i in range(n):
            for j in range(n):
                set_ij(i, j, int(tokens[idx])); idx += 1
        return mat

    # For ATSP, only FULL_MATRIX is safe; otherwise skip
    if typ == "ATSP":
        raise ValueError(f"ATSP with EDGE_WEIGHT_FORMAT={fmt} not supported (expect FULL_MATRIX)")

    # Triangular formats for symmetric TSP
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
    typ = problem["type"]          # normalized "TSP" or "ATSP"
    ew_type = problem["edge_weight_type"]

    if ew_type in {"EUC_2D", "CEIL_2D", "ATT", "GEO"}:
        coords = problem["coords"]
        if len(coords) != n:
            coords = coords[:n]
        if not coords:
            raise ValueError("No coords for coord-based instance")
        return build_distance_matrix_from_coords(coords, ew_type)

    if ew_type == "EXPLICIT":
        fmt = problem["edge_weight_format"]
        tokens = problem["weights_tokens"]
        if not tokens:
            raise ValueError("No EDGE_WEIGHT_SECTION tokens for EXPLICIT")
        return build_matrix_from_explicit(tokens, n, fmt, typ)

    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE={ew_type}")


# ---------------------------
# OR-Tools solver
# ---------------------------
def solve_tsp_ortools(distance_matrix, time_limit_s: float, meta: str) -> Tuple[Optional[int], Optional[list[int]]]:
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

    search_params.time_limit.FromSeconds(int(time_limit_s))

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
    time_limit: float,
    meta: str,
    save_tour: bool
) -> Dict[str, Any]:
    stem = tsp.stem
    try:
        prob = parse_tsplib_problem(tsp)
        n = prob["dimension"]

        t_build0 = time.perf_counter()
        dist_mat = build_distance_matrix_tsplib(prob)
        t_build1 = time.perf_counter()

        t0 = time.perf_counter()
        best, route = solve_tsp_ortools(dist_mat, time_limit, meta)
        t1 = time.perf_counter()

        wall_solve = t1 - t0
        wall_build = t_build1 - t_build0

        optimum = opt_map.get(stem) or opt_map.get(prob["name"])

        if best is None:
            return {
                "name": stem,
                "n": n,
                "type": prob["type"],
                "edge_weight_type": prob["edge_weight_type"],
                "edge_weight_format": prob["edge_weight_format"] or "",
                "opt": optimum if optimum is not None else "",
                "best": "",
                "gap_percent": "",
                "time_solve_sec": f"{wall_solve:.6f}",
                "time_build_sec": f"{wall_build:.6f}",
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
            "opt": optimum if optimum is not None else "",
            "best": best,
            "gap_percent": gap,
            "time_solve_sec": f"{wall_solve:.6f}",
            "time_build_sec": f"{wall_build:.6f}",
            "status": "OK",
        }

    except Exception as e:
        return {
            "name": stem,
            "n": "",
            "type": "",
            "edge_weight_type": "",
            "edge_weight_format": "",
            "opt": "",
            "best": "",
            "gap_percent": "",
            "time_solve_sec": "",
            "time_build_sec": "",
            "status": f"EXCEPTION:{type(e).__name__}",
        }


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsplib_root", type=str, required=True)
    ap.add_argument("--solutions", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="ortools_out")
    ap.add_argument("--csv", type=str, default="ortools_tsplib_results.csv")
    ap.add_argument("--max_n", type=int, default=1002, help="ONLY run instances with DIMENSION <= max_n (default 1002)")
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--meta", type=str, default="GLS", help="NONE / GLS / TABU / SA")
    ap.add_argument("--save_tour", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="0 => min(8, cpu_count)")

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
    tsp_files = sorted(tsplib_root.rglob("*.tsp"))

    out_dir.mkdir(parents=True, exist_ok=True)

    # prefilter by DIMENSION <= max_n
    selected = []
    skipped_dim = 0
    skipped_nodim = 0
    for tsp in tsp_files:
        dim = parse_dimension_fast(tsp)
        if dim is None:
            skipped_nodim += 1
            continue
        if dim > args.max_n:
            skipped_dim += 1
            continue
        selected.append(tsp)

    cpu = os.cpu_count() or 4
    workers = args.workers if args.workers and args.workers > 0 else min(8, cpu)

    print(f"Found {len(tsp_files)} .tsp files")
    print(f"Selected (DIM<= {args.max_n}): {len(selected)}")
    print(f"Skipped: no_dim={skipped_nodim}, dim>{args.max_n}={skipped_dim}")
    print(f"Workers: {workers} | meta={args.meta} | time_limit={args.time_limit}s")

    rows = []
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(run_one, tsp, opt_map, out_dir, args.time_limit, args.meta, args.save_tour)
            for tsp in selected
        ]
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            with print_lock:
                if row["status"] == "OK":
                    if row["gap_percent"] != "":
                        print(f"[{row['name']}] n={row['n']} type={row['type']} "
                              f"{row['edge_weight_type']}/{row['edge_weight_format']} "
                              f"best={row['best']} opt={row['opt']} gap={row['gap_percent']}% "
                              f"solve={float(row['time_solve_sec']):.2f}s build={float(row['time_build_sec']):.2f}s")
                    else:
                        print(f"[{row['name']}] n={row['n']} type={row['type']} "
                              f"{row['edge_weight_type']}/{row['edge_weight_format']} "
                              f"best={row['best']} solve={float(row['time_solve_sec']):.2f}s build={float(row['time_build_sec']):.2f}s")
                else:
                    print(f"[{row['name']}] status={row['status']}")

    rows.sort(key=lambda r: r["name"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name", "n", "type",
                "edge_weight_type", "edge_weight_format",
                "opt", "best", "gap_percent",
                "time_solve_sec", "time_build_sec",
                "status",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. CSV saved to: {csv_path}")
    print(f"Outputs saved under: {out_dir}")

if __name__ == "__main__":
    main()
