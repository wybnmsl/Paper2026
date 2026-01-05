#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a TSP dataset like TSPAEL64.pkl but with 800-1200 node instances,
using LKH as "ground truth" solver for each instance.

Output .pkl keys match the original:
  - 'coordinate'      : list of (N,2) float64 arrays in [0,1]^2
  - 'distance_matrix' : list of (N,N) float64 symmetric, zero-diag
  - 'optimal_tour'    : list of (N,) int32 arrays, 0-based order
  - 'cost'            : list of float (tour length on the float matrix)

CLI:
  python zTSP/test/make_TSPAEL64_1000node.py \
    --out zTSP/TrainingData/TSPAEL64_1000node.pkl \
    --lkh_bin /path/to/LKH \
    --num 32 \
    --nmin 800 --nmax 1200 \
    --time_limit 30 \
    --runs 1 \
    --seed 42

Notes:
- LKH uses integer arc costs; we write EXPLICIT/FULL_MATRIX with scaling (default 1e6).
- If LKH fails on an instance (very rare), we fall back to a greedy NN tour so the dataset is still complete.
"""

import os
import sys
import math
import time
import json
import pickle
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

# -----------------------------
# Utilities
# -----------------------------

def gen_coords(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform coordinates in [0,1]^2 (float64, shape (n,2))."""
    return rng.random((n, 2), dtype=np.float64)

def euclidean_dist_matrix(coords: np.ndarray) -> np.ndarray:
    """Full pairwise Euclidean distance matrix (float64, symmetric, zero diag)."""
    # (x - y)^2 trick with broadcasting
    X = coords
    G = np.dot(X, X.T)                       # Gram
    sq = np.maximum(np.diag(G)[:, None] - 2*G + np.diag(G)[None, :], 0.0)
    D = np.sqrt(sq, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    # force symmetry
    D = 0.5 * (D + D.T)
    return D

def write_tsplib_full_matrix(tsp_path: str, D_float: np.ndarray, int_scale: float = 1_000_000.0):
    """Write EXPLICIT/FULL_MATRIX TSPLIB file with integer weights."""
    N = int(D_float.shape[0])
    D = np.array(D_float, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Distance matrix must be non-negative and finite.")
    M = np.rint(D * float(int_scale)).astype(np.int64)
    np.fill_diagonal(M, 0)
    lines = []
    lines.append(f"NAME : TSP_INST")
    lines.append(f"TYPE : TSP")
    lines.append(f"DIMENSION : {N}")
    lines.append(f"EDGE_WEIGHT_TYPE : EXPLICIT")
    lines.append(f"EDGE_WEIGHT_FORMAT : FULL_MATRIX")
    lines.append(f"EDGE_WEIGHT_SECTION")
    for i in range(N):
        lines.append(" ".join(str(int(x)) for x in M[i, :]))
    lines.append("EOF")
    os.makedirs(os.path.dirname(tsp_path) or ".", exist_ok=True)
    with open(tsp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return int_scale

def write_lkh_par(par_path: str,
                  tsp_path: str,
                  tour_path: str,
                  time_limit_s: float = 30.0,
                  runs: int = 1,
                  seed: int = 123456,
                  trace_level: int = 0):
    """Robust symmetric-TSP defaults commonly used with LKH."""
    lines = []
    lines.append(f"PROBLEM_FILE = {os.path.abspath(tsp_path)}")
    lines.append(f"OUTPUT_TOUR_FILE = {os.path.abspath(tour_path)}")
    lines.append(f"RUNS = {int(runs)}")
    lines.append(f"SEED = {int(seed)}")
    lines.append(f"TIME_LIMIT = {int(math.ceil(float(time_limit_s)))}")
    lines.append(f"TRACE_LEVEL = {int(trace_level)}")
    # classic:
    lines.append(f"MOVE_TYPE = 5")
    lines.append(f"PATCHING_C = 3")
    lines.append(f"PATCHING_A = 2")
    os.makedirs(os.path.dirname(par_path) or ".", exist_ok=True)
    with open(par_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def run_lkh(lkh_bin: str, par_path: str, cwd: str | None = None, timeout: float | None = None):
    """Run LKH and return (returncode, elapsed, stdout, stderr). With preflight checks."""
    bin_abs = os.path.abspath(lkh_bin)
    par_abs = os.path.abspath(par_path)
    if not os.path.isfile(bin_abs):
        return 1, 0.0, "", f"BIN_NOT_FOUND:{bin_abs}"
    if not os.access(bin_abs, os.X_OK):
        return 1, 0.0, "", f"BIN_NOT_EXECUTABLE:{bin_abs}"
    if not os.path.isfile(par_abs):
        return 1, 0.0, "", f"PAR_NOT_FOUND:{par_abs}"
    cmd = [bin_abs, par_abs]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=timeout, check=False, text=True)
        dt = time.time() - t0
        return res.returncode, dt, res.stdout, res.stderr
    except subprocess.TimeoutExpired as e:
        return 124, float(timeout or 0.0), "", f"Timeout: {e}"
    except Exception as e:
        return 1, time.time() - t0, "", f"Exception: {e}"

def parse_tour(tour_path: str, N_expected: int):
    """Parse TSPLIB TOUR to 0-based order list of length N_expected."""
    if not os.path.exists(tour_path):
        return None, "NO_TOUR_FILE"
    order = []
    in_section = False
    with open(tour_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not in_section:
                if s.upper() == "TOUR_SECTION":
                    in_section = True
                continue
            if s == "-1" or s.upper() == "EOF":
                break
            if not s:
                continue
            # Accept both one-per-line or space-separated
            for tok in s.split():
                if tok == "-1":
                    break
                try:
                    v = int(tok) - 1
                    order.append(v)
                except:
                    pass
    order = [int(x) for x in order if 0 <= x < N_expected]
    seen = set()
    uniq = []
    for v in order:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    if len(uniq) != N_expected:
        # try to complete
        for v in range(N_expected):
            if v not in seen:
                uniq.append(v)
    if len(uniq) != N_expected:
        return None, f"BAD_TOUR_LEN:{len(uniq)}"
    return uniq, ""

def greedy_nn_tour(D: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor as a safety fallback (0-based)."""
    n = D.shape[0]
    if n == 0:
        return []
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    cur = 0
    for _ in range(n - 1):
        # find nearest unvisited
        drow = D[cur].copy()
        drow[visited] = np.inf
        nxt = int(np.argmin(drow))
        if not np.isfinite(drow[nxt]):  # in case of pathological matrix
            # pick any unvisited
            candidates = np.where(~visited)[0]
            nxt = int(candidates[0])
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return order

def cycle_cost_from_order(D: np.ndarray, order: list[int]) -> float:
    n = len(order)
    s = 0.0
    for i in range(n):
        a = order[i]; b = order[(i + 1) % n]
        s += float(D[a, b])
    return float(s)

# -----------------------------
# Main build function
# -----------------------------

def make_dataset(out_path: str,
                 lkh_bin: str,
                 num_instances: int = 32,
                 nmin: int = 800,
                 nmax: int = 1200,
                 time_limit_s: float = 30.0,
                 runs: int = 1,
                 seed: int = 42,
                 int_scale: float = 1_000_000.0,
                 workdir: str = "zTSP/evaluation/lkh_build_tmp",
                 trace_level: int = 0):
    """
    Create dataset and write to out_path as .pkl with the TSPAEL64 signature.
    """
    rng = np.random.default_rng(seed)

    coords_list = []
    dists_list  = []
    tour_list   = []
    cost_list   = []

    os.makedirs(workdir, exist_ok=True)

    it = tqdm(range(num_instances), desc="Building TSPAEL64_1000node via LKH", unit="inst")
    for idx in it:
        # 1) sample N and generate instance
        N = int(rng.integers(nmin, nmax + 1))
        coords = gen_coords(N, rng)
        D = euclidean_dist_matrix(coords)

        # 2) prepare LKH files per-instance
        inst_dir = os.path.join(workdir, f"inst_{idx:03d}")
        os.makedirs(inst_dir, exist_ok=True)
        tsp_path  = os.path.join(inst_dir, "problem.tsp")
        par_path  = os.path.join(inst_dir, "params.par")
        tour_path = os.path.join(inst_dir, "solution.tour")

        write_tsplib_full_matrix(tsp_path, D, int_scale=int_scale)
        write_lkh_par(par_path, tsp_path, tour_path,
                      time_limit_s=time_limit_s, runs=runs, seed=seed + idx, trace_level=trace_level)

        # 3) run LKH (timeout = time_limit + 5s guard)
        rc, lkh_time, out, err = run_lkh(lkh_bin, par_path, cwd=inst_dir, timeout=time_limit_s + 5.0)

        # 4) parse tour or fallback
        if rc == 0 and os.path.exists(tour_path):
            order, perr = parse_tour(tour_path, N_expected=N)
            if order is None:
                # Fallback to greedy NN (rare)
                order = greedy_nn_tour(D)
        else:
            order = greedy_nn_tour(D)

        # 5) compute float cost
        cost = cycle_cost_from_order(D, order)

        # 6) append to lists (cast types like the original file)
        coords_list.append(coords.astype(np.float64))
        dists_list.append(D.astype(np.float64))
        tour_list.append(np.asarray(order, dtype=np.int32))
        cost_list.append(float(cost))

        it.set_postfix(rc=rc, N=N, cost=cost)

    # 7) pack and dump
    dataset = {
        "coordinate": coords_list,
        "distance_matrix": dists_list,
        "optimal_tour": tour_list,
        "cost": np.asarray(cost_list, dtype=np.float64)
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] saved dataset to: {out_path}")
    print(f"  instances={num_instances} | N range=[{nmin},{nmax}] | LKH time_limit={time_limit_s}s | runs={runs}")

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Make TSPAEL64_1000node.pkl using LKH as ground truth.")
    parser.add_argument("--out", type=str, default="zTSP/TrainingData/TSPAEL64_1000node.pkl",
                        help="Output .pkl path (default: zTSP/TrainingData/TSPAEL64_1000node.pkl)")
    parser.add_argument("--lkh_bin", type=str, required=True,
                        help="Path to LKH executable (e.g., external/LKH-3.0.13/LKH)")
    parser.add_argument("--num", type=int, default=32, help="Number of instances (default: 32)")
    parser.add_argument("--nmin", type=int, default=800, help="Min number of nodes per instance (default: 800)")
    parser.add_argument("--nmax", type=int, default=1200, help="Max number of nodes per instance (default: 1200)")
    parser.add_argument("--time_limit", type=float, default=30.0, help="LKH time limit per instance, seconds (default: 30)")
    parser.add_argument("--runs", type=int, default=1, help="LKH RUNS parameter (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--int_scale", type=float, default=1_000_000.0, help="Float->int scaling for TSPLIB matrix (default: 1e6)")
    parser.add_argument("--workdir", type=str, default="zTSP/evaluation/lkh_build_tmp",
                        help="Working directory for per-instance TSPLIB files")
    parser.add_argument("--trace", type=int, default=0, help="LKH TRACE_LEVEL (default: 0)")
    args = parser.parse_args()

    make_dataset(
        out_path=args.out,
        lkh_bin=args.lkh_bin,
        num_instances=args.num,
        nmin=args.nmin, nmax=args.nmax,
        time_limit_s=args.time_limit,
        runs=args.runs,
        seed=args.seed,
        int_scale=args.int_scale,
        workdir=args.workdir,
        trace_level=args.trace
    )

if __name__ == "__main__":
    main()
