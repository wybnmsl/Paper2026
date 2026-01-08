#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSPLIB KGLS (Knowledge-Guided Guided Local Search) baseline for TSP with progress logging.

Knowledge used:
  - k-nearest neighbor candidate edges (common TSP "knowledge")
  - Candidate-guided 2-opt neighborhood (only try k positions induced by candidate nodes, plus a small fallback window)
  - Knowledge-weighted utility in GLS penalty update to penalize non-candidate edges more aggressively

Algorithm:
  - Init: Nearest Neighbor tour
  - Inner LS: 2-opt first-improvement optimizing augmented objective
        Aug = length + lambda * sum(penalty(edge))
    but neighborhood is guided by candidate lists (knowledge).
  - Outer loop: At local optimum, compute utilities on edges in current tour:
        util(e) = dist(e) / (1 + penalty(e)) * (1 + beta * (1 - is_candidate_edge(e)))
    penalize all edges with maximal utility (ties).
  - Stop by time limit (10s for classes 20/50/100/500, 60s for 1000) or max_iters.

Run:
  python run_kgls_tsplib.py \
    --tsp_root zTSP_visual/tsplib-master \
    --solutions_file zTSP_visual/tsplib-master/solutions \
    --out_csv kgls_results.csv \
    --verbose
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Set

import numpy as np


# -----------------------------
# TSPLIB parsing
# -----------------------------
@dataclass
class TSPLIBInstance:
    name: str
    dimension: int
    edge_weight_type: str
    edge_weight_format: Optional[str]
    coords: Optional[np.ndarray]
    dist: np.ndarray  # int32, shape (n,n)


def _parse_keyval(line: str) -> Optional[Tuple[str, str]]:
    if ":" not in line:
        return None
    k, v = line.split(":", 1)
    k = k.strip().upper()
    v = v.strip()
    if not k:
        return None
    return k, v


def _geo_to_rad(x: float) -> float:
    deg = int(x)
    minutes = x - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0


def _build_dist_from_coords(coords: np.ndarray, ewt: str) -> np.ndarray:
    ewt = ewt.upper()
    n = coords.shape[0]

    if ewt in ("EUC_2D", "CEIL_2D", "ATT"):
        x = coords[:, 0].astype(np.float64)
        y = coords[:, 1].astype(np.float64)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx * dx + dy * dy)

        if ewt == "EUC_2D":
            dist = (d + 0.5).astype(np.int32)  # TSPLIB rounding
        elif ewt == "CEIL_2D":
            dist = np.ceil(d).astype(np.int32)
        else:  # ATT
            rij = np.sqrt((dx * dx + dy * dy) / 10.0)
            tij = np.floor(rij + 0.5)
            dist = np.where(tij < rij, tij + 1.0, tij).astype(np.int32)

        np.fill_diagonal(dist, 0)
        return dist

    if ewt == "GEO":
        lat = np.array([_geo_to_rad(float(v)) for v in coords[:, 0]], dtype=np.float64)
        lon = np.array([_geo_to_rad(float(v)) for v in coords[:, 1]], dtype=np.float64)
        RRR = 6378.388
        dlon = lon[:, None] - lon[None, :]
        q1 = np.cos(dlon)
        q2 = np.cos(lat[:, None] - lat[None, :])
        q3 = np.cos(lat[:, None] + lat[None, :])
        arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
        arg = np.clip(arg, -1.0, 1.0)
        dist = (RRR * np.arccos(arg) + 1.0).astype(np.int32)
        np.fill_diagonal(dist, 0)
        return dist

    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE for coords: {ewt}")


def _build_dist_from_explicit(weights: List[int], n: int, fmt: str) -> np.ndarray:
    fmt = (fmt or "").upper()
    dist = np.zeros((n, n), dtype=np.int32)

    idx = 0
    if fmt == "FULL_MATRIX":
        needed = n * n
        if len(weights) < needed:
            raise ValueError(f"EXPLICIT FULL_MATRIX needs {needed} weights, got {len(weights)}")
        return np.array(weights[:needed], dtype=np.int32).reshape((n, n))

    if fmt == "UPPER_ROW":
        for i in range(n):
            for j in range(i + 1, n):
                w = weights[idx]
                dist[i, j] = w
                dist[j, i] = w
                idx += 1
        return dist

    if fmt == "LOWER_ROW":
        for i in range(n):
            for j in range(0, i):
                w = weights[idx]
                dist[i, j] = w
                dist[j, i] = w
                idx += 1
        return dist

    if fmt == "UPPER_DIAG_ROW":
        for i in range(n):
            for j in range(i, n):
                w = weights[idx]
                dist[i, j] = w
                dist[j, i] = w
                idx += 1
        np.fill_diagonal(dist, 0)
        return dist

    if fmt == "LOWER_DIAG_ROW":
        for i in range(n):
            for j in range(0, i + 1):
                w = weights[idx]
                dist[i, j] = w
                dist[j, i] = w
                idx += 1
        np.fill_diagonal(dist, 0)
        return dist

    raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")


def read_tsplib_tsp(path: str) -> TSPLIBInstance:
    name = os.path.splitext(os.path.basename(path))[0]
    dimension = None
    edge_weight_type = None
    edge_weight_format = None

    coords: List[Tuple[float, float]] = []
    weights_tokens: List[int] = []

    section = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            u = line.upper()
            if u.startswith("NODE_COORD_SECTION"):
                section = "NODE_COORD"
                continue
            if u.startswith("DISPLAY_DATA_SECTION"):
                section = "DISPLAY_DATA"
                continue
            if u.startswith("EDGE_WEIGHT_SECTION"):
                section = "EDGE_WEIGHT"
                continue
            if u.startswith("EOF"):
                break

            kv = _parse_keyval(line)
            if kv and section is None:
                k, v = kv
                if k == "NAME":
                    name = v.split()[0]
                elif k == "DIMENSION":
                    dimension = int(re.findall(r"\d+", v)[0])
                elif k == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = v.split()[0].upper()
                elif k == "EDGE_WEIGHT_FORMAT":
                    edge_weight_format = v.split()[0].upper()
                continue

            if section in ("NODE_COORD", "DISPLAY_DATA"):
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
                continue

            if section == "EDGE_WEIGHT":
                for tok in line.split():
                    if re.fullmatch(r"[-+]?\d+", tok):
                        weights_tokens.append(int(tok))
                continue

    if dimension is None:
        raise ValueError(f"Missing DIMENSION in {path}")
    if edge_weight_type is None:
        raise ValueError(f"Missing EDGE_WEIGHT_TYPE in {path}")

    n = dimension
    ewt = edge_weight_type.upper()

    if ewt == "EXPLICIT":
        if not edge_weight_format:
            raise ValueError(f"EXPLICIT requires EDGE_WEIGHT_FORMAT in {path}")
        dist = _build_dist_from_explicit(weights_tokens, n, edge_weight_format)
        return TSPLIBInstance(name, n, ewt, edge_weight_format, None, dist)

    if len(coords) != n:
        raise ValueError(f"{path}: expected {n} coords, got {len(coords)}")
    coord_arr = np.array(coords, dtype=np.float64)
    dist = _build_dist_from_coords(coord_arr, ewt)
    return TSPLIBInstance(name, n, ewt, edge_weight_format, coord_arr, dist)


def load_solution_lengths(solutions_file: str) -> Dict[str, int]:
    sol: Dict[str, int] = {}
    if not solutions_file or not os.path.exists(solutions_file):
        return sol
    with open(solutions_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            nm = left.strip()
            m = re.search(r"(\d+)", right)
            if m:
                sol[nm] = int(m.group(1))
    return sol


# -----------------------------
# TSP utilities
# -----------------------------
def tour_edges(tour: List[int]) -> List[Tuple[int, int]]:
    n = len(tour)
    return [(tour[i], tour[(i + 1) % n]) for i in range(n)]


def tour_length(tour: List[int], dist: np.ndarray) -> int:
    total = 0
    n = len(tour)
    for i in range(n):
        total += int(dist[tour[i], tour[(i + 1) % n]])
    return total


def tour_penalty_sum(tour: List[int], pen: np.ndarray) -> int:
    total = 0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += int(pen[a, b])
    return total


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> List[int]:
    n = dist.shape[0]
    unvisited = np.ones(n, dtype=bool)
    tour = [start]
    unvisited[start] = False
    last = start

    for _ in range(n - 1):
        drow = dist[last].astype(np.int64)
        masked = np.where(unvisited, drow, np.iinfo(np.int64).max)
        nxt = int(np.argmin(masked))
        tour.append(nxt)
        unvisited[nxt] = False
        last = nxt

    return tour


# -----------------------------
# Candidate knowledge: kNN lists
# -----------------------------
def build_knn(dist: np.ndarray, cand_k: int) -> np.ndarray:
    """
    Return knn indices array shape (n, k') with k' = min(cand_k, n-1).
    Uses argpartition for speed.
    """
    n = dist.shape[0]
    k = int(min(max(cand_k, 1), n - 1))
    d = dist.astype(np.int64).copy()
    np.fill_diagonal(d, np.iinfo(np.int64).max)

    # argpartition gives k smallest (unordered)
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    return idx.astype(np.int32)


def build_candidate_matrix(knn: np.ndarray, n: int) -> np.ndarray:
    """
    Build symmetric boolean candidate edge matrix cand[a,b]=True if b in knn[a] or a in knn[b].
    """
    cand = np.zeros((n, n), dtype=np.bool_)
    rows = np.repeat(np.arange(n, dtype=np.int32), knn.shape[1])
    cols = knn.reshape(-1)
    cand[rows, cols] = True
    # symmetrize
    cand = np.logical_or(cand, cand.T)
    np.fill_diagonal(cand, False)
    return cand


# -----------------------------
# Candidate-guided 2-opt on augmented objective
# -----------------------------
def two_opt_aug_first_improvement_candidate_guided(
    tour: List[int],
    dist: np.ndarray,
    pen: np.ndarray,
    lam: float,
    knn: np.ndarray,
    fallback_window: int,
    time_limit_s: float,
    t0: float,
    ls_max_moves: int,
    verbose: bool = False,
    progress_interval: float = 1.0,
    log_fn=None,
) -> Tuple[List[int], int, float, int]:
    """
    Optimize augmented cost: Aug = length + lam * penalty_sum
    Using 2-opt first-improvement.
    Neighborhood is knowledge-guided via kNN-induced candidate positions + fallback window.
    """
    n = len(tour)
    true_len = tour_length(tour, dist)
    pen_sum = tour_penalty_sum(tour, pen)
    aug_cost = float(true_len) + lam * float(pen_sum)
    moves = 0

    last_print = t0

    def _log(msg: str):
        if log_fn is not None:
            log_fn(msg)
        if verbose:
            print(msg, flush=True)

    while (time.perf_counter() - t0) < time_limit_s and moves < ls_max_moves:
        improved = False

        now = time.perf_counter()
        if verbose and (now - last_print) >= progress_interval:
            elapsed = now - t0
            _log(f"    [ls] elapsed={elapsed:.2f}s  aug={aug_cost:.2f}  len={true_len}  moves={moves}")
            last_print = now

        # build position map for candidate-to-index conversion
        pos = np.empty(n, dtype=np.int32)
        for idx, node in enumerate(tour):
            pos[node] = idx

        for i in range(1, n - 1):
            a = tour[i - 1]
            b = tour[i]

            # Candidate ks induced by knn[a] U knn[b]
            ks_set: Set[int] = set()
            for v in knn[a]:
                ks_set.add(int(pos[int(v)]))
            for v in knn[b]:
                ks_set.add(int(pos[int(v)]))

            # Add a small deterministic fallback window to avoid over-narrow neighborhood
            fw = int(max(0, fallback_window))
            for k in range(i + 1, min(n, i + 1 + fw)):
                ks_set.add(k)

            # Filter valid ks: need k > i and k != i (already), also avoid k==0 (would make d=tour[0] but allowed)
            ks = [k for k in ks_set if k > i and k < n]
            if not ks:
                continue
            ks.sort()

            for k in ks:
                if (time.perf_counter() - t0) >= time_limit_s:
                    return tour, true_len, aug_cost, moves

                c = tour[k]
                d = tour[(k + 1) % n] if (k + 1) < n else tour[0]

                delta_dist = int(dist[a, c]) + int(dist[b, d]) - int(dist[a, b]) - int(dist[c, d])
                delta_pen = int(pen[a, c]) + int(pen[b, d]) - int(pen[a, b]) - int(pen[c, d])
                delta_aug = float(delta_dist) + lam * float(delta_pen)

                if delta_aug < 0.0:
                    tour[i : k + 1] = reversed(tour[i : k + 1])
                    true_len += delta_dist
                    aug_cost += delta_aug
                    moves += 1
                    improved = True

                    if verbose:
                        elapsed = time.perf_counter() - t0
                        _log(
                            f"    [ls-improve] elapsed={elapsed:.2f}s  d_aug={delta_aug:.2f}  "
                            f"aug={aug_cost:.2f}  len={true_len}  moves={moves}"
                        )
                    break

            if improved or moves >= ls_max_moves:
                break

        if not improved:
            break

    return tour, true_len, aug_cost, moves


# -----------------------------
# KGLS outer loop
# -----------------------------
def kgls_solve(
    dist: np.ndarray,
    time_limit_s: float,
    max_iters: int,
    alpha: float,
    beta: float,
    cand_k: int,
    fallback_window: int,
    ls_max_moves: int,
    seed_start: int = 0,
    verbose: bool = False,
    progress_interval: float = 1.0,
    log_fn=None,
) -> Tuple[List[int], int, int, float, int]:
    """
    KGLS for TSP (edge features) with candidate-edge knowledge.
    lambda = alpha * (best_len / n)

    Utility(edge) = dist(edge)/(1+pen(edge)) * (1 + beta*(1 - cand(edge)))
      - cand(edge)=1 if edge is in candidate set (kNN-based), else 0
      - beta>0 => non-candidate edges get higher utility => more likely penalized
    """
    n = dist.shape[0]
    pen = np.zeros((n, n), dtype=np.int32)

    knn = build_knn(dist, cand_k=cand_k)
    cand = build_candidate_matrix(knn, n)

    t0 = time.perf_counter()

    tour = nearest_neighbor_tour(dist, start=min(max(seed_start, 0), n - 1))
    cur_len = tour_length(tour, dist)
    best_tour = tour.copy()
    best_len = cur_len

    lam = max(1e-9, alpha * (float(cur_len) / float(n)))
    iters_done = 0
    total_ls_moves = 0

    last_outer_print = t0

    def _log(msg: str):
        if log_fn is not None:
            log_fn(msg)
        if verbose:
            print(msg, flush=True)

    while iters_done < max_iters and (time.perf_counter() - t0) < time_limit_s:
        now = time.perf_counter()
        if verbose and (now - last_outer_print) >= progress_interval:
            elapsed = now - t0
            _log(
                f"  [kgls] elapsed={elapsed:.2f}s  iter={iters_done}/{max_iters}  "
                f"best_len={best_len}  lambda={lam:.6f}  cand_k={cand_k}  beta={beta:.3f}"
            )
            last_outer_print = now

        # candidate-guided local search
        tour, cur_len, cur_aug, ls_moves = two_opt_aug_first_improvement_candidate_guided(
            tour=tour,
            dist=dist,
            pen=pen,
            lam=lam,
            knn=knn,
            fallback_window=fallback_window,
            time_limit_s=time_limit_s,
            t0=t0,
            ls_max_moves=ls_max_moves,
            verbose=verbose,
            progress_interval=progress_interval,
            log_fn=log_fn,
        )
        total_ls_moves += ls_moves

        if cur_len < best_len:
            best_len = cur_len
            best_tour = tour.copy()
            lam = max(1e-9, alpha * (float(best_len) / float(n)))
            if verbose:
                elapsed = time.perf_counter() - t0
                _log(f"  [best] elapsed={elapsed:.2f}s  best_len={best_len}  lambda={lam:.6f}")

        if (time.perf_counter() - t0) >= time_limit_s:
            break

        # knowledge-weighted utility on edges of current tour
        edges = tour_edges(tour)
        max_u = -1.0
        utils = []

        for (a, b) in edges:
            base = float(dist[a, b]) / (1.0 + float(pen[a, b]))
            # cand True => multiplier 1, non-cand => 1+beta
            mult = 1.0 + float(beta) * (0.0 if cand[a, b] else 1.0)
            u = base * mult
            utils.append(u)
            if u > max_u:
                max_u = u

        # penalize all edges with maximal utility (ties)
        cnt = 0
        for (a, b), u in zip(edges, utils):
            if abs(u - max_u) <= 1e-12:
                pen[a, b] += 1
                pen[b, a] += 1
                cnt += 1

        iters_done += 1

        if verbose:
            elapsed = time.perf_counter() - t0
            _log(
                f"  [penalize] elapsed={elapsed:.2f}s  iter={iters_done}/{max_iters}  "
                f"penalized_edges={cnt}  max_util={max_u:.6f}"
            )

    return best_tour, best_len, iters_done, lam, total_ls_moves


# -----------------------------
# Instance categorization / budgets
# -----------------------------
SIZE_BINS = [20, 50, 100, 500, 1000]


def classify_n(n: int) -> int:
    return min(SIZE_BINS, key=lambda b: abs(n - b))


def time_limit_for_class(cls: int) -> float:
    return 60.0 if cls == 1000 else 10.0


def iter_tsp_files(root: str) -> List[str]:
    paths = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".tsp"):
                paths.append(os.path.join(r, fn))
    paths.sort()
    return paths


class TeeLogger:
    def __init__(self, file_path: str):
        self.f = open(file_path, "a", encoding="utf-8")

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.f.write(f"[{ts}] {msg}\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsp_root", type=str, required=True, help="Root directory containing TSPLIB .tsp files")
    ap.add_argument("--solutions_file", type=str, default="", help="Path to solutions mapping file (name : optimum)")
    ap.add_argument("--out_csv", type=str, default="kgls_results.csv", help="Output CSV path")
    ap.add_argument("--max_dimension", type=int, default=1002, help="Only run instances with DIMENSION <= this")

    # KGLS controls
    ap.add_argument("--max_iters", type=int, default=1000, help="Max KGLS outer iterations (penalty updates)")
    ap.add_argument("--alpha", type=float, default=0.1, help="Lambda scaling: lambda = alpha * (best_len / n)")
    ap.add_argument("--beta", type=float, default=0.5, help="Knowledge weight: penalize non-candidate edges more (>=0)")
    ap.add_argument("--cand_k", type=int, default=20, help="k in kNN candidate edges")
    ap.add_argument("--fallback_window", type=int, default=30, help="Extra ks per i to avoid too narrow neighborhood")
    ap.add_argument("--ls_max_moves", type=int, default=200000, help="Max accepted 2-opt moves per LS call")

    # misc
    ap.add_argument("--seed_start", type=int, default=0, help="Start node for nearest neighbor init")
    ap.add_argument("--verbose", action="store_true", help="Print per-case and in-run progress logs")
    ap.add_argument("--progress_interval", type=float, default=1.0, help="Heartbeat interval (seconds)")
    ap.add_argument("--log_file", type=str, default="", help="Optional log file to append progress logs")
    args = ap.parse_args()

    logger = TeeLogger(args.log_file) if args.log_file else None

    def log(msg: str):
        if logger:
            logger.log(msg)

    sol_map = load_solution_lengths(args.solutions_file)
    tsp_files = iter_tsp_files(args.tsp_root)
    if not tsp_files:
        raise SystemExit(f"No .tsp files found under: {args.tsp_root}")

    total_files = len(tsp_files)

    out_fields = [
        "name", "path", "n", "class",
        "edge_weight_type", "edge_weight_format",
        "time_limit_s",
        "max_iters", "iters_done",
        "alpha", "beta", "cand_k", "fallback_window",
        "lambda_final",
        "ls_max_moves",
        "best_length", "opt_length", "gap_percent",
        "runtime_s",
        "total_ls_moves",
        "status",
    ]

    if args.verbose:
        print(f"[KGLS] Found {total_files} .tsp files under {args.tsp_root}", flush=True)
        if args.log_file:
            print(f"[KGLS] Logging enabled: {args.log_file}", flush=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=out_fields)
        writer.writeheader()

        done = 0
        skipped = 0
        errors = 0

        for idx, p in enumerate(tsp_files, start=1):
            base_name = os.path.splitext(os.path.basename(p))[0]
            status = "ok"

            try:
                inst = read_tsplib_tsp(p)
                n = inst.dimension

                if n > args.max_dimension:
                    skipped += 1
                    status = f"skipped: dimension {n} > max_dimension {args.max_dimension}"
                    if args.verbose:
                        msg = f"[{idx}/{total_files}] SKIP {inst.name} (n={n})  reason={status}"
                        print(msg, flush=True)
                        log(msg)
                    writer.writerow({
                        "name": inst.name, "path": p, "n": n, "class": "",
                        "edge_weight_type": inst.edge_weight_type,
                        "edge_weight_format": inst.edge_weight_format or "",
                        "time_limit_s": "",
                        "max_iters": args.max_iters, "iters_done": "",
                        "alpha": args.alpha, "beta": args.beta, "cand_k": args.cand_k, "fallback_window": args.fallback_window,
                        "lambda_final": "",
                        "ls_max_moves": args.ls_max_moves,
                        "best_length": "", "opt_length": "", "gap_percent": "",
                        "runtime_s": "", "total_ls_moves": "",
                        "status": status,
                    })
                    wf.flush()
                    continue

                cls = classify_n(n)
                tlim = time_limit_for_class(cls)

                if args.verbose:
                    msg = (f"[{idx}/{total_files}] RUN  {inst.name} "
                           f"(n={n}, class={cls}, tlim={tlim:.0f}s, ewt={inst.edge_weight_type}"
                           f"{('/' + inst.edge_weight_format) if inst.edge_weight_format else ''}, "
                           f"max_iters={args.max_iters}, alpha={args.alpha}, beta={args.beta}, cand_k={args.cand_k})")
                    print(msg, flush=True)
                    log(msg)

                t0 = time.perf_counter()
                best_tour, best_len, iters_done, lam_final, total_ls_moves = kgls_solve(
                    dist=inst.dist,
                    time_limit_s=tlim,
                    max_iters=args.max_iters,
                    alpha=args.alpha,
                    beta=args.beta,
                    cand_k=args.cand_k,
                    fallback_window=args.fallback_window,
                    ls_max_moves=args.ls_max_moves,
                    seed_start=args.seed_start,
                    verbose=args.verbose,
                    progress_interval=args.progress_interval,
                    log_fn=log,
                )
                runtime = time.perf_counter() - t0

                opt = sol_map.get(inst.name)
                gap = None
                if opt is not None and opt > 0:
                    gap = (best_len - opt) / opt * 100.0

                writer.writerow({
                    "name": inst.name,
                    "path": p,
                    "n": n,
                    "class": cls,
                    "edge_weight_type": inst.edge_weight_type,
                    "edge_weight_format": inst.edge_weight_format or "",
                    "time_limit_s": f"{tlim:.3f}",
                    "max_iters": args.max_iters,
                    "iters_done": iters_done,
                    "alpha": f"{args.alpha:.6f}",
                    "beta": f"{args.beta:.6f}",
                    "cand_k": args.cand_k,
                    "fallback_window": args.fallback_window,
                    "lambda_final": f"{lam_final:.9f}",
                    "ls_max_moves": args.ls_max_moves,
                    "best_length": best_len,
                    "opt_length": opt if opt is not None else "",
                    "gap_percent": f"{gap:.6f}" if gap is not None else "",
                    "runtime_s": f"{runtime:.6f}",
                    "total_ls_moves": total_ls_moves,
                    "status": status,
                })
                wf.flush()

                done += 1
                if args.verbose:
                    end_msg = (f"[{idx}/{total_files}] DONE {inst.name}  "
                               f"runtime={runtime:.3f}s  best_len={best_len}  "
                               f"iters={iters_done}/{args.max_iters}  ls_moves={total_ls_moves}"
                               + (f"  gap={gap:.4f}%" if gap is not None else "  gap=N/A"))
                    print(end_msg, flush=True)
                    log(end_msg)

            except Exception as e:
                errors += 1
                status = f"error: {type(e).__name__}: {e}"
                if args.verbose:
                    msg = f"[{idx}/{total_files}] ERROR {base_name}  {status}"
                    print(msg, flush=True)
                    log(msg)

                writer.writerow({
                    "name": base_name, "path": p, "n": "", "class": "",
                    "edge_weight_type": "", "edge_weight_format": "",
                    "time_limit_s": "",
                    "max_iters": args.max_iters, "iters_done": "",
                    "alpha": args.alpha, "beta": args.beta, "cand_k": args.cand_k, "fallback_window": args.fallback_window,
                    "lambda_final": "",
                    "ls_max_moves": args.ls_max_moves,
                    "best_length": "", "opt_length": "", "gap_percent": "",
                    "runtime_s": "", "total_ls_moves": "",
                    "status": status,
                })
                wf.flush()

        summary = f"[KGLS] Finished. done={done}, skipped={skipped}, errors={errors}. CSV saved to: {args.out_csv}"
        print(summary, flush=True)
        if logger:
            logger.log(summary)
            logger.close()


if __name__ == "__main__":
    main()
