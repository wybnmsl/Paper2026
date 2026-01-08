#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSPLIB Local Search (LS) baseline for TSP with progress logging.

Features:
- Parse multiple TSPLIB .tsp formats (NODE_COORD_SECTION / EDGE_WEIGHT_SECTION).
- Support EDGE_WEIGHT_TYPE: EUC_2D, CEIL_2D, ATT, GEO, EXPLICIT
- Support EDGE_WEIGHT_FORMAT (EXPLICIT): FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW, LOWER_DIAG_ROW
- Nearest-Neighbor init + 2-opt first-improvement local search
- Max accepted improving moves: 1000
- Time limit: 10s for categories {20,50,100,500}, 60s for category {1000}
- Classify instances to nearest of {20,50,100,500,1000}
- Export CSV results
- Print per-case progress + optional periodic in-run heartbeat
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    coords: Optional[np.ndarray]          # shape (n,2) float64, if coordinate-based
    dist: np.ndarray                      # shape (n,n) int32 distance matrix


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
            dist = (d + 0.5).astype(np.int32)
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
        mat = np.array(weights[:needed], dtype=np.int32).reshape((n, n))
        return mat

    if fmt == "UPPER_ROW":
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = weights[idx]
                dist[j, i] = weights[idx]
                idx += 1
        return dist

    if fmt == "LOWER_ROW":
        for i in range(n):
            for j in range(0, i):
                dist[i, j] = weights[idx]
                dist[j, i] = weights[idx]
                idx += 1
        return dist

    if fmt == "UPPER_DIAG_ROW":
        for i in range(n):
            for j in range(i, n):
                dist[i, j] = weights[idx]
                dist[j, i] = weights[idx]
                idx += 1
        np.fill_diagonal(dist, 0)
        return dist

    if fmt == "LOWER_DIAG_ROW":
        for i in range(n):
            for j in range(0, i + 1):
                dist[i, j] = weights[idx]
                dist[j, i] = weights[idx]
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
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
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
        return TSPLIBInstance(
            name=name,
            dimension=n,
            edge_weight_type=ewt,
            edge_weight_format=edge_weight_format,
            coords=None,
            dist=dist,
        )

    if len(coords) != n:
        raise ValueError(f"{path}: expected {n} coords, got {len(coords)}")
    coord_arr = np.array(coords, dtype=np.float64)
    dist = _build_dist_from_coords(coord_arr, ewt)
    return TSPLIBInstance(
        name=name,
        dimension=n,
        edge_weight_type=ewt,
        edge_weight_format=edge_weight_format,
        coords=coord_arr,
        dist=dist,
    )


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
            name = left.strip()
            right = right.strip()
            m = re.search(r"(\d+)", right)
            if m:
                sol[name] = int(m.group(1))
    return sol


# -----------------------------
# LS: Nearest Neighbor + 2-opt
# -----------------------------
def tour_length(tour: List[int], dist: np.ndarray) -> int:
    n = len(tour)
    total = 0
    for i in range(n):
        total += int(dist[tour[i], tour[(i + 1) % n]])
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


def two_opt_first_improvement(
    tour: List[int],
    dist: np.ndarray,
    max_moves: int,
    time_limit_s: float,
    t0: float,
    verbose: bool = False,
    progress_interval: float = 1.0,
    log_fn=None,
) -> Tuple[List[int], int, int]:
    """
    2-opt first-improvement with optional heartbeat logs.
    """
    n = len(tour)
    best_len = tour_length(tour, dist)
    moves = 0

    last_print = t0

    def _log(msg: str):
        if log_fn is not None:
            log_fn(msg)
        if verbose:
            print(msg, flush=True)

    while moves < max_moves and (time.perf_counter() - t0) < time_limit_s:
        improved = False

        # heartbeat
        now = time.perf_counter()
        if verbose and (now - last_print) >= progress_interval:
            elapsed = now - t0
            _log(f"    [progress] elapsed={elapsed:.2f}s  best_len={best_len}  moves={moves}")
            last_print = now

        for i in range(1, n - 1):
            a = tour[i - 1]
            b = tour[i]

            for k in range(i + 1, n):
                now = time.perf_counter()
                if (now - t0) >= time_limit_s:
                    return tour, best_len, moves

                c = tour[k]
                d = tour[(k + 1) % n] if (k + 1) < n else tour[0]

                delta = int(dist[a, c]) + int(dist[b, d]) - int(dist[a, b]) - int(dist[c, d])
                if delta < 0:
                    tour[i : k + 1] = reversed(tour[i : k + 1])
                    best_len += delta
                    moves += 1
                    improved = True

                    if verbose:
                        elapsed = time.perf_counter() - t0
                        _log(f"    [improve] elapsed={elapsed:.2f}s  delta={delta}  best_len={best_len}  moves={moves}")

                    break
            if improved or moves >= max_moves:
                break

        if not improved:
            break

    return tour, best_len, moves


# -----------------------------
# Instance categorization
# -----------------------------
SIZE_BINS = [20, 50, 100, 500, 1000]


def classify_n(n: int) -> int:
    return min(SIZE_BINS, key=lambda b: abs(n - b))


def time_limit_for_class(cls: int) -> float:
    return 60.0 if cls == 1000 else 10.0


# -----------------------------
# Runner
# -----------------------------
def iter_tsp_files(root: str) -> List[str]:
    paths = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".tsp"):
                paths.append(os.path.join(r, fn))
    paths.sort()
    return paths


class TeeLogger:
    """Optional logging to both stdout and a file."""
    def __init__(self, file_path: str):
        self.file_path = file_path
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsp_root", type=str, required=True, help="Root directory containing TSPLIB .tsp files")
    ap.add_argument("--solutions_file", type=str, default="", help="Path to solutions mapping file (name : optimum)")
    ap.add_argument("--out_csv", type=str, default="ls_results.csv", help="Output CSV path")
    ap.add_argument("--max_dimension", type=int, default=1002, help="Only run instances with DIMENSION <= this")
    ap.add_argument("--max_moves", type=int, default=1000, help="Max accepted improving 2-opt moves")
    ap.add_argument("--seed_start", type=int, default=0, help="Start node for nearest neighbor init")
    ap.add_argument("--verbose", action="store_true", help="Print per-case and in-run progress logs")
    ap.add_argument("--progress_interval", type=float, default=1.0, help="Heartbeat interval (seconds) in 2-opt")
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

    # We can't know DIMENSION without parsing, so total here is file count;
    # We will still print "skipped" when dimension > max_dimension after parsing.
    total_files = len(tsp_files)

    out_fields = [
        "name", "path", "n", "class",
        "edge_weight_type", "edge_weight_format",
        "time_limit_s", "max_moves",
        "best_length", "opt_length", "gap_percent",
        "runtime_s", "moves_accepted",
        "status",
    ]

    if args.verbose:
        print(f"[LS] Found {total_files} .tsp files under {args.tsp_root}", flush=True)
        if args.log_file:
            print(f"[LS] Logging enabled: {args.log_file}", flush=True)

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
                        "name": inst.name,
                        "path": p,
                        "n": n,
                        "class": "",
                        "edge_weight_type": inst.edge_weight_type,
                        "edge_weight_format": inst.edge_weight_format or "",
                        "time_limit_s": "",
                        "max_moves": args.max_moves,
                        "best_length": "",
                        "opt_length": "",
                        "gap_percent": "",
                        "runtime_s": "",
                        "moves_accepted": "",
                        "status": status,
                    })
                    wf.flush()
                    continue

                cls = classify_n(n)
                tlim = time_limit_for_class(cls)

                if args.verbose:
                    msg = (f"[{idx}/{total_files}] RUN  {inst.name} "
                           f"(n={n}, class={cls}, tlim={tlim:.0f}s, ewt={inst.edge_weight_type}"
                           f"{('/' + inst.edge_weight_format) if inst.edge_weight_format else ''})")
                    print(msg, flush=True)
                    log(msg)

                t0 = time.perf_counter()
                tour0 = nearest_neighbor_tour(inst.dist, start=min(max(args.seed_start, 0), n - 1))
                tour_best, best_len, moves = two_opt_first_improvement(
                    tour=tour0,
                    dist=inst.dist,
                    max_moves=args.max_moves,
                    time_limit_s=tlim,
                    t0=t0,
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
                    "max_moves": args.max_moves,
                    "best_length": best_len,
                    "opt_length": opt if opt is not None else "",
                    "gap_percent": f"{gap:.6f}" if gap is not None else "",
                    "runtime_s": f"{runtime:.6f}",
                    "moves_accepted": moves,
                    "status": status,
                })
                wf.flush()

                done += 1
                if args.verbose:
                    end_msg = (f"[{idx}/{total_files}] DONE {inst.name}  "
                               f"runtime={runtime:.3f}s  best_len={best_len}  "
                               f"moves={moves}" + (f"  gap={gap:.4f}%" if gap is not None else "  gap=N/A"))
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
                    "name": base_name,
                    "path": p,
                    "n": "",
                    "class": "",
                    "edge_weight_type": "",
                    "edge_weight_format": "",
                    "time_limit_s": "",
                    "max_moves": args.max_moves,
                    "best_length": "",
                    "opt_length": "",
                    "gap_percent": "",
                    "runtime_s": "",
                    "moves_accepted": "",
                    "status": status,
                })
                wf.flush()

        summary = f"[LS] Finished. done={done}, skipped={skipped}, errors={errors}. CSV saved to: {args.out_csv}"
        print(summary, flush=True)
        if logger:
            logger.log(summary)
            logger.close()


if __name__ == "__main__":
    main()
