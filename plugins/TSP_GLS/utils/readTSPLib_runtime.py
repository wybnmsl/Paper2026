# plugins/TSP_GLS/utils/readTSPLib_runtime.py

from __future__ import annotations

import os
import gzip
import tempfile
from typing import Iterable, List, Tuple, Dict, Optional
import math
import numpy as np

try:
    import tsplib95
except Exception:
    tsplib95 = None


def read_solutions_file(path: str) -> Dict[str, float]:
    """Parse a TSPLIB solutions file in common formats."""
    mapping: Dict[str, float] = {}
    if not path or not os.path.exists(path):
        return mapping

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" in s:
                name_part, val_part = s.split(":", 1)
            else:
                parts = s.split()
                if len(parts) < 2:
                    continue
                name_part, val_part = parts[0], parts[1]
            name = name_part.strip()
            try:
                val = float(val_part.strip())
            except Exception:
                continue
            mapping[name] = val
    return mapping


def _load_text(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_tsplib_problem_from_text(text: str):
    """Support both tsplib95.parse(text) and tsplib95.load(file)."""
    if tsplib95 is None:
        raise RuntimeError("tsplib95 is not installed, but TSPLIB parsing is required.")

    parse_fn = getattr(tsplib95, "parse", None)
    if callable(parse_fn):
        return parse_fn(text)

    with tempfile.NamedTemporaryFile("w+", suffix=".tsp", delete=False) as f:
        tmp_path = f.name
        f.write(text)
    try:
        prob = tsplib95.load(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return prob


def coords_to_array(prob) -> np.ndarray:
    """Extract (N,2) coordinates from tsplib95 problem.

    Priority:
      1) node_coords (NODE_COORD_SECTION)
      2) display_data (DISPLAY_DATA_SECTION)
      3) fallback: construct pseudo coordinates on a unit circle (for EXPLICIT instances)
    """
    coords = getattr(prob, "node_coords", None)
    if coords:
        xs: list[tuple[float, float]] = []
        for i in range(1, len(coords) + 1):
            vals = coords[i]
            x, y = float(vals[0]), float(vals[1])
            xs.append((x, y))
        arr = np.asarray(xs, dtype=np.float64)
        if arr.shape[0] > 0:
            return arr

    disp = getattr(prob, "display_data", None)
    if disp:
        xs: list[tuple[float, float]] = []
        for i in range(1, len(disp) + 1):
            vals = disp[i]
            x, y = float(vals[0]), float(vals[1])
            xs.append((x, y))
        arr = np.asarray(xs, dtype=np.float64)
        if arr.shape[0] > 0:
            return arr

    try:
        n = int(getattr(prob, "dimension", 0))
    except Exception:
        n = 0

    if n <= 0:
        try:
            nodes = list(prob.get_nodes())
            n = len(nodes)
        except Exception:
            n = 0

    if n <= 0:
        raise ValueError("TSPLIB problem has neither coords/display_data nor a valid dimension.")

    xs: list[tuple[float, float]] = []
    for i in range(n):
        theta = 2.0 * math.pi * i / max(n, 1)
        xs.append((math.cos(theta), math.sin(theta)))
    return np.asarray(xs, dtype=np.float64)


def build_distance_matrix(prob) -> np.ndarray:
    """Build an (N,N) distance matrix via tsplib95.get_weight(i,j)."""
    nodes = sorted(list(prob.get_nodes()))
    n = len(nodes)
    mat = np.zeros((n, n), dtype=np.float64)
    for ii, i in enumerate(nodes):
        for jj, j in enumerate(nodes):
            w = prob.get_weight(i, j)
            mat[ii, jj] = float(w)
    return mat


def _load_opt_tour_indices(root: str, name: str) -> Optional[np.ndarray]:
    """Load {name}.opt.tour[.gz] and parse TOUR_SECTION into a 0-based order."""
    candidates = [
        os.path.join(root, f"{name}.opt.tour"),
        os.path.join(root, f"{name}.opt.tour.gz"),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return None

    text = _load_text(path)
    lines = text.splitlines()
    in_section = False
    nodes: List[int] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("TOUR_SECTION"):
            in_section = True
            continue
        if up.startswith("EOF"):
            break
        if not in_section:
            continue
        toks = s.split()
        for tok in toks:
            try:
                v = int(tok)
            except Exception:
                continue
            if v == -1:
                in_section = False
                break
            nodes.append(v)
        if not in_section:
            break

    if not nodes:
        return None

    return np.asarray(nodes, dtype=np.int64) - 1


def tour_cost(distmat: np.ndarray, order: np.ndarray) -> float:
    """Compute the cost of a closed tour under distmat."""
    if order is None or len(order) < 2:
        raise ValueError("Empty tour.")
    n = len(order)
    s = 0.0
    for i in range(n):
        a = int(order[i])
        b = int(order[(i + 1) % n])
        s += float(distmat[a, b])
    return float(s)


def find_instances(root: str, names_filter: Optional[Iterable[str]] = None) -> List[str]:
    """Find all *.tsp / *.tsp.gz under root and filter by names_filter if provided."""
    all_names = set()
    for fn in os.listdir(root):
        low = fn.lower()
        if low.endswith(".tsp") or low.endswith(".tsp.gz"):
            base = fn.split(".", 1)[0]
            all_names.add(base)

    if names_filter is None:
        return sorted(all_names)

    out: List[str] = []
    for nm in names_filter:
        if nm in all_names:
            out.append(nm)
        else:
            nm_lower = nm.lower()
            for cand in all_names:
                if cand.lower() == nm_lower:
                    out.append(cand)
                    break

    seen = set()
    ordered = []
    for nm in out:
        if nm not in seen:
            ordered.append(nm)
            seen.add(nm)
    return ordered


def _load_tsplib_problem_from_root(root: str, name: str):
    """Find and parse {name}.tsp or {name}.tsp.gz under root."""
    candidates = [
        os.path.join(root, f"{name}.tsp"),
        os.path.join(root, f"{name}.tsp.gz"),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"Cannot find TSPLIB instance for name={name} under {root}")
    text = _load_text(path)
    return load_tsplib_problem_from_text(text)


def load_instances(
    root: str,
    solutions_file: Optional[str] = None,
    names: Optional[Iterable[str]] = None,
    min_nodes: int = 2,
    max_nodes: int = 999999,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Load a batch of TSPLIB instances and return (names, coords, distmats, opt_costs).

    Instances without an optimal cost (neither solutions file nor .opt.tour) are skipped.
    """
    if not os.path.isdir(root):
        raise NotADirectoryError(f"TSPLIB root not found: {root}")

    sol_map: Dict[str, float] = {}
    if solutions_file:
        sol_map = read_solutions_file(solutions_file)

    selected_names = find_instances(root, names)

    coords_list: List[np.ndarray] = []
    dist_list: List[np.ndarray] = []
    costs: List[float] = []
    kept_names: List[str] = []

    for nm in selected_names:
        try:
            prob = _load_tsplib_problem_from_root(root, nm)

            ew_type = getattr(prob, "edge_weight_type", None)
            if ew_type and str(ew_type).upper() == "SPECIAL":
                print(f"[TSPLIB] Skip {nm}: EDGE_WEIGHT_TYPE=SPECIAL (not supported in this loader).")
                continue

            coords = coords_to_array(prob)
            n = coords.shape[0]
            if n < min_nodes or n > max_nodes:
                print(f"[TSPLIB] Skip {nm}: node count {n} out of range [{min_nodes}, {max_nodes}].")
                continue

            distmat = build_distance_matrix(prob)

            opt_cost = None
            if nm in sol_map:
                opt_cost = float(sol_map[nm])

            if opt_cost is None:
                tour = _load_opt_tour_indices(root, nm)
                if tour is not None:
                    opt_cost = tour_cost(distmat, tour)

            if opt_cost is None:
                print(f"[TSPLIB] Skip {nm}: no optimal cost found (no solutions entry and no .opt.tour).")
                continue

            kept_names.append(nm)
            coords_list.append(coords)
            dist_list.append(distmat)
            costs.append(float(opt_cost))

        except Exception as ex:
            print(f"[TSPLIB] Error while loading {nm}: {ex}")
            continue

    if not kept_names:
        return [], [], [], np.asarray([], dtype=np.float64)

    opt_costs = np.asarray(costs, dtype=np.float64)
    return kept_names, coords_list, dist_list, opt_costs
