# zCVRP/utils/readCVRPLib.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .utils import euclidean_distance_matrix_int


@dataclass(frozen=True)
class CVRPInstance:
    name: str
    dimension: int
    capacity: int
    coords: np.ndarray          # (N,2), internal indexing, depot at 0
    demand: np.ndarray          # (N,), internal indexing, demand[0]=0
    dist: np.ndarray            # (N,N) int32
    depot_orig_id: int          # original node id in file (e.g., 1)
    internal_to_orig: List[int] # length N
    orig_to_internal: Dict[int, int]
    edge_weight_type: str = "EUC_2D"
    opt_cost: Optional[float] = None


@dataclass(frozen=True)
class CVRPSolution:
    name: str
    cost: Optional[float]
    routes_raw: List[List[int]]           # as in .sol file (integers)
    routes_orig_ids: Optional[List[List[int]]]  # mapped to VRP file node IDs
    inferred_offset: int                  # 0 or 1 typically (sol_id + offset => orig_id)


_SECTION_NODE_COORD = "NODE_COORD_SECTION"
_SECTION_DEMAND = "DEMAND_SECTION"
_SECTION_DEPOT = "DEPOT_SECTION"
_EOF = "EOF"


def _read_text(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_vrplib_instance(vrp_path: Union[str, Path]) -> CVRPInstance:
    """
    Parse a standard VRPLIB/CVRPLIB .vrp file (EUC_2D supported).
    Canonicalize to internal indexing: depot at 0, others follow by ascending orig node id.
    """
    text = _read_text(vrp_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    meta: Dict[str, str] = {}
    coords: Dict[int, Tuple[float, float]] = {}
    demand: Dict[int, int] = {}
    depot_ids: List[int] = []

    state = "header"

    for ln in lines:
        up = ln.upper()

        if up.startswith(_SECTION_NODE_COORD):
            state = "coords"
            continue
        if up.startswith(_SECTION_DEMAND):
            state = "demand"
            continue
        if up.startswith(_SECTION_DEPOT):
            state = "depot"
            continue
        if up.startswith(_EOF):
            break

        if state == "header":
            # Formats often like: KEY : value (with tabs/spaces)
            if ":" in ln:
                k, v = ln.split(":", 1)
                meta[k.strip().upper()] = v.strip().strip('"')
            else:
                # tolerate weird header lines
                pass

        elif state == "coords":
            parts = ln.split()
            if len(parts) >= 3:
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[nid] = (x, y)

        elif state == "demand":
            parts = ln.split()
            if len(parts) >= 2:
                nid = int(parts[0])
                dem = int(float(parts[1]))
                demand[nid] = dem

        elif state == "depot":
            # depot list ends at -1
            try:
                v = int(ln.split()[0])
            except Exception:
                continue
            if v == -1:
                state = "after_depot"
                continue
            depot_ids.append(v)

    name = meta.get("NAME", Path(vrp_path).stem).strip()
    dimension = int(meta.get("DIMENSION", str(len(coords))))
    capacity = int(meta.get("CAPACITY", "0"))
    ewt = meta.get("EDGE_WEIGHT_TYPE", "EUC_2D").strip().upper()

    if capacity <= 0:
        raise ValueError(f"{name}: invalid CAPACITY={capacity}")
    if len(coords) == 0:
        raise ValueError(f"{name}: no NODE_COORD_SECTION parsed")
    if len(demand) == 0:
        raise ValueError(f"{name}: no DEMAND_SECTION parsed")
    if len(coords) != dimension:
        # not fatal, but suspicious
        # allow if file has extra spaces/lines; still proceed
        pass

    if len(depot_ids) == 0:
        # Common default depot is 1
        depot_orig = 1
    else:
        depot_orig = depot_ids[0]

    # Canonical internal ordering: depot first, then others ascending by original node id
    orig_ids_sorted = sorted(coords.keys())
    if depot_orig not in coords:
        raise ValueError(f"{name}: depot id {depot_orig} not found in coords")

    non_depot = [oid for oid in orig_ids_sorted if oid != depot_orig]
    internal_to_orig = [depot_orig] + non_depot
    orig_to_internal = {oid: i for i, oid in enumerate(internal_to_orig)}

    coords_arr = np.array([coords[oid] for oid in internal_to_orig], dtype=np.float64)
    demand_arr = np.array([demand.get(oid, 0) for oid in internal_to_orig], dtype=np.int32)

    # enforce depot demand=0 if not
    demand_arr[0] = 0

    if ewt != "EUC_2D":
        raise NotImplementedError(f"{name}: EDGE_WEIGHT_TYPE={ewt} not supported yet (only EUC_2D)")

    dist = euclidean_distance_matrix_int(coords_arr)

    return CVRPInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        coords=coords_arr,
        demand=demand_arr,
        dist=dist,
        depot_orig_id=depot_orig,
        internal_to_orig=internal_to_orig,
        orig_to_internal=orig_to_internal,
        edge_weight_type=ewt,
        opt_cost=None,
    )


def load_vrplib_solution(sol_path: Union[str, Path], instance: Optional[CVRPInstance] = None) -> CVRPSolution:
    """
    Parse a VRPLIB/CVRPLIB .sol file:
      - Route lines like: "Route #k: 31 46 35"
      - Cost line like: "Cost 27591"
    If instance is provided, we attempt to map route node ids to .vrp original node ids,
    inferring whether solution uses customer indexing 1..(N-1) excluding depot (offset=1).
    """
    text = _read_text(sol_path)
    name = Path(sol_path).stem

    routes_raw: List[List[int]] = []
    cost: Optional[float] = None

    # cost
    m = re.search(r"\bCOST\b\s*[: ]\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            cost = float(m.group(1))
        except Exception:
            cost = None

    # routes
    for ln in text.splitlines():
        if "ROUTE" in ln.upper() and ":" in ln:
            _, rhs = ln.split(":", 1)
            nodes = []
            for tok in rhs.strip().split():
                try:
                    nodes.append(int(tok))
                except Exception:
                    pass
            if nodes:
                routes_raw.append(nodes)

    inferred_offset = 0
    routes_orig: Optional[List[List[int]]] = None

    if instance is not None and routes_raw:
        # infer whether solution indices should be shifted by +1 to match VRP node IDs
        depot_orig = instance.depot_orig_id
        dim = instance.dimension
        all_orig_ids = set(instance.internal_to_orig)

        flat_raw = [v for r in routes_raw for v in r]
        if flat_raw:
            def score_offset(off: int) -> Tuple[int, int, int]:
                # lower is better
                mapped = [v + off for v in flat_raw]
                # invalid count
                invalid = sum(1 for x in mapped if x not in all_orig_ids)
                depot_hits = sum(1 for x in mapped if x == depot_orig)
                uniq = len(set(mapped))
                # prefer mapping that does NOT include depot and covers many unique customers
                # expected unique customers ~= dim-1
                uniq_gap = abs((dim - 1) - uniq)
                return (invalid, depot_hits, uniq_gap)

            cand = [(score_offset(0), 0), (score_offset(1), 1)]
            cand.sort(key=lambda x: x[0])
            inferred_offset = cand[0][1]

            routes_orig = [[v + inferred_offset for v in r] for r in routes_raw]

    return CVRPSolution(
        name=name,
        cost=cost,
        routes_raw=routes_raw,
        routes_orig_ids=routes_orig,
        inferred_offset=inferred_offset,
    )


def load_instances(
    data_root: Union[str, Path],
    case_names: Sequence[str],
    require_sol: bool = False,
) -> List[CVRPInstance]:
    """
    Load a list of instances from data_root:
      - expects <name>.vrp exists
      - if <name>.sol exists, read best known cost into instance.opt_cost
    """
    root = Path(data_root)
    out: List[CVRPInstance] = []
    for name in case_names:
        vrp_path = root / f"{name}.vrp"
        sol_path = root / f"{name}.sol"

        if not vrp_path.exists():
            raise FileNotFoundError(f"Missing VRP file: {vrp_path}")

        inst = load_vrplib_instance(vrp_path)

        if sol_path.exists():
            sol = load_vrplib_solution(sol_path, instance=inst)
            inst = CVRPInstance(
                **{**inst.__dict__, "opt_cost": sol.cost}  # type: ignore[arg-type]
            )
        else:
            if require_sol:
                raise FileNotFoundError(f"Missing SOL file: {sol_path}")

        out.append(inst)
    return out
