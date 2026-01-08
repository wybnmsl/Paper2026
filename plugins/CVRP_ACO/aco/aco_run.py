from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from .aco_engine import aco_solve
from .spec import ACOSpec, default_aco_spec, from_json
from ..utils.utils import gap_percent, summarize_instance, profile_instance


def _compute_tldr_traj_from_events(
    incumbent_events: List[Tuple[float, float]],
    opt_cost: float,
    T_used: float,
    delta: float = 1e-6,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Same definition as DASH/TSP side:
      gap%(t) = (best_cost(t)-opt_cost)/opt_cost*100
      ell(t)=log(max(gap%(t), delta))
      J(T)=(1/T)∫ ell(t) dt  (piecewise constant between events)
      tLDR_traj(T)=(2/T)*(ell(0)-J(T))
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
        g = gap_percent(cost, opt_cost)
        V = max(float(g), float(delta))
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


def solve_instance_with_spec(
    instance,
    spec: Union[ACOSpec, Dict[str, Any], None] = None,
    *,
    guidance_module=None,
    time_limit_s: Optional[float] = None,
    return_trace: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Standard DASH plugin interface:
      returns (gap%, meta)

    - instance: zCVRP.utils.readCVRPLib.CVRPInstance
    - spec: ACOSpec or dict (LLM-generated), None -> default_aco_spec
    - time_limit_s: optional override for spec.stopping.time_limit_s

    meta includes:
      - best_cost, opt_cost, gap
      - incumbent_events (t,cost) and tLDR_traj if opt_cost exists
      - profiling vector (for PLR): profile_phi
    """
    if spec is None:
        spec_obj = default_aco_spec()
    elif isinstance(spec, ACOSpec):
        spec_obj = spec
    else:
        spec_obj = from_json(spec)

    if time_limit_s is not None:
        spec_obj.stopping.time_limit_s = float(time_limit_s)

    t0 = time.time()
    routes, best_cost, events, solve_meta = aco_solve(instance, spec_obj, guidance_module=guidance_module, return_events=return_trace)
    T_used = float(time.time() - t0)

    opt = getattr(instance, "opt_cost", None)
    gap = gap_percent(best_cost, opt) if opt is not None else float("nan")

    tldr_traj, tldr_aux = (None, {"reason": "no_opt_cost"})
    if opt is not None and opt > 0 and return_trace:
        tldr_traj, tldr_aux = _compute_tldr_traj_from_events(events, float(opt), float(spec_obj.stopping.time_limit_s))

    meta: Dict[str, Any] = {
        "problem": "CVRP",
        "instance": summarize_instance(instance),
        "spec": asdict(spec_obj),
        "best_cost": float(best_cost),
        "opt_cost": float(opt) if opt is not None else None,
        "gap": float(gap),
        "routes": routes,  # optional, can be large
        "incumbent_events": events if return_trace else [],
        "tLDR_traj": tldr_traj,
        "tLDR_aux": tldr_aux,
        "T_used_wall": float(T_used),
        "solve_meta": solve_meta,
        "profile_phi": profile_instance(instance).tolist(),
    }
    return float(gap), meta
