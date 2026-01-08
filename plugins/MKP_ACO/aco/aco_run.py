# aco/aco_run.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..utils.readMKP import MKPInstance
from ..utils.utils import gap_from_profit, summarize_instance, profile_instance
from .spec import ACOSpec
from .aco_evol import run_aco_mkp


def _compute_tldr_from_profit_events(
    events: List[Tuple[float, int]],
    T_used: float,
    opt_profit: Optional[float],
    ub_profit: Optional[float],
    delta: float = 1e-4,
) -> Dict[str, float]:
    if T_used <= 1e-12:
        return {"tLDR_traj": 0.0, "ell0": 0.0, "J": 0.0, "n_events": float(len(events))}

    if not events:
        events = [(0.0, 0)]
    if events[0][0] > 1e-9:
        events = [(0.0, events[0][1])] + events

    ev = [(max(0.0, min(float(t), float(T_used))), int(p)) for t, p in events]
    ev.sort(key=lambda x: x[0])

    gaps = []
    times = []
    best_p = -10**18
    for t, p in ev:
        if p > best_p:
            best_p = p
        g = gap_from_profit(best_p, opt_profit=opt_profit, ub_profit=ub_profit)
        g = max(g, delta)
        times.append(t)
        gaps.append(g)

    ell = np.log(np.asarray(gaps, dtype=np.float64))
    ts = np.asarray(times, dtype=np.float64)

    if ts[-1] < T_used:
        ts = np.append(ts, T_used)
        ell = np.append(ell, ell[-1])

    area = 0.0
    for i in range(len(ts) - 1):
        dt = float(ts[i + 1] - ts[i])
        if dt > 0:
            area += float(ell[i] * dt)

    J = area / float(T_used)
    ell0 = float(ell[0])
    tldr = (2.0 / float(T_used)) * (ell0 - J)

    return {"tLDR_traj": float(tldr), "ell0": float(ell0), "J": float(J), "n_events": float(len(events))}


def solve_instance_with_spec(
    inst: MKPInstance,
    spec: ACOSpec,
    guide_module: Optional[Any] = None,
    return_meta: bool = True,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    spec = ACOSpec.from_json(spec.to_dict(), n_items=inst.n)

    best_x, best_profit, meta = run_aco_mkp(inst, spec, guide_module=guide_module)

    opt_profit = float(inst.best_known) if getattr(inst, "best_known", 0) and inst.best_known > 0 else None
    ub_profit = float(inst.lp_ub) if getattr(inst, "lp_ub", 0.0) and inst.lp_ub > 0 else meta.get("ub_proxy", None)

    gap = gap_from_profit(best_profit, opt_profit=opt_profit, ub_profit=ub_profit)

    tldr_fields = _compute_tldr_from_profit_events(
        events=meta.get("events", []),
        T_used=float(meta.get("time_used", spec.time_limit_s)),
        opt_profit=opt_profit,
        ub_profit=ub_profit,
    )

    meta2 = dict(meta)
    meta2.update({
        "best_known_profit": opt_profit,
        "lp_ub": ub_profit,
        "gap": float(gap),
        **tldr_fields,
        "spec": spec.to_dict(),
    })

    if return_meta:
        return best_x, best_profit, meta2
    return best_x, best_profit, {}
def solve_instance_with_spec_gap(
    inst: MKPInstance,
    spec: ACOSpec,
    guide_module: Optional[Any] = None,
    return_meta: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    DASH-style interface:
      returns (gap_fraction, meta)

    gap_fraction is in [0,1] based on best_known if available, otherwise LP/UB proxy.
    meta contains tLDR_traj, events, spec, and profiling vector (for PLR).
    """
    best_x, best_profit, meta = solve_instance_with_spec(inst, spec, guide_module=guide_module, return_meta=True)
    gap = float(meta.get("gap", 0.0))

    meta2 = dict(meta)
    meta2.update({
        "problem": "MKP",
        "instance": summarize_instance(inst),
        "best_x": best_x,  # optional large payload
        "best_profit": int(best_profit),
        "profile_phi": profile_instance(inst).tolist(),
    })
    if return_meta:
        return gap, meta2
    return gap, {}
