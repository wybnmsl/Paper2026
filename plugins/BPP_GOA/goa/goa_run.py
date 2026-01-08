"""GOA solver for Bin Packing Problem (BPP).

This module supports two execution modes:
  - strict_online=True: only constructive online decisions (no relocation).
  - strict_online=False: hybrid/offline LNS (bin-empty, k-repack, ruin-recreate).

The implementation keeps your original search logic intact and only adjusts:
  - package-relative imports for the DASH plugin layout
  - naming cleanup (ACO -> GOA) for consistency with BPP_GOA plugin name
  - removal of legacy commented-out blocks
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .spec import GOASpec
from .goa_operators import (
    Solution,
    pack_ffd_or_bfd,
    try_empty_one_bin,
    try_k_bin_repack,
    ruin_and_recreate,
)
from ..utils.utils import lb_relaxation, gap_percent


@dataclass
class GOAResult:
    best_bins: int
    best_gap: float
    best_solution: Solution
    trace: list  # list of (t, best_gap, best_bins)
    info: Dict[str, Any]


def _init_tau(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64)


def _sample_choice(rng: np.random.Generator, tau: np.ndarray) -> int:
    t = np.asarray(tau, dtype=np.float64)
    t = np.clip(t, 1e-12, None)
    p = t / float(np.sum(t))
    return int(rng.choice(len(p), p=p))


def _evaporate_and_deposit(
    tau: np.ndarray,
    rho: float,
    deposit: float,
    choice_idx: int,
    best_bins: int,
    tau_min: float,
    tau_max: float,
):
    tau *= (1.0 - float(rho))
    add = float(deposit) / max(1.0, float(best_bins))
    tau[choice_idx] += add
    np.clip(tau, tau_min, tau_max, out=tau)


def _softmax_sample(rng: np.random.Generator, scores: np.ndarray, temp: float) -> int:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 1:
        return 0
    t = max(float(temp), 1e-9)
    z = (s - np.max(s)) / t
    p = np.exp(np.clip(z, -50, 50))
    ps = float(np.sum(p))
    if ps <= 0:
        return int(np.argmax(s))
    p /= ps
    return int(rng.choice(len(p), p=p))


def _construct_online_solution_bucketed(
    items: np.ndarray,
    capacity: int,
    *,
    priority_fn,
    gamma: float,
    temp: float,
    q0: float,
    open_bias: float,
    cand_k: int,
    rng: np.random.Generator,
) -> Solution:
    """
    STRICT ONLINE constructive policy:
      - process items in given arrival order (items array order)
      - decide bin for each arriving item; no later relocation

    Uses bucket scanning by remaining capacity (C is small, e.g., 100):
      buckets[r] keeps (possibly stale) bin ids that *once* had remaining r.
      We use lazy validation: check current remain[bin] == r.
    """
    items = np.asarray(items)
    C = int(capacity)

    # We assume integer weights for bucketed speed.
    # If floats appear, we fallback to rounding up.
    if not np.issubdtype(items.dtype, np.integer):
        w_int = np.ceil(items).astype(np.int64)
    else:
        w_int = items.astype(np.int64, copy=False)

    bins: list[list[int]] = []
    loads: list[int] = []
    remain: list[int] = []

    buckets: list[list[int]] = [[] for _ in range(C + 1)]  # 0..C

    def add_new_bin(item_idx: int, w: int):
        b = len(bins)
        bins.append([item_idx])
        ld = w
        loads.append(ld)
        r = C - ld
        remain.append(r)
        buckets[r].append(b)

    def place_in_bin(b: int, item_idx: int, w: int):
        bins[b].append(item_idx)
        loads[b] += w
        r = C - loads[b]
        remain[b] = r
        buckets[r].append(b)  # lazy; old bucket entries remain

    for i in range(int(w_int.shape[0])):
        w = int(w_int[i])
        if w <= 0:
            # ignore weird items
            continue
        if w > C:
            # infeasible item, still force new bin (will violate), but dataset should not have this
            add_new_bin(i, w)
            continue

        if len(bins) == 0:
            add_new_bin(i, w)
            continue

        # collect candidates: best-fit oriented (smallest remaining >= w)
        cand_bins: list[int] = []
        for r in range(w, C + 1):
            if len(cand_bins) >= cand_k:
                break
            bucket = buckets[r]
            if not bucket:
                continue
            # iterate from end for cache locality; include stale ids check
            # do not pop (keep structure stable)
            for b in reversed(bucket):
                if len(cand_bins) >= cand_k:
                    break
                if b < 0 or b >= len(bins):
                    continue
                if remain[b] != r:
                    continue
                # r >= w by loop
                cand_bins.append(b)
                if len(cand_bins) >= cand_k:
                    break

        # if no feasible, open new bin
        if not cand_bins:
            add_new_bin(i, w)
            continue

        cand_rem = np.array([remain[b] for b in cand_bins], dtype=np.float64)

        # best-fit score (tight fit is better): -(rem - w)
        bestfit_score = -(cand_rem - float(w))

        # optional learned priority
        if priority_fn is not None and gamma > 0.0:
            try:
                pr = np.asarray(priority_fn(float(w), cand_rem.copy()), dtype=np.float64)
                if pr.shape[0] != cand_rem.shape[0]:
                    pr = np.zeros_like(cand_rem)
            except Exception:
                pr = np.zeros_like(cand_rem)
        else:
            pr = np.zeros_like(cand_rem)

        scores = (1.0 - float(gamma)) * bestfit_score + float(gamma) * pr

        # optionally include "open new bin" as an action (rarely helpful for bin count, but allowed)
        include_new = float(open_bias) > -1e8
        if include_new:
            # new-bin score: just bias (usually negative)
            scores2 = np.concatenate([scores, np.array([float(open_bias)], dtype=np.float64)])
            # exploitation vs sampling
            if rng.random() < float(q0):
                j = int(np.argmax(scores2))
            else:
                j = _softmax_sample(rng, scores2, temp=float(temp))
            if j == len(cand_bins):
                add_new_bin(i, w)
            else:
                place_in_bin(cand_bins[j], i, w)
        else:
            if rng.random() < float(q0):
                j = int(np.argmax(scores))
            else:
                j = _softmax_sample(rng, scores, temp=float(temp))
            place_in_bin(cand_bins[j], i, w)

    # build Solution
    loads_arr = np.array(loads, dtype=np.float64)
    return Solution(bins=bins, loads=loads_arr)


def solve_bpp_goa(
    items: np.ndarray,
    capacity: int,
    priority_fn=None,
    spec: Optional[GOASpec] = None,
) -> GOAResult:
    """
    Two modes:
      - strict_online=True: only constructive online decisions (no relocation).
      - strict_online=False: hybrid/offline LNS (previous behavior).
    """
    if spec is None:
        spec = GOASpec()

    rng = np.random.default_rng(int(spec.seed))
    items = np.asarray(items)
    start = time.perf_counter()
    time_end = start + float(spec.time_limit_s)

    lb = lb_relaxation(items, int(capacity))

    # --- init best solution ---
    # For strict online, user should set init_method=online_*
    best_sol = pack_ffd_or_bfd(items, capacity, method=spec.init_method)
    best_bins = len(best_sol.bins)
    best_gap = gap_percent(best_bins, lb)

    trace = [(0.0, best_gap, best_bins)]

    if spec.strict_online:
        # GOA over discrete policy parameters
        tau_k = _init_tau(len(spec.cand_k_choices))
        tau_g = _init_tau(len(spec.gamma_choices))
        tau_t = _init_tau(len(spec.temp_choices))
        tau_q = _init_tau(len(spec.q0_choices))
        tau_o = _init_tau(len(spec.open_bias_choices))

        iters = int(spec.iters)
        ants = int(spec.ants)

        for it in range(iters):
            if time.perf_counter() >= time_end:
                break

            iter_best_bins = None
            iter_best_choice = None

            for a in range(ants):
                if time.perf_counter() >= time_end:
                    break

                i_k = _sample_choice(rng, tau_k)
                i_g = _sample_choice(rng, tau_g)
                i_t = _sample_choice(rng, tau_t)
                i_q = _sample_choice(rng, tau_q)
                i_o = _sample_choice(rng, tau_o)

                cand_k = int(spec.cand_k_choices[i_k])
                gamma = float(spec.gamma_choices[i_g])
                temp = float(spec.temp_choices[i_t])
                q0 = float(spec.q0_choices[i_q])
                open_bias = float(spec.open_bias_choices[i_o])

                sol = _construct_online_solution_bucketed(
                    items,
                    int(capacity),
                    priority_fn=priority_fn if spec.use_priority_in_construct else None,
                    gamma=gamma,
                    temp=temp,
                    q0=q0,
                    open_bias=open_bias,
                    cand_k=cand_k,
                    rng=rng,
                )

                bins_used = len(sol.bins)

                if bins_used < best_bins:
                    best_sol = sol
                    best_bins = bins_used
                    best_gap = gap_percent(best_bins, lb)
                    t = time.perf_counter() - start
                    trace.append((t, best_gap, best_bins))

                if iter_best_bins is None or bins_used < iter_best_bins:
                    iter_best_bins = bins_used
                    iter_best_choice = (i_k, i_g, i_t, i_q, i_o)

            # pheromone update with iteration-best policy
            if iter_best_bins is not None and iter_best_choice is not None:
                i_k, i_g, i_t, i_q, i_o = iter_best_choice
                _evaporate_and_deposit(tau_k, spec.rho, spec.deposit, i_k, iter_best_bins, spec.tau_min, spec.tau_max)
                _evaporate_and_deposit(tau_g, spec.rho, spec.deposit, i_g, iter_best_bins, spec.tau_min, spec.tau_max)
                _evaporate_and_deposit(tau_t, spec.rho, spec.deposit, i_t, iter_best_bins, spec.tau_min, spec.tau_max)
                _evaporate_and_deposit(tau_q, spec.rho, spec.deposit, i_q, iter_best_bins, spec.tau_min, spec.tau_max)
                _evaporate_and_deposit(tau_o, spec.rho, spec.deposit, i_o, iter_best_bins, spec.tau_min, spec.tau_max)

        info = {
            "mode": "strict_online",
            "lb": int(lb),
            "best_bins": int(best_bins),
            "best_gap": float(best_gap),
            "tau_cand_k": tau_k.tolist(),
            "tau_gamma": tau_g.tolist(),
            "tau_temp": tau_t.tolist(),
            "tau_q0": tau_q.tolist(),
            "tau_open_bias": tau_o.tolist(),
            "spec": spec.__dict__,
        }
        return GOAResult(best_bins=best_bins, best_gap=best_gap, best_solution=best_sol, trace=trace, info=info)

    # --------------------------------------------------------------------
    # Hybrid/offline mode (kept for compatibility; identical to your previous version)
    # --------------------------------------------------------------------
    tau_ruin = _init_tau(len(spec.ruin_frac_choices))
    tau_k = _init_tau(len(spec.repack_k_choices))
    tau_empty = _init_tau(len(spec.empty_trials_choices))

    for it in range(int(spec.iters)):
        if time.perf_counter() >= time_end:
            break

        iter_best_bins = None
        iter_best_choices = None

        for a in range(int(spec.ants)):
            if time.perf_counter() >= time_end:
                break

            idx_ruin = _sample_choice(rng, tau_ruin)
            idx_k = _sample_choice(rng, tau_k)
            idx_empty = _sample_choice(rng, tau_empty)

            ruin_frac = float(spec.ruin_frac_choices[idx_ruin])
            repack_k = int(spec.repack_k_choices[idx_k])
            empty_trials = int(spec.empty_trials_choices[idx_empty])

            # per-ant slice
            remain = time_end - time.perf_counter()
            slice_s = max(0.02, min(remain, float(spec.time_limit_s) * spec.inner_time_frac / max(1, spec.iters * spec.ants)))

            sol = pack_ffd_or_bfd(items, capacity, method=spec.init_method)
            cur_bins = len(sol.bins)

            _lns_improve_in_time(
                items, capacity, sol,
                priority_fn=priority_fn if spec.use_priority_in_construct else None,
                rng=rng,
                cand_k=int(getattr(spec, "cand_k", 32)) if hasattr(spec, "cand_k") else 32,
                slice_s=float(slice_s),
                repack_k=repack_k,
                empty_trials=empty_trials,
                ruin_frac=ruin_frac,
                spec=spec,
                time_end=time_end,
            )

            new_bins = len(sol.bins)

            if new_bins < best_bins:
                best_sol = sol
                best_bins = new_bins
                best_gap = gap_percent(best_bins, lb)
                t = time.perf_counter() - start
                trace.append((t, best_gap, best_bins))

            if iter_best_bins is None or new_bins < iter_best_bins:
                iter_best_bins = new_bins
                iter_best_choices = (idx_ruin, idx_k, idx_empty)

        if iter_best_bins is not None and iter_best_choices is not None:
            i_ruin, i_k, i_empty = iter_best_choices
            _evaporate_and_deposit(tau_ruin, spec.rho, spec.deposit, i_ruin, iter_best_bins, spec.tau_min, spec.tau_max)
            _evaporate_and_deposit(tau_k, spec.rho, spec.deposit, i_k, iter_best_bins, spec.tau_min, spec.tau_max)
            _evaporate_and_deposit(tau_empty, spec.rho, spec.deposit, i_empty, iter_best_bins, spec.tau_min, spec.tau_max)

    info = {
        "mode": "hybrid_offline",
        "lb": int(lb),
        "best_bins": int(best_bins),
        "best_gap": float(best_gap),
        "tau_ruin": tau_ruin.tolist(),
        "tau_k": tau_k.tolist(),
        "tau_empty": tau_empty.tolist(),
        "spec": spec.__dict__,
    }
    return GOAResult(best_bins=best_bins, best_gap=best_gap, best_solution=best_sol, trace=trace, info=info)


def _lns_improve_in_time(
    items: np.ndarray,
    capacity: int,
    sol: Solution,
    *,
    priority_fn,
    rng: np.random.Generator,
    cand_k: int,
    slice_s: float,
    repack_k: int,
    empty_trials: int,
    ruin_frac: float,
    spec: GOASpec,
    time_end: float,
):
    # identical structure as before; additionally respects global time_end
    t_end = min(time.perf_counter() + float(slice_s), time_end)

    best_local_bins = len(sol.bins)
    best_local = Solution(bins=[b[:] for b in sol.bins], loads=np.copy(sol.loads))

    p_empty = 0.55 if spec.enable_bin_empty else 0.0
    p_repack = 0.35 if spec.enable_k_repack else 0.0
    p_ruin = 0.10 if spec.enable_ruin_recreate else 0.0
    probs = np.array([p_empty, p_repack, p_ruin], dtype=np.float64)
    if probs.sum() <= 0:
        return
    probs = probs / probs.sum()

    while time.perf_counter() < t_end:
        if len(sol.bins) <= 1:
            break

        op = int(rng.choice(3, p=probs))
        if op == 0 and spec.enable_bin_empty:
            for _ in range(max(1, empty_trials // 8)):
                if time.perf_counter() >= t_end:
                    break
                ok = try_empty_one_bin(items, capacity, sol, priority_fn=priority_fn, rng=rng, cand_k=cand_k)
                if ok:
                    break

        elif op == 1 and spec.enable_k_repack:
            try_k_bin_repack(items, capacity, sol, k=repack_k, rng=rng)

        elif op == 2 and spec.enable_ruin_recreate:
            ruin_and_recreate(
                items, capacity, sol,
                ruin_frac=ruin_frac,
                priority_fn=priority_fn,
                rng=rng,
                cand_k=cand_k,
                init_method=spec.init_method,
            )

        new_bins = len(sol.bins)

        if new_bins < best_local_bins:
            best_local_bins = new_bins
            best_local = Solution(bins=[b[:] for b in sol.bins], loads=np.copy(sol.loads))
        elif new_bins == best_local_bins:
            if rng.random() > float(spec.accept_equal_prob):
                sol.bins = [b[:] for b in best_local.bins]
                sol.loads = np.copy(best_local.loads)
        else:
            if rng.random() > float(spec.worsen_accept_prob):
                sol.bins = [b[:] for b in best_local.bins]
                sol.loads = np.copy(best_local.loads)

    sol.bins = [b[:] for b in best_local.bins]
    sol.loads = np.copy(best_local.loads)
