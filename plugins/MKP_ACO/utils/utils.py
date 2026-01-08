# utils/utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import time
import numpy as np
from .readMKP import MKPInstance


def objective_profit(inst: MKPInstance, x: np.ndarray) -> int:
    return int(np.dot(inst.profits, x.astype(np.int64)))


def total_weights(inst: MKPInstance, x: np.ndarray) -> np.ndarray:
    return inst.weights @ x.astype(np.int64)


def is_feasible(inst: MKPInstance, x: np.ndarray) -> bool:
    return bool(np.all(total_weights(inst, x) <= inst.capacities))


def normalize_solution(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return (x > 0).astype(np.int8)


def compute_eta(inst: MKPInstance, eps: float = 1e-12) -> np.ndarray:
    caps = inst.capacities.astype(np.float64)
    denom = (inst.weights.astype(np.float64) / caps[:, None]).sum(axis=0)
    eta = inst.profits.astype(np.float64) / (eps + denom)
    return np.maximum(eta, eps)


def compute_surrogate_weight(inst: MKPInstance, multipliers: Optional[np.ndarray] = None) -> np.ndarray:
    if multipliers is None:
        multipliers = 1.0 / inst.capacities.astype(np.float64)
    multipliers = np.asarray(multipliers, dtype=np.float64)
    return (multipliers[:, None] * inst.weights.astype(np.float64)).sum(axis=0)


def greedy_construct(
    inst: MKPInstance,
    score: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    randomize_topk: int = 0,
) -> np.ndarray:
    n = inst.n
    if score is None:
        score = compute_eta(inst)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score)

    x = np.zeros(n, dtype=np.int8)
    cur_w = np.zeros(inst.m, dtype=np.int64)

    if randomize_topk > 0 and rng is not None:
        remaining = order.tolist()
        while remaining:
            k = min(randomize_topk, len(remaining))
            pick = int(rng.integers(0, k))
            i = remaining.pop(pick)
            wi = inst.weights[:, i]
            if np.all(cur_w + wi <= inst.capacities):
                x[i] = 1
                cur_w += wi
        return x

    for i in order:
        wi = inst.weights[:, i]
        if np.all(cur_w + wi <= inst.capacities):
            x[i] = 1
            cur_w += wi
    return x


def repair_drop(
    inst: MKPInstance,
    x: np.ndarray,
    drop_rule: str = "min_ratio",
    eta: Optional[np.ndarray] = None,
) -> np.ndarray:
    x = normalize_solution(x)
    if is_feasible(inst, x):
        return x

    selected = np.where(x > 0)[0]
    if selected.size == 0:
        return x

    if eta is None:
        eta = compute_eta(inst)
    eta_sel = eta[selected].astype(np.float64)
    profits_sel = inst.profits[selected].astype(np.float64)

    if drop_rule == "min_ratio":
        drop_order = selected[np.argsort(eta_sel)]
    elif drop_rule == "min_profit":
        drop_order = selected[np.argsort(profits_sel)]
    elif drop_rule == "max_weight":
        w_hat = compute_surrogate_weight(inst)[selected]
        drop_order = selected[np.argsort(-w_hat)]
    else:
        raise ValueError(f"Unknown drop_rule: {drop_rule}")

    cur_w = total_weights(inst, x)
    for i in drop_order:
        if np.all(cur_w <= inst.capacities):
            break
        if x[i] == 1:
            x[i] = 0
            cur_w -= inst.weights[:, i]
    return x


def greedy_refill(
    inst: MKPInstance,
    x: np.ndarray,
    score: Optional[np.ndarray] = None,
    deadline: Optional[float] = None,
) -> np.ndarray:
    x = normalize_solution(x)
    if score is None:
        score = compute_eta(inst)
    score = np.asarray(score, dtype=np.float64)

    cur_w = total_weights(inst, x)
    candidates = np.where(x == 0)[0]
    order = candidates[np.argsort(-score[candidates])]

    for i in order:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        wi = inst.weights[:, i]
        if np.all(cur_w + wi <= inst.capacities):
            x[i] = 1
            cur_w += wi
    return x


def local_search_1swap(
    inst: MKPInstance,
    x: np.ndarray,
    score_add: Optional[np.ndarray] = None,
    max_steps: int = 200,
    deadline: Optional[float] = None,
) -> np.ndarray:
    x = normalize_solution(x)
    if score_add is None:
        score_add = compute_eta(inst)
    score_add = np.asarray(score_add, dtype=np.float64)

    cur_profit = objective_profit(inst, x)
    cur_w = total_weights(inst, x)

    sel = np.where(x == 1)[0].astype(np.int64)
    nosel = np.where(x == 0)[0].astype(np.int64)

    add_order = nosel[np.argsort(-score_add[nosel])]

    if sel.size > 0:
        w_free = compute_surrogate_weight(inst)[sel] + 1e-9
        loss = inst.profits[sel].astype(np.float64)
        rem_order = sel[np.argsort(loss / w_free)]
    else:
        rem_order = np.array([], dtype=np.int64)

    steps = 0
    for i in add_order:
        if steps >= max_steps:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break

        wi = inst.weights[:, i]

        if np.all(cur_w + wi <= inst.capacities):
            x[i] = 1
            cur_w += wi
            cur_profit += int(inst.profits[i])
            steps += 1
            continue

        for j in rem_order:
            if deadline is not None and time.perf_counter() >= deadline:
                return x
            new_w = cur_w - inst.weights[:, j] + wi
            if np.all(new_w <= inst.capacities):
                new_profit = cur_profit - int(inst.profits[j]) + int(inst.profits[i])
                if new_profit > cur_profit:
                    x[j] = 0
                    x[i] = 1
                    cur_w = new_w
                    cur_profit = new_profit
                    steps += 1
                break

    return x


def surrogate_upper_bound(inst: MKPInstance, multipliers: Optional[np.ndarray] = None) -> float:
    if multipliers is None:
        multipliers = 1.0 / inst.capacities.astype(np.float64)
    multipliers = np.asarray(multipliers, dtype=np.float64)

    p = inst.profits.astype(np.float64)
    w_hat = (multipliers[:, None] * inst.weights.astype(np.float64)).sum(axis=0)
    w_hat = np.maximum(w_hat, 1e-12)

    cap_hat = float(np.dot(multipliers, inst.capacities.astype(np.float64)))
    if not np.isfinite(cap_hat) or cap_hat <= 0:
        cap_hat = float(inst.m)

    ratio = p / w_hat
    order = np.argsort(-ratio)

    ub = 0.0
    used = 0.0
    for i in order:
        if used >= cap_hat:
            break
        if used + w_hat[i] <= cap_hat:
            ub += p[i]
            used += w_hat[i]
        else:
            frac = (cap_hat - used) / w_hat[i]
            if frac > 0:
                ub += p[i] * frac
            break
    return float(ub)


def gap_from_profit(best_profit: float, opt_profit: Optional[float] = None, ub_profit: Optional[float] = None) -> float:
    if opt_profit is not None and opt_profit > 0:
        return float(max(0.0, (opt_profit - best_profit) / opt_profit))
    if ub_profit is not None and ub_profit > 0:
        return float(max(0.0, (ub_profit - best_profit) / ub_profit))
    return 0.0 if best_profit > 0 else 1.0
# -----------------------------
# DASH PLR hooks (profiling)
# -----------------------------

def summarize_instance(inst: MKPInstance) -> dict:
    """
    Lightweight summary for logging and PLR bookkeeping.
    """
    p = inst.profits.astype(np.float64)
    w = inst.weights.astype(np.float64)
    b = inst.capacities.astype(np.float64)
    out = {
        "name": getattr(inst, "name", ""),
        "n": int(inst.n),
        "m": int(inst.m),
        "best_known": int(getattr(inst, "best_known", 0) or 0),
        "lp_ub": float(getattr(inst, "lp_ub", 0.0) or 0.0),
        "profit_mean": float(p.mean()) if p.size else 0.0,
        "profit_std": float(p.std()) if p.size else 0.0,
        "weight_mean": float(w.mean()) if w.size else 0.0,
        "weight_std": float(w.std()) if w.size else 0.0,
        "cap_mean": float(b.mean()) if b.size else 0.0,
        "cap_std": float(b.std()) if b.size else 0.0,
    }
    return out


def profile_instance(inst: MKPInstance) -> np.ndarray:
    """
    Instance profiling vector phi(x) for PLR.

    Features (float32):
      [n_items, m_constraints,
       profit_mean, profit_cv,
       weight_mean, weight_cv,
       cap_mean, cap_cv,
       tightness_mean, tightness_std,
       density_mean, density_std]
    where
      tightness_i = cap_i / sum_j w_{i,j}
      density_j = profit_j / sum_i w_{i,j}
    """
    n = float(inst.n)
    m = float(inst.m)
    p = inst.profits.astype(np.float64)
    W = inst.weights.astype(np.float64)   # (m,n)
    b = inst.capacities.astype(np.float64)

    p_mean = float(p.mean()) if p.size else 0.0
    p_std = float(p.std()) if p.size else 0.0
    p_cv = float(p_std / (p_mean + 1e-9)) if p.size else 0.0

    w_mean = float(W.mean()) if W.size else 0.0
    w_std = float(W.std()) if W.size else 0.0
    w_cv = float(w_std / (w_mean + 1e-9)) if W.size else 0.0

    b_mean = float(b.mean()) if b.size else 0.0
    b_std = float(b.std()) if b.size else 0.0
    b_cv = float(b_std / (b_mean + 1e-9)) if b.size else 0.0

    if W.size and b.size:
        sum_c = W.sum(axis=1)  # (m,)
        tight = b / (sum_c + 1e-12)
        tight_mean = float(np.mean(tight))
        tight_std = float(np.std(tight))
        sum_item = W.sum(axis=0)  # (n,)
        dens = p / (sum_item + 1e-12)
        dens_mean = float(np.mean(dens))
        dens_std = float(np.std(dens))
    else:
        tight_mean = tight_std = dens_mean = dens_std = 0.0

    return np.array(
        [n, m, p_mean, p_cv, w_mean, w_cv, b_mean, b_cv, tight_mean, tight_std, dens_mean, dens_std],
        dtype=np.float32
    )
