# aco/aco_operators.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional
import time
import numpy as np

from ..utils.readMKP import MKPInstance
from ..utils.utils import (
    objective_profit,
    total_weights,
    is_feasible,
    repair_drop,
    greedy_refill,
    local_search_1swap,
    normalize_solution,
)


def repair_solution(inst: MKPInstance, x: np.ndarray, repair: str, drop_rule: str, eta: Optional[np.ndarray], deadline: Optional[float] = None) -> np.ndarray:
    x = normalize_solution(x)
    if repair == "none":
        return x
    if is_feasible(inst, x):
        if repair == "drop_refill":
            return greedy_refill(inst, x, score=eta, deadline=deadline)
        return x

    x = repair_drop(inst, x, drop_rule=drop_rule, eta=eta)
    if repair == "drop_refill":
        x = greedy_refill(inst, x, score=eta, deadline=deadline)
    return x


def local_search_add_drop(inst: MKPInstance, x: np.ndarray, eta: np.ndarray, max_steps: int, deadline: Optional[float] = None) -> np.ndarray:
    x = greedy_refill(inst, x, score=eta, deadline=deadline)
    x = local_search_1swap(inst, x, score_add=eta, max_steps=max_steps, deadline=deadline)
    return x


def local_search_kflip(
    inst: MKPInstance,
    x: np.ndarray,
    eta: np.ndarray,
    max_steps: int = 300,
    pair_budget: int = 40,
    deadline: Optional[float] = None,
) -> np.ndarray:
    x = normalize_solution(x)
    if not is_feasible(inst, x):
        x = repair_drop(inst, x, drop_rule="min_ratio", eta=eta)

    cur_profit = objective_profit(inst, x)
    cur_w = total_weights(inst, x)

    sel = np.where(x == 1)[0].astype(np.int64)
    nosel = np.where(x == 0)[0].astype(np.int64)

    add_order = nosel[np.argsort(-eta[nosel])]
    bad_order = sel[np.argsort(eta[sel])] if sel.size > 0 else np.array([], dtype=np.int64)

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
            sel = np.where(x == 1)[0].astype(np.int64)
            bad_order = sel[np.argsort(eta[sel])] if sel.size > 0 else bad_order
            steps += 1
            continue

        improved = False
        worst = bad_order[: min(pair_budget, bad_order.size)]

        for j in worst:
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
                    improved = True
                    steps += 1
                break

        if improved:
            sel = np.where(x == 1)[0].astype(np.int64)
            bad_order = sel[np.argsort(eta[sel])] if sel.size > 0 else bad_order
            continue

        L = worst.size
        if L >= 2:
            for a in range(L - 1):
                if deadline is not None and time.perf_counter() >= deadline:
                    return x
                j = worst[a]
                wj = inst.weights[:, j]
                for b in range(a + 1, L):
                    if deadline is not None and time.perf_counter() >= deadline:
                        return x
                    k = worst[b]
                    new_w = cur_w - wj - inst.weights[:, k] + wi
                    if np.all(new_w <= inst.capacities):
                        new_profit = cur_profit - int(inst.profits[j]) - int(inst.profits[k]) + int(inst.profits[i])
                        if new_profit > cur_profit:
                            x[j] = 0
                            x[k] = 0
                            x[i] = 1
                            cur_w = new_w
                            cur_profit = new_profit
                            improved = True
                            steps += 1
                        break
                if improved:
                    break

        if improved:
            sel = np.where(x == 1)[0].astype(np.int64)
            bad_order = sel[np.argsort(eta[sel])] if sel.size > 0 else bad_order

    return x


def improve_solution(
    inst: MKPInstance,
    x: np.ndarray,
    eta: np.ndarray,
    ls: str,
    max_steps: int,
    deadline: Optional[float] = None,
) -> np.ndarray:
    if ls == "none" or max_steps <= 0:
        return normalize_solution(x)

    if ls == "1swap":
        return local_search_1swap(inst, x, score_add=eta, max_steps=max_steps, deadline=deadline)

    if ls == "add_drop":
        return local_search_add_drop(inst, x, eta=eta, max_steps=max_steps, deadline=deadline)

    if ls == "kflip":
        x = local_search_kflip(inst, x, eta=eta, max_steps=max_steps, pair_budget=40, deadline=deadline)
        x = local_search_1swap(inst, x, score_add=eta, max_steps=max_steps // 3, deadline=deadline)
        return x

    if ls == "hybrid":
        x = greedy_refill(inst, normalize_solution(x), score=eta, deadline=deadline)
        x = local_search_kflip(inst, x, eta=eta, max_steps=max_steps, pair_budget=40, deadline=deadline)
        x = local_search_1swap(inst, x, score_add=eta, max_steps=max_steps // 2, deadline=deadline)
        return x

    return normalize_solution(x)
