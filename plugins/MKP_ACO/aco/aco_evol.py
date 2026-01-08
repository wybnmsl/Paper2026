# aco/aco_evol.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..utils.readMKP import MKPInstance
from ..utils.utils import (
    compute_eta,
    surrogate_upper_bound,
    objective_profit,
    is_feasible,
    greedy_construct,
)
from .spec import ACOSpec
from .aco_operators import repair_solution, improve_solution


@dataclass
class AntSolution:
    x: np.ndarray
    profit: int


def _candidate_order_from_eta(eta: np.ndarray) -> np.ndarray:
    return np.argsort(-eta)


def _update_multipliers_from_slack(inst: MKPInstance, x: np.ndarray) -> np.ndarray:
    w = inst.weights @ x.astype(np.int64)
    slack = (inst.capacities - w).astype(np.float64)
    lam = 1.0 / (slack + 1.0)
    lam = lam / np.maximum(inst.capacities.astype(np.float64), 1.0)
    lam = lam * (lam.size / (lam.sum() + 1e-12))
    return lam


def _eta_with_multipliers(inst: MKPInstance, base_eta: np.ndarray, lam: np.ndarray, power: float) -> np.ndarray:
    if power <= 0.0:
        return base_eta
    penalty = (lam[:, None] * inst.weights.astype(np.float64)).sum(axis=0)
    penalty = np.maximum(penalty, 1e-12)
    return base_eta / (penalty ** power)


def _roulette_choice(rng: np.random.Generator, items: np.ndarray, scores: np.ndarray) -> int:
    s = scores.sum()
    if not np.isfinite(s) or s <= 0:
        return int(items[int(rng.integers(0, items.size))])
    r = float(rng.random() * s)
    c = 0.0
    for idx, v in enumerate(scores):
        c += float(v)
        if c >= r:
            return int(items[idx])
    return int(items[-1])


def construct_ant_solution(
    inst: MKPInstance,
    tau: np.ndarray,
    eta: np.ndarray,
    spec: ACOSpec,
    rng: np.random.Generator,
    cand_order: np.ndarray,
    q0: float,
    guide_module: Optional[Any] = None,
    deadline: Optional[float] = None,
) -> np.ndarray:
    n = inst.n
    x = np.zeros(n, dtype=np.int8)
    rem = inst.capacities.astype(np.int64).copy()
    not_sel = np.ones(n, dtype=bool)

    eta_term = np.power(eta, spec.beta)
    item_n_used = np.zeros(n, dtype=np.int64)

    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            break

        cand: List[int] = []
        if spec.allow_infeasible_construct:
            if spec.candk > 0:
                for i in cand_order:
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    if not_sel[i]:
                        cand.append(int(i))
                        if len(cand) >= spec.candk:
                            break
            else:
                cand = np.where(not_sel)[0].astype(int).tolist()
        else:
            if spec.candk > 0:
                for i in cand_order:
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    if not_sel[i] and np.all(inst.weights[:, i] <= rem):
                        cand.append(int(i))
                        if len(cand) >= spec.candk:
                            break
            else:
                for i in cand_order:
                    if deadline is not None and time.perf_counter() >= deadline:
                        break
                    if not_sel[i] and np.all(inst.weights[:, i] <= rem):
                        cand.append(int(i))

        if not cand:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break

        cand_items = np.array(cand, dtype=np.int64)
        tau_term = np.power(tau[cand_items], spec.alpha)
        scores = tau_term * eta_term[cand_items]

        if guide_module is not None and hasattr(guide_module, "update_item_score"):
            try:
                scores2 = guide_module.update_item_score(
                    scores, x.copy(), rem.copy(), item_n_used.copy(), cand_items.copy()
                )
                scores2 = np.asarray(scores2, dtype=np.float64)
                if scores2.shape == scores.shape and np.all(np.isfinite(scores2)):
                    scores = scores2
            except Exception:
                pass

        if q0 > 0 and rng.random() < q0:
            chosen = int(cand_items[int(np.argmax(scores))])
        else:
            chosen = _roulette_choice(rng, cand_items, scores)

        x[chosen] = 1
        not_sel[chosen] = False
        item_n_used[chosen] += 1

        if not spec.allow_infeasible_construct:
            rem -= inst.weights[:, chosen]

        if spec.use_local_pheromone_update:
            tau[chosen] = (1.0 - spec.phi) * tau[chosen] + spec.phi * spec.tau0

    return x


def _rank_weights(mu: int) -> np.ndarray:
    denom = mu * (mu + 1) / 2.0
    return np.array([(mu - r) / denom for r in range(mu)], dtype=np.float64)


def _pheromone_global_update(tau: np.ndarray, sols: List[AntSolution], gbest: AntSolution, ub: float, spec: ACOSpec) -> None:
    tau *= (1.0 - spec.rho)
    eps = 1e-12

    if spec.deposit == "ibest":
        best = max(sols, key=lambda s: s.profit)
        tau[best.x.astype(bool)] += spec.elite_weight * (best.profit / max(ub, eps))
    elif spec.deposit == "gbest":
        tau[gbest.x.astype(bool)] += spec.elite_weight * (gbest.profit / max(ub, eps))
    else:
        mu = min(spec.rank_mu, len(sols))
        top = sorted(sols, key=lambda s: s.profit, reverse=True)[:mu]
        w = _rank_weights(mu)
        for r, s in enumerate(top):
            tau[s.x.astype(bool)] += spec.elite_weight * w[r] * (s.profit / max(ub, eps))
        if spec.gbest_weight > 0:
            tau[gbest.x.astype(bool)] += spec.gbest_weight * (gbest.profit / max(ub, eps))

    if spec.use_tau_bounds:
        tau_max = max(spec.tau_max, spec.tau0)
        tau_min = spec.tau_min_ratio * tau_max
        np.clip(tau, tau_min, tau_max, out=tau)


def run_aco_mkp(inst: MKPInstance, spec: ACOSpec, guide_module: Optional[Any] = None) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    t_start = time.perf_counter()
    deadline = t_start + spec.time_limit_s - spec.time_guard_margin_s
    rng = np.random.default_rng(int(spec.seed))

    base_eta = compute_eta(inst)

    ub = float(inst.lp_ub) if (getattr(inst, "lp_ub", 0.0) and inst.lp_ub > 0) else float(surrogate_upper_bound(inst))
    if not np.isfinite(ub) or ub <= 0:
        ub = float(np.sum(inst.profits))

    greedy_x = greedy_construct(inst, score=base_eta, rng=rng, randomize_topk=0)
    if not is_feasible(inst, greedy_x):
        greedy_x[:] = 0
    gbest = AntSolution(x=greedy_x.copy(), profit=int(objective_profit(inst, greedy_x)))

    events: List[Tuple[float, int]] = []
    if spec.trace:
        events.append((0.0, int(gbest.profit)))

    lam = (1.0 / inst.capacities.astype(np.float64))
    lam = lam * (lam.size / (lam.sum() + 1e-12))

    eta = base_eta.copy()
    if spec.use_multipliers:
        eta = _eta_with_multipliers(inst, base_eta, lam, spec.mult_eta_power)
        eta = np.maximum(eta, 1e-12)

    cand_order = _candidate_order_from_eta(eta)

    tau = np.full(inst.n, float(spec.tau0), dtype=np.float64)
    if gbest.profit > 0:
        tau[gbest.x.astype(bool)] = min(spec.tau_max, spec.tau0 * 1.2)

    iters = 0            # “代数”：完成一次 ants 批次就计 1
    stagn = 0
    ls_time = 0.0
    last_mult_update = 0

    while True:
        if time.perf_counter() >= deadline:
            break
        if iters >= spec.max_iters:
            break

        # ✅ 修正：只要进入并开始这一代，就先计数
        iters += 1

        if spec.use_multipliers and iters > 1 and (iters - 1 - last_mult_update) >= spec.mult_update_every:
            lam_new = _update_multipliers_from_slack(inst, gbest.x)
            lam = spec.mult_smooth * lam + (1.0 - spec.mult_smooth) * lam_new
            eta = _eta_with_multipliers(inst, base_eta, lam, spec.mult_eta_power)
            eta = np.maximum(eta, 1e-12)
            cand_order = _candidate_order_from_eta(eta)
            last_mult_update = (iters - 1)

        sols: List[AntSolution] = []
        n_greedy = int(round(spec.n_ants * spec.ants_greedy_frac))

        # --- ants batch (construct + repair + light LS) ---
        for a in range(spec.n_ants):
            if time.perf_counter() >= deadline:
                break
            q0 = 1.0 if a < n_greedy else spec.q0

            x = construct_ant_solution(
                inst, tau, eta, spec, rng, cand_order,
                q0=q0, guide_module=guide_module, deadline=deadline
            )

            x = repair_solution(
                inst, x,
                repair=("drop_refill" if spec.repair == "drop_refill" else spec.repair),
                drop_rule=spec.drop_rule,
                eta=eta,
                deadline=deadline,
            )

            if spec.ls_all_ants_steps > 0 and time.perf_counter() < deadline:
                t0 = time.perf_counter()
                x = improve_solution(inst, x, eta=eta, ls="1swap", max_steps=spec.ls_all_ants_steps, deadline=deadline)
                ls_time += (time.perf_counter() - t0)

            if not is_feasible(inst, x):
                x = repair_solution(inst, x, repair="drop_refill", drop_rule="min_ratio", eta=eta, deadline=deadline)

            sols.append(AntSolution(x=x, profit=int(objective_profit(inst, x))))

        if not sols:
            break

        # --- update gbest (cheap) ---
        sols.sort(key=lambda s: s.profit, reverse=True)
        ibest = sols[0]
        if ibest.profit > gbest.profit:
            gbest = AntSolution(x=ibest.x.copy(), profit=int(ibest.profit))
            stagn = 0
            if spec.trace:
                events.append((time.perf_counter() - t_start, int(gbest.profit)))
        else:
            stagn += 1

        # --- elite strong LS (skip if time left is small) ---
        time_left = deadline - time.perf_counter()
        if spec.ls != "none" and spec.ls_elite_k > 0 and spec.ls_elite_steps > 0 and time_left >= spec.skip_elite_if_time_left_s:
            k = min(spec.ls_elite_k, len(sols))
            for i in range(k):
                if time.perf_counter() >= deadline:
                    break
                t0 = time.perf_counter()
                x2 = improve_solution(inst, sols[i].x.copy(), eta=eta, ls=spec.ls, max_steps=spec.ls_elite_steps, deadline=deadline)
                ls_time += (time.perf_counter() - t0)
                if is_feasible(inst, x2):
                    p2 = int(objective_profit(inst, x2))
                    if p2 > sols[i].profit:
                        sols[i] = AntSolution(x=x2, profit=p2)

            sols.sort(key=lambda s: s.profit, reverse=True)
            ib2 = sols[0]
            if ib2.profit > gbest.profit:
                gbest = AntSolution(x=ib2.x.copy(), profit=int(ib2.profit))
                stagn = 0
                if spec.trace:
                    events.append((time.perf_counter() - t_start, int(gbest.profit)))

        # --- daemon (skip if time left is small) ---
        time_left = deadline - time.perf_counter()
        if spec.daemon_every > 0 and (iters % spec.daemon_every == 0) and spec.daemon_ls_steps > 0 and time_left >= spec.skip_daemon_if_time_left_s:
            t0 = time.perf_counter()
            x2 = improve_solution(inst, gbest.x.copy(), eta=eta, ls="kflip", max_steps=spec.daemon_ls_steps, deadline=deadline)
            ls_time += (time.perf_counter() - t0)
            if is_feasible(inst, x2):
                p2 = int(objective_profit(inst, x2))
                if p2 > gbest.profit:
                    gbest = AntSolution(x=x2.copy(), profit=p2)
                    stagn = 0
                    if spec.trace:
                        events.append((time.perf_counter() - t_start, int(gbest.profit)))

        # --- pheromone update: 尽量完成（这是“多代 ACO”的关键）---
        time_left = deadline - time.perf_counter()
        if time_left >= spec.skip_pheromone_if_time_left_s:
            _pheromone_global_update(tau, sols, gbest, ub=ub, spec=spec)

        # restart
        if stagn >= spec.stagnation_iters:
            tau0 = spec.restart_tau0 if spec.restart_tau0 is not None else spec.tau0
            tau[:] = float(tau0)
            if spec.restart_keep_best:
                tau[gbest.x.astype(bool)] = min(spec.tau_max, tau0 * 1.25)
            stagn = 0

    time_used = time.perf_counter() - t_start
    meta: Dict[str, Any] = {
        "time_used": float(time_used),
        "iters": int(iters),
        "best_profit": int(gbest.profit),
        "ub_proxy": float(ub),
        "ls_time": float(ls_time),
        "events": events,
    }
    return gbest.x.copy(), int(gbest.profit), meta
