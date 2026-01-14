# zCVRP/aco/aco_engine.py
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from utils.utils import routes_cost, edges_from_routes
except ImportError:  # pragma: no cover
    from ..utils.utils import routes_cost, edges_from_routes

from aco.spec import ACOSpec
from aco.aco_operators import (
    vnd_local_search,
    clarke_wright_savings_init,
    destroy_random,
    destroy_shaw,
    destroy_worst,
    destroy_route,
    repair_regret_k,
)


def _base_eta(dist: np.ndarray) -> np.ndarray:
    eps = 1e-9
    d = dist.astype(np.float64)
    eta = 1.0 / (d + eps)
    np.fill_diagonal(eta, 0.0)
    return eta


def _candidate_lists(dist: np.ndarray, k: int) -> List[np.ndarray]:
    N = dist.shape[0]
    kk = max(1, min(k, N - 1))
    cand = []
    for i in range(N):
        order = np.argsort(dist[i])
        order = order[order != i]
        cand.append(order[:kk].astype(np.int32))
    return cand


def _init_tau(dist: np.ndarray, spec: ACOSpec, best_cost_hint: Optional[float]) -> Tuple[np.ndarray, float, float, float]:
    N = dist.shape[0]
    rho = float(spec.pheromone.rho)
    if spec.pheromone.tau_init != "auto":
        tau0 = float(spec.pheromone.tau_init)
    else:
        if best_cost_hint is None or best_cost_hint <= 0:
            avg = float(np.mean(dist[dist > 0]))
            best_cost_hint = avg * (N - 1)
        tau0 = 1.0 / max(1e-9, rho * float(best_cost_hint))

    tau = np.full((N, N), tau0, dtype=np.float64)
    np.fill_diagonal(tau, 0.0)
    tau_max = tau0
    tau_min = tau_max * float(spec.pheromone.tau_min_factor)
    return tau, tau_min, tau_max, tau0


def _evaporate(tau: np.ndarray, rho: float) -> None:
    tau *= (1.0 - rho)
    np.maximum(tau, 0.0, out=tau)


def _clip_tau(tau: np.ndarray, tau_min: float, tau_max: float) -> None:
    np.clip(tau, tau_min, tau_max, out=tau)
    np.fill_diagonal(tau, 0.0)


def _deposit_edges(tau: np.ndarray, routes: List[List[int]], Q: float, cost: float, weight: float = 1.0) -> None:
    if cost <= 0:
        return
    delta = weight * (Q / float(cost))
    for (i, j) in edges_from_routes(routes, depot=0, symmetric=True):
        tau[i, j] += delta
        tau[j, i] += delta


def _weighted_choice(cands: List[int], weights: List[float], rng: random.Random) -> int:
    s = float(sum(weights))
    if s <= 0.0 or not math.isfinite(s):
        return rng.choice(cands)
    r = rng.random() * s
    acc = 0.0
    for j, w in zip(cands, weights):
        acc += float(w)
        if acc >= r:
            return j
    return cands[-1]


def _choose_next(
    cur: int,
    feasible: List[int],
    tau: np.ndarray,
    eta: np.ndarray,
    spec: ACOSpec,
    rng: random.Random,
    cand_list: Optional[np.ndarray] = None,
    remaining_cap: Optional[int] = None,
    demand: Optional[np.ndarray] = None,
) -> int:
    alpha = float(spec.construct.alpha)
    beta = float(spec.construct.beta)
    variant = spec.engine.variant
    q0 = float(spec.construct.q0)
    dem_gamma = float(spec.construct.demand_gamma)

    if cand_list is not None:
        feas2 = [int(j) for j in cand_list if j in feasible]
        if feas2:
            feasible = feas2

    if not feasible:
        return -1

    w = []
    for j in feasible:
        tij = float(tau[cur, j])
        eij = float(eta[cur, j])
        val = (tij ** alpha) * (eij ** beta)
        if demand is not None and remaining_cap is not None and dem_gamma != 0.0:
            dj = float(demand[j])
            ratio = max(1e-9, dj / max(1.0, float(remaining_cap)))
            val *= (ratio ** dem_gamma)
        w.append(val)

    if variant == "ACS" and rng.random() < q0:
        return feasible[int(np.argmax(w))]

    return _weighted_choice(feasible, w, rng)


def construct_routes_sequential(
    dist: np.ndarray,
    demand: np.ndarray,
    capacity: int,
    tau: np.ndarray,
    eta: np.ndarray,
    cand: List[np.ndarray],
    spec: ACOSpec,
    rng: random.Random,
) -> List[List[int]]:
    N = dist.shape[0]
    unserved = set(range(1, N))
    routes: List[List[int]] = []

    while unserved:
        cur = 0
        rem = capacity
        route: List[int] = []

        feasible0 = [int(j) for j in cand[0] if int(j) in unserved and int(demand[int(j)]) <= rem]
        if feasible0:
            start = int(rng.choice(feasible0[: min(len(feasible0), 10)]))
        else:
            feasible_any = [j for j in unserved if int(demand[j]) <= rem]
            start = int(rng.choice(feasible_any)) if feasible_any else int(next(iter(unserved)))

        if int(demand[start]) <= rem:
            route.append(start)
            unserved.remove(start)
            rem -= int(demand[start])
            cur = start

        while True:
            feas = [j for j in unserved if int(demand[j]) <= rem]
            if not feas:
                break
            nxt = _choose_next(
                cur=cur,
                feasible=feas,
                tau=tau,
                eta=eta,
                spec=spec,
                rng=rng,
                cand_list=cand[cur],
                remaining_cap=rem,
                demand=demand,
            )
            if nxt < 0 or nxt not in unserved:
                break
            route.append(nxt)
            unserved.remove(nxt)
            rem -= int(demand[nxt])
            cur = nxt

        routes.append(route if route else [int(next(iter(unserved)))])

        # safety
        routes = [r for r in routes if r]

    return routes


# ----------------------- ALNS loop -----------------------

def _accept_rrt(cost_new: int, cost_best: int, eps: float) -> bool:
    return cost_new <= int((1.0 + eps) * float(cost_best))


def _accept_sa(cost_new: int, cost_cur: int, T: float, rng: random.Random) -> bool:
    if cost_new <= cost_cur:
        return True
    if T <= 1e-12:
        return False
    prob = math.exp(-(float(cost_new - cost_cur) / T))
    return rng.random() < prob


def alns_improve(
    routes0: List[List[int]],
    dist: np.ndarray,
    demand: np.ndarray,
    capacity: int,
    spec: ACOSpec,
    rng: random.Random,
    cand: List[np.ndarray],
    start_time: float,
    end_time: float,
    best_cost_global: int,
    light_ls_neigh: Optional[List[str]] = None,
) -> Tuple[List[List[int]], int]:
    """
    Improve incumbent using ALNS until end_time (time budget).
    Uses destroy ops + regret-k repair + acceptance criterion + optional inner LS.
    """
    routes_cur = [r[:] for r in routes0 if r]
    cost_cur = int(routes_cost(routes_cur, dist, depot=0))

    routes_best = [r[:] for r in routes_cur]
    cost_best = int(cost_cur)

    # operator weights (light self-adaptation)
    w = {
        "random": float(spec.alns.w_random),
        "shaw": float(spec.alns.w_shaw),
        "worst": float(spec.alns.w_worst),
        "route": float(spec.alns.w_route),
    }

    T0 = float(spec.alns.sa_init_factor) * max(1.0, float(cost_cur))

    while time.time() < end_time:
        t = time.time()
        denom = max(1e-9, (end_time - start_time))
        prog = min(1.0, max(0.0, (t - start_time) / denom))

        # ruin size schedule
        ruin_frac = float(spec.alns.ruin_frac_min) + (float(spec.alns.ruin_frac_max) - float(spec.alns.ruin_frac_min)) * (0.3 + 0.7 * prog)
        n_customers = sum(len(r) for r in routes_cur)
        if n_customers <= 0:
            break
        k_remove = max(4, int(round(ruin_frac * n_customers)))
        k_remove = min(k_remove, n_customers)

        # choose destroy operator
        ops = list(w.keys())
        ws = [max(1e-9, w[k]) for k in ops]
        if rng.random() < 0.20:
            op = ops[int(np.argmax(ws))]
        else:
            op = rng.choices(ops, weights=ws, k=1)[0]

        # destroy
        if op == "random":
            r_partial, removed = destroy_random(routes_cur, k_remove, rng)
        elif op == "shaw":
            r_partial, removed = destroy_shaw(routes_cur, dist, demand, k_remove, rng)
        elif op == "worst":
            r_partial, removed = destroy_worst(routes_cur, dist, k_remove, rng)
        else:  # "route"
            r_partial, removed = destroy_route(routes_cur, k_remove, rng)

        # repair (regret-k)
        r_new = repair_regret_k(
            r_partial, removed, dist, demand, capacity,
            regret_k=int(spec.alns.regret_k),
            rng=rng,
        )

        # optional inner LS (light cap)
        if spec.alns.inner_ls:
            ls_cap = float(spec.alns.inner_ls_time_cap_s)
            ls_end = min(end_time, time.time() + ls_cap)
            neigh = light_ls_neigh if light_ls_neigh is not None else list(spec.local_search.neigh)
            r_new = vnd_local_search(
                r_new, dist, demand, capacity,
                neigh=neigh,
                max_moves=int(spec.alns.inner_ls_moves),
                end_time=ls_end,
                first_improve=True,
                rng=rng,
                cand=cand,
                granular_k=int(spec.local_search.granular_k),
            )

        cost_new = int(routes_cost(r_new, dist, depot=0))

        # update local best inside this ALNS call
        improved_global = cost_new < best_cost_global
        improved_local = cost_new < cost_best
        if improved_local:
            routes_best = [r[:] for r in r_new]
            cost_best = cost_new

        # acceptance
        if spec.alns.accept == "greedy":
            accepted = (cost_new <= cost_cur)
        elif spec.alns.accept == "rrt":
            eps = float(spec.alns.rrt_epsilon) + (float(spec.alns.rrt_final_epsilon) - float(spec.alns.rrt_epsilon)) * prog
            # NOTE: RRT acceptance should compare against the global best cost.
            accepted = _accept_rrt(cost_new, best_cost_global, eps)
        else:  # "sa"
            Tcur = T0 * (float(spec.alns.sa_alpha) ** (1000.0 * prog))
            accepted = _accept_sa(cost_new, cost_cur, Tcur, rng)

        if accepted:
            routes_cur = [r[:] for r in r_new]
            cost_cur = cost_new

        # very light weight adaptation
        if improved_global:
            w[op] *= 1.08
        elif improved_local:
            w[op] *= 1.03
        else:
            w[op] *= 0.995

        # normalize occasionally
        if rng.random() < 0.05:
            s = sum(w.values())
            if s > 0:
                for k in w:
                    w[k] = max(0.05, w[k] / s)

    return routes_best, int(cost_best)


# ----------------------- Main ACO solve -----------------------

def aco_solve(
    instance,
    spec: ACOSpec,
    guidance_module=None,
    return_events: bool = True,
) -> Tuple[List[List[int]], int, List[Tuple[float, int]], Dict[str, Any]]:
    dist = instance.dist
    demand = instance.demand
    capacity = int(instance.capacity)
    N = dist.shape[0]

    rng = random.Random(int(spec.engine.seed))
    start_time = time.time()
    T = float(spec.stopping.time_limit_s)
    end_time = start_time + T

    explore_end = start_time + float(spec.engine.explore_frac) * T
    exploit_end = start_time + (float(spec.engine.explore_frac) + float(spec.engine.exploit_frac)) * T
    intensify_last = float(spec.engine.intensify_last_s)

    cand = _candidate_lists(dist, int(spec.construct.candidate_k))
    eta0 = _base_eta(dist)
    eta = eta0.copy()

    # ---- init with savings ----
    best_routes: List[List[int]] = []
    best_cost: int = 10**18
    ls_time_used = 0.0
    alns_time_used = 0.0

    if spec.init.method == "savings":
        try:
            init_routes = clarke_wright_savings_init(dist, demand, capacity, rng)
            if spec.init.ls_after_init and spec.local_search.enabled:
                ls_end = min(end_time, time.time() + 0.9)
                t0 = time.time()
                init_routes = vnd_local_search(
                    init_routes, dist, demand, capacity,
                    neigh=list(spec.local_search.neigh),
                    max_moves=min(120000, int(spec.local_search.max_moves)),
                    end_time=ls_end,
                    first_improve=True,
                    rng=rng,
                    cand=cand,
                    granular_k=int(spec.local_search.granular_k),
                )
                ls_time_used += (time.time() - t0)
            init_cost = int(routes_cost(init_routes, dist, depot=0))
            best_routes = init_routes
            best_cost = init_cost
        except Exception:
            best_routes = []
            best_cost = 10**18

    # ---- tau init based on incumbent ----
    hint = float(best_cost) if best_cost < 10**18 else None
    tau, tau_min, tau_max, tau0 = _init_tau(dist, spec, best_cost_hint=hint)

    if best_routes and best_cost < 10**18:
        _deposit_edges(tau, best_routes, Q=float(spec.pheromone.q), cost=float(best_cost), weight=1.0)
        if spec.engine.variant == "MMAS":
            rho = float(spec.pheromone.rho)
            tau_max = 1.0 / max(1e-9, rho * float(best_cost))
            tau_min = tau_max * float(spec.pheromone.tau_min_factor)
            _clip_tau(tau, tau_min, tau_max)

    # events
    events: List[Tuple[float, int]] = []
    if return_events and best_cost < 10**18:
        events.append((0.0, int(best_cost)))

    # time budgets
    ls_budget_total = T * float(spec.local_search.time_share) if spec.local_search.enabled else 0.0
    alns_budget_total = T * float(spec.alns.time_share) if spec.alns.enabled else 0.0

    no_improve = 0
    it = 0

    # light LS neigh for exploration/ALNS inner
    light_ls_neigh = ["2opt_intra", "relocate1", "2opt_star"]

    while True:
        now = time.time()
        if now >= end_time:
            break
        if it >= int(spec.stopping.max_iters):
            break

        remaining = end_time - now
        intensify = remaining <= intensify_last

        # choose phase
        if intensify:
            phase = "intensify"
        elif now < explore_end:
            phase = "explore"
        elif now < exploit_end:
            phase = "exploit"
        else:
            phase = "exploit2"

        # guidance (optional)
        if (
            spec.guidance.type == "llm"
            and guidance_module is not None
            and best_routes
            and (it % int(spec.guidance.call_interval_iters) == 0)
        ):
            try:
                if hasattr(guidance_module, "update_eta_matrix"):
                    eta = np.asarray(guidance_module.update_eta_matrix(eta, best_routes, None), dtype=np.float64)
                elif hasattr(guidance_module, "update_score_matrix"):
                    eta = np.asarray(guidance_module.update_score_matrix(eta, best_routes, None), dtype=np.float64)
                eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
                eta = np.maximum(eta, 0.0)
                np.fill_diagonal(eta, 0.0)
            except Exception:
                eta = eta0.copy()

        # ---------------- ACO construct ----------------
        if phase == "explore":
            n_ants = int(spec.construct.n_ants)
            ls_apply = False
        elif phase == "exploit":
            n_ants = max(16, int(spec.construct.n_ants * 0.6))
            ls_apply = True
        elif phase == "exploit2":
            n_ants = max(12, int(spec.construct.n_ants * 0.5))
            ls_apply = True
        else:
            # intensify: few ants or none
            n_ants = max(6, int(spec.construct.n_ants * 0.3))
            ls_apply = False

        candidates: List[Tuple[int, List[List[int]]]] = []
        for _a in range(n_ants):
            if time.time() >= end_time:
                break
            routes = construct_routes_sequential(dist, demand, capacity, tau, eta, cand, spec, rng)
            c = int(routes_cost(routes, dist, depot=0))
            candidates.append((c, routes))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        best_it_cost, best_it_routes = candidates[0]

        # ---------------- LS only on best/top_k (time-budgeted) ----------------
        if spec.local_search.enabled and ls_apply:
            ls_rem = max(0.0, ls_budget_total - ls_time_used)
            if ls_rem > 1e-6:
                # per-iter cap
                per_cap = min(ls_rem, 0.18 if phase.startswith("exploit") else 0.10)
                ls_end = min(end_time, time.time() + per_cap)
                t0 = time.time()

                if spec.local_search.apply_to == "all":
                    selected = candidates
                elif spec.local_search.apply_to == "top_k":
                    selected = candidates[: max(1, int(spec.local_search.top_k))]
                else:
                    selected = candidates[:1]

                improved_best = (best_it_cost, best_it_routes)
                for (c0, r0) in selected:
                    if time.time() >= ls_end:
                        break
                    neigh = light_ls_neigh if phase == "explore" else list(spec.local_search.neigh)
                    r1 = vnd_local_search(
                        r0, dist, demand, capacity,
                        neigh=neigh,
                        max_moves=int(spec.local_search.max_moves),
                        end_time=ls_end,
                        first_improve=bool(spec.local_search.first_improve),
                        rng=rng,
                        cand=cand,
                        granular_k=int(spec.local_search.granular_k),
                    )
                    c1 = int(routes_cost(r1, dist, depot=0))
                    if c1 < improved_best[0]:
                        improved_best = (c1, r1)

                ls_time_used += (time.time() - t0)
                best_it_cost, best_it_routes = improved_best

        # ---------------- update incumbent ----------------
        if best_it_cost < best_cost:
            best_cost = int(best_it_cost)
            best_routes = best_it_routes
            no_improve = 0
            if return_events:
                events.append((float(time.time() - start_time), int(best_cost)))
        else:
            no_improve += 1

        # ---------------- pheromone update ----------------
        rho = float(spec.pheromone.rho)
        _evaporate(tau, rho)

        dep_mode = spec.pheromone.deposit
        Q = float(spec.pheromone.q)

        if dep_mode == "ibest":
            # rank deposit (top mu) if enabled
            mu = max(1, int(spec.pheromone.rank_mu))
            decay = float(spec.pheromone.rank_weight)
            for rank, (c, r) in enumerate(candidates[:mu]):
                w = (decay ** rank) if decay > 0 else 1.0
                _deposit_edges(tau, r, Q=Q, cost=float(c), weight=w)
        elif dep_mode == "gbest":
            if best_routes and best_cost < 10**18:
                _deposit_edges(tau, best_routes, Q=Q, cost=float(best_cost), weight=1.0)
        else:
            if best_routes and best_cost < 10**18:
                for _ in range(max(1, int(spec.pheromone.elite_m))):
                    _deposit_edges(tau, best_routes, Q=Q, cost=float(best_cost), weight=1.0)

        if spec.engine.variant == "MMAS" and best_cost < 10**18:
            tau_max = 1.0 / max(1e-9, rho * float(best_cost))
            tau_min = tau_max * float(spec.pheromone.tau_min_factor)
            _clip_tau(tau, tau_min, tau_max)

        # ---------------- ALNS intensification ----------------
        if spec.alns.enabled:
            alns_rem = max(0.0, alns_budget_total - alns_time_used)
            if alns_rem > 1e-6:
                if phase.startswith("exploit"):
                    # occasional short alns on incumbent
                    if it % 3 == 0 and best_routes:
                        cap = min(alns_rem, 0.22)
                        t0 = time.time()
                        alns_end = min(end_time, time.time() + cap)
                        r2, c2 = alns_improve(
                            best_routes, dist, demand, capacity,
                            spec=spec, rng=rng, cand=cand,
                            start_time=time.time(), end_time=alns_end,
                            best_cost_global=best_cost,
                            light_ls_neigh=light_ls_neigh,
                        )
                        alns_time_used += (time.time() - t0)
                        if c2 < best_cost:
                            best_cost = int(c2)
                            best_routes = r2
                            no_improve = 0
                            if return_events:
                                events.append((float(time.time() - start_time), int(best_cost)))
                elif intensify and best_routes:
                    # use remaining time mainly for ALNS+LS until end
                    cap = min(alns_rem, remaining - 0.01)
                    if cap > 0.02:
                        t0 = time.time()
                        alns_end = min(end_time, time.time() + cap)
                        r2, c2 = alns_improve(
                            best_routes, dist, demand, capacity,
                            spec=spec, rng=rng, cand=cand,
                            start_time=time.time(), end_time=alns_end,
                            best_cost_global=best_cost,
                            light_ls_neigh=light_ls_neigh,
                        )
                        alns_time_used += (time.time() - t0)
                        if c2 < best_cost:
                            best_cost = int(c2)
                            best_routes = r2
                            no_improve = 0
                            if return_events:
                                events.append((float(time.time() - start_time), int(best_cost)))

        # ---------------- restart if stagnation ----------------
        if spec.engine.variant == "MMAS" and no_improve >= int(spec.pheromone.restart_no_improve) and best_cost < 10**18:
            strength = float(spec.pheromone.restart_reset_strength)
            tau_reset = np.full_like(tau, tau0, dtype=np.float64)
            np.fill_diagonal(tau_reset, 0.0)
            tau = strength * tau_reset + (1.0 - strength) * tau
            _deposit_edges(tau, best_routes, Q=Q, cost=float(best_cost), weight=1.0)
            tau_max = 1.0 / max(1e-9, rho * float(best_cost))
            tau_min = tau_max * float(spec.pheromone.tau_min_factor)
            _clip_tau(tau, tau_min, tau_max)
            no_improve = 0

        it += 1

    meta = {
        "iters": it,
        "no_improve": no_improve,
        "time_used": float(time.time() - start_time),
        "variant": spec.engine.variant,
        "deposit": spec.pheromone.deposit,
        "ls_time_used": float(ls_time_used),
        "alns_time_used": float(alns_time_used),
    }

    if not events and best_cost < 10**18:
        events = [(0.0, int(best_cost))]

    return best_routes, int(best_cost), events, meta
