# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ACOSpec:
    time_limit_s: float = 10.0
    max_iters: int = 1_000_000
    seed: int = 0

    # Strict budget control
    # Smaller margin allows more iterations under short budgets, while keeping a safety buffer
    time_guard_margin_s: float = 0.03
    # If too close to deadline, skip daemon (heavy local search)
    skip_daemon_if_time_left_s: float = 0.25
    # If too close to deadline, skip elite strong local search
    skip_elite_if_time_left_s: float = 0.20
    # If too close to deadline, optionally skip pheromone update (usually fast, but guarded)
    skip_pheromone_if_time_left_s: float = 0.008

    # Colony (default encourages multi-iteration learning)
    n_ants: int = 25
    ants_greedy_frac: float = 0.25
    candk: int = 25

    # Transition
    alpha: float = 1.0
    beta: float = 3.2
    q0: float = 0.20
    use_local_pheromone_update: bool = True
    phi: float = 0.10

    # Pheromone update
    rho: float = 0.12
    deposit: str = "rank"          # {"ibest","gbest","rank"}
    rank_mu: int = 10
    elite_weight: float = 1.0
    gbest_weight: float = 0.6

    # Tau bounds
    tau0: float = 0.5
    use_tau_bounds: bool = True
    tau_min_ratio: float = 0.05
    tau_max: float = 1.0

    # Stagnation & restart
    stagnation_iters: int = 60
    restart_keep_best: bool = True
    restart_tau0: Optional[float] = None

    # Repair
    allow_infeasible_construct: bool = False
    repair: str = "drop_refill"      # {"none","drop","drop_refill"}
    drop_rule: str = "min_ratio"     # {"min_ratio","min_profit","max_weight"}

    # Local search schedule (kept light by default to preserve ACO iterations)
    ls_all_ants_steps: int = 12
    ls_elite_k: int = 3
    ls_elite_steps: int = 120
    ls: str = "kflip"                # {"none","1swap","kflip","add_drop","hybrid"}

    # Daemon (disabled by default; can consume the budget under short time limits)
    daemon_every: int = 0
    daemon_ls_steps: int = 500

    # Dynamic multipliers (useful under short budgets, but avoid too frequent updates)
    use_multipliers: bool = True
    mult_update_every: int = 2
    mult_eta_power: float = 1.0
    mult_smooth: float = 0.4

    trace: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: Dict[str, Any], n_items: int) -> "ACOSpec":
        spec = ACOSpec()
        for k, v in (d or {}).items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        spec.clip(n_items)
        return spec

    def clip(self, n_items: int) -> None:
        self.time_limit_s = float(max(0.05, self.time_limit_s))
        self.max_iters = int(max(1, self.max_iters))

        self.time_guard_margin_s = float(min(max(0.0, self.time_guard_margin_s), 0.5))
        self.skip_daemon_if_time_left_s = float(min(max(0.0, self.skip_daemon_if_time_left_s), 5.0))
        self.skip_elite_if_time_left_s = float(min(max(0.0, self.skip_elite_if_time_left_s), 5.0))
        self.skip_pheromone_if_time_left_s = float(min(max(0.0, self.skip_pheromone_if_time_left_s), 1.0))

        self.n_ants = int(min(max(5, self.n_ants), 300))
        self.ants_greedy_frac = float(min(max(0.0, self.ants_greedy_frac), 1.0))
        self.candk = int(min(max(0, self.candk), n_items))

        self.alpha = float(min(max(0.0, self.alpha), 10.0))
        self.beta = float(min(max(0.0, self.beta), 20.0))
        self.q0 = float(min(max(0.0, self.q0), 1.0))
        self.phi = float(min(max(0.0, self.phi), 1.0))

        self.rho = float(min(max(1e-6, self.rho), 0.999999))
        if self.deposit not in ("ibest", "gbest", "rank"):
            self.deposit = "rank"
        self.rank_mu = int(min(max(1, self.rank_mu), self.n_ants))

        self.elite_weight = float(max(0.0, self.elite_weight))
        self.gbest_weight = float(max(0.0, self.gbest_weight))

        self.tau0 = float(max(1e-6, self.tau0))
        self.tau_max = float(max(self.tau0, self.tau_max))
        self.tau_min_ratio = float(min(max(1e-4, self.tau_min_ratio), 0.999))

        self.stagnation_iters = int(max(5, self.stagnation_iters))

        if self.repair not in ("none", "drop", "drop_refill"):
            self.repair = "drop_refill"
        if self.drop_rule not in ("min_ratio", "min_profit", "max_weight"):
            self.drop_rule = "min_ratio"

        self.ls_all_ants_steps = int(max(0, self.ls_all_ants_steps))
        self.ls_elite_k = int(min(max(0, self.ls_elite_k), self.n_ants))
        self.ls_elite_steps = int(max(0, self.ls_elite_steps))

        if self.ls not in ("none", "1swap", "kflip", "add_drop", "hybrid"):
            self.ls = "kflip"

        self.daemon_every = int(max(0, self.daemon_every))
        self.daemon_ls_steps = int(max(0, self.daemon_ls_steps))

        self.mult_update_every = int(max(1, self.mult_update_every))
        self.mult_eta_power = float(max(0.0, self.mult_eta_power))
        self.mult_smooth = float(min(max(0.0, self.mult_smooth), 0.99))
