from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class EngineSpec:
    variant: str = "MMAS"           # "ACS" | "MMAS"
    seed: int = 0
    explore_frac: float = 0.45      # fraction of time for explore
    exploit_frac: float = 0.45      # fraction of time for exploit (rest is late exploit + intensify)
    intensify_last_s: float = 1.0   # last seconds reserved for intensification


@dataclass
class InitSpec:
    method: str = "savings"         # "savings" | "random"
    ls_after_init: bool = True


@dataclass
class ConstructSpec:
    n_ants: int = 40
    alpha: float = 1.0
    beta: float = 2.5
    q0: float = 0.25                # ACS exploitation prob
    candidate_k: int = 24
    demand_gamma: float = 0.0       # optional bias toward high demand when remaining cap is small


@dataclass
class PheromoneSpec:
    rho: float = 0.15
    q: float = 1.0
    tau_init: str = "auto"          # "auto" or numeric (string accepted)
    tau_min_factor: float = 0.05

    deposit: str = "gbest"          # "gbest" | "ibest" | "elite"
    elite_m: int = 2

    rank_mu: int = 5                # for "ibest"
    rank_weight: float = 0.85

    restart_no_improve: int = 35
    restart_reset_strength: float = 0.65


@dataclass
class LocalSearchSpec:
    enabled: bool = True
    time_share: float = 0.22

    apply_to: str = "top_k"         # "best" | "top_k" | "all"
    top_k: int = 3

    neigh: List[str] = field(default_factory=lambda: ["2opt_intra", "relocate1", "swap1", "2opt_star"])
    max_moves: int = 160000
    first_improve: bool = True
    granular_k: int = 24            # for candidate-based neighborhood pruning


@dataclass
class ALNSSpec:
    enabled: bool = True
    time_share: float = 0.28

    # destroy weights
    w_random: float = 1.0
    w_shaw: float = 1.0
    w_worst: float = 1.0
    w_route: float = 0.7

    ruin_frac_min: float = 0.10
    ruin_frac_max: float = 0.30

    regret_k: int = 3

    accept: str = "rrt"             # "greedy" | "rrt" | "sa"
    rrt_epsilon: float = 0.03
    rrt_final_epsilon: float = 0.01

    sa_init_factor: float = 0.02
    sa_alpha: float = 0.995

    inner_ls: bool = True
    inner_ls_time_cap_s: float = 0.20
    inner_ls_moves: int = 5000


@dataclass
class GuidanceSpec:
    type: str = "none"              # "none" | "llm" | "builtin"
    call_interval_iters: int = 5


@dataclass
class StoppingSpec:
    time_limit_s: float = 10.0
    max_iters: int = 10**9


@dataclass
class ACOSpec:
    engine: EngineSpec = field(default_factory=EngineSpec)
    init: InitSpec = field(default_factory=InitSpec)
    construct: ConstructSpec = field(default_factory=ConstructSpec)
    pheromone: PheromoneSpec = field(default_factory=PheromoneSpec)
    local_search: LocalSearchSpec = field(default_factory=LocalSearchSpec)
    alns: ALNSSpec = field(default_factory=ALNSSpec)
    guidance: GuidanceSpec = field(default_factory=GuidanceSpec)
    stopping: StoppingSpec = field(default_factory=StoppingSpec)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_aco_spec() -> ACOSpec:
    return ACOSpec()


def from_json(js: Optional[Dict[str, Any]]) -> ACOSpec:
    """
    Tolerant parser for LLM-generated dict specs.
    Missing fields use defaults; unknown fields are ignored.
    """
    base = default_aco_spec()
    if not isinstance(js, dict):
        return base

    def merge_field(field_name: str, cls):
        val = js.get(field_name, None)
        if isinstance(val, dict):
            cur = asdict(getattr(base, field_name))
            cur.update(val)
            return cls(**cur)
        return getattr(base, field_name)

    return ACOSpec(
        engine=merge_field("engine", EngineSpec),
        init=merge_field("init", InitSpec),
        construct=merge_field("construct", ConstructSpec),
        pheromone=merge_field("pheromone", PheromoneSpec),
        local_search=merge_field("local_search", LocalSearchSpec),
        alns=merge_field("alns", ALNSSpec),
        guidance=merge("guidance", "guidance"),
        stopping=merge("stopping", "stopping"),
    )
