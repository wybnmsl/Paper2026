from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class GOASpec:
    # -----------------------------
    # Budget / outer loop
    # -----------------------------
    time_limit_s: float = 10.0
    seed: int = 0
    iters: int = 12
    ants: int = 16

    # -----------------------------
    # Mode
    # -----------------------------
    # True  => STRICT ONLINE: construct only, NO moving past items.
    # False => Hybrid/offline: allow LNS operators (bin-empty, k-repack, ruin-recreate)
    strict_online: bool = True

    # -----------------------------
    # "GOA" parameters (pheromone on discrete choices)
    # -----------------------------
    rho: float = 0.15          # evaporation
    deposit: float = 1.0       # deposit scale
    tau_min: float = 1e-4
    tau_max: float = 1e3

    # -----------------------------
    # Initialization method
    # -----------------------------
    init_method: str = "online_bf"   # "online_bf" | "online_ff" | "bfd" | "ffd"

    # -----------------------------
    # Compatibility: used by hybrid/offline insertion candidate pruning
    # -----------------------------
    cand_k: int = 32

    # -----------------------------
    # STRICT ONLINE: GOA over policy parameters (discrete choices)
    # -----------------------------
    cand_k_choices: List[int] = None
    gamma_choices: List[float] = None
    temp_choices: List[float] = None
    q0_choices: List[float] = None
    open_bias_choices: List[float] = None

    use_priority_in_construct: bool = True

    # logging
    log_on_improve: bool = True
    log_interval_s: float = 0.5

    # -----------------------------
    # HYBRID/OFFLINE fields (ignored when strict_online=True)
    # -----------------------------
    enable_bin_empty: bool = True
    enable_k_repack: bool = True
    enable_ruin_recreate: bool = True

    accept_equal_prob: float = 0.05
    worsen_accept_prob: float = 0.01

    ruin_frac_choices: List[float] = None
    repack_k_choices: List[int] = None
    empty_trials_choices: List[int] = None

    inner_time_frac: float = 0.85

    def __post_init__(self):
        if self.cand_k_choices is None:
            self.cand_k_choices = [8, 16, 32, 64]
        if self.gamma_choices is None:
            self.gamma_choices = [0.0, 0.5, 1.0]
        if self.temp_choices is None:
            self.temp_choices = [0.05, 0.1, 0.2, 0.4]
        if self.q0_choices is None:
            self.q0_choices = [0.0, 0.2, 0.5, 0.8]
        if self.open_bias_choices is None:
            self.open_bias_choices = [-1e9, -10.0, -2.0]

        if self.ruin_frac_choices is None:
            self.ruin_frac_choices = [0.02, 0.04, 0.08, 0.12]
        if self.repack_k_choices is None:
            self.repack_k_choices = [2, 3, 4, 5]
        if self.empty_trials_choices is None:
            self.empty_trials_choices = [16, 32, 64]


def default_goa_spec() -> GOASpec:
    return GOASpec()


def from_json(obj: Dict[str, Any]) -> GOASpec:
    base = asdict(default_goa_spec())
    for k, v in obj.items():
        if k in base:
            base[k] = v
    return GOASpec(**base)


def to_json(spec: GOASpec) -> Dict[str, Any]:
    return asdict(spec)
