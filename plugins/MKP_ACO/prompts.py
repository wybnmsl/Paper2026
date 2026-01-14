# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PromptPack:
    system: str
    mdl: str
    mcl: str
    ssl: str


SYSTEM_PROMPT = """You are an expert in combinatorial optimization and heuristic design.
You will propose an improved solver configuration for a DASH-style evolution step.

Output format (STRICT):
- Return ONLY a JSON object (no markdown, no comments).
- The JSON keys must be valid ACOSpec fields.
- Do NOT invent new keys.
- Use numeric values where appropriate.
"""

MDL_PROMPT = """Task: Mechanism Design Layer (MDL)
You will improve the solver *mechanism* by tuning search behavior primitives.

MKP-ACO Mechanism knobs (examples, not exhaustive):
- construction/transition: alpha, beta, q0, candk, ants_greedy_frac
- pheromone update: rho, deposit, rank_mu, elite_weight, gbest_weight, tau0, tau_min_ratio, tau_max
- feasibility/repair: allow_infeasible_construct, repair, drop_rule
- local search operators: ls, ls_all_ants_steps, ls_elite_k, ls_elite_steps
- dynamic multipliers: use_multipliers, mult_update_every, mult_eta_power, mult_smooth

Objective (paper-aligned):
- Favor candidates with better trajectory-aware efficiency (tLDR) AND good final quality (gap).
- Avoid configs that improve only at the end (endpoint-only behavior).

Given:
- current_spec_json: {current_spec_json}
- feedback_json (contains mean_gap, mean_tLDR, and per-instance meta): {feedback_json}

Return a new ACOSpec JSON (only changed keys are allowed, but full keys also acceptable).
"""

MCL_PROMPT = """Task: Macro Composition Layer (MCL)
You will improve the solver *macro structure* by composing/coordinating multiple components.

Note:
- MKP-ACO here is spec-driven; if your pipeline supports macro composition, express it as a staged
  plan by adjusting schedule-related fields (e.g., LS intensity timing, daemon settings, restarts).

Given:
- current_spec_json: {current_spec_json}
- feedback_json: {feedback_json}

Return a new ACOSpec JSON focusing on higher-level workflow improvements.
"""

SSL_PROMPT = """Task: Schedule & Selection Layer (SSL)
You will improve the runtime *schedule* to maximize anytime efficiency under a time budget.

Schedule knobs in MKP-ACO:
- LS allocation: ls_all_ants_steps, ls_elite_k, ls_elite_steps
- daemon usage: daemon_every, daemon_ls_steps
- budget guards (keep within time): time_guard_margin_s, skip_daemon_if_time_left_s,
  skip_elite_if_time_left_s, skip_pheromone_if_time_left_s
- stagnation/restart: stagnation_iters, restart_keep_best, restart_tau0

Objective (paper-aligned):
- Improve tLDR (early and sustained progress), not only endpoint gap.
- Keep budget safety margins so the run respects the time limit.

Given:
- current_spec_json: {current_spec_json}
- feedback_json: {feedback_json}

Return a new ACOSpec JSON.
"""


def build_prompts(
    current_spec: Dict[str, Any],
    feedback: Optional[Dict[str, Any]] = None,
) -> PromptPack:
    cur = json.dumps(current_spec or {}, ensure_ascii=False)
    fb = json.dumps(feedback or {}, ensure_ascii=False)

    return PromptPack(
        system=SYSTEM_PROMPT,
        mdl=MDL_PROMPT.format(current_spec_json=cur, feedback_json=fb),
        mcl=MCL_PROMPT.format(current_spec_json=cur, feedback_json=fb),
        ssl=SSL_PROMPT.format(current_spec_json=cur, feedback_json=fb),
    )


# Compatibility helper (some plugins use GetPrompts())
class GetPrompts:
    def __call__(self, current_spec: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> PromptPack:
        return build_prompts(current_spec=current_spec, feedback=feedback)
