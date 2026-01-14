from __future__ import annotations

"""
CVRP-ACO plugin prompts.

These prompts are intentionally lightweight. In DASH, the evolution engine drives
LLM editing in three layers (MDL / MCL / SSL). For ACO-style solvers, we expose a
single JSON spec (ACOSpec) and let each layer focus on a subset of fields:

- MDL: mechanism parameters (operators, neighborhood, acceptance logic)
- MCL: macro structure (phase coordination / components on-off switches)
- SSL: runtime schedule (time shares, stage allocation, early intensification)

The LLM should output a JSON object that partially updates the current spec.
Unknown fields will be ignored by the tolerant spec parser.
"""

PROMPT_HEADER = """You are editing a CVRP solver configuration (ACOSpec) used by DASH.
Return ONLY a valid JSON object (no markdown). The JSON will be merged into the current spec.
Rules:
- Only include fields you want to change.
- Keep values within reasonable ranges.
- Do not remove required fields; do not invent new top-level modules.
"""

PROMPT_MDL = PROMPT_HEADER + """
MDL (Mechanism Design Layer) task:
Improve the solver's search mechanism (operators and acceptance logic) to achieve better anytime efficiency.

You may adjust (examples):
- pheromone: rho, deposit, rank_mu, rank_weight, restart parameters
- construct: alpha, beta, q0, candidate_k, demand_gamma
- local_search: neigh, max_moves, first_improve, granular_k
- alns: destroy weights, ruin fractions, accept rule, regret_k, SA/RRT parameters
"""

PROMPT_MCL = PROMPT_HEADER + """
MCL (Macro Composition Layer) task:
Improve the solver's macro structure by composing / coordinating components into a stronger workflow.

You may adjust (examples):
- engine: variant (ACS/MMAS), explore_frac, exploit_frac, intensify_last_s
- init: method, ls_after_init
- toggle components: local_search.enabled, alns.enabled
- how local_search is applied: apply_to, top_k
"""

PROMPT_SSL = PROMPT_HEADER + """
SSL (Schedule & Selection Layer) task:
Improve the runtime schedule to enhance anytime performance under a fixed time budget.

You may adjust (examples):
- local_search.time_share, alns.time_share
- engine.explore_frac, engine.exploit_frac, engine.intensify_last_s
- stopping-related parameters (if needed): stopping.max_iters (time_limit is fixed outside)
"""


def build_prompt(layer: str) -> str:
    layer = layer.strip().upper()
    if layer == "MDL":
        return PROMPT_MDL
    if layer == "MCL":
        return PROMPT_MCL
    if layer == "SSL":
        return PROMPT_SSL
    return PROMPT_HEADER
