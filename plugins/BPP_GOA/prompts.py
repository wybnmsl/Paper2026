from __future__ import annotations

"""
BPP-GOA plugin prompts.

DASH evolves solvers through three complementary layers (MDL / MCL / SSL).
For BPP-GOA, the solver is parameterized by a discrete-choice GOASpec.

The LLM should output ONLY a valid JSON object (no markdown). The JSON will be merged into
the current spec by a tolerant parser (unknown keys are ignored).

Layer guidance (paper-aligned):
- MDL: mechanism (priority usage, candidate pruning, pheromone/choice dynamics, operators toggles)
- MCL: macro composition (strict_online vs hybrid/offline, which components are enabled, how they are coordinated)
- SSL: runtime schedule (iters/ants allocation under time_limit_s, inner_time_frac, logging cadence)
"""

PROMPT_HEADER = """You are editing a Bin Packing (BPP) GOA solver configuration (GOASpec) used by DASH.
Return ONLY a valid JSON object (no markdown). The JSON will be merged into the current spec.

Rules:
- Only include fields you want to change.
- Keep values within reasonable ranges.
- Do not delete required fields; do not invent new top-level modules.
"""

PROMPT_MDL = PROMPT_HEADER + """
MDL (Mechanism Design Layer) task:
Improve the solver mechanism (search behavior) to achieve better anytime efficiency.

You may adjust (examples):
- use_priority_in_construct (true/false)
- cand_k and cand_k_choices (candidate pruning for placement decisions)
- gamma_choices, temp_choices, q0_choices, open_bias_choices (policy search space)
- rho, deposit, tau_min, tau_max (GOA pheromone dynamics)
- in hybrid/offline mode: enable_bin_empty, enable_k_repack, enable_ruin_recreate
- accept_equal_prob, worsen_accept_prob (stochastic acceptance / rollback)
"""

PROMPT_MCL = PROMPT_HEADER + """
MCL (Macro Composition Layer) task:
Improve the macro structure by composing / coordinating components into a stronger workflow.

You may adjust (examples):
- strict_online (true/false) to switch between STRICT-ONLINE construct-only and HYBRID/OFFLINE LNS mode
- init_method: online_bf / online_ff / bfd / ffd
- in hybrid/offline mode: operator toggles and the relative emphasis implied by choices lists:
  ruin_frac_choices, repack_k_choices, empty_trials_choices
"""

PROMPT_SSL = PROMPT_HEADER + """
SSL (Schedule & Selection Layer) task:
Improve the runtime schedule to enhance anytime performance under a fixed time budget.

You may adjust (examples):
- iters, ants (outer-loop allocation)
- inner_time_frac (hybrid/offline per-ant slice fraction)
- log_on_improve, log_interval_s (trajectory logging cadence)
"""


def build_prompt(layer: str) -> str:
    layer = (layer or "").strip().upper()
    if layer == "MDL":
        return PROMPT_MDL
    if layer == "MCL":
        return PROMPT_MCL
    if layer == "SSL":
        return PROMPT_SSL
    return PROMPT_HEADER
