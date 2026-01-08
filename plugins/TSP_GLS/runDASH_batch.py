#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch runner for DASH + TSPGLS (TSPLIB).

This script is optional. It shows how to run DASH on a list of TSPLIB instances
using the TSPGLS plugin.

Notes:
- It expects your DASH framework package to be importable from repo root (src/).
- LLM credentials are read from environment variables:
    DASH_LLM_API_ENDPOINT, DASH_LLM_API_KEY, DASH_LLM_MODEL
"""

from __future__ import annotations

import os
import sys
import time

# Ensure `src/` is importable when running from repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DASH.plugins.tsp_gls.problem import TSPGLS


# -------------------------
# TSPLIB dataset settings
# -------------------------

TSPLIB_ROOT = os.path.join(REPO_ROOT, "tsplib-master")
TSPLIB_SOLUTIONS = os.path.join(TSPLIB_ROOT, "solutions")

TSPLIB_CASES = [
    "a280",
]


def _get_llm_cfg():
    endpoint = os.environ.get("DASH_LLM_API_ENDPOINT", "")
    api_key = os.environ.get("DASH_LLM_API_KEY", "")
    model = os.environ.get("DASH_LLM_MODEL", "")
    return endpoint, api_key, model


def main():
    # Import inside main so this script can still be opened without full DASH deps.
    from DASH.utils.getParas import Paras
    from DASH.ec import methods as ec_methods
    from DASH.runner.evol import EVOL

    endpoint, api_key, model = _get_llm_cfg()

    paras = Paras()
    prob_name = "TSPGLS"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{prob_name}_{timestamp}"
    setattr(paras, "exp_run_name", run_name)

    # Set DASH parameters (adjust as needed)
    paras.set_paras(
        method="dash",
        problem=None,
        llm_api_endpoint=endpoint,
        llm_api_key=api_key,
        llm_model=model,
        ec_pop_size=6,
        ec_n_pop=6,
        exp_n_proc=6,
        exp_debug_mode=False,
        eva_numba_decorator=False,
        eva_timeout=30,
    )

    # Build selection / management (depends on your ec.methods implementation)
    select = getattr(ec_methods, "build_selection", None)
    manage = getattr(ec_methods, "build_management", None)
    if not callable(select) or not callable(manage):
        raise RuntimeError(
            "Cannot find ec_methods.build_selection/build_management. "
            "Please implement them or change this runner accordingly."
        )
    selector = select(paras)
    manager = manage(paras)

    print(f"[Batch] run_name = {run_name}")
    print(f"[Batch] cases    = {TSPLIB_CASES}")

    for case in TSPLIB_CASES:
        print("\n" + "=" * 60)
        print(f"[Batch] Start case: {case}")
        print("=" * 60)

        try:
            problem = TSPGLS(
                tsplib_root=TSPLIB_ROOT,
                solutions_file=TSPLIB_SOLUTIONS,
                case_names=[case],
                n_inst_eva=None,
                time_limit=float(paras.eva_timeout),
                debug_mode=bool(paras.exp_debug_mode),
            )
        except Exception as e:
            print(f"[Batch][WARN] Skip case {case}: {e}")
            continue

        setattr(problem, "case_tag", case)
        paras.problem = problem

        runner = EVOL(paras, problem, selector, manager)
        runner.run()

    print("\n[Batch] All cases finished.")


if __name__ == "__main__":
    main()
