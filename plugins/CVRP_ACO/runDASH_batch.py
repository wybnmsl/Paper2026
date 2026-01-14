from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional

from .utils.readCVRPLib import load_instances
from .aco.spec import default_aco_spec
from .aco.aco_run import solve_instance_with_spec
from .plr import PLRConfig, PLRLibrary


def _parse_cases(cases_arg: str) -> List[str]:
    if os.path.isfile(cases_arg):
        with open(cases_arg, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]
    # comma-separated
    return [x.strip() for x in cases_arg.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Run DASH CVRP-ACO plugin in batch (with optional PLR warm-start).")
    ap.add_argument("--data-root", type=str, required=True, help="Folder containing *.vrp and (optional) *.sol")
    ap.add_argument("--cases", type=str, required=True, help="Comma list or a text file of instance names (without suffix)")
    ap.add_argument("--time", type=float, default=10.0, help="Time limit seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="artifacts/cvrp_aco_runs.jsonl")

    # PLR
    ap.add_argument("--plr", action="store_true", help="Enable PLR retrieval + library update")
    ap.add_argument("--plr-path", type=str, default="", help="Path to PLR json. If empty, uses default under artifacts/plr/")
    ap.add_argument("--plr-fit", action="store_true", help="Fit centroids from the loaded instances before running")
    args = ap.parse_args()

    case_names = _parse_cases(args.cases)
    insts = load_instances(args.data_root, case_names, require_sol=False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # PLR init
    plr_lib: Optional[PLRLibrary] = None
    if args.plr:
        cfg = PLRConfig(enabled=True, seed=int(args.seed))
        plr_path = args.plr_path.strip() or cfg.save_path
        if os.path.isfile(plr_path):
            plr_lib = PLRLibrary.load(plr_path)
        else:
            plr_lib = PLRLibrary(cfg)
        if args.plr_fit:
            plr_lib.fit_groups(insts)

    with open(args.out, "w", encoding="utf-8") as fout:
        for inst in insts:
            spec = default_aco_spec()
            spec.engine.seed = int(args.seed)
            spec.stopping.time_limit_s = float(args.time)

            # PLR warm-start: replace spec by retrieved archived spec if exists
            if plr_lib is not None:
                retrieved = plr_lib.retrieve(inst, allow_global_fallback=True)
                if retrieved is not None:
                    spec = retrieved  # dict spec is accepted by solve_instance_with_spec

            t0 = time.time()
            gap, meta = solve_instance_with_spec(inst, spec, time_limit_s=float(args.time), return_trace=True)
            wall = time.time() - t0

            rec = {
                "case": getattr(inst, "name", ""),
                "gap": float(gap),
                "best_cost": meta.get("best_cost", None),
                "opt_cost": meta.get("opt_cost", None),
                "tLDR_traj": meta.get("tLDR_traj", None),
                "T_used_wall": float(wall),
                "profile_phi": meta.get("profile_phi", None),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            # Update PLR with the run result (for library building / debugging)
            if plr_lib is not None:
                spec_dict = meta.get("spec", None)
                if isinstance(spec_dict, dict):
                    plr_lib.update_from_run(inst, spec_dict, meta, tag="batch_run")

    if plr_lib is not None:
        plr_path = args.plr_path.strip() or plr_lib.cfg.save_path
        plr_lib.save(plr_path)
        print(f"[PLR] saved -> {plr_path}")

    print(f"[DONE] wrote -> {args.out}")


if __name__ == "__main__":
    main()
