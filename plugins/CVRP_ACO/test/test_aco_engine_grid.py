from __future__ import annotations

import argparse
import os
import time

from zCVRP.utils.readCVRPLib import load_instances
from zCVRP.aco.spec import default_aco_spec
from zCVRP.aco.aco_run import solve_instance_with_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True, help="Folder containing *.vrp and (optional) *.sol")
    ap.add_argument("--case", type=str, required=True, help="Instance name without suffix, e.g., X-n101-k25")
    ap.add_argument("--time", type=float, default=10.0, help="Time limit seconds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inst = load_instances(args.data_root, [args.case], require_sol=False)[0]

    spec = default_aco_spec()
    spec.engine.seed = int(args.seed)
    spec.stopping.time_limit_s = float(args.time)

    t0 = time.time()
    gap, meta = solve_instance_with_spec(inst, spec, return_trace=True)
    t1 = time.time()

    print(f"[CVRP_ACO] case={args.case}  cost={meta['best_cost']:.0f}  opt={meta['opt_cost']}  gap={gap:.3f}%  wall={t1-t0:.3f}s")
    print(f"tLDR_traj={meta['tLDR_traj']}  events={len(meta['incumbent_events'])}")
    print(f"profile_phi={meta['profile_phi']}")


if __name__ == "__main__":
    main()
