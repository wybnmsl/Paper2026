# zMKP/test_aco_engine_grid.py
# -*- coding: utf-8 -*-
"""
Manual tuning / grid test for MKP ACO engine.

Run:
  python zMKP/test_aco_engine_grid.py --data Data/MKP/mknapcb3.txt --res Data/MKP/mkcbres.txt --idx 0 --time 1 --runs 1
"""

from __future__ import annotations
import argparse
import os
from zMKP.utils.readMKP import load_instance, sanity_check_instance
from zMKP.aco.spec import ACOSpec
from zMKP.aco.aco_run import solve_instance_with_spec_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="Data/MKP/mknapcb1.txt")
    ap.add_argument("--res", type=str, default="Data/MKP/mkcbres.txt")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    inst = load_instance(args.data, instance_index=args.idx, res_path=args.res)
    sanity_check_instance(inst)

    print("============================================================")
    print(f"[MKP-ACO GRID] file={args.data}, idx={args.idx}, n={inst.n}, m={inst.m}, T={args.time}s")
    print(f"  best_known={inst.best_known} | lp_ub={inst.lp_ub:.6f} | name={inst.name}")
    print("alpha beta rho q0 ants candk dep mu | best_profit gap% tLDR iters ls_time ls% time_used")
    print("------------------------------------------------------------")

    # 这组配置偏“多代学习”，让 1 秒能跑多代（尤其 n>=500）
    grid = [
        dict(alpha=1.0, beta=3.0, rho=0.12, q0=0.20, n_ants=24, candk=25, deposit="rank", rank_mu=10,
             ls="kflip", ls_elite_k=3, ls_elite_steps=120, ls_all_ants_steps=10,
             daemon_every=0, daemon_ls_steps=0, use_multipliers=True, mult_eta_power=1.0, mult_update_every=2),
        dict(alpha=1.0, beta=3.4, rho=0.10, q0=0.25, n_ants=26, candk=30, deposit="rank", rank_mu=12,
             ls="hybrid", ls_elite_k=3, ls_elite_steps=110, ls_all_ants_steps=8,
             daemon_every=0, daemon_ls_steps=0, use_multipliers=True, mult_eta_power=1.1, mult_update_every=2),
        dict(alpha=0.9, beta=3.2, rho=0.15, q0=0.15, n_ants=28, candk=20, deposit="rank", rank_mu=12,
             ls="kflip", ls_elite_k=2, ls_elite_steps=90, ls_all_ants_steps=12,
             daemon_every=0, daemon_ls_steps=0, use_multipliers=True, mult_eta_power=0.9, mult_update_every=3),
        dict(alpha=1.2, beta=2.6, rho=0.12, q0=0.10, n_ants=22, candk=25, deposit="ibest", rank_mu=10,
             ls="add_drop", ls_elite_k=2, ls_elite_steps=90, ls_all_ants_steps=12,
             daemon_every=0, daemon_ls_steps=0, use_multipliers=False, mult_eta_power=0.0),
    ]

    best_overall = None

    for gi, cfg in enumerate(grid):
        best_profit_runs = []
        gap_runs = []
        tldr_runs = []
        it_runs = []
        ls_runs = []
        tm_runs = []

        for r in range(args.runs):
            spec = ACOSpec(time_limit_s=args.time, seed=args.seed + 1000 * gi + r, **cfg)
            gap, meta = solve_instance_with_spec_gap(inst, spec, guide_module=None, return_meta=True)
            best_profit = meta.get('best_profit', 0)

            best_profit_runs.append(best_profit)
            gap_runs.append(meta.get("gap", 0.0))
            tldr_runs.append(meta.get("tLDR_traj", 0.0))
            it_runs.append(meta.get("iters", 0))
            ls_runs.append(meta.get("ls_time", 0.0))
            tm_runs.append(meta.get("time_used", 0.0))

        bp = sum(best_profit_runs) / len(best_profit_runs)
        gp = 100.0 * (sum(gap_runs) / len(gap_runs))
        tl = sum(tldr_runs) / len(tldr_runs)
        it = int(round(sum(it_runs) / len(it_runs)))
        ls = sum(ls_runs) / len(ls_runs)
        tm = sum(tm_runs) / len(tm_runs)
        lsp = 100.0 * (ls / max(tm, 1e-9))

        print(f"{cfg['alpha']:>4.1f} {cfg['beta']:>4.1f} {cfg['rho']:>4.2f} {cfg['q0']:>3.2f} "
              f"{cfg['n_ants']:>4d} {cfg['candk']:>5d} {cfg['deposit']:<5s} {cfg['rank_mu']:>2d} | "
              f"{bp:>11.1f} {gp:>5.2f} {tl:>6.3f} {it:>5d} {ls:>7.2f}s {lsp:>4.0f}% {tm:>7.2f}s")

        if best_overall is None or bp > best_overall[0]:
            best_overall = (bp, cfg)

    print("============================================================")
    if best_overall is not None:
        bp, cfg = best_overall
        print(f"Best(avg_profit)={bp:.1f} with cfg={cfg}")


if __name__ == "__main__":
    main()
