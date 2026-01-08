from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from typing import Optional

import numpy as np

from DASH.plugins.BPP_GOA.utils.read_bpp_pickle import load_bpp_pickle
from DASH.plugins.BPP_GOA.utils.utils import make_logger, lb_relaxation, gap_percent, summarize_instance
from DASH.plugins.BPP_GOA.goa.spec import GOASpec
from DASH.plugins.BPP_GOA.goa.goa_run import solve_bpp_goa
from DASH.plugins.BPP_GOA.goa.goa_operators import pack_ffd_or_bfd, Solution


class BestFitHeuristic:
    def priority(self, item: float, bins_remain: np.ndarray) -> np.ndarray:
        r = np.asarray(bins_remain, dtype=np.float64)
        w = float(item)
        return -(r - w)


def validate_solution(items: np.ndarray, C: int, sol: Solution, *, name: str = "sol") -> tuple[bool, str]:
    items = np.asarray(items)
    n = int(items.shape[0])

    flat: list[int] = []
    for b, bin_items in enumerate(sol.bins):
        for idx in bin_items:
            flat.append(int(idx))

    if len(flat) != n:
        return False, f"{name}: item count mismatch. assigned={len(flat)} vs n={n}"

    arr = np.asarray(flat, dtype=np.int64)
    if np.min(arr) < 0 or np.max(arr) >= n:
        return False, f"{name}: out-of-range item index detected."

    cnt = np.bincount(arr, minlength=n)
    if np.any(cnt != 1):
        dup = int(np.sum(cnt > 1))
        miss = int(np.sum(cnt == 0))
        return False, f"{name}: duplicates/missing detected. dup_items={dup}, missing_items={miss}"

    eps = 1e-9
    for b, bin_items in enumerate(sol.bins):
        load = float(np.sum(items[np.asarray(bin_items, dtype=np.int64)])) if bin_items else 0.0
        if load - float(C) > eps:
            return False, f"{name}: capacity violated in bin {b}"
    return True, f"{name}: OK (bins={len(sol.bins)})"


def maybe_shuffle_items(
    items: np.ndarray,
    *,
    do_shuffle: bool,
    perm_seed: Optional[int],
    fallback_seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if not do_shuffle:
        return items, None
    seed = int(fallback_seed if perm_seed is None else perm_seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(items))
    return items[perm], perm


def run_one(logger, items, C, spec: GOASpec, seed_offset: int = 0):
    h = BestFitHeuristic()
    spec = GOASpec(**{**asdict(spec), "seed": int(spec.seed) + int(seed_offset)})

    lb = lb_relaxation(items, C)

    init_sol = pack_ffd_or_bfd(items, C, method=spec.init_method)
    init_bins = len(init_sol.bins)
    init_gap = gap_percent(init_bins, lb)
    ok0, msg0 = validate_solution(items, C, init_sol, name="init")

    logger.info(f"Init ({spec.init_method.upper()}): bins={init_bins}  lb={lb}  gap={init_gap:.3f}%  | {msg0}")
    if not ok0:
        raise RuntimeError(f"Init solution invalid: {msg0}")

    t0 = time.perf_counter()
    res = solve_bpp_goa(items, C, priority_fn=h.priority, spec=spec)
    t1 = time.perf_counter()

    gap = gap_percent(res.best_bins, lb)

    ok1, msg1 = validate_solution(items, C, res.best_solution, name="best")
    if not ok1:
        raise RuntimeError(f"Best solution invalid: {msg1}")

    logger.info(
        f"BEST: bins={res.best_bins}  lb={lb}  gap={gap:.3f}%  "
        f"time={t1 - t0:.3f}s  (iters={spec.iters}, ants={spec.ants}, rho={spec.rho})  | {msg1}"
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--time", type=float, default=10.0)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--init", type=str, default="online_bf", choices=["bfd", "ffd", "online_bf", "online_ff"])
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--perm-seed", type=int, default=None)
    ap.add_argument("--hybrid", action="store_true")

    args = ap.parse_args()
    logger = make_logger("bpp_goa_test", level=20)

    ds = load_bpp_pickle(args.pkl)
    items0, C, case_meta = ds.get_case(args.case)
    items, perm = maybe_shuffle_items(items0, do_shuffle=bool(args.shuffle), perm_seed=args.perm_seed, fallback_seed=int(args.seed) + int(args.case))

    strict_online = not bool(args.hybrid)

    logger.info(f"Case meta: case_id={case_meta['case_id']}  n_items={len(items)}  C={C}")
    logger.info(f"Mode: {'STRICT-ONLINE' if strict_online else 'HYBRID/OFFLINE'}")
    logger.info(f"Instance summary: {summarize_instance(items, C)}")

    base = GOASpec(
        time_limit_s=float(args.time),
        seed=int(args.seed),
        iters=12 if strict_online else 6,
        ants=16 if strict_online else 10,
        rho=0.15,
        init_method=str(args.init),
        strict_online=bool(strict_online),
        cand_k=32,
        use_priority_in_construct=True,
        enable_bin_empty=True,
        enable_k_repack=True,
        enable_ruin_recreate=True,
    )

    if not args.grid:
        run_one(logger, items, C, base)
        return

    iters_list = [8, 12] if strict_online else [5, 6]
    ants_list = [12, 16] if strict_online else [8, 10, 12]
    rho_list = [0.10, 0.15, 0.20]

    best = None
    best_cfg = None

    for iters in iters_list:
        for ants in ants_list:
            for rho in rho_list:
                spec = GOASpec(**{**asdict(base), "iters": iters, "ants": ants, "rho": rho})
                res = run_one(logger, items, C, spec, seed_offset=iters + ants)
                score = res.best_bins
                if best is None or score < best:
                    best = score
                    best_cfg = spec

    logger.info("=== GRID BEST ===")
    logger.info(f"best_bins={best}  spec={best_cfg}")


if __name__ == "__main__":
    main()
