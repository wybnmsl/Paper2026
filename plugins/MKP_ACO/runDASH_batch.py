#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _repo_root_from_here() -> str:
    # .../plugins/MKP_ACO/runDASH_batch.py -> repo_root = .../ (two levels up)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _ensure_import_paths() -> None:
    repo_root = _repo_root_from_here()

    # Make plugin importable as "MKP_ACO.*"
    plugins_dir = os.path.join(repo_root, "plugins")
    if plugins_dir not in sys.path:
        sys.path.insert(0, plugins_dir)

    # If you also want "src/" importable (optional)
    src_dir = os.path.join(repo_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _try_import_problem_class():
    # Prefer MKPACOProblem (as in your plugin design)
    from MKP_ACO.problem import MKPACOProblem  # type: ignore
    return MKPACOProblem


def _choose_indices(n: int, k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx[: min(k, n)]


def main():
    _ensure_import_paths()

    from MKP_ACO.aco.spec import ACOSpec  # type: ignore
    from MKP_ACO.utils.readMKP import read_mknapcb  # type: ignore
    from MKP_ACO.aco.aco_run import solve_instance_with_spec_gap  # type: ignore
    from MKP_ACO.plr import PLRConfig, PLRLibrary  # type: ignore

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="OR-Library mknapcb*.txt")
    ap.add_argument("--res", type=str, default="", help="OR-Library mkcbres.txt (best-known profits)")
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=50)
    ap.add_argument("--test_n", type=int, default=50)

    ap.add_argument("--plr", action="store_true", help="Enable PLR")
    ap.add_argument("--groups", type=int, default=8, help="PLR num_groups")
    ap.add_argument("--k", type=int, default=5, help="PLR archive_k (top-k per group)")
    ap.add_argument("--plr_dir", type=str, default="outputs/plr_mkp", help="PLR save_dir")
    args = ap.parse_args()

    instances = read_mknapcb(args.data, res_path=(args.res or None))
    if not instances:
        raise RuntimeError(f"No instances loaded from: {args.data}")

    # Default spec (you can later replace it by evolved ones)
    spec = ACOSpec(time_limit_s=float(args.time), seed=int(args.seed))
    spec_dict: Dict[str, Any] = spec.to_dict()

    # -------------------------
    # PLR setup
    # -------------------------
    plr_cfg = PLRConfig(
        enabled=bool(args.plr),
        num_groups=int(args.groups),
        archive_k=int(args.k),
        save_dir=str(args.plr_dir),
        seed=int(args.seed),
        score_key="tLDR_traj",
    )
    plr = PLRLibrary(plr_cfg)

    if plr_cfg.enabled:
        # Fit centroids on a subset (paper-style profiling & grouping)
        plr.fit_groups(instances[: min(200, len(instances))])

    # -------------------------
    # TRAIN: evaluate + update PLR
    # -------------------------
    train_idx = _choose_indices(len(instances), int(args.train_n), seed=int(args.seed) + 11)
    print("=" * 80)
    print(f"[MKP-ACO][TRAIN] n_total={len(instances)} train_n={len(train_idx)} time={args.time}s plr={args.plr}")
    print("=" * 80)

    train_gaps = []
    train_tldrs = []

    for i in train_idx:
        inst = instances[i]
        gap, meta = solve_instance_with_spec_gap(inst, spec, guide_module=None, return_meta=True)
        train_gaps.append(float(gap))
        train_tldrs.append(float(meta.get("tLDR_traj", 0.0)))

        if plr_cfg.enabled:
            # Payload for warm-start retrieval (paper: archived specialized solver)
            payload = {"spec": spec_dict}
            plr.update(inst, payload=payload, meta=meta)

    print(f"[TRAIN] mean_gap={sum(train_gaps)/max(len(train_gaps),1):.6f} "
          f"mean_tLDR={sum(train_tldrs)/max(len(train_tldrs),1):.6f}")

    saved_path: Optional[str] = None
    if plr_cfg.enabled:
        saved_path = plr.save("plr_library.json")
        print(f"[TRAIN] PLR saved: {saved_path}")

    # -------------------------
    # TEST: baseline vs PLR warm-start (retrieve spec by group)
    # -------------------------
    test_idx = _choose_indices(len(instances), int(args.test_n), seed=int(args.seed) + 29)

    print("\n" + "=" * 80)
    print(f"[MKP-ACO][TEST] test_n={len(test_idx)} compare: default spec vs PLR-retrieved spec")
    print("=" * 80)

    # Baseline
    base_gaps = []
    base_tldrs = []
    for i in test_idx:
        inst = instances[i]
        gap, meta = solve_instance_with_spec_gap(inst, spec, guide_module=None, return_meta=True)
        base_gaps.append(float(gap))
        base_tldrs.append(float(meta.get("tLDR_traj", 0.0)))

    print(f"[TEST][BASE] mean_gap={sum(base_gaps)/max(len(base_gaps),1):.6f} "
          f"mean_tLDR={sum(base_tldrs)/max(len(base_tldrs),1):.6f}")

    if not plr_cfg.enabled:
        print("[TEST] PLR disabled. Done.")
        return

    # Load library back (simulates real use)
    if saved_path is None:
        raise RuntimeError("PLR was enabled but no library file was saved.")
    plr2 = PLRLibrary.load(saved_path)

    plr_gaps = []
    plr_tldrs = []
    for i in test_idx:
        inst = instances[i]
        gid, entries = plr2.retrieve(inst, k=1)
        if entries and "payload" in entries[0] and "spec" in entries[0]["payload"]:
            spec2 = ACOSpec.from_json(entries[0]["payload"]["spec"], n_items=inst.n)
        else:
            spec2 = spec

        gap, meta = solve_instance_with_spec_gap(inst, spec2, guide_module=None, return_meta=True)
        plr_gaps.append(float(gap))
        plr_tldrs.append(float(meta.get("tLDR_traj", 0.0)))

    print(f"[TEST][PLR ] mean_gap={sum(plr_gaps)/max(len(plr_gaps),1):.6f} "
          f"mean_tLDR={sum(plr_tldrs)/max(len(plr_tldrs),1):.6f}")
    print("[TEST] Done.")


if __name__ == "__main__":
    main()
