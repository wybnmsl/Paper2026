# utils/gen_weibull_bpp.py
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict, Any

import numpy as np


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _weibull_case_int(
    rng: np.random.Generator,
    n_items: int,
    capacity: int,
    *,
    shape_k: float,
    scale_lambda: float,
    clip_q: float,
) -> np.ndarray:
    """
    Generate ONE case: int weights in [1, capacity], shape (n_items,).
    Uses quantile scaling (clip_q) to avoid a single outlier compressing all weights.
    """
    raw = rng.weibull(shape_k, size=int(n_items)).astype(np.float64) * float(scale_lambda)
    raw = np.maximum(raw, 0.0)

    denom = float(np.quantile(raw, float(clip_q)))
    if denom <= 1e-12:
        mx = float(np.max(raw))
        denom = mx if mx > 1e-12 else 1.0

    x = raw / (denom + 1e-12)  # roughly (0, ~1]
    w = np.rint(x * float(capacity)).astype(np.int32)
    w = np.clip(w, 1, int(capacity))

    # capacity=500 fits int16
    return w.astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="output pkl path, e.g. ./TrainingData/BPP5k_C500.pkl")
    ap.add_argument("--seed", type=int, default=0)

    # fixed target defaults: 5k items, C=500
    ap.add_argument("--n-cases", type=int, default=100, help="number of cases in dataset")
    ap.add_argument("--n-items", type=int, default=5000, help="items per case (default 5000)")
    ap.add_argument("--capacity", type=int, default=500, help="bin capacity (default 500)")

    # Weibull params
    ap.add_argument("--shape-k", type=float, default=1.5)
    ap.add_argument("--scale-lambda", type=float, default=1.0)
    ap.add_argument("--clip-q", type=float, default=0.995)

    ap.add_argument("--force", action="store_true", help="overwrite if out exists")
    args = ap.parse_args()

    out_path = str(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if os.path.exists(out_path) and (not args.force):
        print(f"[gen_weibull_bpp] Output exists (use --force to overwrite): {out_path}")
        return

    n_cases = int(args.n_cases)
    n_items = int(args.n_items)
    C = int(args.capacity)

    # Make per-case seeds deterministic
    base_rng = np.random.default_rng(int(args.seed))
    case_seeds = base_rng.integers(low=0, high=2**31 - 1, size=n_cases, dtype=np.int64)

    t0 = time.perf_counter()

    # IMPORTANT: instances as a 2D numeric array: (n_cases, n_items)
    instances = np.empty((n_cases, n_items), dtype=np.int16)

    for cid in range(n_cases):
        rng = np.random.default_rng(int(case_seeds[cid]))
        instances[cid] = _weibull_case_int(
            rng,
            n_items=n_items,
            capacity=C,
            shape_k=float(args.shape_k),
            scale_lambda=float(args.scale_lambda),
            clip_q=float(args.clip_q),
        )

        if cid < 3:
            s = int(instances[cid].sum())
            lb = int((s + C - 1) // C)
            print(
                f"[case {cid}] n={n_items} C={C} sum={s} "
                f"mean={float(instances[cid].mean()):.3f} "
                f"min={int(instances[cid].min())} max={int(instances[cid].max())} lb={lb}"
            )

    t1 = time.perf_counter()

    meta: Dict[str, Any] = {
        "name": "Weibull-BPP-5k-C500",
        "created_at": _now_str(),
        "seed": int(args.seed),
        "n_cases": n_cases,
        "n_items": n_items,
        "capacity": C,
        "weibull_shape_k": float(args.shape_k),
        "weibull_scale_lambda": float(args.scale_lambda),
        "clip_quantile": float(args.clip_q),
        "seconds": float(t1 - t0),
        "format_version": 1,
        "instances_shape": tuple(instances.shape),
        "instances_dtype": str(instances.dtype),
    }

    payload = {
        "meta": meta,
        "instances": instances, 
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[gen_weibull_bpp] Saved: {out_path}")
    print(f"[gen_weibull_bpp] instances shape={instances.shape}, dtype={instances.dtype}, C={C}")


if __name__ == "__main__":
    main()
