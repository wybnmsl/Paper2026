from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional

import numpy as np

from .goa.spec import GOASpec, to_json as spec_to_json, from_json as spec_from_json
from .problem import BPPGOAProblem
from .plr import PLRConfig, PLRLibrary, bpp_profile
from .utils.read_bpp_pickle import load_bpp_pickle


def _parse_case_ids(arg: str, n_total: int) -> List[int]:
    """
    Accept:
      - "0,1,2"
      - "0-99"
      - a text file with one int per line
      - "all"
    """
    s = arg.strip()
    if s.lower() == "all":
        return list(range(n_total))

    if os.path.isfile(s):
        ids: List[int] = []
        with open(s, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                ids.append(int(ln))
        return ids

    if "-" in s and s.replace("-", "").isdigit():
        a, b = s.split("-", 1)
        lo, hi = int(a), int(b)
        lo = max(0, lo)
        hi = min(n_total - 1, hi)
        return list(range(lo, hi + 1))

    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Run BPP-GOA plugin in batch (optionally with PLR warm-start).")
    ap.add_argument("--pkl", type=str, required=True, help="BPP dataset pickle path")
    ap.add_argument("--cases", type=str, default="0-9", help='Case ids: "0,1,2" | "0-99" | file | "all"')
    ap.add_argument("--time", type=float, default=10.0, help="Time budget (seconds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", type=str, default="test", choices=["train", "test"], help="Whether to update PLR")

    # Spec overrides
    ap.add_argument("--strict-online", action="store_true", help="Force strict_online=True")
    ap.add_argument("--hybrid", action="store_true", help="Force strict_online=False (hybrid/offline)")
    ap.add_argument("--spec-json", type=str, default="", help="Optional JSON string/file to override GOASpec fields")

    # PLR
    ap.add_argument("--plr", action="store_true", help="Enable PLR retrieval + (optional) library update")
    ap.add_argument("--plr-path", type=str, default="", help="Path to PLR library JSON (default uses cfg.save_dir/name)")
    ap.add_argument("--plr-fit", action="store_true", help="Fit centroids from the selected cases before running")
    ap.add_argument("--plr-groups", type=int, default=8, help="Number of groups for PLR")
    ap.add_argument("--plr-k", type=int, default=5, help="Top-k per group/global archive size")
    ap.add_argument("--out", type=str, default="artifacts/bpp_goa_runs.jsonl", help="Output jsonl")

    args = ap.parse_args()

    ds = load_bpp_pickle(args.pkl)
    n_total = len(ds.instances)
    case_ids = _parse_case_ids(args.cases, n_total)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Base spec
    spec = GOASpec(time_limit_s=float(args.time), seed=int(args.seed))
    if args.strict_online and args.hybrid:
        raise SystemExit("Do not set both --strict-online and --hybrid.")
    if args.strict_online:
        spec.strict_online = True
    if args.hybrid:
        spec.strict_online = False

    # Apply external spec override (JSON string or file path)
    if args.spec_json.strip():
        s = args.spec_json.strip()
        if os.path.isfile(s):
            with open(s, "r", encoding="utf-8") as f:
                override = json.load(f)
        else:
            override = json.loads(s)
        spec = spec_from_json({**spec_to_json(spec), **dict(override)})

    # PLR init (optional external load + optional fit)
    plr_cfg = None
    plr_lib: Optional[PLRLibrary] = None
    if args.plr:
        plr_cfg = PLRConfig(
            enabled=True,
            num_groups=int(args.plr_groups),
            archive_k=int(args.plr_k),
            seed=int(args.seed),
        )
        if args.plr_path.strip():
            plr_cfg.save_dir = os.path.dirname(args.plr_path.strip()) or plr_cfg.save_dir
            plr_cfg.save_name = os.path.basename(args.plr_path.strip()) or plr_cfg.save_name

        path = plr_cfg.path()
        if os.path.exists(path):
            plr_lib = PLRLibrary.load(path)
        else:
            plr_lib = PLRLibrary(plr_cfg)

        if args.plr_fit:
            profiles = [bpp_profile(ds.instances[cid], ds.capacity) for cid in case_ids]
            plr_lib.fit_groups(profiles)

    # Use problem wrapper (computes tLDR and updates PLR in train mode)
    prob = BPPGOAProblem(
        dataset_pkl=args.pkl,
        case_ids=case_ids,
        plr_cfg=plr_cfg if args.plr else PLRConfig(enabled=False),
        mode=str(args.mode),
        load_plr_if_exists=True,
    )

    # If we built/loaded PLR explicitly above, plug it in (ensures consistent state)
    if plr_lib is not None:
        prob.plr = plr_lib
        prob.plr_cfg = plr_cfg  # type: ignore[assignment]

    with open(args.out, "w", encoding="utf-8") as fout:
        for cid in case_ids:
            items, C, _ = prob.get_case(cid)

            # PLR warm-start: retrieve spec payload and merge into current spec (test-time behavior)
            warm_payload = None
            gid = -1
            if prob.plr is not None and prob.plr_cfg.enabled:
                gid, payloads = prob.plr_retrieve_warm_start(items, C, topk=1)
                if payloads:
                    warm_payload = payloads[0].get("goa_spec", None)

            run_spec = spec
            if isinstance(warm_payload, dict):
                # Merge warm-start spec with current spec; keep time/seed from CLI
                merged = {**warm_payload, "time_limit_s": float(args.time), "seed": int(args.seed)}
                run_spec = spec_from_json(merged)

            t0 = time.time()
            out = prob.evaluate_one(
                cid,
                spec=run_spec,
                priority_fn=None,  # LLM-generated priority can be wired here later
                payload={"warm_start_group": int(gid)},
            )
            out["wall_time_s"] = float(time.time() - t0)
            out["warm_start_group"] = int(gid)
            out["warm_started"] = bool(warm_payload is not None)

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    # Save PLR if enabled
    if prob.plr is not None and prob.plr_cfg.enabled:
        path = prob.plr_save()
        print(f"[PLR] saved -> {path}")

    print(f"[DONE] wrote -> {args.out}")


if __name__ == "__main__":
    main()
