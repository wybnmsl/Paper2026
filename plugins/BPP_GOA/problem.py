from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .goa import GOASpec, solve_bpp_goa, to_json as spec_to_json
from .plr import PLRConfig, PLRLibrary, bpp_profile
from .utils.read_bpp_pickle import load_bpp_pickle, BPPDataset
from .utils.utils import (
    make_logger,
    lb_relaxation,
    gap_percent,
    summarize_instance,
    compute_tldr_from_trace,
)


class BPPGOAProblem:
    """BPP task wrapper for DASH, with optional PLR (Profiled Library Retrieval).

    You can use this as the problem-side interface:
      - load dataset
      - run GOA solver on a case
      - compute trajectory-aware metrics (tLDR-style score from trace)
      - maintain a profiled solver library (group-wise archives) for warm-start
    """

    def __init__(
        self,
        *,
        dataset_pkl: str,
        case_ids: Optional[List[int]] = None,
        plr_cfg: Optional[Union[PLRConfig, Dict[str, Any]]] = None,
        mode: str = "train",
        logger=None,
        load_plr_if_exists: bool = True,
    ):
        self.ds: BPPDataset = load_bpp_pickle(dataset_pkl)
        self.dataset_pkl = str(dataset_pkl)
        self.mode = str(mode)

        if case_ids is None:
            case_ids = list(range(len(self.ds.instances)))
        self.case_ids = list(map(int, case_ids))

        self.logger = logger if logger is not None else make_logger("BPP_GOA", level=20)

        # PLR
        if plr_cfg is None:
            self.plr_cfg = PLRConfig(enabled=False)
        elif isinstance(plr_cfg, PLRConfig):
            self.plr_cfg = plr_cfg
        else:
            self.plr_cfg = PLRConfig(**dict(plr_cfg))

        self.plr: Optional[PLRLibrary] = None
        if self.plr_cfg.enabled:
            # try load existing library
            if load_plr_if_exists and os.path.exists(self.plr_cfg.path()):
                self.plr = PLRLibrary.load(self.plr_cfg.path())
                self.logger.info(f"[PLR] Loaded library: {self.plr_cfg.path()}")
            else:
                self.plr = PLRLibrary(self.plr_cfg)
                # Fit groups from current case pool profiles
                profiles = [bpp_profile(self.ds.instances[cid], self.ds.capacity) for cid in self.case_ids]
                self.plr.fit_groups(profiles)
                self.logger.info(f"[PLR] Fitted groups: G={self.plr_cfg.num_groups} from {len(profiles)} cases")

    @property
    def capacity(self) -> int:
        return int(self.ds.capacity)

    def get_case(self, case_id: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        return self.ds.get_case(case_id)

    # -----------------------------
    # Evaluation
    # -----------------------------
    def evaluate_one(
        self,
        case_id: int,
        *,
        spec: GOASpec,
        priority_fn=None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run solver on one case and optionally update PLR."""
        items, C, meta = self.get_case(case_id)

        lb = lb_relaxation(items, C)
        res = solve_bpp_goa(items, C, priority_fn=priority_fn, spec=spec)

        best_bins = int(res.best_bins)
        final_gap = float(gap_percent(best_bins, lb))

        # trace is (t, gap, bins)
        tldr = float(compute_tldr_from_trace(res.trace, time_limit_s=float(spec.time_limit_s)))

        out = {
            "case_id": int(case_id),
            "capacity": int(C),
            "lb": int(lb),
            "best_bins": int(best_bins),
            "final_gap": float(final_gap),
            "tldr": float(tldr),
            "trace": list(res.trace),
            "meta": meta,
            "spec": spec_to_json(spec),
            "instance_summary": summarize_instance(items, C),
        }

        if self.plr is not None and self.plr_cfg.enabled and self.mode == "train":
            prof = bpp_profile(items, C)
            gid = self.plr.assign_group(prof)
            pl = dict(payload or {})
            # Always store spec; caller can add code strings or other fields
            pl.setdefault("goa_spec", spec_to_json(spec))
            self.plr.update(
                profile=prof,
                score=float(tldr),
                final_gap=float(final_gap),
                final_bins=int(best_bins),
                payload=pl,
                group_id=int(gid),
            )
        return out

    def evaluate_batch(
        self,
        case_ids: Optional[List[int]] = None,
        *,
        spec: GOASpec,
        priority_fn=None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ids = self.case_ids if case_ids is None else list(map(int, case_ids))
        return [self.evaluate_one(cid, spec=spec, priority_fn=priority_fn, payload=payload) for cid in ids]

    # -----------------------------
    # PLR retrieve / save
    # -----------------------------
    def plr_retrieve_warm_start(
        self,
        items: np.ndarray,
        capacity: Optional[int] = None,
        *,
        topk: int = 1,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Return (group_id, [payload,...]) for the nearest group archive (fallback to global)."""
        if self.plr is None or not self.plr_cfg.enabled:
            return -1, []
        C = int(self.capacity if capacity is None else capacity)
        prof = bpp_profile(items, C)
        gid, entries = self.plr.retrieve(prof, topk=int(topk))
        return int(gid), [dict(e.payload) for e in entries]

    def plr_save(self) -> Optional[str]:
        if self.plr is None or not self.plr_cfg.enabled:
            return None
        path = self.plr.save()
        self.logger.info(f"[PLR] Saved library: {path}")
        return path
