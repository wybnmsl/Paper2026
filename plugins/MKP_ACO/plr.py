from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils.utils import profile_instance


@dataclass
class PLRConfig:
    enabled: bool = True
    num_groups: int = 8
    archive_k: int = 5
    save_dir: str = "outputs/plr_mkp"
    seed: int = 0

    # how to update: keep top-k by a scalar score (default: tLDR_traj, larger is better)
    score_key: str = "tLDR_traj"

    def ensure_dirs(self) -> None:
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)


class PLRLibrary:
    """
    Profiled Library Retrieval (PLR) for MKP:
      - compute phi(x) by profile_instance()
      - cluster into G groups (k-means)
      - maintain top-k archive per group based on a score (e.g., tLDR_traj)
      - support retrieve for warm-start
    """

    def __init__(self, cfg: PLRConfig):
        self.cfg = cfg
        self.cfg.ensure_dirs()
        self.rng = np.random.default_rng(int(cfg.seed))
        self.centroids: Optional[np.ndarray] = None  # (G, d)
        self.group_archives: List[List[Dict[str, Any]]] = [[] for _ in range(int(cfg.num_groups))]
        self.global_archive: List[Dict[str, Any]] = []

    def _kmeans_fit(self, X: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
        n = X.shape[0]
        k = min(k, n)
        # init centroids by random sample
        idx = self.rng.choice(n, size=k, replace=False)
        C = X[idx].copy()
        for _ in range(iters):
            # assign
            dist2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            a = dist2.argmin(axis=1)
            # update
            C_new = C.copy()
            for j in range(k):
                mask = (a == j)
                if mask.any():
                    C_new[j] = X[mask].mean(axis=0)
            if np.allclose(C_new, C):
                break
            C = C_new
        return C

    def fit_groups(self, instances: List[Any]) -> None:
        """
        Fit group centroids using a list of instances (profiles only).
        """
        if not instances:
            return
        X = np.stack([profile_instance(inst) for inst in instances], axis=0).astype(np.float32)
        self.centroids = self._kmeans_fit(X, int(self.cfg.num_groups))
        # reset archives (optional)
        self.group_archives = [[] for _ in range(int(self.cfg.num_groups))]

    def _assign_group(self, inst: Any) -> int:
        if self.centroids is None:
            return 0
        x = profile_instance(inst).astype(np.float32)
        dist2 = ((self.centroids - x[None, :]) ** 2).sum(axis=1)
        return int(dist2.argmin())

    def update(self, inst: Any, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
        """
        Update archives with a new solver candidate evaluated on `inst`.
        payload: anything needed for warm-start (e.g., spec dict, code, params)
        meta: includes score_key (default tLDR_traj) and other metrics.
        """
        if not self.cfg.enabled:
            return
        gid = self._assign_group(inst)
        score = float(meta.get(self.cfg.score_key, float("-inf")))
        entry = {
            "score": score,
            "payload": payload,
            "meta": meta,
            "phi": profile_instance(inst).tolist(),
            "gid": gid,
        }

        def insert_topk(lst: List[Dict[str, Any]]) -> None:
            lst.append(entry)
            lst.sort(key=lambda e: float(e.get("score", float("-inf"))), reverse=True)
            del lst[int(self.cfg.archive_k):]

        insert_topk(self.group_archives[gid])
        insert_topk(self.global_archive)

    def retrieve(self, inst: Any, k: int = 1) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Retrieve top-k entries from the assigned group; fallback to global archive.
        Returns (group_id, entries)
        """
        gid = self._assign_group(inst)
        group_list = self.group_archives[gid]
        if group_list:
            return gid, group_list[:k]
        return gid, self.global_archive[:k]

    def save(self, filename: str = "plr_library.json") -> str:
        self.cfg.ensure_dirs()
        path = os.path.join(self.cfg.save_dir, filename)
        obj = {
            "cfg": asdict(self.cfg),
            "centroids": self.centroids.tolist() if self.centroids is not None else None,
            "group_archives": self.group_archives,
            "global_archive": self.global_archive,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        return path

    @staticmethod
    def load(path: str) -> "PLRLibrary":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cfg = PLRConfig(**obj.get("cfg", {}))
        lib = PLRLibrary(cfg)
        C = obj.get("centroids", None)
        lib.centroids = np.asarray(C, dtype=np.float32) if C is not None else None
        lib.group_archives = obj.get("group_archives", [[] for _ in range(int(cfg.num_groups))])
        lib.global_archive = obj.get("global_archive", [])
        return lib
