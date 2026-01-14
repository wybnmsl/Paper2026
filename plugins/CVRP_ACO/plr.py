from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils.utils import profile_instance


@dataclass
class PLRConfig:
    """
    Profiled Library Retrieval (PLR) config.

    PLR maintains:
      - a global archive of top-k solver specs
      - group-wise archives for instances grouped by profile phi(x)
    """
    enabled: bool = True

    # grouping
    num_groups: int = 8
    seed: int = 0
    kmeans_iters: int = 25

    # archive
    top_k_per_group: int = 5
    top_k_global: int = 10

    # persistence
    save_dir: str = "artifacts/plr"
    save_name: str = "plr_cvrp_aco.json"

    def ensure_dirs(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def save_path(self) -> str:
        return os.path.join(self.save_dir, self.save_name)


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint_spec(spec_dict: Dict[str, Any]) -> str:
    h = hashlib.sha1(_stable_json(spec_dict).encode("utf-8")).hexdigest()
    return h


class PLRLibrary:
    """
    PLR for DASH plugins (CVRP-ACO version).

    Paper-aligned behavior:
      - Profiling: compute phi(x) by profile_instance(instance)
      - Grouping: assign each instance to a group by nearest centroid (k-means centroids)
      - Archiving: maintain top-k solver specs per group (and a global archive) during evolution
      - Retrieval: warm-start test-time runs by retrieving the best archived spec for the matched group

    Notes
    - This implementation is lightweight (numpy only), so it can run anywhere.
    - The "score" used for ranking prefers higher tLDR_traj when available, otherwise uses -gap%.
    """

    def __init__(self, cfg: PLRConfig):
        self.cfg = cfg
        self.cfg.ensure_dirs()
        self.rng = np.random.default_rng(int(cfg.seed))

        self.centroids: Optional[np.ndarray] = None  # (G, d)
        self.group_archives: List[List[Dict[str, Any]]] = [[] for _ in range(int(cfg.num_groups))]
        self.global_archive: List[Dict[str, Any]] = []

    # ---------------------------- grouping ----------------------------

    def _kmeans_fit(self, X: np.ndarray, k: int) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        kk = max(1, min(int(k), n))

        # init centroids by sampling points
        idx = self.rng.choice(n, size=kk, replace=False)
        C = X[idx].copy()

        for _ in range(int(self.cfg.kmeans_iters)):
            # assign
            dist2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=-1)  # (n,kk)
            labels = np.argmin(dist2, axis=1)

            # update
            C_new = C.copy()
            for g in range(kk):
                mask = labels == g
                if np.any(mask):
                    C_new[g] = X[mask].mean(axis=0)
                else:
                    # re-seed empty cluster
                    C_new[g] = X[int(self.rng.integers(0, n))]
            if np.max(np.abs(C_new - C)) < 1e-6:
                C = C_new
                break
            C = C_new

        # If requested groups > available, pad by repeating
        if kk < int(k):
            pad = np.repeat(C[-1:, :], repeats=(int(k) - kk), axis=0)
            C = np.concatenate([C, pad], axis=0)

        return C.astype(np.float32)

    def fit_groups(self, instances: List[Any]) -> None:
        """
        Fit group centroids using profiles from a list of instances.
        Use this once before evolution (or periodically) if you want stable grouping.
        """
        if not instances:
            return
        X = np.stack([profile_instance(inst) for inst in instances], axis=0).astype(np.float32)
        self.centroids = self._kmeans_fit(X, int(self.cfg.num_groups))
        self.group_archives = [[] for _ in range(int(self.cfg.num_groups))]

    def _assign_group_by_phi(self, phi: np.ndarray) -> int:
        G = int(self.cfg.num_groups)
        if self.centroids is None:
            # lazily initialize centroids using this instance (repeat)
            c0 = np.asarray(phi, dtype=np.float32)[None, :]
            self.centroids = np.repeat(c0, repeats=G, axis=0)
            return 0

        C = np.asarray(self.centroids, dtype=np.float32)
        x = np.asarray(phi, dtype=np.float32).reshape(1, -1)
        dist2 = np.sum((C - x) ** 2, axis=1)
        gid = int(np.argmin(dist2))
        return max(0, min(G - 1, gid))

    def assign_group(self, instance: Any) -> int:
        phi = profile_instance(instance)
        return self._assign_group_by_phi(phi)

    # ---------------------------- archiving ----------------------------

    @staticmethod
    def _score_from_meta(meta: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Return (score, aux) for ranking.
        Prefer larger tLDR_traj (higher is better).
        Fall back to -gap% (lower gap => higher score).
        """
        aux: Dict[str, Any] = {}
        tldr = meta.get("tLDR_traj", None)
        gap = meta.get("gap", None)
        if isinstance(tldr, (int, float)) and np.isfinite(float(tldr)):
            aux["score_source"] = "tLDR_traj"
            return float(tldr), aux
        if isinstance(gap, (int, float)) and np.isfinite(float(gap)):
            aux["score_source"] = "neg_gap"
            return -float(gap), aux
        aux["score_source"] = "unknown"
        return float("-inf"), aux

    def _push_topk(self, lst: List[Dict[str, Any]], item: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        # de-duplicate by fingerprint, keep the better score
        fp = item.get("fingerprint", "")
        merged: Dict[str, Dict[str, Any]] = {x.get("fingerprint", ""): x for x in lst}
        if fp in merged:
            if float(item.get("score", -1e18)) > float(merged[fp].get("score", -1e18)):
                merged[fp] = item
        else:
            merged[fp] = item

        out = list(merged.values())
        out.sort(key=lambda x: float(x.get("score", -1e18)), reverse=True)
        return out[: max(0, int(k))]

    def update_from_run(
        self,
        instance: Any,
        spec_dict: Dict[str, Any],
        meta: Dict[str, Any],
        *,
        tag: str = "",
    ) -> None:
        """
        Update group-wise archive and global archive from a finished solver run.
        """
        if not self.cfg.enabled:
            return

        gid = self.assign_group(instance)
        score, aux = self._score_from_meta(meta)

        entry = {
            "fingerprint": _fingerprint_spec(spec_dict),
            "spec": spec_dict,
            "score": float(score),
            "tag": str(tag),
            "group_id": int(gid),
            "aux": {**aux, "gap": meta.get("gap", None), "tLDR_traj": meta.get("tLDR_traj", None)},
        }

        self.group_archives[gid] = self._push_topk(self.group_archives[gid], entry, int(self.cfg.top_k_per_group))
        self.global_archive = self._push_topk(self.global_archive, entry, int(self.cfg.top_k_global))

    def retrieve(self, instance: Any, *, allow_global_fallback: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve the best archived spec for the instance's group.
        Returns a spec dict, or None if no archive exists.
        """
        if not self.cfg.enabled:
            return None

        gid = self.assign_group(instance)
        group = self.group_archives[gid] if 0 <= gid < len(self.group_archives) else []
        if group:
            return dict(group[0]["spec"])

        if allow_global_fallback and self.global_archive:
            return dict(self.global_archive[0]["spec"])

        return None

    # ---------------------------- persistence ----------------------------

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.cfg.save_path
        obj = {
            "cfg": asdict(self.cfg),
            "centroids": self.centroids.tolist() if self.centroids is not None else None,
            "group_archives": self.group_archives,
            "global_archive": self.global_archive,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
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
        # ensure size
        if len(lib.group_archives) != int(cfg.num_groups):
            lib.group_archives = (lib.group_archives + [[] for _ in range(int(cfg.num_groups))])[: int(cfg.num_groups)]
        return lib
