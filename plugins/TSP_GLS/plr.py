# plugins/TSP_GLS/plr.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import os
import time

import numpy as np


# -----------------------------
# Profiling (phi(x))
# -----------------------------

_PROFILE_KEYS: Tuple[str, ...] = (
    "n_nodes",
    "mean_dist",
    "std_dist",
    "min_dist",
    "max_dist",
    "bbox_w",
    "bbox_h",
)

def compute_instance_profile(coords: Optional[np.ndarray], distmat: Optional[np.ndarray]) -> Dict[str, float]:
    """Compute a lightweight instance profile dict.

    This mirrors the features already computed in problem.py for logging:
      - n_nodes
      - mean/std/min/max distance statistics (upper triangle)
      - bounding box width/height from coordinates
    """
    try:
        n = int(distmat.shape[0]) if distmat is not None else 0

        mean_d = std_d = min_d = max_d = 0.0
        if n > 1 and distmat is not None:
            dm = np.asarray(distmat, dtype=float)
            iu = np.triu_indices(n, k=1)
            dvals = dm[iu]
            dvals = dvals[np.isfinite(dvals)]
            if dvals.size > 0:
                mean_d = float(np.mean(dvals))
                std_d = float(np.std(dvals))
                min_d = float(np.min(dvals))
                max_d = float(np.max(dvals))

        bbox_w = bbox_h = 0.0
        if coords is not None:
            arr = np.asarray(coords, dtype=float)
            if arr.ndim == 2 and arr.shape[0] > 0:
                xs = arr[:, 0]
                ys = arr[:, 1] if arr.shape[1] > 1 else xs
                bbox_w = float(xs.max() - xs.min())
                bbox_h = float(ys.max() - ys.min())

        return {
            "n_nodes": float(n),
            "mean_dist": float(mean_d),
            "std_dist": float(std_d),
            "min_dist": float(min_d),
            "max_dist": float(max_d),
            "bbox_w": float(bbox_w),
            "bbox_h": float(bbox_h),
        }
    except Exception:
        return {k: 0.0 for k in _PROFILE_KEYS}


def profile_to_vector(profile: Dict[str, float], normalize: bool = True) -> np.ndarray:
    """Convert profile dict to a fixed-order numeric vector.

    If normalize=True, apply a simple, stable normalization to reduce scale issues:
      - n_nodes -> log1p
      - distance stats -> log1p
      - bbox -> log1p
    """
    vec = np.array([float(profile.get(k, 0.0)) for k in _PROFILE_KEYS], dtype=np.float64)
    if not normalize:
        return vec

    out = vec.copy()
    # log1p for positive-ish features
    for i in range(out.size):
        out[i] = math.log1p(max(0.0, float(out[i])))
    return out


# -----------------------------
# Grouping
# -----------------------------

def bucket_group_id(profile: Dict[str, float]) -> str:
    """A deterministic grouping policy without fitting.

    Groups are defined by:
      - node-count bucket
      - mean-distance bucket (coarse)
      - bbox aspect bucket

    This is intentionally simple and stable, useful when you don't want k-means.
    """
    n = float(profile.get("n_nodes", 0.0))
    mean_d = float(profile.get("mean_dist", 0.0))
    bw = float(profile.get("bbox_w", 0.0))
    bh = float(profile.get("bbox_h", 0.0))

    # node buckets
    if n <= 50:
        n_b = "N<=50"
    elif n <= 100:
        n_b = "50<N<=100"
    elif n <= 200:
        n_b = "100<N<=200"
    else:
        n_b = "N>200"

    # distance bucket (log scale)
    md = math.log1p(max(0.0, mean_d))
    if md <= 2.0:
        d_b = "D:small"
    elif md <= 4.0:
        d_b = "D:mid"
    else:
        d_b = "D:large"

    # aspect
    ar = (bw + 1e-9) / (bh + 1e-9)
    if ar < 0.67:
        a_b = "AR:tall"
    elif ar > 1.5:
        a_b = "AR:wide"
    else:
        a_b = "AR:mid"

    return f"{n_b}|{d_b}|{a_b}"


def _kmeans_fit(X: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight numpy-only k-means.

    Returns:
      centers: (k, d)
      labels:  (n,)
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    k = max(1, min(int(k), n))

    # init: choose k random points
    idx = rng.choice(n, size=k, replace=False)
    centers = X[idx].copy()

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max(1, int(iters))):
        # assign
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int64)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        # update
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                # re-seed an empty cluster
                centers[j] = X[rng.randint(0, n)]
            else:
                centers[j] = X[mask].mean(axis=0)

    return centers, labels


@dataclass
class PLRGrouping:
    """Stores grouping prototypes for PLR retrieval."""
    mode: str = "bucket"  # "bucket" | "kmeans"
    normalize: bool = True

    # k-means params
    k: int = 8
    seed: int = 0
    iters: int = 30

    # learned prototypes for kmeans
    centers: Optional[np.ndarray] = None  # (k, d)

    def fit(self, profiles: List[Dict[str, float]]) -> List[str]:
        """Fit grouping if needed and return group ids for training instances."""
        if self.mode == "bucket":
            return [bucket_group_id(p) for p in profiles]

        # kmeans
        X = np.stack([profile_to_vector(p, normalize=self.normalize) for p in profiles], axis=0)
        centers, labels = _kmeans_fit(X, k=self.k, iters=self.iters, seed=self.seed)
        self.centers = centers
        return [f"G{int(i)}" for i in labels.tolist()]

    def assign(self, profile: Dict[str, float]) -> str:
        """Assign a single instance to a group."""
        if self.mode == "bucket":
            return bucket_group_id(profile)

        if self.centers is None or not isinstance(self.centers, np.ndarray):
            # fallback if not fitted
            return bucket_group_id(profile)

        x = profile_to_vector(profile, normalize=self.normalize)[None, :]
        dists = ((x - self.centers[None, :, :]) ** 2).sum(axis=2)[0]
        gid = int(np.argmin(dists))
        return f"G{gid}"


# -----------------------------
# Archive (group-wise top-k)
# -----------------------------

@dataclass
class SolverEntry:
    """One archived solver entry."""
    score: float
    payload: Dict[str, Any]  # e.g., {"code":..., "gls_spec":..., "name":...}
    meta: Dict[str, Any]
    created_at: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "payload": self.payload,
            "meta": self.meta,
            "created_at": float(self.created_at),
        }

    @staticmethod
    def from_json(js: Dict[str, Any]) -> "SolverEntry":
        return SolverEntry(
            score=float(js.get("score", 1e10)),
            payload=dict(js.get("payload", {}) or {}),
            meta=dict(js.get("meta", {}) or {}),
            created_at=float(js.get("created_at", 0.0)),
        )


class PLRArchive:
    """Group-wise top-k archive + optional global archive."""
    def __init__(self, top_k: int = 5, keep_global: bool = True) -> None:
        self.top_k = max(1, int(top_k))
        self.keep_global = bool(keep_global)
        self.group_archives: Dict[str, List[SolverEntry]] = {}
        self.global_archive: List[SolverEntry] = []

    def _insert_topk(self, arr: List[SolverEntry], entry: SolverEntry) -> List[SolverEntry]:
        arr2 = list(arr) + [entry]
        arr2.sort(key=lambda e: float(e.score))
        return arr2[: self.top_k]

    def update(self, group_id: str, score: float, payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        """Update the archive with a new evaluated solver."""
        entry = SolverEntry(
            score=float(score),
            payload=dict(payload),
            meta=dict(meta or {}),
            created_at=time.time(),
        )
        gid = str(group_id)

        self.group_archives[gid] = self._insert_topk(self.group_archives.get(gid, []), entry)
        if self.keep_global:
            self.global_archive = self._insert_topk(self.global_archive, entry)

    def retrieve(self, group_id: str, fallback_global: bool = True) -> Optional[SolverEntry]:
        """Retrieve the best archived solver for a group (or fallback global)."""
        gid = str(group_id)
        lst = self.group_archives.get(gid, [])
        if lst:
            return lst[0]
        if fallback_global and self.keep_global and self.global_archive:
            return self.global_archive[0]
        return None

    def to_json(self) -> Dict[str, Any]:
        return {
            "top_k": int(self.top_k),
            "keep_global": bool(self.keep_global),
            "group_archives": {gid: [e.to_json() for e in lst] for gid, lst in self.group_archives.items()},
            "global_archive": [e.to_json() for e in self.global_archive],
        }

    @staticmethod
    def from_json(js: Dict[str, Any]) -> "PLRArchive":
        arch = PLRArchive(top_k=int(js.get("top_k", 5)), keep_global=bool(js.get("keep_global", True)))
        ga = js.get("group_archives", {}) or {}
        for gid, lst in ga.items():
            arch.group_archives[str(gid)] = [SolverEntry.from_json(e) for e in (lst or [])]
        arch.global_archive = [SolverEntry.from_json(e) for e in (js.get("global_archive", []) or [])]
        return arch

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "PLRArchive":
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        return PLRArchive.from_json(js)


# -----------------------------
# Convenience: end-to-end PLR pack
# -----------------------------

@dataclass
class PLRPack:
    """A single container holding grouping + archive."""
    grouping: PLRGrouping
    archive: PLRArchive

    def assign_group(self, coords: Optional[np.ndarray], distmat: Optional[np.ndarray]) -> str:
        prof = compute_instance_profile(coords, distmat)
        return self.grouping.assign(prof)

    def warm_start_solver(self, coords: Optional[np.ndarray], distmat: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Return the best archived solver payload for the instance group."""
        gid = self.assign_group(coords, distmat)
        ent = self.archive.retrieve(gid, fallback_global=True)
        return None if ent is None else dict(ent.payload)
