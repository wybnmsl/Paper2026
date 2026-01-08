from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils.utils import ensure_dir, stable_json_dumps


@dataclass
class PLRConfig:
    enabled: bool = False
    num_groups: int = 8
    archive_k: int = 5
    m_per_group: int = 2
    seed: int = 0
    save_dir: str = "outputs/plr_bpp_goa"
    save_name: str = "plr_library.json"

    # assignment / fitting
    kmeans_iters: int = 30
    min_points_per_group: int = 5

    # scoring
    primary_metric: str = "tldr"  # "tldr" | "final_gap" | "final_bins"
    higher_is_better: bool = True  # for primary_metric

    def path(self) -> str:
        return str(Path(self.save_dir) / self.save_name)


def bpp_profile(items: np.ndarray, capacity: int) -> np.ndarray:
    """Instance profile for PLR (lightweight, fast, distribution-oriented).

    Features are normalized to be scale-insensitive and stable:
      0) n_items / 1e4
      1) mean(items) / C
      2) std(items) / C
      3) min(items) / C
      4) max(items) / C
      5) sum(items) / (n*C)  (fill ratio)
      6) lb_relax / n_items  (tightness proxy)
    """
    x = np.asarray(items, dtype=np.float64)
    C = float(max(int(capacity), 1))
    n = float(max(int(x.shape[0]), 1))
    s = float(np.sum(x))
    mu = float(np.mean(x))
    sd = float(np.std(x))
    mn = float(np.min(x))
    mx = float(np.max(x))

    lb = int((s + C - 1.0) // C)

    feats = np.array(
        [
            n / 1e4,
            mu / C,
            sd / C,
            mn / C,
            mx / C,
            (s / (n * C)),
            float(lb) / n,
        ],
        dtype=np.float64,
    )
    return feats


def _kmeans_fit(X: np.ndarray, k: int, *, seed: int, iters: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simple numpy k-means (Lloyd). Returns (centroids, labels)."""
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("Empty X for kmeans.")
    k = max(1, min(int(k), n))

    rng = np.random.default_rng(int(seed))
    init_ids = rng.choice(n, size=k, replace=False)
    C = X[init_ids].copy()

    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max(1, int(iters))):
        # assign
        d2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)  # (n,k)
        labels = np.argmin(d2, axis=1)

        # update
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                C[j] = np.mean(X[mask], axis=0)
            else:
                # re-seed empty cluster
                C[j] = X[int(rng.integers(0, n))]
    return C, labels


@dataclass
class LibraryEntry:
    score: float
    final_gap: float
    final_bins: int
    group_id: int
    payload: Dict[str, Any]
    profile: List[float]
    created_at: float

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "LibraryEntry":
        return LibraryEntry(
            score=float(obj["score"]),
            final_gap=float(obj.get("final_gap", 0.0)),
            final_bins=int(obj.get("final_bins", 0)),
            group_id=int(obj.get("group_id", -1)),
            payload=dict(obj.get("payload", {})),
            profile=list(obj.get("profile", [])),
            created_at=float(obj.get("created_at", 0.0)),
        )


class PLRLibrary:
    """Group-wise top-k archive + nearest-prototype retrieval."""

    def __init__(self, cfg: PLRConfig):
        self.cfg = cfg
        self.centroids: Optional[np.ndarray] = None  # (G, d)
        self.group_archives: Dict[int, List[LibraryEntry]] = {}
        self.global_archive: List[LibraryEntry] = []

    # -------------------
    # persistence
    # -------------------
    def save(self) -> str:
        if not self.cfg.enabled:
            return self.cfg.path()
        ensure_dir(self.cfg.save_dir)
        obj = {
            "cfg": asdict(self.cfg),
            "centroids": self.centroids.tolist() if self.centroids is not None else None,
            "group_archives": {
                str(g): [e.to_json() for e in lst] for g, lst in self.group_archives.items()
            },
            "global_archive": [e.to_json() for e in self.global_archive],
            "saved_at": time.time(),
        }
        p = Path(self.cfg.path())
        p.write_text(stable_json_dumps(obj, indent=2), encoding="utf-8")
        return str(p)

    @staticmethod
    def load(path: str) -> "PLRLibrary":
        p = Path(path)
        obj = json.loads(p.read_text(encoding="utf-8"))
        cfg = PLRConfig(**obj.get("cfg", {}))
        lib = PLRLibrary(cfg)
        c = obj.get("centroids", None)
        if c is not None:
            lib.centroids = np.asarray(c, dtype=np.float64)
        ga = obj.get("group_archives", {})
        lib.group_archives = {int(k): [LibraryEntry.from_json(x) for x in v] for k, v in ga.items()}
        lib.global_archive = [LibraryEntry.from_json(x) for x in obj.get("global_archive", [])]
        return lib

    # -------------------
    # fit/assign
    # -------------------
    def fit_groups(self, profiles: List[np.ndarray]) -> None:
        X = np.asarray(profiles, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("profiles must be a non-empty 2D array-like.")
        self.centroids, _ = _kmeans_fit(
            X,
            int(self.cfg.num_groups),
            seed=int(self.cfg.seed),
            iters=int(self.cfg.kmeans_iters),
        )

    def assign_group(self, profile: np.ndarray) -> int:
        if self.centroids is None:
            return -1
        p = np.asarray(profile, dtype=np.float64)
        d2 = np.sum((self.centroids - p[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    # -------------------
    # update/retrieve
    # -------------------
    def _better(self, a: LibraryEntry, b: LibraryEntry) -> bool:
        # True if a is better than b
        if self.cfg.primary_metric == "final_gap":
            return a.final_gap < b.final_gap
        if self.cfg.primary_metric == "final_bins":
            return a.final_bins < b.final_bins
        # default: tldr-style score
        if self.cfg.higher_is_better:
            return a.score > b.score
        return a.score < b.score

    def _insert_topk(self, lst: List[LibraryEntry], entry: LibraryEntry, k: int) -> List[LibraryEntry]:
        lst2 = lst + [entry]
        # sort best-first
        lst2.sort(key=lambda e: (e.score if self.cfg.higher_is_better else -e.score), reverse=True)
        # tie-break by final_gap then final_bins (lower better)
        lst2.sort(key=lambda e: (-(e.score) if self.cfg.higher_is_better else e.score, e.final_gap, e.final_bins))
        return lst2[: max(1, int(k))]

    def update(
        self,
        *,
        profile: np.ndarray,
        score: float,
        final_gap: float,
        final_bins: int,
        payload: Dict[str, Any],
        group_id: Optional[int] = None,
    ) -> None:
        if not self.cfg.enabled:
            return
        gid = int(group_id) if group_id is not None else self.assign_group(profile)

        e = LibraryEntry(
            score=float(score),
            final_gap=float(final_gap),
            final_bins=int(final_bins),
            group_id=int(gid),
            payload=dict(payload),
            profile=[float(x) for x in np.asarray(profile, dtype=np.float64).tolist()],
            created_at=time.time(),
        )

        self.global_archive = self._insert_topk(self.global_archive, e, int(self.cfg.archive_k))

        if gid not in self.group_archives:
            self.group_archives[gid] = []
        self.group_archives[gid] = self._insert_topk(self.group_archives[gid], e, int(self.cfg.archive_k))

    def retrieve(self, profile: np.ndarray, *, topk: int = 1) -> Tuple[int, List[LibraryEntry]]:
        if not self.cfg.enabled:
            return -1, []
        gid = self.assign_group(profile)
        cand = list(self.group_archives.get(gid, []))
        if not cand:
            cand = list(self.global_archive)
        return gid, cand[: max(1, int(topk))]
