from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class BPPDataset:
    meta: Dict[str, Any]
    instances: List[np.ndarray]

    @property
    def capacity(self) -> int:
        return int(self.meta.get("capacity", 0))

    @property
    def n_instances(self) -> int:
        return int(self.meta.get("n_instances", len(self.instances)))

    @property
    def n_items(self) -> int:
        return int(self.meta.get("n_items", len(self.instances[0]) if self.instances else 0))

    def get_case(self, case_id: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        if not self.instances:
            raise ValueError("Empty dataset: no instances found.")
        n = len(self.instances)
        idx = int(case_id)
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx >= n:
            raise IndexError(f"case_id out of range: {case_id}, dataset has {n} instances.")
        items = np.asarray(self.instances[idx])
        if items.ndim != 1:
            raise ValueError(f"Each instance must be 1D array, got shape={items.shape}")
        case_meta = {
            "case_id": idx,
            "capacity": self.capacity,
            "n_items": int(items.shape[0]),
            "dataset_meta": self.meta,
        }
        return items, self.capacity, case_meta


def load_bpp_pickle(path: str | Path) -> BPPDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "meta" in obj and "instances" in obj:
        meta = dict(obj["meta"])
        instances = obj["instances"]
    else:
        raise ValueError("Unknown dataset format. Expect dict with keys: 'meta' and 'instances'.")

    arr = np.asarray(instances)
    inst_list: List[np.ndarray] = []

    # Support both list-of-1D and 2D array formats.
    if arr.ndim == 2:
        for i in range(arr.shape[0]):
            inst_list.append(np.asarray(arr[i]))
    else:
        for x in instances:
            inst_list.append(np.asarray(x))

    for a in inst_list:
        if a.ndim != 1:
            raise ValueError(f"Each instance must be 1D array, got shape={a.shape}")

    return BPPDataset(meta=meta, instances=inst_list)
