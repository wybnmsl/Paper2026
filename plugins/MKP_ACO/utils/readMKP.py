# utils/readMKP.py
# -*- coding: utf-8 -*-
"""
Read MKP instances in OR-Library mknapcb format (e.g., mknapcb1.txt),
and best-known / LP-relaxation values from mkcbres.txt.

mkcbres.txt structure:
- A table: "Problem Name  Best Feasible Solution Value"
- A table: "Problem Name  LP optimal" with scientific notation values (e.g. 2.458e+04).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import os
import re
import numpy as np


@dataclass(frozen=True)
class MKPInstance:
    n: int
    m: int
    profits: np.ndarray          # (n,), int64
    weights: np.ndarray          # (m,n), int64
    capacities: np.ndarray       # (m,), int64

    # from dataset header (often 0)
    opt: int = 0

    # from mkcbres.txt
    best_known: int = 0          # best feasible (best-known)
    lp_ub: float = 0.0           # LP relaxation optimal (upper bound)

    name: str = ""               # e.g. "5.100-00"


def _read_all_ints_fast(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    arr = np.fromstring(text, dtype=np.int64, sep=" ")
    if arr.size > 0:
        return arr.tolist()

    toks = text.split()
    out: List[int] = []
    for t in toks:
        try:
            out.append(int(t))
        except ValueError:
            continue
    return out


_KEY_PAT = re.compile(r"(\d+)\.(\d+)-(\d+)")
# float regex that supports scientific notation like 2.458e+04
_NUM_SCI_PAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def read_mkcbres(path: str) -> Dict[str, Dict[str, float]]:
    """
    Parse mkcbres.txt into:
      key -> {"best_known": int, "lp_ub": float}

    We explicitly track which table we are in:
      - after header containing "Best Feasible Solution Value" => BEST section
      - after header containing "LP optimal" => LP section

    This avoids accidental mode flips and handles scientific notation correctly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out: Dict[str, Dict[str, float]] = {}

    section: Optional[str] = None  # "best" | "lp" | None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            low = line.lower()

            # section switching
            if "best feasible solution value" in low:
                section = "best"
                continue
            if low.startswith("problem name") and "lp optimal" in low:
                section = "lp"
                continue
            # sometimes appears as "Problem Name     LP optimal"
            if "lp optimal" in low and low.startswith("problem"):
                section = "lp"
                continue

            m = _KEY_PAT.search(line)
            if not m:
                continue
            key = m.group(0)

            # parse numbers after the key
            rest = line[m.end():]
            nums = _NUM_SCI_PAT.findall(rest)
            if not nums:
                continue

            if key not in out:
                out[key] = {"best_known": 0.0, "lp_ub": 0.0}

            # In mkcbres:
            # - best table lines: key + integer
            # - lp table lines:   key + sci float
            # But be robust: if section unknown, infer by presence of exponent or decimal.
            v = float(nums[0])

            if section == "best":
                out[key]["best_known"] = max(out[key]["best_known"], v)
            elif section == "lp":
                out[key]["lp_ub"] = max(out[key]["lp_ub"], v)
            else:
                token = nums[0]
                if ("e" in token.lower()) or ("." in token):
                    out[key]["lp_ub"] = max(out[key]["lp_ub"], v)
                else:
                    out[key]["best_known"] = max(out[key]["best_known"], v)

    # cast best_known to int-like (still keep float dict for safety)
    return out


def _parse_instances_from_tokens(
    tokens: Sequence[int],
    res_map: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[MKPInstance]:
    if len(tokens) < 1:
        raise ValueError("Empty token stream; cannot parse MKP dataset.")

    idx = 0
    K = int(tokens[idx]); idx += 1
    if K <= 0:
        raise ValueError(f"Invalid K={K} in dataset.")

    instances: List[MKPInstance] = []
    for k in range(K):
        if idx + 3 > len(tokens):
            raise ValueError(f"Unexpected EOF while reading header of instance {k}.")
        n = int(tokens[idx]); m = int(tokens[idx + 1]); opt = int(tokens[idx + 2])
        idx += 3

        if n <= 0 or m <= 0:
            raise ValueError(f"Invalid (n,m)=({n},{m}) in instance {k}.")

        if idx + n > len(tokens):
            raise ValueError(f"Unexpected EOF while reading profits of instance {k}.")
        profits = np.array(tokens[idx:idx + n], dtype=np.int64)
        idx += n

        need = m * n
        if idx + need > len(tokens):
            raise ValueError(f"Unexpected EOF while reading weights of instance {k}.")
        weights_flat = np.array(tokens[idx:idx + need], dtype=np.int64)
        idx += need
        weights = weights_flat.reshape(m, n)

        if idx + m > len(tokens):
            raise ValueError(f"Unexpected EOF while reading capacities of instance {k}.")
        capacities = np.array(tokens[idx:idx + m], dtype=np.int64)
        idx += m

        name = f"{m}.{n}-{k:02d}"

        best_known = 0
        lp_ub = 0.0
        if res_map is not None:
            rec = res_map.get(name)
            if rec:
                best_known = int(round(float(rec.get("best_known", 0.0))))
                lp_ub = float(rec.get("lp_ub", 0.0))

        instances.append(MKPInstance(
            n=n, m=m,
            profits=profits, weights=weights, capacities=capacities,
            opt=opt, best_known=best_known, lp_ub=lp_ub,
            name=name
        ))

    return instances


def read_mknapcb(path: str, *, res_path: Optional[str] = None) -> List[MKPInstance]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    res_map = None
    if res_path is not None and os.path.exists(res_path):
        res_map = read_mkcbres(res_path)

    tokens = _read_all_ints_fast(path)
    return _parse_instances_from_tokens(tokens, res_map=res_map)


def load_instance(path: str, instance_index: int = 0, *, res_path: Optional[str] = None) -> MKPInstance:
    instances = read_mknapcb(path, res_path=res_path)
    if not (0 <= instance_index < len(instances)):
        raise IndexError(f"instance_index {instance_index} out of range (0..{len(instances)-1})")
    return instances[instance_index]


def sanity_check_instance(inst: MKPInstance) -> None:
    if inst.profits.shape != (inst.n,):
        raise ValueError(f"profits shape mismatch: {inst.profits.shape} vs (n,)={(inst.n,)}")
    if inst.weights.shape != (inst.m, inst.n):
        raise ValueError(f"weights shape mismatch: {inst.weights.shape} vs (m,n)={(inst.m, inst.n)}")
    if inst.capacities.shape != (inst.m,):
        raise ValueError(f"capacities shape mismatch: {inst.capacities.shape} vs (m,)={(inst.m,)}")
    if np.any(inst.capacities <= 0):
        raise ValueError("Non-positive capacity found.")
    if np.any(inst.weights < 0) or np.any(inst.profits < 0):
        raise ValueError("Negative weights/profits found (unexpected for mknapcb).")
