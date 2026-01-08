"""Operators and constructive heuristics for BPP GOA solver.

This file is a cleaned version of your original operators module:
- removed legacy commented-out implementations
- kept the functional implementations unchanged
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

PriorityFn = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class Solution:
    """
    bins: list of bins, each bin is list of item indices (indices refer to the items array passed in)
    loads: loads for each bin
    """
    bins: List[List[int]]
    loads: np.ndarray  # shape (nbins,)


def _bin_load(items: np.ndarray, bin_items: Sequence[int]) -> float:
    return float(np.sum(items[np.array(bin_items, dtype=np.int64)])) if len(bin_items) > 0 else 0.0


def compute_loads(items: np.ndarray, bins: List[List[int]]) -> np.ndarray:
    loads = np.zeros(len(bins), dtype=np.float64)
    for b, lst in enumerate(bins):
        if lst:
            loads[b] = float(np.sum(items[np.array(lst, dtype=np.int64)]))
    return loads


def compact_bins(bins: List[List[int]]) -> List[List[int]]:
    return [b for b in bins if len(b) > 0]


def pack_ffd_or_bfd(
    items: np.ndarray,
    capacity: int,
    *,
    method: str = "bfd",
) -> Solution:
    """
    Greedy initialization.

    Supported methods:
      - "ffd": first-fit decreasing (sort by weight desc)
      - "bfd": best-fit decreasing  (sort by weight desc)
      - "online_ff": first-fit online (NO sorting; use given arrival order)
      - "online_bf": best-fit online  (NO sorting; use given arrival order)

    NOTE:
      - For online_* methods, the arrival order is the order of the `items` array.
        If you want random arrival, shuffle items BEFORE calling this function (see test script).
    """
    items = np.asarray(items)
    n = int(items.shape[0])

    if method in ("ffd", "bfd"):
        order = np.argsort(-items)  # decreasing by weight
    elif method in ("online_ff", "online_bf"):
        order = np.arange(n, dtype=np.int64)  # keep arrival order
    else:
        raise ValueError(f"Unknown init method: {method}")

    bins: List[List[int]] = []
    loads: List[float] = []

    for idx in order:
        i = int(idx)
        w = float(items[i])

        best_bin = -1
        if method in ("ffd", "online_ff"):
            # first feasible bin
            for b in range(len(bins)):
                if loads[b] + w <= capacity + 1e-9:
                    best_bin = b
                    break

        elif method in ("bfd", "online_bf"):
            # choose feasible bin that leaves minimal remaining capacity
            best_rem = None
            for b in range(len(bins)):
                rem = capacity - (loads[b] + w)
                if rem >= -1e-9:
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_bin = b

        if best_bin == -1:
            bins.append([i])
            loads.append(w)
        else:
            bins[best_bin].append(i)
            loads[best_bin] += w

    bins = compact_bins(bins)
    loads_arr = np.array(loads, dtype=np.float64)
    return Solution(bins=bins, loads=loads_arr)


def _choose_bin_for_item(
    items: np.ndarray,
    capacity: int,
    sol_bins: List[List[int]],
    sol_loads: np.ndarray,
    item_idx: int,
    priority_fn: Optional[PriorityFn],
    rng: np.random.Generator,
    cand_k: int,
) -> int:
    """
    Choose an existing bin id to place item_idx, or return -1 to open new bin.
    """
    w = float(items[item_idx])
    nb = len(sol_bins)
    if nb == 0:
        return -1

    remain = capacity - sol_loads
    feasible = np.where(remain + 1e-9 >= w)[0]
    if feasible.size == 0:
        return -1

    # candidate pruning
    feas_rem = remain[feasible]
    if priority_fn is not None:
        scores = priority_fn(w, feas_rem.astype(np.float64))
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape[0] != feasible.shape[0]:
            # fallback: best-fit
            scores = -np.abs(feas_rem - w)
    else:
        scores = -np.abs(feas_rem - w)  # best-fit-ish

    # take top-k by score
    if cand_k is not None and cand_k > 0 and feasible.size > cand_k:
        top_idx = np.argpartition(-scores, cand_k - 1)[:cand_k]
        feasible = feasible[top_idx]
        scores = scores[top_idx]

    # softmax sampling for diversity
    s = scores - np.max(scores)
    p = np.exp(np.clip(s, -50, 50))
    p_sum = float(np.sum(p))
    if p_sum <= 0:
        return int(feasible[int(np.argmax(scores))])
    p = p / p_sum
    return int(rng.choice(feasible, p=p))


def try_empty_one_bin(
    items: np.ndarray,
    capacity: int,
    sol: Solution,
    *,
    priority_fn: Optional[PriorityFn],
    rng: np.random.Generator,
    cand_k: int,
) -> bool:
    """
    Attempt to remove one bin by reinserting its items into other bins.
    Accept only if no new bin is created.
    """
    nb = len(sol.bins)
    if nb <= 1:
        return False

    loads = sol.loads
    rank = np.argsort(loads)  # light -> heavy
    head = max(1, int(0.3 * nb))
    b_idx = int(rng.choice(rank[:head]))

    victim_items = sol.bins[b_idx][:]
    if not victim_items:
        return False

    kept_bins = [sol.bins[i][:] for i in range(nb) if i != b_idx]
    kept_loads = np.delete(sol.loads, b_idx)

    victim_items.sort(key=lambda i: float(items[i]), reverse=True)

    for item_idx in victim_items:
        choose_b = _choose_bin_for_item(
            items, capacity, kept_bins, kept_loads, item_idx,
            priority_fn, rng, cand_k
        )
        if choose_b == -1:
            return False
        kept_bins[choose_b].append(item_idx)
        kept_loads[choose_b] += float(items[item_idx])

    sol.bins = compact_bins(kept_bins)
    sol.loads = kept_loads
    return True


def _subset_sum_best_fit(indices: List[int], weights: np.ndarray, cap: int) -> List[int]:
    """
    Subset-sum DP (O(m*cap)) with reconstruction; cap is small (C=100).
    Find subset whose total weight is as large as possible <= cap.
    """
    m = len(indices)
    if m == 0:
        return []

    dp = np.full(cap + 1, -1, dtype=np.int32)      # store chosen position in indices
    prev = np.full(cap + 1, -1, dtype=np.int32)    # previous sum
    dp[0] = -2

    for pos in range(m):
        w = int(weights[indices[pos]])
        if w <= 0 or w > cap:
            continue
        for s in range(cap - w, -1, -1):
            if dp[s] != -1 and dp[s + w] == -1:
                dp[s + w] = pos
                prev[s + w] = s

    best_s = int(np.max(np.where(dp != -1)[0]))
    if best_s <= 0:
        return []

    chosen_pos = []
    s = best_s
    while s > 0:
        pos = int(dp[s])
        chosen_pos.append(pos)
        s = int(prev[s])

    return [indices[pos] for pos in chosen_pos]


def _pack_into_m_bins(indices: List[int], items: np.ndarray, capacity: int, m: int) -> Optional[List[List[int]]]:
    """
    Try to pack given indices into <= m bins using repeated best-fit subset-sum.
    Returns bins if succeed, else None.
    """
    remaining = indices[:]
    bins: List[List[int]] = []

    for _ in range(m):
        if not remaining:
            break

        subset = _subset_sum_best_fit(remaining, items, int(capacity))
        if not subset:
            # greedy fallback
            subset = []
            load = 0.0
            remaining.sort(key=lambda i: float(items[i]), reverse=True)
            for idx in remaining:
                w = float(items[idx])
                if load + w <= capacity + 1e-9:
                    subset.append(idx)
                    load += w
            if not subset:
                return None

        bins.append(subset)
        subset_set = set(subset)
        remaining = [i for i in remaining if i not in subset_set]

    if remaining:
        return None
    return bins


def try_k_bin_repack(
    items: np.ndarray,
    capacity: int,
    sol: Solution,
    *,
    k: int,
    rng: np.random.Generator,
) -> bool:
    """
    Select k bins, pool items, try repack into (k-1) bins.
    If succeed, replace and accept.
    """
    if k < 2:
        return False
    nb = len(sol.bins)
    if nb < k:
        return False

    loads = sol.loads
    rank = np.argsort(loads)
    head = max(k, int(0.5 * nb))
    chosen_bins = rng.choice(rank[:head], size=k, replace=False)
    chosen_bins = list(map(int, chosen_bins))

    pooled: List[int] = []
    for b in chosen_bins:
        pooled.extend(sol.bins[b])
    if not pooled:
        return False

    new_bins = _pack_into_m_bins(pooled, items, capacity, m=k - 1)
    if new_bins is None:
        return False

    keep_bins = [sol.bins[i] for i in range(nb) if i not in chosen_bins]
    keep_bins.extend(new_bins)
    keep_bins = compact_bins(keep_bins)
    sol.bins = keep_bins
    sol.loads = compute_loads(items, keep_bins)
    return True


def ruin_and_recreate(
    items: np.ndarray,
    capacity: int,
    sol: Solution,
    *,
    ruin_frac: float,
    priority_fn: Optional[PriorityFn],
    rng: np.random.Generator,
    cand_k: int,
    init_method: str = "bfd",
) -> bool:
    """
    Remove a fraction of bins, then reinsert their items.
    Accept if does not increase bin count.
    """
    nb = len(sol.bins)
    if nb <= 2:
        return False

    rm = max(1, int(round(ruin_frac * nb)))
    rm = min(rm, nb - 1)

    rank = np.argsort(sol.loads)
    head = max(rm, int(0.7 * nb))
    remove_bins = rng.choice(rank[:head], size=rm, replace=False)
    remove_bins = set(map(int, remove_bins))

    removed_items: List[int] = []
    kept_bins: List[List[int]] = []
    for i in range(nb):
        if i in remove_bins:
            removed_items.extend(sol.bins[i])
        else:
            kept_bins.append(sol.bins[i][:])

    kept_bins = compact_bins(kept_bins)
    kept_loads = compute_loads(items, kept_bins)

    removed_items.sort(key=lambda idx: float(items[idx]), reverse=True)

    for item_idx in removed_items:
        b = _choose_bin_for_item(items, capacity, kept_bins, kept_loads, item_idx, priority_fn, rng, cand_k)
        if b == -1:
            kept_bins.append([item_idx])
            kept_loads = np.append(kept_loads, float(items[item_idx]))
        else:
            kept_bins[b].append(item_idx)
            kept_loads[b] += float(items[item_idx])

    kept_bins = compact_bins(kept_bins)
    new_nb = len(kept_bins)
    old_nb = len(sol.bins)
    if new_nb <= old_nb:
        sol.bins = kept_bins
        sol.loads = compute_loads(items, kept_bins)
        return True

    return False
