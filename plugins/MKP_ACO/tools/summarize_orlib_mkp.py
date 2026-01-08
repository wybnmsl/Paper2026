#!/usr/bin/env python3
"""
Summarize OR-Library MKP-style datasets (mknap1/mknapcb*/mknap2).

Outputs:
  1) per-instance CSV with size + basic stats
  2) per-(file, m, n) size count CSV
Optionally: xlsx with 2 sheets (requires pandas + openpyxl).

Supported formats:
- Count-based (mknap1, mknapcb*):
    K
    m n opt
    p[1..m]
    w[1..(m*n)]   (constraint-major: n blocks, each has m numbers)
    b[1..n]
    (repeat K times)

- "problem NAME" marked (mknap2.txt):
    problem XXX.DAT
    n m
    p[1..m]
    b[1..n]
    A[1..(m*n)]   (object-major: m rows x n cols)
    opt
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

_NUM_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def tokenize_numbers(text: str) -> List[float]:
    return [float(x) for x in _NUM_RE.findall(text)]


def _as_np(a: List[float]):
    if np is None:
        return a
    return np.asarray(a, dtype=float)


def _mean(x):
    if np is not None and hasattr(x, "mean"):
        return float(x.mean())
    return sum(x) / (len(x) if x else 1)


def _std(x):
    if not x:
        return float("nan")
    if np is not None and hasattr(x, "std"):
        return float(x.std(ddof=0))
    mu = _mean(x)
    return math.sqrt(sum((v - mu) ** 2 for v in x) / len(x))


def _min(x): return float(min(x)) if x else float("nan")
def _max(x): return float(max(x)) if x else float("nan")
def _sum(x): return float(sum(x)) if x else 0.0


def _corr(x, y) -> float:
    if len(x) < 2:
        return float("nan")
    if np is not None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sx = float(x.std(ddof=0))
        sy = float(y.std(ddof=0))
        if sx == 0.0 or sy == 0.0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    mx = _mean(x)
    my = _mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


@dataclass
class Instance:
    source_file: str
    instance_id: str
    m_items: int
    n_constraints: int
    opt_value: Optional[float]  # None if unknown
    profits: List[float]         # length m
    weights: List[float]         # length m*n
    capacities: List[float]      # length n
    weights_layout: str          # "n_by_m" or "m_by_n"


def parse_count_file(path: Path) -> List[Instance]:
    nums = tokenize_numbers(path.read_text(errors="ignore"))
    if not nums:
        return []
    k = int(nums[0])
    idx = 1
    out: List[Instance] = []
    for i in range(k):
        m = int(nums[idx]); n = int(nums[idx + 1]); opt = float(nums[idx + 2]); idx += 3
        profits = nums[idx:idx + m]; idx += m
        weights = nums[idx:idx + (m * n)]; idx += (m * n)
        caps = nums[idx:idx + n]; idx += n
        out.append(
            Instance(
                source_file=path.name,
                instance_id=f"{i+1:02d}",
                m_items=m,
                n_constraints=n,
                opt_value=None if opt == 0 else opt,
                profits=profits,
                weights=weights,
                capacities=caps,
                weights_layout="n_by_m",
            )
        )
    return out


def parse_problem_marked_file(path: Path) -> List[Instance]:
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    prob_lines: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*problem\s+([A-Za-z0-9_.-]+)", line, flags=re.I)
        if m:
            prob_lines.append((i, m.group(1)))
    if not prob_lines:
        return []

    out: List[Instance] = []
    for (i, name), (j, _) in zip(prob_lines, prob_lines[1:] + [(len(lines), "__END__")]):
        seg = "\n".join(lines[i + 1: j])
        nums = tokenize_numbers(seg)
        n = int(nums[0]); m = int(nums[1])
        idx = 2
        profits = nums[idx:idx + m]; idx += m
        caps = nums[idx:idx + n]; idx += n
        weights = nums[idx:idx + (m * n)]; idx += (m * n)
        opt = nums[idx] if idx < len(nums) else float("nan")
        out.append(
            Instance(
                source_file=path.name,
                instance_id=name,
                m_items=m,
                n_constraints=n,
                opt_value=opt,
                profits=profits,
                weights=weights,
                capacities=caps,
                weights_layout="m_by_n",
            )
        )
    return out


def compute_features(inst: Instance) -> Dict[str, float]:
    m, n = inst.m_items, inst.n_constraints
    p = _as_np(inst.profits)
    w = _as_np(inst.weights)
    b = _as_np(inst.capacities)

    if np is not None:
        if inst.weights_layout == "n_by_m":
            W = np.asarray(inst.weights, dtype=float).reshape(n, m)
            sum_c = W.sum(axis=1)
            sum_item = W.sum(axis=0)
        else:
            W = np.asarray(inst.weights, dtype=float).reshape(m, n)
            sum_c = W.sum(axis=0)
            sum_item = W.sum(axis=1)

        tight = b / sum_c
        dens = p / np.where(sum_item == 0, np.nan, sum_item)
        zero_ratio = float(np.mean(np.isclose(w, 0.0)))
        tight_mean = float(np.nanmean(tight))
        tight_std  = float(np.nanstd(tight))
        tight_min  = float(np.nanmin(tight))
        tight_max  = float(np.nanmax(tight))
        dens_mean  = float(np.nanmean(dens))
        dens_std   = float(np.nanstd(dens))
    else:
        zero_ratio = sum(1 for x in w if x == 0) / (len(w) if w else 1)
        sum_c = []
        if inst.weights_layout == "n_by_m":
            for ci in range(n):
                chunk = inst.weights[ci * m:(ci + 1) * m]
                sum_c.append(sum(chunk))
            sum_item = [sum(inst.weights[ci * m + j] for ci in range(n)) for j in range(m)]
        else:
            sum_c = [sum(inst.weights[j * n + ci] for j in range(m)) for ci in range(n)]
            sum_item = [sum(inst.weights[j * n:(j + 1) * n]) for j in range(m)]
        tight = [b[i] / sum_c[i] if sum_c[i] else float("nan") for i in range(n)]
        dens = [p[j] / sum_item[j] if sum_item[j] else float("nan") for j in range(m)]
        tight_v = [t for t in tight if not math.isnan(t)]
        dens_v = [d for d in dens if not math.isnan(d)]
        tight_mean = _mean(tight_v); tight_std = _std(tight_v); tight_min = _min(tight_v); tight_max = _max(tight_v)
        dens_mean = _mean(dens_v); dens_std = _std(dens_v)

    return {
        "m_items": m,
        "n_constraints": n,
        "opt_value": float("nan") if inst.opt_value is None else float(inst.opt_value),

        "profit_sum": _sum(p),
        "profit_mean": _mean(p),
        "profit_std": _std(p),
        "profit_min": _min(p),
        "profit_max": _max(p),

        "weight_sum": _sum(w),
        "weight_mean": _mean(w),
        "weight_std": _std(w),
        "weight_min": _min(w),
        "weight_max": _max(w),
        "weight_zero_ratio": zero_ratio,

        "cap_sum": _sum(b),
        "cap_mean": _mean(b),
        "cap_std": _std(b),
        "cap_min": _min(b),
        "cap_max": _max(b),

        # tightness = b_i / sum_j w_{i,j}
        "tightness_mean": tight_mean,
        "tightness_std": tight_std,
        "tightness_min": tight_min,
        "tightness_max": tight_max,

        "profit_weight_corr": _corr(list(p), list(sum_item)),
        "density_mean": dens_mean,
        "density_std": dens_std,
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Files or directories (directories scan *.txt).")
    ap.add_argument("--out", default="mkp_instances_summary.csv", help="Per-instance CSV.")
    ap.add_argument("--out-sizes", default="mkp_sizes_summary.csv", help="Size-count CSV.")
    ap.add_argument("--xlsx", default="", help="Optional XLSX path (requires pandas).")
    args = ap.parse_args()

    in_paths: List[Path] = []
    for x in args.inputs:
        p = Path(x)
        if p.is_dir():
            in_paths.extend(sorted(p.glob("*.txt")))
        else:
            in_paths.append(p)

    instances: List[Instance] = []
    for p in in_paths:
        if not p.exists():
            print(f"[warn] missing: {p}")
            continue
        text = p.read_text(errors="ignore")
        if re.search(r"^\s*problem\s+", text, flags=re.I | re.M):
            instances.extend(parse_problem_marked_file(p))
        else:
            try:
                instances.extend(parse_count_file(p))
            except Exception as e:
                print(f"[warn] skip {p.name}: {e}")

    rows: List[Dict[str, object]] = []
    for inst in instances:
        feats = compute_features(inst)
        rows.append({
            "case_id": f"{Path(inst.source_file).stem}::{inst.instance_id}",
            "source_file": inst.source_file,
            "instance": inst.instance_id,
            **feats,
        })

    if rows:
        write_csv(Path(args.out), rows, list(rows[0].keys()))
    else:
        write_csv(Path(args.out), [], ["case_id","source_file","instance"])

    # size summary
    size_map: Dict[Tuple[str, int, int], int] = {}
    for r in rows:
        key = (r["source_file"], int(r["m_items"]), int(r["n_constraints"]))
        size_map[key] = size_map.get(key, 0) + 1
    size_rows = [
        {"source_file": k[0], "m_items": k[1], "n_constraints": k[2], "num_instances": v}
        for k, v in sorted(size_map.items())
    ]
    write_csv(Path(args.out_sizes), size_rows, ["source_file", "m_items", "n_constraints", "num_instances"])

    # optional xlsx
    if args.xlsx:
        try:
            import pandas as pd
            with pd.ExcelWriter(args.xlsx) as xw:
                pd.DataFrame(rows).to_excel(xw, index=False, sheet_name="instances")
                pd.DataFrame(size_rows).to_excel(xw, index=False, sheet_name="sizes")
        except Exception as e:
            print(f"[warn] failed to write xlsx: {e}")

    print(f"[ok] instances: {len(rows)} -> {args.out}")
    print(f"[ok] sizes:     {len(size_rows)} -> {args.out_sizes}")


if __name__ == "__main__":
    main()
