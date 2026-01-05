#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a TSPAEL64.pkl-style dataset from locally downloaded TSPLIB files.

- 读取 {name}.tsp[.gz] 与 {name}.opt.tour[.gz]
- 仅保留：有坐标 & 有官方最优解 & min_nodes <= N <= max_nodes
- 距离矩阵按 TSPLIB 的度量由 tsplib95.get_weight(i,j) 构造（整数权重→以 float 存）
- 输出 dict：
    {
      "coordinate":      list[np.ndarray(N,2)],    # 原始坐标（不归一化），float64
      "distance_matrix": list[np.ndarray(N,N)],    # TSPLIB 权重矩阵，float64
      "optimal_tour":    list[np.ndarray(N,)],     # 0-based 顺序，int32
      "cost":            list[float],              # 在 distance_matrix 上计算的最优环长度
    }
"""

import os
import io
import sys
import gzip
import pickle
import argparse
import tempfile
from typing import List, Optional

import numpy as np
from tqdm import tqdm

try:
    import tsplib95  # pip install tsplib95 tqdm
except Exception as e:
    print("Please `pip install tsplib95 tqdm` first.", file=sys.stderr)
    raise

# ------------------------------
# File helpers
# ------------------------------

def read_text_auto(path: str) -> str:
    """Read .gz or plain text and return str."""
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    else:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")

def find_instances(root: str, names_filter: Optional[List[str]] = None) -> List[str]:
    """
    在 root 中自动发现同时存在 tsp 与 opt.tour 的基名列表（不含扩展名）。
    也支持 names_filter 指定的子集（自动过滤掉缺文件的）。
    """
    cand = set()
    for fn in os.listdir(root):
        if fn.lower().endswith((".tsp", ".tsp.gz", ".opt.tour", ".opt.tour.gz")):
            base = fn
            if base.lower().endswith(".opt.tour.gz"):
                base = base[: -len(".opt.tour.gz")]
            elif base.lower().endswith(".opt.tour"):
                base = base[: -len(".opt.tour")]
            elif base.lower().endswith(".tsp.gz"):
                base = base[: -len(".tsp.gz")]
            elif base.lower().endswith(".tsp"):
                base = base[: -len(".tsp")]
            if base:
                cand.add(base)

    def has_both(b: str) -> bool:
        tsp_ok = any(os.path.isfile(os.path.join(root, b + ext)) for ext in [".tsp", ".tsp.gz"])
        tour_ok = any(os.path.isfile(os.path.join(root, b + ext)) for ext in [".opt.tour", ".opt.tour.gz"])
        return tsp_ok and tour_ok

    all_bases = sorted([b for b in cand if has_both(b)])
    if names_filter:
        names_set = set(names_filter)
        return [b for b in all_bases if b in names_set]
    return all_bases

# ------------------------------
# TSPLIB parsing helpers
# ------------------------------

def load_tsplib_problem_from_text(text: str):
    """
    兼容不同版本 tsplib95：
    - 若存在 tsplib95.parse(text)，优先用它（直接从字符串解析）
    - 否则落地到临时文件，再用 tsplib95.load(tmp_path)
    """
    # 1) 尝试 parse
    parse_fn = getattr(tsplib95, "parse", None)
    if callable(parse_fn):
        return parse_fn(text)

    # 2) 退回临时文件 + load
    with tempfile.NamedTemporaryFile("w+", suffix=".tsp", delete=False) as f:
        tmp_path = f.name
        f.write(text)
    try:
        prob = tsplib95.load(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return prob

def parse_opt_tour_to_order(tour_text: str, n_expected: Optional[int] = None) -> List[int]:
    """
    解析官方 .opt.tour，返回 0-based 顺序列表。
    兼容一行多个数/重复起点结尾的写法。
    """
    order_1 = []
    in_section = False
    for line in tour_text.splitlines():
        s = line.strip()
        if not in_section:
            if s.upper() == "TOUR_SECTION":
                in_section = True
            continue
        if s == "-1" or s.upper() == "EOF":
            break
        if not s:
            continue
        for tok in s.split():
            if tok == "-1":
                break
            try:
                v = int(tok)
                order_1.append(v)
            except:
                pass
    # 去重保序
    seen = set()
    tmp = []
    for v in order_1:
        if v not in seen:
            seen.add(v)
            tmp.append(v)
    if n_expected is not None:
        if len(tmp) == n_expected + 1 and tmp[0] == tmp[-1]:
            tmp = tmp[:-1]
    order0 = [v - 1 for v in tmp]
    return order0

def has_coordinates(problem) -> bool:
    try:
        nc = problem.node_coords
        if nc and len(nc) > 0:
            k = next(iter(nc.keys()))
            return len(problem.node_coords[k]) >= 2
    except Exception:
        pass
    return False

def coords_to_array(problem) -> np.ndarray:
    """node_coords dict -> (N,2) float64 array (node id 按 1..N 排序)."""
    nodes = sorted(list(problem.get_nodes()))
    arr = np.zeros((len(nodes), 2), dtype=np.float64)
    for idx, node in enumerate(nodes):
        x, y = problem.node_coords[node][:2]
        arr[idx, 0] = float(x)
        arr[idx, 1] = float(y)
    return arr

def build_distance_matrix(problem) -> np.ndarray:
    """用 TSPLIB 的度量构造完整矩阵（get_weight）。"""
    nodes = list(problem.get_nodes())  # 1..N
    N = len(nodes)
    W = np.zeros((N, N), dtype=np.float64)
    for i_idx, i in enumerate(nodes):
        for j_idx, j in enumerate(nodes):
            if i_idx == j_idx:
                continue
            W[i_idx, j_idx] = float(problem.get_weight(i, j))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W

def tour_cost(W: np.ndarray, order0: List[int]) -> float:
    n = len(order0)
    s = 0.0
    for i in range(n):
        a = order0[i]
        b = order0[(i + 1) % n]
        s += float(W[a, b])
    return float(s)

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pack local TSPLIB GT into TSPAEL64-like .pkl")
    parser.add_argument("--root", type=str,
                        default="/home/data2t1/wangrongzheng/EoH-main/zTSP/test/zTSP/external/tsplib_gt",
                        help="Directory containing {name}.tsp[.gz] and {name}.opt.tour[.gz].")
    parser.add_argument("--out", type=str, default="zTSP/TrainingData/TSPAEL64_real.pkl",
                        help="Output pickle path.")
    parser.add_argument("--names", type=str, nargs="*", default=None,
                        help="Optional subset of basenames to include (no extension). If omitted, auto-discover all valid pairs in root.")
    parser.add_argument("--max_nodes", type=int, default=999, help="Keep instances with N <= max_nodes.")
    parser.add_argument("--min_nodes", type=int, default=4, help="Keep instances with N >= min_nodes.")
    parser.add_argument("--fail_on_missing", action="store_true",
                        help="Raise on missing coords/tour/bad length instead of skipping.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    bases = find_instances(root, args.names)

    coords_list: List[np.ndarray] = []
    dists_list:  List[np.ndarray] = []
    tours_list:  List[np.ndarray] = []
    costs_list:  List[float] = []

    kept = 0
    skipped = []

    it = tqdm(bases, desc="Packing TSPLIB GT", unit="inst")
    for base in it:
        try:
            # 读取 .tsp / .tsp.gz
            tsp_path = None
            for ext in (".tsp", ".tsp.gz"):
                p = os.path.join(root, base + ext)
                if os.path.isfile(p):
                    tsp_path = p
                    break
            if tsp_path is None:
                msg = "missing_tsp"
                if args.fail_on_missing: raise FileNotFoundError(msg)
                skipped.append((base, msg)); continue

            tsp_text = read_text_auto(tsp_path)
            prob = load_tsplib_problem_from_text(tsp_text)
            N = int(prob.dimension)

            if N < args.min_nodes or N > args.max_nodes:
                skipped.append((base, f"N={N} out_of_range")); continue

            if not has_coordinates(prob):
                msg = "no_coordinates"
                if args.fail_on_missing: raise RuntimeError(msg)
                skipped.append((base, msg)); continue

            # 读取 .opt.tour / .opt.tour.gz
            tour_path = None
            for ext in (".opt.tour", ".opt.tour.gz"):
                p = os.path.join(root, base + ext)
                if os.path.isfile(p):
                    tour_path = p
                    break
            if tour_path is None:
                msg = "missing_opt_tour"
                if args.fail_on_missing: raise FileNotFoundError(msg)
                skipped.append((base, msg)); continue

            tour_text = read_text_auto(tour_path)
            order0 = parse_opt_tour_to_order(tour_text, n_expected=N)
            if len(order0) != N:
                msg = f"bad_tour_len:{len(order0)} vs N:{N}"
                if args.fail_on_missing: raise ValueError(msg)
                skipped.append((base, msg)); continue

            # 构造矩阵 & 坐标
            W = build_distance_matrix(prob)
            C = coords_to_array(prob)

            cost = tour_cost(W, order0)

            coords_list.append(C.astype(np.float64))
            dists_list.append(W.astype(np.float64))
            tours_list.append(np.asarray(order0, dtype=np.int32))
            costs_list.append(float(cost))

            kept += 1
            it.set_postfix(keep=kept, N=N, cost=cost)

        except Exception as e:
            skipped.append((base, f"error:{type(e).__name__}:{e}"))
            if args.fail_on_missing:
                raise

    dataset = {
        "coordinate": coords_list,
        "distance_matrix": dists_list,
        "optimal_tour": tours_list,
        "cost": np.asarray(costs_list, dtype=np.float64),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] saved: {args.out}")
    print(f"[stats] kept={kept} / candidates={len(bases)}")
    if skipped:
        print("[skipped]")
        for nm, why in skipped:
            print(f"  - {nm}: {why}")

if __name__ == "__main__":
    main()
