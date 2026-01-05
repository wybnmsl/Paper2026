#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a TSPAEL64.pkl-style dataset from locally downloaded TSPLIB files.

- 读取 {name}.tsp[.gz]，可选读取 {name}.opt.tour[.gz]
- 若提供 solutions 文件，则优先从 solutions 中读取最优目标值
- 仅保留：有坐标 & min_nodes <= N <= max_nodes
- 距离矩阵按 TSPLIB 的度量由 tsplib95.get_weight(i,j) 构造（整数权重→以 float 存）
- 输出 dict：
    {
      "coordinate":      list[np.ndarray(N,2)],    # 原始坐标（不归一化），float64
      "distance_matrix": list[np.ndarray(N,N)],    # TSPLIB 权重矩阵，float64
      "optimal_tour":    list[np.ndarray(N,)],     # 0-based 顺序，int32；若无 tour 用 -1 填充
      "cost":            np.ndarray[float],        # 在 distance_matrix 上的“最优”环长度
    }
"""

"""
python zTSP_1202/utils/make_TSPAEL64_from_tsplib.py \
  --root zTSP_1202/tsplib-master \
  --solutions solutions \
  --names kroA100 \
  --out zTSP_1202/TrainingData/TSPAEL64.pkl
"""


import os
import sys
import gzip
import pickle
import argparse
import tempfile
from typing import List, Optional, Dict

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


def read_solutions_file(path: str) -> Dict[str, float]:
    """
    解析 mastqe/tsplib 仓库中的 `solutions` 文件：
        a280 : 2579
        ali535 : 202339
        ...
    返回 {basename -> best_cost} 字典。
    """
    sol = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # 兼容 "name : value" 或 "name value"
            if ":" in s:
                left, right = s.split(":", 1)
            else:
                parts = s.split()
                if len(parts) < 2:
                    continue
                left, right = parts[0], parts[1]
            name = left.strip()
            val_str = right.strip().split()[0]
            try:
                val = float(val_str)
            except Exception:
                continue
            if name:
                sol[name] = val
    return sol


def find_instances(root: str,
                   names_filter: Optional[List[str]] = None,
                   require_opt_tour: bool = True) -> List[str]:
    """
    在 root 中自动发现实例基名（不含扩展名）。

    - require_opt_tour=True: 只保留同时存在 tsp 与 opt.tour 的基名
    - require_opt_tour=False: 只要有 tsp 即可
    - names_filter: 限定要打包的子集
    """
    bases_tsp = set()
    bases_tour = set()

    for fn in os.listdir(root):
        low = fn.lower()
        base = fn
        if low.endswith(".tsp.gz"):
            base = base[: -len(".tsp.gz")]
            bases_tsp.add(base)
        elif low.endswith(".tsp"):
            base = base[: -len(".tsp")]
            bases_tsp.add(base)
        elif low.endswith(".opt.tour.gz"):
            base = base[: -len(".opt.tour.gz")]
            bases_tour.add(base)
        elif low.endswith(".opt.tour"):
            base = base[: -len(".opt.tour")]
            bases_tour.add(base)

    if require_opt_tour:
        bases = sorted(b for b in bases_tsp if b in bases_tour)
    else:
        bases = sorted(bases_tsp)

    if names_filter:
        names_set = set(names_filter)
        bases = [b for b in bases if b in names_set]

    return bases


# ------------------------------
# TSPLIB parsing helpers
# ------------------------------


def load_tsplib_problem_from_text(text: str):
    """
    兼容不同版本 tsplib95：
    - 若存在 tsplib95.parse(text)，优先用它（直接从字符串解析）
    - 否则落地到临时文件，再用 tsplib95.load(tmp_path)
    """
    parse_fn = getattr(tsplib95, "parse", None)
    if callable(parse_fn):
        return parse_fn(text)

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
            except Exception:
                pass

    # 去重保序
    seen = set()
    tmp = []
    for v in order_1:
        if v not in seen:
            seen.add(v)
            tmp.append(v)

    if n_expected is not None:
        # 有些 tour 会首尾重复一个点
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
    # 对称化 + 对角线为 0
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def tour_cost(W: np.ndarray, order0: List[int]) -> float:
    """根据 0-based 顺序在距离矩阵 W 上计算闭环成本。"""
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
    parser = argparse.ArgumentParser(
        description="Pack local TSPLIB GT into TSPAEL64-like .pkl"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./tsplib-master",
        help="Directory containing {name}.tsp[.gz] and (optionally) {name}.opt.tour[.gz].",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="zTSP/TrainingData/TSPAEL64_real.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of basenames to include (no extension). If omitted, auto-discover all valid .tsp in root.",
    )
    parser.add_argument(
        "--solutions",
        type=str,
        default=None,
        help="Optional solutions file path (e.g. 'solutions' in mastqe/tsplib). "
             "If provided, optimal costs are read from here.",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=999999,
        help="Keep instances with N <= max_nodes.",
    )
    parser.add_argument(
        "--min_nodes",
        type=int,
        default=4,
        help="Keep instances with N >= min_nodes.",
    )
    parser.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Raise on missing coords/tour/bad length instead of skipping.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)

    # 若提供 solutions 文件，则不再强制需要 .opt.tour
    solutions_map: Dict[str, float] = {}
    if args.solutions:
        sol_path = args.solutions
        if not os.path.isabs(sol_path):
            sol_path = os.path.join(root, sol_path)
        if not os.path.isfile(sol_path):
            raise FileNotFoundError(f"solutions file not found: {sol_path}")
        solutions_map = read_solutions_file(sol_path)
        print(f"[info] loaded {len(solutions_map)} solutions from {sol_path}")

    bases = find_instances(
        root,
        args.names,
        require_opt_tour=(len(solutions_map) == 0),
    )

    coords_list: List[np.ndarray] = []
    dists_list: List[np.ndarray] = []
    tours_list: List[np.ndarray] = []
    costs_list: List[float] = []

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
                if args.fail_on_missing:
                    raise FileNotFoundError(msg)
                skipped.append((base, msg))
                continue

            tsp_text = read_text_auto(tsp_path)
            prob = load_tsplib_problem_from_text(tsp_text)
            N = int(prob.dimension)

            if N < args.min_nodes or N > args.max_nodes:
                skipped.append((base, f"N={N} out_of_range"))
                continue

            if not has_coordinates(prob):
                msg = "no_coordinates"
                if args.fail_on_missing:
                    raise RuntimeError(msg)
                skipped.append((base, msg))
                continue

            # 构造矩阵 & 坐标（先算好）
            W = build_distance_matrix(prob)
            C = coords_to_array(prob)

            order0: Optional[List[int]] = None
            cost: Optional[float] = None

            if solutions_map:
                # 只依赖 solutions 提供的最优值
                if base not in solutions_map:
                    msg = "no_solution_cost"
                    if args.fail_on_missing:
                        raise KeyError(msg)
                    skipped.append((base, msg))
                    continue
                cost = float(solutions_map[base])

            else:
                # 旧路径：需要 .opt.tour
                tour_path = None
                for ext in (".opt.tour", ".opt.tour.gz"):
                    p = os.path.join(root, base + ext)
                    if os.path.isfile(p):
                        tour_path = p
                        break
                if tour_path is None:
                    msg = "missing_opt_tour"
                    if args.fail_on_missing:
                        raise FileNotFoundError(msg)
                    skipped.append((base, msg))
                    continue

                tour_text = read_text_auto(tour_path)
                order0 = parse_opt_tour_to_order(tour_text, n_expected=N)
                if len(order0) != N:
                    msg = f"bad_tour_len:{len(order0)} vs N:{N}"
                    if args.fail_on_missing:
                        raise ValueError(msg)
                    skipped.append((base, msg))
                    continue

                cost = tour_cost(W, order0)

            coords_list.append(C.astype(np.float64))
            dists_list.append(W.astype(np.float64))

            if order0 is None:
                # 无 tour，用 -1 占位，保证数据结构一致
                tours_list.append(np.full((N,), -1, dtype=np.int32))
            else:
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
