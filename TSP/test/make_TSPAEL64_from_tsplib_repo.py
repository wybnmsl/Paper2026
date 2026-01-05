#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a TSPAEL64.pkl-style dataset from a local clone of mastqe/tsplib (offline).

- 递归扫描 --root 下的 .tsp 文件（如 /home/.../zTSP/tsplib-master）
- 仅保留有坐标(node_coords)的 STSP 实例
- 使用 tsplib95.get_weight(i,j) 构造距离矩阵（复现 TSPLIB 的度量/四舍五入规则）
- Ground-truth 只写入 cost（来自内置/外部 BKS 表），optimal_tour 用空数组占位
- 输出结构与 TSPAEL64.pkl 相同的 4 个键

依赖: pip install tsplib95 tqdm
BKS 来源参考（镜像页面，等价于 TSPLIB 官方 STSP 表）：https://www.cse.unr.edu/~sushil/class/gas/TSP/STSP.html
"""

import os
import sys
import argparse
import json
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

try:
    import tsplib95  # pip install tsplib95
except Exception as e:
    print("Please `pip install tsplib95 tqdm` first.", file=sys.stderr)
    raise

# -----------------------
# 内置 BKS（常用 <1000 节点；键一律小写）
# 来源：STSP best-known/optimal 值表（与 TSPLIB 官方一致的镜像页）
# https://www.cse.unr.edu/~sushil/class/gas/TSP/STSP.html
# -----------------------
BKS_STSP: Dict[str, int] = {
    # 小规模（含欧式/ATT/GEO等，均为整数权重）
    "burma14": 3323,
    "fri26": 937,
    "ulysses16": 6859,
    "ulysses22": 7013,
    "dantzig42": 699,
    "swiss42": 1273,
    "att48": 10628,
    "hk48": 11461,
    "berlin52": 7542,
    "eil51": 426,
    "st70": 675,
    "eil76": 538,
    "pr76": 108159,
    "rd100": 7910,
    "kroa100": 21282,
    "krob100": 22141,
    "kroc100": 20749,
    "krod100": 21294,
    "kroe100": 22068,
    "lin105": 14379,
    "ch130": 6110,
    "gr120": 6942,     # 无坐标，多为 MATRIX；若无坐标会被跳过
    "pr107": 44303,
    "pr124": 59030,
    "bier127": 118282,
    "ch150": 6528,
    "kroa150": 26524,
    "krob150": 26130,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "rat195": 2323,
    "d198": 15780,
    "rd400": 15281,
    "pcb442": 50778,
    # 其它常见 <1000
    "a280": 2579,
    "bayg29": 1610,
    "bays29": 2020,
    "gil262": 2378,
    "lin318": 42029,
    "linhp318": 41345,
    "p654": 34643,
    "pa561": 2763,
    "si175": 21407,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    # 注意：gr* 多数无坐标 (FULL_MATRIX)；脚本会自动跳过无坐标
    "gr48": 5046,
    "gr96": 55209,
    "gr137": 69853,
    "gr202": 40160,
    "gr229": 134602,
    "gr431": 171414,
}

def load_bks(extra_json: Optional[str]) -> Dict[str, float]:
    """合并内置 BKS 与外部 JSON（若提供）"""
    bks = {k.lower(): float(v) for k, v in BKS_STSP.items()}
    if extra_json:
        with open(extra_json, "r", encoding="utf-8") as f:
            ext = json.load(f)
        for k, v in ext.items():
            bks[str(k).lower()] = float(v)
    return bks

# -----------------------
# 工具函数
# -----------------------

def find_tsp_files(root: str) -> List[str]:
    """递归搜集 .tsp 文件路径"""
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".tsp"):
                paths.append(os.path.join(dp, fn))
    return sorted(paths)

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
    nodes = sorted(list(problem.get_nodes()))
    arr = np.zeros((len(nodes), 2), dtype=np.float64)
    for idx, node in enumerate(nodes):
        x, y = problem.node_coords[node][:2]
        arr[idx, 0] = float(x)
        arr[idx, 1] = float(y)
    return arr

def build_distance_matrix(problem) -> np.ndarray:
    nodes = list(problem.get_nodes())
    N = len(nodes)
    W = np.zeros((N, N), dtype=np.float64)
    for i_idx, i in enumerate(nodes):
        for j_idx, j in enumerate(nodes):
            if i_idx == j_idx:
                continue
            W[i_idx, j_idx] = float(problem.get_weight(i, j))
    # 保证对称与 0 对角
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W

# -----------------------
# 主流程
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str,
                    default="/home/data2t1/wangrongzheng/EoH-main/zTSP/tsplib-master",
                    help="mastqe/tsplib 本地根目录（递归扫描 .tsp）")
    ap.add_argument("--out", type=str, default="zTSP/TrainingData/TSPAEL64_real_from_repo.pkl",
                    help="输出 pkl 路径")
    ap.add_argument("--csv", type=str, default=None,
                    help="可选：输出一个 CSV 汇总（实例名, N, cost）")
    ap.add_argument("--names", type=str, nargs="*", default=None,
                    help="可选：仅打包指定 basename（不含扩展名），大小写不敏感")
    ap.add_argument("--bks-json", type=str, default=None,
                    help="可选：附加/覆盖 BKS 的 JSON 文件（{name: cost}）")
    ap.add_argument("--min_nodes", type=int, default=4)
    ap.add_argument("--max_nodes", type=int, default=999)
    args = ap.parse_args()

    bks = load_bks(args.bks_json)  # name->float

    all_tsp = find_tsp_files(args.root)
    if args.names:
        name_set = {n.lower() for n in args.names}
        cand = [p for p in all_tsp if os.path.splitext(os.path.basename(p))[0].lower() in name_set]
    else:
        cand = all_tsp

    coords_list:  List[np.ndarray] = []
    dists_list:   List[np.ndarray] = []
    tours_list:   List[np.ndarray] = []   # 空数组占位
    costs_list:   List[float]       = []
    rows_csv:     List[Tuple[str,int,float]] = []

    kept = 0
    skipped = []

    it = tqdm(cand, desc="Packing tsplib-master", unit="file")
    for tsp_path in it:
        base = os.path.splitext(os.path.basename(tsp_path))[0]
        base_l = base.lower()

        try:
            problem = tsplib95.load(tsp_path)

            N = int(problem.dimension)
            if N < args.min_nodes or N > args.max_nodes:
                skipped.append((base, f"N={N} out_of_range"))
                continue

            # 仅保留有坐标的实例（以契合你的数据格式）
            if not has_coordinates(problem):
                skipped.append((base, "no_coordinates"))
                continue

            # BKS（ground-truth cost）查表
            if base_l not in bks:
                skipped.append((base, "bks_missing"))
                continue
            cost = float(bks[base_l])

            # 生成矩阵与坐标
            W = build_distance_matrix(problem)
            C = coords_to_array(problem)

            # 写入
            coords_list.append(C.astype(np.float64))
            dists_list.append(W.astype(np.float64))
            tours_list.append(np.asarray([], dtype=np.int32))  # 空占位
            costs_list.append(cost)

            kept += 1
            rows_csv.append((base, N, cost))
            it.set_postfix(keep=kept, N=N, cost=cost)

        except Exception as e:
            skipped.append((base, f"error:{type(e).__name__}:{e}"))
            continue

    # 打包
    dataset = {
        "coordinate":      coords_list,
        "distance_matrix": dists_list,
        "optimal_tour":    tours_list,                     # 空数组占位；你当前流程无需使用
        "cost":            np.asarray(costs_list, dtype=np.float64),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] saved: {args.out}")
    print(f"[stats] kept={kept} / candidates={len(cand)}")

    if skipped:
        print("[skipped]")
        for nm, why in skipped:
            print(f"  - {nm}: {why}")

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "N", "opt_cost"])
            for nm, N, c in rows_csv:
                w.writerow([nm, N, f"{c:.6f}"])
        print(f"[csv] wrote: {args.csv}")

if __name__ == "__main__":
    main()
