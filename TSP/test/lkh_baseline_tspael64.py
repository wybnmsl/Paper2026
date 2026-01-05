# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# LKH baseline for TSPAEL64.pkl
# - 将浮点 distance_matrix 缩放为整数，写成 TSPLIB EXPLICIT/FULL_MATRIX
# - 生成 LKH 参数文件，按时间限制求解
# - 解析 LKH 输出 TOUR，回到原始浮点矩阵计算真实解成本与 gap%
# - tqdm 进度条与 CSV 输出，列与 EoH/OR-Tools 基线保持一致

# 用法：
#   python zTSP/test/lkh_baseline_tspael64.py \
#     --dataset zTSP/TrainingData/TSPAEL64.pkl \
#     --csv zTSP/evaluation/tspael64_lkh_baseline.csv \
#     --lkh_bin external/LKH-3/LKH \
#     --time_limit 10 \
#     --limit 64

# 可选参数（均有默认）：
#   --int_scale 1000000        # 浮点转整数的放大因子
#   --runs 1                   # LKH RUNS
#   --seed 123456              # 随机种子
#   --trace 0                  # LKH TRACE_LEVEL
#   --workdir zTSP/evaluation/lkh_tmp   # 临时文件根目录
# """

# import os
# import sys
# import time
# import math
# import json
# import pickle
# import argparse
# import subprocess
# import numpy as np
# from tqdm import tqdm

# # ---------------------------
# # 数据集读取
# # ---------------------------
# def load_tspael64(pkl_path: str):
#     with open(pkl_path, "rb") as f:
#         data = pickle.load(f)
#     coords = data.get("coordinate")
#     dists  = data.get("distance_matrix")
#     costs  = data.get("cost")  # 最优环路成本（浮点）
#     if coords is None or dists is None or costs is None:
#         raise ValueError("TSPAEL64.pkl 缺少必需键：coordinate / distance_matrix / cost")
#     coords = [np.asarray(c) for c in coords]
#     dists  = [np.asarray(D) for D in dists]
#     costs  = [float(x) for x in costs]
#     if not (len(coords) == len(dists) == len(costs)):
#         raise ValueError("coordinate / distance_matrix / cost 长度不一致")
#     return coords, dists, costs

# # ---------------------------
# # TSPLIB 写入（EXPLICIT/FULL_MATRIX, 整数）
# # ---------------------------
# def write_tsplib_full_matrix(tsp_path: str, D_float: np.ndarray, int_scale: float = 1_000_000.0):
#     N = int(D_float.shape[0])
#     D = np.array(D_float, dtype=np.float64, copy=True)
#     if not np.all(np.isfinite(D)) or np.any(D < 0):
#         raise ValueError("距离矩阵必须非负且有限")
#     # 缩放为整数
#     M = np.rint(D * float(int_scale)).astype(np.int64)
#     np.fill_diagonal(M, 0)

#     # TSPLIB 内容
#     lines = []
#     lines.append(f"NAME : TSP_INST")
#     lines.append(f"TYPE : TSP")
#     lines.append(f"DIMENSION : {N}")
#     lines.append(f"EDGE_WEIGHT_TYPE : EXPLICIT")
#     lines.append(f"EDGE_WEIGHT_FORMAT : FULL_MATRIX")
#     lines.append(f"EDGE_WEIGHT_SECTION")
#     # 每行写 N 个整数
#     for i in range(N):
#         row = " ".join(str(int(x)) for x in M[i, :])
#         lines.append(row)
#     lines.append("EOF")

#     os.makedirs(os.path.dirname(tsp_path) or ".", exist_ok=True)
#     with open(tsp_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines))
#     return int_scale  # 返回缩放因子，供记录

# # ---------------------------
# # LKH 参数文件
# # ---------------------------
# def write_lkh_par(par_path: str,
#                   tsp_path: str,
#                   tour_path: str,
#                   time_limit_s: float = 10.0,
#                   runs: int = 1,
#                   seed: int = 123456,
#                   trace_level: int = 0):
#     """
#     常用稳健配置：MOVE_TYPE=5, PATCHING_C=3, PATCHING_A=2（经典推荐）
#     其余保持默认。你可以按需在此追加/修改参数。
#     """
#     lines = []
#     lines.append(f"PROBLEM_FILE = {os.path.abspath(tsp_path)}")
#     lines.append(f"OUTPUT_TOUR_FILE = {os.path.abspath(tour_path)}")
#     lines.append(f"RUNS = {int(runs)}")
#     lines.append(f"SEED = {int(seed)}")
#     lines.append(f"TIME_LIMIT = {int(math.ceil(float(time_limit_s)))}")
#     lines.append(f"TRACE_LEVEL = {int(trace_level)}")
#     # 经典的稳健局部搜索配置（对称 TSP）
#     lines.append(f"MOVE_TYPE = 5")
#     lines.append(f"PATCHING_C = 3")
#     lines.append(f"PATCHING_A = 2")
#     # 可选：候选边数、禁忌强度、初始周期等，默认即可
#     # lines.append("CANDIDATE_SET_TYPE = ALPHA")
#     # lines.append("MAX_CANDIDATES = 20")
#     # ...
#     os.makedirs(os.path.dirname(par_path) or ".", exist_ok=True)
#     with open(par_path, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines))

# # ---------------------------
# # 运行 LKH
# # ---------------------------
# def run_lkh(lkh_bin: str, par_path: str, cwd: str | None = None, timeout: float | None = None):
#     cmd = [os.path.abspath(lkh_bin), os.path.abspath(par_path)]
#     t0 = time.time()
#     try:
#         res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                              timeout=timeout, check=False, text=True)
#         dt = time.time() - t0
#         return res.returncode, dt, res.stdout, res.stderr
#     except subprocess.TimeoutExpired as e:
#         return 124, float(timeout or 0.0), "", f"Timeout: {e}"
#     except Exception as e:
#         return 1, time.time() - t0, "", f"Exception: {e}"

# # ---------------------------
# # 解析 TOUR 文件
# # ---------------------------
# def parse_tour(tour_path: str, N_expected: int):
#     """
#     TSPLIB TOUR 格式：
#       TYPE : TOUR
#       DIMENSION : N
#       TOUR_SECTION
#       v1
#       v2
#       ...
#       -1
#       EOF
#     节点编号从 1 开始。返回 0-based 的顺序列表（长度 N）。
#     """
#     if not os.path.exists(tour_path):
#         return None, "NO_TOUR_FILE"

#     order = []
#     in_section = False
#     with open(tour_path, "r", encoding="utf-8", errors="ignore") as f:
#         for line in f:
#             s = line.strip()
#             if not in_section:
#                 if s.upper() == "TOUR_SECTION":
#                     in_section = True
#                 continue
#             # in_section
#             if s == "-1" or s.upper() == "EOF":
#                 break
#             if not s:
#                 continue
#             try:
#                 v = int(s)
#                 order.append(v - 1)  # to 0-based
#             except:
#                 # 有些实现一行多个数字，这里兼容一下
#                 for tok in s.split():
#                     if tok == "-1":
#                         break
#                     try:
#                         v = int(tok)
#                         order.append(v - 1)
#                     except:
#                         pass

#     # 清洗、补齐
#     order = [int(x) for x in order if 0 <= x < N_expected]
#     # 去重保序
#     seen = set()
#     uniq = []
#     for v in order:
#         if v not in seen:
#             seen.add(v)
#             uniq.append(v)
#     if len(uniq) != N_expected:
#         # 若缺失，则尝试补齐
#         for v in range(N_expected):
#             if v not in seen:
#                 uniq.append(v)
#     if len(uniq) != N_expected:
#         return None, f"BAD_TOUR_LEN:{len(uniq)}"
#     return uniq, ""

# # ---------------------------
# # 解环成本（在原始浮点矩阵上）
# # ---------------------------
# def cycle_cost_from_order(D: np.ndarray, order: list[int]) -> float:
#     n = len(order)
#     s = 0.0
#     for i in range(n):
#         a = order[i]
#         b = order[(i + 1) % n]
#         s += float(D[a, b])
#     return float(s)

# # ---------------------------
# # 批量评测 & CSV
# # ---------------------------
# def lkh_baseline_to_csv(dataset_path: str,
#                         csv_out: str,
#                         lkh_bin: str,
#                         time_limit: float = 10.0,
#                         limit: int | None = None,
#                         int_scale: float = 1_000_000.0,
#                         runs: int = 1,
#                         seed: int = 123456,
#                         trace_level: int = 0,
#                         workdir: str = "zTSP/evaluation/lkh_tmp",
#                         tqdm_disable: bool = False):
#     try:
#         import pandas as pd
#         use_pandas = True
#     except Exception:
#         import csv as _csv
#         use_pandas = False

#     coords, dists, costs = load_tspael64(dataset_path)
#     total = len(dists)
#     n = min(total, int(limit)) if limit is not None else total

#     rows = []
#     it = tqdm(range(n), disable=tqdm_disable, desc="LKH baseline", unit="inst")

#     os.makedirs(workdir, exist_ok=True)

#     for i in it:
#         D = dists[i]
#         opt = float(costs[i])
#         N  = int(D.shape[0])

#         inst_dir = os.path.join(workdir, f"inst_{i:03d}")
#         os.makedirs(inst_dir, exist_ok=True)
#         tsp_path  = os.path.join(inst_dir, "problem.tsp")
#         par_path  = os.path.join(inst_dir, "params.par")
#         tour_path = os.path.join(inst_dir, "solution.tour")

#         # 生成 TSPLIB & PAR
#         used_scale = write_tsplib_full_matrix(tsp_path, D, int_scale=int_scale)
#         write_lkh_par(par_path, tsp_path, tour_path,
#                       time_limit_s=time_limit, runs=runs, seed=seed, trace_level=trace_level)

#         t0 = time.time()
#         rc, lkh_time, out, err = run_lkh(lkh_bin, par_path, cwd=inst_dir, timeout=time_limit + 5.0)
#         dt = time.time() - t0

#         error_msg = ""
#         sol_cost = float("nan")
#         gap = float("nan")

#         if rc == 0 and os.path.exists(tour_path):
#             order, perr = parse_tour(tour_path, N_expected=N)
#             if order is None:
#                 error_msg = f"parse_tour:{perr}"
#             else:
#                 sol_cost = cycle_cost_from_order(D, order)
#                 gap = (sol_cost - opt) / opt * 100.0
#                 if abs(gap) < 1e-9:
#                     gap = 0.0
#         else:
#             # 记录 LKH 的报错
#             if rc == 124:
#                 error_msg = "LKH_timeout"
#             else:
#                 error_msg = f"LKH_rc={rc}"
#             if err:
#                 error_msg += f"|{err.strip().splitlines()[-1][:200]}"

#         rows.append({
#             "idx": i,
#             "N": N,
#             "opt_cost": opt,
#             "sol_cost_est": sol_cost,      # 与其他基线同列名，写真实成本
#             "gap_percent": gap,
#             "time_s": dt,
#             "time_limit_s": float(time_limit),
#             "loop_max": "",                # LKH 不适用，留空
#             "max_no_improve": "",
#             "k": "",
#             "top_k": "",
#             "perturb": "",
#             "perturb_interval": "",
#             "error": error_msg,
#         })

#     os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
#     if use_pandas:
#         import pandas as pd
#         pd.DataFrame(rows).to_csv(csv_out, index=False)
#     else:
#         import csv as _csv
#         keys = list(rows[0].keys()) if rows else [
#             "idx","N","opt_cost","sol_cost_est","gap_percent","time_s","time_limit_s",
#             "loop_max","max_no_improve","k","top_k","perturb","perturb_interval","error"
#         ]
#         with open(csv_out, "w", newline="", encoding="utf-8") as f:
#             w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

#     valid = [r for r in rows if isinstance(r["gap_percent"], (int, float)) and math.isfinite(r["gap_percent"])]
#     avg_gap  = float(np.mean([r["gap_percent"] for r in valid])) if valid else float("nan")
#     avg_time = float(np.mean([r["time_s"] for r in rows])) if rows else float("nan")
#     print(f"[lkh-summary] instances={n}/{total} | avg_gap%={avg_gap:.6f} | avg_time_s={avg_time:.3f} | csv -> {csv_out}")
#     return rows

# # ---------------------------
# # CLI
# # ---------------------------
# def main():
#     parser = argparse.ArgumentParser(description="LKH baseline on TSPAEL64.pkl with CSV output (tqdm).")
#     parser.add_argument("--dataset", type=str, default="zTSP/TrainingData/TSPAEL64.pkl",
#                         help="Path to TSPAEL64.pkl (default: zTSP/TrainingData/TSPAEL64.pkl)")
#     parser.add_argument("--csv", type=str, default="zTSP/evaluation/tspael64_lkh_baseline.csv",
#                         help="Output CSV path (default: zTSP/evaluation/tspael64_lkh_baseline.csv)")
#     parser.add_argument("--lkh_bin", type=str, required=True,
#                         help="Path to LKH executable (e.g., external/LKH-3/LKH)")
#     parser.add_argument("--time_limit", type=float, default=10.0,
#                         help="Time limit per instance in seconds (default: 10)")
#     parser.add_argument("--limit", type=int, default=None,
#                         help="Only evaluate first N instances (default: all)")
#     parser.add_argument("--int_scale", type=float, default=1_000_000.0,
#                         help="Scale factor from float distances to integer (default: 1e6)")
#     parser.add_argument("--runs", type=int, default=1, help="LKH RUNS (default: 1)")
#     parser.add_argument("--seed", type=int, default=123456, help="LKH SEED (default: 123456)")
#     parser.add_argument("--trace", type=int, default=0, help="LKH TRACE_LEVEL (default: 0)")
#     parser.add_argument("--workdir", type=str, default="zTSP/evaluation/lkh_tmp",
#                         help="Working dir to store temp tsp/par/tour files")
#     parser.add_argument("--no_tqdm", action="store_true", help="Disable progress bar")
#     args = parser.parse_args()

#     lkh_baseline_to_csv(
#         dataset_path=args.dataset,
#         csv_out=args.csv,
#         lkh_bin=args.lkh_bin,
#         time_limit=args.time_limit,
#         limit=args.limit,
#         int_scale=args.int_scale,
#         runs=args.runs,
#         seed=args.seed,
#         trace_level=args.trace,
#         workdir=args.workdir,
#         tqdm_disable=args.no_tqdm
#     )

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LKH baseline for TSPAEL64.pkl (robust auto-scaling)

- 将浮点/整数 distance_matrix 写为 TSPLIB EXPLICIT/FULL_MATRIX（整数）
- 自适应缩放：若矩阵本为整数 -> scale=1；否则将用户给定的 int_scale 下调到不超阈值
- 生成 LKH 参数文件（显式 PRECISION=1），按时间限制求解
- 解析 LKH TOUR，回到原始浮点矩阵计算真实解成本与 gap%
- tqdm 进度条与 CSV 输出，列与 EoH/OR-Tools 基线保持一致，并新增 scale_used

用法（示例）：
  python zTSP/test/lkh_baseline_tspael64.py \
    --dataset zTSP/TrainingData/TSPAEL64.pkl \
    --csv zTSP/evaluation/tspael64_lkh_baseline.csv \
    --lkh_bin external/LKH-3.0.13/LKH \
    --time_limit 10 \
    --limit 64
"""

import os
import sys
import time
import math
import json
import pickle
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

INT_MAX = 2_147_483_647

# ---------------------------
# 数据集读取
# ---------------------------
def load_tspael64(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    coords = data.get("coordinate")
    dists  = data.get("distance_matrix")
    costs  = data.get("cost")
    if coords is None or dists is None or costs is None:
        raise ValueError("TSPAEL64.pkl 缺少必需键：coordinate / distance_matrix / cost")
    coords = [np.asarray(c) for c in coords]
    dists  = [np.asarray(D) for D in dists]
    costs  = [float(x) for x in (costs.tolist() if hasattr(costs, "tolist") else costs)]
    if not (len(coords) == len(dists) == len(costs)):
        raise ValueError("coordinate / distance_matrix / cost 长度不一致")
    return coords, dists, costs

# ---------------------------
# 自适应整数缩放（关键修复）
# ---------------------------
def choose_safe_scale(D: np.ndarray, user_scale: float, precision: int = 1) -> int:
    """
    返回一个不超过 user_scale 的安全整数缩放因子，使得 max(round(D*scale)) 不会超过 INT_MAX/2/PRECISION。
    若 D 近似整数（<=1e-9），则直接返回 1。
    """
    D = np.asarray(D, dtype=float)
    # 近似整数判定
    if np.allclose(D, np.rint(D), atol=1e-9):
        return 1

    max_allowed = (INT_MAX // 2) // max(1, precision)  # 与报错阈值一致
    dmax = float(np.max(D))
    if dmax <= 0:
        return 1

    # 用户建议缩放
    s_user = int(max(1, round(float(user_scale))))
    # 理论上限（向下取整）
    s_bound = int(max_allowed // max(1.0, dmax))

    if s_bound <= 0:
        # 极端情况下仍保证至少为 1（同时提醒：可能损失精度）
        return 1
    return int(min(s_user, s_bound))

# ---------------------------
# TSPLIB 写入（EXPLICIT/FULL_MATRIX, 整数）
# ---------------------------
def write_tsplib_full_matrix(tsp_path: str, D_float: np.ndarray,
                             int_scale: float = 1_000_000.0,
                             precision: int = 1):
    """
    将 D_float 写成 TSPLIB EXPLICIT/FULL_MATRIX。
    - 若 D 近似整数：scale=1
    - 否则按 user_scale 自适应下调，确保不超 INT_MAX/2/PRECISION
    返回实际使用的 scale。
    """
    N = int(D_float.shape[0])
    D = np.array(D_float, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("距离矩阵必须非负且有限")

    scale = choose_safe_scale(D, int_scale, precision=precision)
    M = np.rint(D * scale).astype(np.int64)
    np.fill_diagonal(M, 0)

    # 再次安全检查
    max_allowed = (INT_MAX // 2) // max(1, precision)
    if int(M.max(initial=0)) > max_allowed:
        # 再退一步（极少见）
        tight = int(max_allowed // max(1.0, float(np.max(D))))
        scale = max(1, tight)
        M = np.rint(D * scale).astype(np.int64)
        np.fill_diagonal(M, 0)

    lines = []
    lines.append(f"NAME : TSP_INST")
    lines.append(f"TYPE : TSP")
    lines.append(f"DIMENSION : {N}")
    lines.append(f"EDGE_WEIGHT_TYPE : EXPLICIT")
    lines.append(f"EDGE_WEIGHT_FORMAT : FULL_MATRIX")
    lines.append(f"EDGE_WEIGHT_SECTION")
    for i in range(N):
        lines.append(" ".join(str(int(x)) for x in M[i, :]))
    lines.append("EOF")

    os.makedirs(os.path.dirname(tsp_path) or ".", exist_ok=True)
    with open(tsp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return scale

# ---------------------------
# LKH 参数文件
# ---------------------------
def write_lkh_par(par_path: str,
                  tsp_path: str,
                  tour_path: str,
                  time_limit_s: float = 10.0,
                  runs: int = 1,
                  seed: int = 123456,
                  trace_level: int = 0,
                  precision: int = 1):
    """
    增加 PRECISION=1，避免 LKH 自行推断导致阈值变化。
    """
    lines = []
    lines.append(f"PROBLEM_FILE = {os.path.abspath(tsp_path)}")
    lines.append(f"OUTPUT_TOUR_FILE = {os.path.abspath(tour_path)}")
    lines.append(f"RUNS = {int(runs)}")
    lines.append(f"SEED = {int(seed)}")
    lines.append(f"TIME_LIMIT = {int(math.ceil(float(time_limit_s)))}")
    lines.append(f"TRACE_LEVEL = {int(trace_level)}")
    lines.append(f"PRECISION = {int(precision)}")
    # 经典稳健配置（对称 TSP）
    lines.append(f"MOVE_TYPE = 5")
    lines.append(f"PATCHING_C = 3")
    lines.append(f"PATCHING_A = 2")

    os.makedirs(os.path.dirname(par_path) or ".", exist_ok=True)
    with open(par_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------------------------
# 运行 LKH
# ---------------------------
def run_lkh(lkh_bin: str, par_path: str, cwd: str | None = None, timeout: float | None = None):
    cmd = [os.path.abspath(lkh_bin), os.path.abspath(par_path)]
    t0 = time.time()
    try:
        res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=timeout, check=False, text=True)
        dt = time.time() - t0
        return res.returncode, dt, res.stdout, res.stderr
    except subprocess.TimeoutExpired as e:
        return 124, float(timeout or 0.0), "", f"Timeout: {e}"
    except Exception as e:
        return 1, time.time() - t0, "", f"Exception: {e}"

# ---------------------------
# 解析 TOUR 文件
# ---------------------------
def parse_tour(tour_path: str, N_expected: int):
    if not os.path.exists(tour_path):
        return None, "NO_TOUR_FILE"
    order = []
    in_section = False
    with open(tour_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not in_section:
                if s.upper() == "TOUR_SECTION":
                    in_section = True
                continue
            if s == "-1" or s.upper() == "EOF":
                break
            if not s:
                continue
            # 支持一行多个数
            for tok in s.split():
                if tok == "-1":
                    break
                try:
                    v = int(tok) - 1
                    order.append(v)
                except:
                    pass
    order = [int(x) for x in order if 0 <= x < N_expected]
    seen, uniq = set(), []
    for v in order:
        if v not in seen:
            seen.add(v); uniq.append(v)
    if len(uniq) != N_expected:
        for v in range(N_expected):
            if v not in seen:
                uniq.append(v)
    if len(uniq) != N_expected:
        return None, f"BAD_TOUR_LEN:{len(uniq)}"
    return uniq, ""

# ---------------------------
# 解环成本（在原始浮点矩阵上）
# ---------------------------
def cycle_cost_from_order(D: np.ndarray, order: list[int]) -> float:
    n = len(order)
    s = 0.0
    for i in range(n):
        a = order[i]
        b = order[(i + 1) % n]
        s += float(D[a, b])
    return float(s)

# ---------------------------
# 批量评测 & CSV
# ---------------------------
def lkh_baseline_to_csv(dataset_path: str,
                        csv_out: str,
                        lkh_bin: str,
                        time_limit: float = 10.0,
                        limit: int | None = None,
                        int_scale: float = 1_000_000.0,
                        runs: int = 1,
                        seed: int = 123456,
                        trace_level: int = 0,
                        workdir: str = "zTSP/evaluation/lkh_tmp",
                        tqdm_disable: bool = False):
    try:
        import pandas as pd
        use_pandas = True
    except Exception:
        import csv as _csv
        use_pandas = False

    coords, dists, costs = load_tspael64(dataset_path)
    total = len(dists)
    n = min(total, int(limit)) if limit is not None else total

    rows = []
    it = tqdm(range(n), disable=tqdm_disable, desc="LKH baseline", unit="inst")
    os.makedirs(workdir, exist_ok=True)

    PRECISION = 1  # 显式设定，与阈值计算一致

    for i in it:
        D = dists[i]
        opt = float(costs[i])
        N  = int(D.shape[0])

        inst_dir = os.path.join(workdir, f"inst_{i:03d}")
        os.makedirs(inst_dir, exist_ok=True)
        tsp_path  = os.path.join(inst_dir, "problem.tsp")
        par_path  = os.path.join(inst_dir, "params.par")
        tour_path = os.path.join(inst_dir, "solution.tour")

        # 生成 TSPLIB & PAR（关键：使用自适应 scale）
        used_scale = write_tsplib_full_matrix(tsp_path, D, int_scale=int_scale, precision=PRECISION)
        write_lkh_par(par_path, tsp_path, tour_path,
                      time_limit_s=time_limit, runs=runs, seed=seed,
                      trace_level=trace_level, precision=PRECISION)

        t0 = time.time()
        rc, lkh_time, out, err = run_lkh(lkh_bin, par_path, cwd=inst_dir, timeout=time_limit + 5.0)
        dt = time.time() - t0

        error_msg = ""
        sol_cost = float("nan")
        gap = float("nan")

        if rc == 0 and os.path.exists(tour_path):
            order, perr = parse_tour(tour_path, N_expected=N)
            if order is None:
                error_msg = f"parse_tour:{perr}"
            else:
                sol_cost = cycle_cost_from_order(D, order)
                gap = (sol_cost - opt) / opt * 100.0
                if abs(gap) < 1e-9:
                    gap = 0.0
        else:
            if rc == 124:
                error_msg = "LKH_timeout"
            else:
                error_msg = f"LKH_rc={rc}"
            if err:
                error_msg += f"|{err.strip().splitlines()[-1][:200]}"

        rows.append({
            "idx": i,
            "N": N,
            "opt_cost": opt,
            "sol_cost_est": sol_cost,
            "gap_percent": gap,
            "time_s": dt,
            "time_limit_s": float(time_limit),
            "loop_max": "",
            "max_no_improve": "",
            "k": "",
            "top_k": "",
            "perturb": "",
            "perturb_interval": "",
            "scale_used": used_scale,      # 新增：记录实际 scale
            "error": error_msg,
        })

    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    if use_pandas:
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_out, index=False)
    else:
        import csv as _csv
        keys = list(rows[0].keys()) if rows else [
            "idx","N","opt_cost","sol_cost_est","gap_percent","time_s","time_limit_s",
            "loop_max","max_no_improve","k","top_k","perturb","perturb_interval","scale_used","error"
        ]
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # 控制台摘要
    finite = [r for r in rows if isinstance(r["gap_percent"], (int, float)) and np.isfinite(r["gap_percent"])]
    avg_gap  = float(np.mean([r["gap_percent"] for r in finite])) if finite else float("nan")
    avg_time = float(np.mean([r["time_s"] for r in rows])) if rows else float("nan")
    print(f"[lkh-summary] instances={n}/{total} | avg_gap%={avg_gap:.6f} | avg_time_s={avg_time:.3f} | csv -> {csv_out}")
    return rows

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="LKH baseline on TSPAEL64.pkl with CSV output (tqdm, auto-scaling).")
    parser.add_argument("--dataset", type=str, default="zTSP/TrainingData/TSPAEL64.pkl",
                        help="Path to TSPAEL64.pkl (default: zTSP/TrainingData/TSPAEL64.pkl)")
    parser.add_argument("--csv", type=str, default="zTSP/evaluation/tspael64_lkh_baseline.csv",
                        help="Output CSV path (default: zTSP/evaluation/tspael64_lkh_baseline.csv)")
    parser.add_argument("--lkh_bin", type=str, required=True,
                        help="Path to LKH executable (e.g., external/LKH-3.0.13/LKH)")
    parser.add_argument("--time_limit", type=float, default=10.0,
                        help="Time limit per instance in seconds (default: 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate first N instances (default: all)")
    parser.add_argument("--int_scale", type=float, default=1_000_000.0,
                        help="Preferred scale factor; will be auto-shrunk if needed (default: 1e6)")
    parser.add_argument("--runs", type=int, default=1, help="LKH RUNS (default: 1)")
    parser.add_argument("--seed", type=int, default=123456, help="LKH SEED (default: 123456)")
    parser.add_argument("--trace", type=int, default=0, help="LKH TRACE_LEVEL (default: 0)")
    parser.add_argument("--workdir", type=str, default="zTSP/evaluation/lkh_tmp",
                        help="Working dir to store temp tsp/par/tour files")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    lkh_baseline_to_csv(
        dataset_path=args.dataset,
        csv_out=args.csv,
        lkh_bin=args.lkh_bin,
        time_limit=args.time_limit,
        limit=args.limit,
        int_scale=args.int_scale,
        runs=args.runs,
        seed=args.seed,
        trace_level=args.trace,
        workdir=args.workdir,
        tqdm_disable=args.no_tqdm
    )

if __name__ == "__main__":
    main()
