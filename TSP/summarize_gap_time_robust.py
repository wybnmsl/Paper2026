#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust gap/time summary for an EoH TSP run.

Usage:
    python summarize_gap_time_robust.py [RUN_ROOT]

默认行为：
- 读取 RUN_ROOT 下每个 case 的 population_generation_*.json
- 过滤掉失效解（gap 超大 / time 非法）
- 按 gap 升序取 top-K（默认 K=6），导出 gap 和 time
- 按 time 升序取 top-K（默认 K=6），导出 time 和 gap
- 导出 CSV：analysis_plots/gap_time_top6_list.csv

CSV 列：
    case,
    gap1_by_gap,time1_by_gap,...,gapK_by_gap,timeK_by_gap,
    time1_by_time,gap1_by_time,...,timeK_by_time,gapK_by_time
"""

import os
import sys
import json

import numpy as np

# ===== 默认 run 目录 =====
DEFAULT_RUN_ROOT = "results/TSPGLS_20251209_171205"

# ===== 统计与过滤超参 =====
# 认为是“惩罚解 / 失效解”的 objective 阈值，直接忽略
MAX_VALID_GAP = 1e8

# top-K 的 K
TOP_K = 6

# 下面这些容忍参数目前在本版本中不再使用，但保留以兼容可能的后续扩展
BEST_GAP_REL_TOL = 0.10   # 相对容忍：10% * best_gap
BEST_GAP_ABS_TOL = 0.05   # 绝对容忍：0.05 gap
BEST_TIME_REL_TOL = 0.50  # 允许比 best_gap 差 50%
BEST_TIME_ABS_TOL = 0.50  # 允许绝对多 0.5 gap


def get_run_root_from_argv():
    if len(sys.argv) >= 2:
        return sys.argv[1]
    return DEFAULT_RUN_ROOT


def iter_cases(run_root):
    if not os.path.isdir(run_root):
        print(f"[ERROR] RUN_ROOT not found: {run_root}")
        return
    for name in sorted(os.listdir(run_root)):
        case_dir = os.path.join(run_root, name)
        if not os.path.isdir(case_dir):
            continue
        if name.startswith(".") or name == "analysis_plots":
            continue
        yield name, case_dir


def recursive_find_files(root_dir, prefix, suffix):
    matches = []
    for cur_root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.startswith(prefix) and fname.endswith(suffix):
                matches.append(os.path.join(cur_root, fname))
    return sorted(matches)


def load_population_generations(case_name, case_dir):
    files = recursive_find_files(case_dir, "population_generation_", ".json")
    if not files:
        print(f"[info][{case_name}] no population_generation_*.json found under {case_dir}")
        return []

    records = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            gen_str = fname.replace("population_generation_", "").replace(".json", "")
            gen_idx = int(gen_str)
        except Exception:
            print(f"[warn][{case_name}] skip file (gen parse failed): {fname}")
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                pop = json.load(f)
        except Exception as e:
            print(f"[warn][{case_name}] failed to load {fpath}: {e}")
            continue

        if not isinstance(pop, list):
            print(f"[warn][{case_name}] pop file not a list: {fpath}")
            continue

        for i, indiv in enumerate(pop):
            if not isinstance(indiv, dict):
                continue
            obj = indiv.get("objective", None)
            t_eval = indiv.get("eval_time", None)
            if obj is None or t_eval is None:
                continue
            try:
                obj_f = float(obj)
                t_f = float(t_eval)
            except Exception:
                continue
            # 过滤掉明显异常的时间与 gap
            if not np.isfinite(obj_f) or not np.isfinite(t_f):
                continue
            if obj_f >= MAX_VALID_GAP:
                continue
            if t_f <= 0.0:
                # 0 或负时间一般是记录异常，直接忽略
                continue

            records.append(
                {
                    "case": case_name,
                    "generation": gen_idx,
                    "indiv_idx": i,
                    "objective": obj_f,
                    "eval_time": t_f,
                }
            )

    print(f"[info][{case_name}] loaded {len(records)} valid individual records from {len(files)} population files")
    return records


def summarize_case_topk_by_gap(records, k=TOP_K):
    """
    按 gap 升序选出 top-k: 返回 [(gap, time), ...]
    """
    if not records:
        return []

    objs = np.array([r["objective"] for r in records], dtype=float)
    times = np.array([r["eval_time"] for r in records], dtype=float)

    order = np.argsort(objs)
    top_idx = order[:k]

    top_pairs = []
    for idx in top_idx:
        top_pairs.append((float(objs[idx]), float(times[idx])))

    return top_pairs


def summarize_case_topk_by_time(records, k=TOP_K):
    """
    按 time 升序选出 top-k: 返回 [(time, gap), ...]
    """
    if not records:
        return []

    objs = np.array([r["objective"] for r in records], dtype=float)
    times = np.array([r["eval_time"] for r in records], dtype=float)

    order = np.argsort(times)
    top_idx = order[:k]

    top_pairs = []
    for idx in top_idx:
        top_pairs.append((float(times[idx]), float(objs[idx])))

    return top_pairs


def main():
    run_root = get_run_root_from_argv()
    print(f"[info] RUN_ROOT = {run_root}")
    if not os.path.isdir(run_root):
        print(f"[ERROR] RUN_ROOT not found: {run_root}")
        return

    # 每行：case, gap1_by_gap,time1_by_gap,...,gapK_by_gap,timeK_by_gap,
    #            time1_by_time,gap1_by_time,...,timeK_by_time,gapK_by_time
    summary_rows = []

    any_case = False
    for case_name, case_dir in iter_cases(run_root):
        any_case = True
        print(f"\n[Case] {case_name} | dir={case_dir}")

        recs = load_population_generations(case_name, case_dir)
        if not recs:
            print(f"[info][{case_name}] no valid records, skip.")
            continue

        top_gap_pairs = summarize_case_topk_by_gap(recs, k=TOP_K)
        top_time_pairs = summarize_case_topk_by_time(recs, k=TOP_K)

        if not top_gap_pairs and not top_time_pairs:
            print(f"[info][{case_name}] no top-k pairs found, skip.")
            continue

        # padding 到 TOP_K
        if len(top_gap_pairs) < TOP_K:
            pad_len = TOP_K - len(top_gap_pairs)
            top_gap_pairs = top_gap_pairs + [(np.nan, np.nan)] * pad_len

        if len(top_time_pairs) < TOP_K:
            pad_len = TOP_K - len(top_time_pairs)
            top_time_pairs = top_time_pairs + [(np.nan, np.nan)] * pad_len

        summary_rows.append((case_name, top_gap_pairs, top_time_pairs))

    if not any_case:
        print("[INFO] no case directories found under RUN_ROOT.")
        return

    if not summary_rows:
        print("[INFO] no valid summary rows produced.")
        return

    # 按 case 名排序
    summary_rows.sort(key=lambda x: x[0])

    # 输出到 CSV
    out_dir = os.path.join(run_root, "analysis_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gap_time_top6_list.csv")

    with open(out_path, "w", encoding="utf-8") as f:
        # 表头
        header_cols = ["case"]
        # 按 gap 的 top-K
        for i in range(TOP_K):
            header_cols.append(f"gap{i+1}_by_gap")
            header_cols.append(f"time{i+1}_by_gap")
        # 按 time 的 top-K
        for i in range(TOP_K):
            header_cols.append(f"time{i+1}_by_time")
            header_cols.append(f"gap{i+1}_by_time")
        f.write(",".join(header_cols) + "\n")

        # 写每行
        for case, top_gap_pairs, top_time_pairs in summary_rows:
            row_strs = [case]

            # gap 排序 top-K: (gap, time)
            for g, t in top_gap_pairs:
                if g is None or not np.isfinite(g):
                    row_strs.append("")
                else:
                    row_strs.append(f"{float(g):.6f}")
                if t is None or not np.isfinite(t):
                    row_strs.append("")
                else:
                    row_strs.append(f"{float(t):.6f}")

            # time 排序 top-K: (time, gap)
            for t, g in top_time_pairs:
                if t is None or not np.isfinite(t):
                    row_strs.append("")
                else:
                    row_strs.append(f"{float(t):.6f}")
                if g is None or not np.isfinite(g):
                    row_strs.append("")
                else:
                    row_strs.append(f"{float(g):.6f}")

            f.write(",".join(row_strs) + "\n")

    print(f"\n[done] top-{TOP_K} gap/time list saved to {out_path}")
    print(
        "Columns: case,"
        + ",".join([f"gap{i+1}_by_gap,time{i+1}_by_gap" for i in range(TOP_K)])
        + ","
        + ",".join([f"time{i+1}_by_time,gap{i+1}_by_time" for i in range(TOP_K)])
    )


if __name__ == "__main__":
    main()
