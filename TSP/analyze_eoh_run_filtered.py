#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对现有 LDR 统计做一次“去噪版”分析：
- 读取各 case 的 t_history / em_history；
- 用 IQR + abs clip 去掉极端 tLDR outlier；
- 丢掉样本太少的 case；
- 重新画 T-stage / EM-operator 的 tLDR 分布（*_filtered.png）。
"""

import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import analyze_eoh_run as base  # 依赖同目录下的原分析脚本


# ====== 过滤超参数 ======

# 全局 IQR 过滤 + 绝对值裁剪
T_IQR_K = 2.0
T_ABS_CLIP = 50.0       # T 阶段 tLDR 极端值裁剪

EM_IQR_K = 2.0
EM_ABS_CLIP = 10.0      # e/m 阶段 tLDR 极端值裁剪

# 按 case 的最小样本数阈值
T_MIN_SAMPLES_PER_CASE = 30
EM_MIN_SAMPLES_PER_CASE = 10


# ====== 工具函数 ======

def _finite_tldr(val):
    """把 rec['tLDR'] 转成 float 并检查 finite。无效返回 None。"""
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if math.isnan(v) or not math.isfinite(v):
        return None
    return v


def _compute_iqr_bounds(values, k=1.5, abs_clip=None):
    """给一批值算 IQR range 和可选的绝对值裁剪。"""
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    if abs_clip is not None:
        lo = max(lo, -abs_clip)
        hi = min(hi, abs_clip)
    return lo, hi


def _filter_records_by_tldr(records, lo, hi):
    """按给定上下界过滤 tLDR。保留没有 tLDR 的记录（用于 case 统计时也能落在里边）。"""
    if lo is None or hi is None:
        return records

    filtered = []
    dropped = 0
    for rec in records:
        v = _finite_tldr(rec.get("tLDR"))
        if v is None:
            # 没有 tLDR 的记录直接保留（一般不会用于 boxplot）
            filtered.append(rec)
            continue
        if lo <= v <= hi:
            filtered.append(rec)
        else:
            dropped += 1
    return filtered, dropped


def _filter_records_by_case_size(records, min_samples):
    """按 case 粒度过滤：只保留样本数 >= min_samples 的 case。"""
    count_by_case = defaultdict(int)
    for rec in records:
        case = rec.get("case") or "unknown"
        v = _finite_tldr(rec.get("tLDR"))
        if v is not None:
            count_by_case[case] += 1

    keep_cases = {c for c, n in count_by_case.items() if n >= min_samples}
    filtered = [rec for rec in records if (rec.get("case") or "unknown") in keep_cases]
    return filtered, keep_cases


# ====== 画“去噪版”箱线图 ======

def plot_ldr_t_stage_boxplot_filtered(t_records_all, out_dir):
    if not t_records_all:
        print("[info][filtered] no T-history records to plot")
        return

    values_by_stage = defaultdict(list)
    for rec in t_records_all:
        stage = rec.get("stage") or "unknown"
        v = _finite_tldr(rec.get("tLDR"))
        if v is None:
            continue
        values_by_stage[stage].append(v)

    if not values_by_stage:
        print("[info][filtered] no valid tLDR for T-stages")
        return

    stages = sorted(values_by_stage.keys())
    data = [values_by_stage[s] for s in stages]

    plt.figure(figsize=(max(6, len(stages) * 1.0), 4))
    plt.boxplot(data, labels=stages, showfliers=False)
    plt.xlabel("T-stage (stage in t_history)")
    plt.ylabel("tLDR")
    plt.title("tLDR distribution by T-stage (all cases, filtered)")
    plt.grid(True, axis="y")
    plt.tight_layout()

    fpath = base.os.path.join(out_dir, "ldr_t_stage_boxplot_filtered.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot][filtered] saved {fpath}")


def plot_ldr_em_operator_boxplot_filtered(em_records_all, out_dir, min_count=10):
    if not em_records_all:
        print("[info][filtered] no EM-history records to plot")
        return

    values_by_op = defaultdict(list)
    for rec in em_records_all:
        op = rec.get("operator") or "unknown"
        v = _finite_tldr(rec.get("tLDR"))
        if v is None:
            continue
        values_by_op[op].append(v)

    # 只保留样本数充足的 operator
    values_by_op = {op: vals for op, vals in values_by_op.items() if len(vals) >= min_count}
    if not values_by_op:
        print(f"[info][filtered] no EM-operators with >= {min_count} valid tLDR samples")
        return

    ops = sorted(values_by_op.keys())
    data = [values_by_op[op] for op in ops]

    plt.figure(figsize=(max(6, len(ops) * 1.0), 4))
    plt.boxplot(data, labels=ops, showfliers=False)
    plt.xlabel("Operator (e/m)")
    plt.ylabel("tLDR")
    plt.title(f"tLDR distribution by EM-operator (filtered, n>={min_count})")
    plt.grid(True, axis="y")
    plt.tight_layout()

    fpath = base.os.path.join(out_dir, "ldr_em_operator_boxplot_filtered.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot][filtered] saved {fpath}")


# ====== 主流程 ======

def main():
    run_root = base.get_run_root_from_argv()
    print(f"[info][filtered] RUN_ROOT = {run_root}")

    if not base.os.path.isdir(run_root):
        print(f"[ERROR][filtered] RUN_ROOT not found: {run_root}")
        return

    out_dir = base.ensure_plot_dir(run_root)
    print(f"[info][filtered] plots will be saved under: {out_dir}")

    t_records_all = []
    em_records_all = []

    any_case = False
    for case_name, case_dir in base.iter_cases(run_root):
        any_case = True
        print(f"[Case][filtered] {case_name} | dir={case_dir}")

        t_records, em_records = base.load_history_records(case_name, case_dir)
        t_records_all.extend(t_records)
        em_records_all.extend(em_records)

    if not any_case:
        print("[INFO][filtered] no case directories found under RUN_ROOT.")
        return

    # -------- T 阶段：全局 IQR + abs clip → case 过滤 --------
    all_tldr_t = [
        _finite_tldr(rec.get("tLDR"))
        for rec in t_records_all
        if _finite_tldr(rec.get("tLDR")) is not None
    ]
    t_lo, t_hi = _compute_iqr_bounds(all_tldr_t, k=T_IQR_K, abs_clip=T_ABS_CLIP)
    print(f"[info][filtered] T-stage tLDR bounds: lo={t_lo:.3f}, hi={t_hi:.3f} (IQR k={T_IQR_K}, abs_clip={T_ABS_CLIP})")

    t_records_all, dropped_t = _filter_records_by_tldr(t_records_all, t_lo, t_hi)
    print(f"[info][filtered] T-stage: dropped {dropped_t} outlier records by tLDR")

    t_records_all, keep_cases_t = _filter_records_by_case_size(t_records_all, T_MIN_SAMPLES_PER_CASE)
    print(f"[info][filtered] T-stage: keep {len(keep_cases_t)} cases with >= {T_MIN_SAMPLES_PER_CASE} samples")

    # -------- e/m 阶段：全局 IQR + abs clip → case 过滤 --------
    all_tldr_em = [
        _finite_tldr(rec.get("tLDR"))
        for rec in em_records_all
        if _finite_tldr(rec.get("tLDR")) is not None
    ]
    em_lo, em_hi = _compute_iqr_bounds(all_tldr_em, k=EM_IQR_K, abs_clip=EM_ABS_CLIP)
    print(f"[info][filtered] EM tLDR bounds: lo={em_lo:.3f}, hi={em_hi:.3f} (IQR k={EM_IQR_K}, abs_clip={EM_ABS_CLIP})")

    em_records_all, dropped_em = _filter_records_by_tldr(em_records_all, em_lo, em_hi)
    print(f"[info][filtered] EM: dropped {dropped_em} outlier records by tLDR")

    em_records_all, keep_cases_em = _filter_records_by_case_size(em_records_all, EM_MIN_SAMPLES_PER_CASE)
    print(f"[info][filtered] EM: keep {len(keep_cases_em)} cases with >= {EM_MIN_SAMPLES_PER_CASE} samples")

    # -------- 重新画“去噪版”箱线图 --------
    plot_ldr_t_stage_boxplot_filtered(t_records_all, out_dir)
    plot_ldr_em_operator_boxplot_filtered(em_records_all, out_dir, min_count=EM_MIN_SAMPLES_PER_CASE)

    print("\n[Done][filtered] All filtered plots saved to", out_dir)


if __name__ == "__main__":
    main()
