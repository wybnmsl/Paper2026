#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ===== 默认 run 目录 =====
# 你可以：
#   1) 直接修改这里的默认值，或者
#   2) 用命令行参数传： python analyze_eoh_run.py results/TSPGLS_20251209_171205
DEFAULT_RUN_ROOT = "results/TSPGLS_20251209_171205"


# ---------- 基础工具 ----------

def get_run_root_from_argv():
    import sys
    if len(sys.argv) >= 2:
        return sys.argv[1]
    return DEFAULT_RUN_ROOT


def iter_cases(run_root):
    """
    遍历 run_root 下的 case 子目录：
    例如：results/TSPGLS_xxx/att48, bayg29, ...
    """
    if not os.path.isdir(run_root):
        print(f"[ERROR] RUN_ROOT not found: {run_root}")
        return

    for name in sorted(os.listdir(run_root)):
        case_dir = os.path.join(run_root, name)
        if not os.path.isdir(case_dir):
            continue
        # 略过分析输出目录等
        if name.startswith(".") or name == "analysis_plots":
            continue
        yield name, case_dir


def recursive_find_files(root_dir, prefix, suffix):
    """
    在 root_dir 下递归查找所有以 prefix 开头、suffix 结尾的文件。
    返回完整路径列表。
    """
    matches = []
    for cur_root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.startswith(prefix) and fname.endswith(suffix):
                matches.append(os.path.join(cur_root, fname))
    return sorted(matches)


# ---------- 加载数据 ----------

def load_population_generations(case_name, case_dir):
    """
    递归地找 case 目录里的 population_generation_*.json，
    返回 records: [{case, generation, indiv_idx, objective, eval_time}, ...]
    """
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
            records.append(
                {
                    "case": case_name,
                    "generation": gen_idx,
                    "indiv_idx": i,
                    "objective": obj_f,
                    "eval_time": t_f,
                }
            )

    print(f"[info][{case_name}] loaded {len(records)} individual records from {len(files)} population files")
    return records


def load_history_records(case_name, case_dir):
    """
    递归地找 case 目录里的 pop_*.json，
    把 individual.other_inf.t_history / em_history 展开为两类记录。
    """
    files = recursive_find_files(case_dir, "pop_", ".json")
    if not files:
        print(f"[info][{case_name}] no pop_*.json history files found under {case_dir}")
        return [], []

    t_records = []
    em_records = []

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                h = json.load(f)
        except Exception as e:
            print(f"[warn][{case_name}] failed to load history {fpath}: {e}")
            continue

        if not isinstance(h, dict):
            continue

        phase = h.get("phase")
        op_name = h.get("operator")
        pop_index = h.get("pop_index")

        # parents / offsprings 可能是 list 或 list[list]
        for role in ["parents", "offsprings"]:
            container = h.get(role, []) or []
            # 兼容两层嵌套的情况
            if container and isinstance(container[0], list):
                indiv_list = [ind for sub in container for ind in (sub or [])]
            else:
                indiv_list = container

            for indiv in indiv_list:
                if not isinstance(indiv, dict):
                    continue
                other = indiv.get("other_inf") or {}

                # T 阶段 t_history
                for rec in other.get("t_history", []) or []:
                    if not isinstance(rec, dict):
                        continue
                    t_records.append(
                        {
                            "case": case_name,
                            "phase": phase,
                            "operator": op_name,
                            "pop_index": pop_index,
                            "gen": rec.get("gen"),
                            "stage": rec.get("stage"),
                            "accepted": rec.get("accepted"),
                            "reason": rec.get("reason"),
                            "iLDR": rec.get("iLDR"),
                            "tLDR": rec.get("tLDR"),
                        }
                    )

                # e/m 阶段 em_history
                for rec in other.get("em_history", []) or []:
                    if not isinstance(rec, dict):
                        continue
                    em_records.append(
                        {
                            "case": case_name,
                            "phase": phase,
                            "operator": rec.get("operator") or op_name,
                            "gen": rec.get("gen"),
                            "iLDR": rec.get("iLDR"),
                            "tLDR": rec.get("tLDR"),
                        }
                    )

    print(f"[info][{case_name}] loaded {len(t_records)} T-history records, {len(em_records)} EM-history records from {len(files)} files")
    return t_records, em_records


# ---------- 画图工具 ----------

def ensure_plot_dir(run_root):
    out_dir = os.path.join(run_root, "analysis_plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_best_gap_curve(case_name, pop_records, out_dir):
    if not pop_records:
        return

    best_by_gen = {}
    for rec in pop_records:
        g = rec["generation"]
        obj = rec["objective"]
        if g not in best_by_gen or obj < best_by_gen[g]:
            best_by_gen[g] = obj

    gens = sorted(best_by_gen.keys())
    best_gap = [best_by_gen[g] for g in gens]

    plt.figure()
    plt.plot(gens, best_gap, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best objective (gap)")
    plt.title(f"{case_name}: Best gap per generation")
    plt.grid(True)
    plt.tight_layout()

    fpath = os.path.join(out_dir, f"{case_name}_best_gap_curve.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot][{case_name}] saved {fpath}")


def plot_time_curve(case_name, pop_records, out_dir):
    if not pop_records:
        return

    times_by_gen = defaultdict(list)
    for rec in pop_records:
        g = rec["generation"]
        times_by_gen[g].append(rec["eval_time"])

    gens = sorted(times_by_gen.keys())
    median_time = [float(np.median(times_by_gen[g])) for g in gens]

    plt.figure()
    plt.plot(gens, median_time, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Median eval_time (s)")
    plt.title(f"{case_name}: Eval time (median) per generation")
    plt.grid(True)
    plt.tight_layout()

    fpath = os.path.join(out_dir, f"{case_name}_time_curve.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot][{case_name}] saved {fpath}")


def plot_gap_vs_time(case_name, pop_records, out_dir):
    if not pop_records:
        return

    xs = [rec["eval_time"] for rec in pop_records]
    ys = [rec["objective"] for rec in pop_records]

    plt.figure()
    plt.scatter(xs, ys, s=10)
    plt.xlabel("Eval time (s)")
    plt.ylabel("Objective (gap)")
    plt.title(f"{case_name}: Gap vs eval_time (all individuals)")
    plt.grid(True)
    plt.tight_layout()

    fpath = os.path.join(out_dir, f"{case_name}_gap_vs_time.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot][{case_name}] saved {fpath}")


def plot_gap_by_step(case_name, pop_records, out_dir):
    """
    把所有个体当作“迭代步”（gap）：
    - 先按 generation 升序排序；
    - 每个 generation 内部按 objective (gap) 降序排序；
    - x 轴是 step 索引，y 轴是该个体的 gap。
    """
    if not pop_records:
        return

    recs = sorted(
        pop_records,
        key=lambda r: (r["generation"], -r["objective"])
    )
    steps = np.arange(len(recs))
    gaps = [r["objective"] for r in recs]

    plt.figure()
    plt.plot(steps, gaps, marker=".", linewidth=0.7)
    plt.xlabel("Step index (all individuals, sorted by gen & gap desc)")
    plt.ylabel("Objective (gap)")
    plt.title(f"{case_name}: gap by step (within gen sorted by gap desc)")
    plt.grid(True)
    plt.tight_layout()

    fpath = os.path.join(out_dir, f"{case_name}_gap_by_step.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot][{case_name}] saved {fpath}")


def plot_time_by_step(case_name, pop_records, out_dir):
    """
    把所有个体当作“迭代步”（time）：
    - 先按 generation 升序排序；
    - 每个 generation 内部按 eval_time 降序排序；
    - x 轴是 step 索引，y 轴是该个体的 eval_time。
    """
    if not pop_records:
        return

    recs = sorted(
        pop_records,
        key=lambda r: (r["generation"], -r["eval_time"])
    )
    steps = np.arange(len(recs))
    times = [r["eval_time"] for r in recs]

    plt.figure()
    plt.plot(steps, times, marker=".", linewidth=0.7)
    plt.xlabel("Step index (all individuals, sorted by gen & time desc)")
    plt.ylabel("Eval time (s)")
    plt.title(f"{case_name}: eval time by step (within gen sorted by time desc)")
    plt.grid(True)
    plt.tight_layout()

    fpath = os.path.join(out_dir, f"{case_name}_time_by_step.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot][{case_name}] saved {fpath}")


def plot_initial_vs_final_gap(all_case_stats, out_dir):
    """
    画 allcases_initial_vs_final_gap.png：
    - Gen0: best / mean / worst
    - Last gen: best
    """
    if not all_case_stats:
        print("[info] no case stats to plot (initial vs final)")
        return

    cases = sorted(all_case_stats.keys())

    first_best_vals = []
    first_mean_vals = []
    first_worst_vals = []
    last_best_vals = []

    for c in cases:
        stats = all_case_stats[c]
        first_best_vals.append(stats["first_best"])
        first_mean_vals.append(stats["first_mean"])
        first_worst_vals.append(stats["first_worst"])
        last_best_vals.append(stats["last_best"])

    x = np.arange(len(cases))
    width = 0.18

    plt.figure(figsize=(max(8, len(cases) * 0.5), 4))
    plt.bar(x - 1.5 * width, first_best_vals, width, label="Gen0 best")
    plt.bar(x - 0.5 * width, first_mean_vals, width, label="Gen0 mean")
    plt.bar(x + 0.5 * width, first_worst_vals, width, label="Gen0 worst")
    plt.bar(x + 1.5 * width, last_best_vals, width, label="Last gen best")

    plt.xticks(x, cases, rotation=45, ha="right")
    plt.ylabel("Objective (gap)")
    plt.title("Initial (Gen0) vs final (last gen) gap per case")
    plt.legend()
    plt.tight_layout()

    fpath = os.path.join(out_dir, "allcases_initial_vs_final_gap.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot] saved {fpath}")


def plot_case_gap_time_table(all_case_stats, out_dir):
    """
    画出一个统计表格：
    - 每行一个 case：
        best_gap, time@best_gap, best_time, gap@best_time
    - 最后一行是 MEAN：
        所有 case 的平均 time@best_gap 和 平均 gap@best_time
    """
    if not all_case_stats:
        print("[info] no case stats to plot gap-time table")
        return

    cases = sorted(all_case_stats.keys())

    rows = []
    times_at_best_gap = []
    gaps_at_best_time = []

    for c in cases:
        stats = all_case_stats[c]
        best_gap = stats.get("best_gap")
        time_at_best_gap = stats.get("time_at_best_gap")
        best_time = stats.get("best_time")
        gap_at_best_time = stats.get("gap_at_best_time")

        if best_gap is None or time_at_best_gap is None or \
           best_time is None or gap_at_best_time is None:
            continue

        rows.append([
            c,
            f"{best_gap:.6f}",
            f"{time_at_best_gap:.3f}",
            f"{best_time:.3f}",
            f"{gap_at_best_time:.6f}",
        ])
        times_at_best_gap.append(time_at_best_gap)
        gaps_at_best_time.append(gap_at_best_time)

    if not rows:
        print("[info] no valid case rows to plot gap-time table")
        return

    mean_t_best_gap = float(np.mean(times_at_best_gap))
    mean_gap_best_time = float(np.mean(gaps_at_best_time))

    rows.append([
        "MEAN",
        "-",
        f"{mean_t_best_gap:.3f}",
        "-",
        f"{mean_gap_best_time:.6f}",
    ])

    col_labels = ["case", "best_gap", "time@best_gap", "best_time", "gap@best_time"]

    fig_height = 0.4 * (len(rows) + 1)
    fig, ax = plt.subplots(figsize=(max(8, len(cases) * 0.4), fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    fpath = os.path.join(out_dir, "allcases_gap_time_table.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot] saved {fpath}")
    print(f"[stat] mean time@best_gap = {mean_t_best_gap:.3f} s")
    print(f"[stat] mean gap@best_time = {mean_gap_best_time:.6f}")


def plot_ldr_t_stage_boxplot(t_records_all, out_dir):
    if not t_records_all:
        print("[info] no T-history LDR records to plot")
        return

    values_by_stage = defaultdict(list)
    for rec in t_records_all:
        stage = rec.get("stage") or "unknown"
        tldr = rec.get("tLDR")
        if tldr is None:
            continue
        try:
            t_val = float(tldr)
        except Exception:
            continue
        if math.isnan(t_val) or not math.isfinite(t_val):
            continue
        values_by_stage[stage].append(t_val)

    if not values_by_stage:
        print("[info] no valid tLDR values to plot for T-stages")
        return

    stages = sorted(values_by_stage.keys())
    data = [values_by_stage[s] for s in stages]

    plt.figure(figsize=(max(6, len(stages) * 1.0), 4))
    plt.boxplot(data, labels=stages, showfliers=False)
    plt.xlabel("T-stage (stage in t_history)")
    plt.ylabel("tLDR")
    plt.title("tLDR distribution by T-stage (all cases)")
    plt.grid(True, axis="y")
    plt.tight_layout()

    fpath = os.path.join(out_dir, "ldr_t_stage_boxplot.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot] saved {fpath}")


def plot_ldr_em_operator_boxplot(em_records_all, out_dir, min_count=10):
    if not em_records_all:
        print("[info] no EM-history LDR records to plot")
        return

    values_by_op = defaultdict(list)
    for rec in em_records_all:
        op = rec.get("operator") or "unknown"
        tldr = rec.get("tLDR")
        if tldr is None:
            continue
        try:
            t_val = float(tldr)
        except Exception:
            continue
        if math.isnan(t_val) or not math.isfinite(t_val):
            continue
        values_by_op[op].append(t_val)

    values_by_op = {op: vals for op, vals in values_by_op.items() if len(vals) >= min_count}
    if not values_by_op:
        print(f"[info] no EM-operators with >= {min_count} valid tLDR samples")
        return

    ops = sorted(values_by_op.keys())
    data = [values_by_op[op] for op in ops]

    plt.figure(figsize=(max(6, len(ops) * 1.0), 4))
    plt.boxplot(data, labels=ops, showfliers=False)
    plt.xlabel("Operator (e/m)")
    plt.ylabel("tLDR")
    plt.title(f"tLDR distribution by EM-operator (all cases, n>={min_count})")
    plt.grid(True, axis="y")
    plt.tight_layout()

    fpath = os.path.join(out_dir, "ldr_em_operator_boxplot.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"[plot] saved {fpath}")


# ---------- 主流程 ----------

def main():
    run_root = get_run_root_from_argv()
    print(f"[info] RUN_ROOT = {run_root}")

    if not os.path.isdir(run_root):
        print(f"[ERROR] RUN_ROOT not found: {run_root}")
        return

    out_dir = ensure_plot_dir(run_root)
    print(f"[info] plots will be saved under: {out_dir}")

    all_case_stats = {}
    t_records_all = []
    em_records_all = []

    any_case = False
    for case_name, case_dir in iter_cases(run_root):
        any_case = True
        print(f"\n[Case] {case_name} | dir={case_dir}")

        pop_records = load_population_generations(case_name, case_dir)
        if pop_records:
            # 原有的按 generation 画图
            plot_best_gap_curve(case_name, pop_records, out_dir)
            plot_time_curve(case_name, pop_records, out_dir)
            plot_gap_vs_time(case_name, pop_records, out_dir)

            # 新增：按 step 展开（把子代都当成迭代步）
            plot_gap_by_step(case_name, pop_records, out_dir)
            plot_time_by_step(case_name, pop_records, out_dir)

            gens = [rec["generation"] for rec in pop_records]
            if gens:
                g_min = min(gens)
                g_max = max(gens)

                # Gen0 的所有个体：best / worst / mean
                gen0_recs = [rec for rec in pop_records if rec["generation"] == g_min]
                gen0_objs = [rec["objective"] for rec in gen0_recs]
                first_best = float(min(gen0_objs))
                first_worst = float(max(gen0_objs))
                first_mean = float(np.mean(gen0_objs))

                # 最后一代的 best（按 gap）
                last_best = min(
                    rec["objective"]
                    for rec in pop_records
                    if rec["generation"] == g_max
                )

                # 全局 best gap 及其 time
                best_rec = min(pop_records, key=lambda r: r["objective"])
                best_gap = float(best_rec["objective"])
                time_at_best_gap = float(best_rec["eval_time"])

                # 全局 best time 及其 gap
                fast_rec = min(pop_records, key=lambda r: r["eval_time"])
                best_time = float(fast_rec["eval_time"])
                gap_at_best_time = float(fast_rec["objective"])

                all_case_stats[case_name] = {
                    "first_best": first_best,
                    "first_worst": first_worst,
                    "first_mean": first_mean,
                    "last_best": float(last_best),
                    "best_gap": best_gap,
                    "time_at_best_gap": time_at_best_gap,
                    "best_time": best_time,
                    "gap_at_best_time": gap_at_best_time,
                }

        t_records, em_records = load_history_records(case_name, case_dir)
        t_records_all.extend(t_records)
        em_records_all.extend(em_records)

    if not any_case:
        print("[INFO] no case directories found under RUN_ROOT, nothing to plot.")
        return

    # Gen0 best/mean/worst & last-gen best
    plot_initial_vs_final_gap(all_case_stats, out_dir)
    # gap–time 统计表格
    plot_case_gap_time_table(all_case_stats, out_dir)
    # LDR 相关 boxplot
    plot_ldr_t_stage_boxplot(t_records_all, out_dir)
    plot_ldr_em_operator_boxplot(em_records_all, out_dir, min_count=10)

    print("\n[Done] All plots saved to", out_dir)


if __name__ == "__main__":
    main()
