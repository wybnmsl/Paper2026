#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
汇总一个 EoH TSPGLS 批量实验的报表：

1) 每个 case 的 top-k 个最优解（按 objective 升序） -> topk_report.csv
2) 每个 case objective(=gap) 最低的解一条 -> best_gap_report.csv
3) 每个 case eval_time 最低的解一条 -> best_time_report.csv

用法示例：
    python report_topk_gls.py \
        --run-root results/TSPGLS_20251208_192933 \
        --topk 5
"""

import os
import json
import csv
import argparse
from typing import List, Dict, Any, Optional, Tuple


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] failed to load JSON: {path} ({e})")
        return None


def extract_update_edge_distance(py_path: str) -> str:
    """
    从导出的 solver.py 中粗暴提取 `def update_edge_distance(...)` 源码。
    如果找不到，返回空字符串。
    """
    if not os.path.isfile(py_path):
        return ""

    try:
        with open(py_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"[WARN] failed to read {py_path}: {e}")
        return ""

    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # 允许前面有缩进，但必须是 def update_edge_distance 开头
        if stripped.startswith("def update_edge_distance"):
            start_idx = i
            break

    if start_idx is None:
        # 没找到 update_edge_distance，可能用的是别的名字，这里先不强求
        return ""

    # 从 start_idx 往后收集，直到遇到下一个顶格的 def/class 或文件结束
    collected: List[str] = []
    for j in range(start_idx, len(lines)):
        line = lines[j]
        stripped = line.lstrip()
        if j > start_idx and stripped.startswith("def ") and not line.startswith(" "):
            break
        if j > start_idx and stripped.startswith("class ") and not line.startswith(" "):
            break
        collected.append(line)

    return "\n".join(collected).strip()


def flatten_gls_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 gls_spec 中抽几个关键参数出来平铺。
    """
    if spec is None:
        spec = {}
    candset = spec.get("candset") or {}
    guidance = spec.get("guidance") or {}
    schedule = spec.get("schedule") or {}
    stopping = spec.get("stopping") or {}

    return {
        "candset_k": candset.get("k"),
        "guidance_top_k": guidance.get("top_k"),
        "schedule_loop_max": schedule.get("loop_max"),
        "schedule_max_no_improve": schedule.get("max_no_improve"),
        "stopping_time_limit_s": stopping.get("time_limit_s"),
        "gls_spec_json": json.dumps(spec, ensure_ascii=False),
    }


def collect_from_exports(case_dir: str) -> List[Dict[str, Any]]:
    """
    优先从 results/exports/ 下的 JSON 文件收集个体信息。
    每个 JSON 是 _persist_t_accept 导出的 payload。
    返回列表里的元素结构统一为：
        {
            "source": "exports",
            "json_path": ...,
            "objective": ...,
            "eval_time": ...,
            "gls_spec": ...,
        }
    """
    exports_dir = os.path.join(case_dir, "results", "exports")
    candidates: List[Dict[str, Any]] = []

    if not os.path.isdir(exports_dir):
        return candidates

    for name in sorted(os.listdir(exports_dir)):
        if not name.endswith(".json"):
            continue
        jpath = os.path.join(exports_dir, name)
        data = load_json(jpath)
        if not data:
            continue
        obj = data.get("objective")
        t = data.get("eval_time")
        if obj is None:
            continue
        cand = {
            "source": "exports",
            "json_path": jpath,
            "objective": float(obj),
            "eval_time": float(t) if t is not None else None,
            "gls_spec": data.get("gls_spec"),
        }
        candidates.append(cand)

    return candidates


def collect_from_pops_best(case_dir: str) -> List[Dict[str, Any]]:
    """
    没有 exports 时的 fallback：从 results/pops_best/ 里读每一代的 best 个体。
    注意：这种情况下一般拿不到 code 和 gls_spec（除非你在框架里有写），
    所以只填 objective / eval_time，其余设为 None。
    """
    pops_dir = os.path.join(case_dir, "results", "pops_best")
    candidates: List[Dict[str, Any]] = []

    if not os.path.isdir(pops_dir):
        return candidates

    for name in sorted(os.listdir(pops_dir)):
        if not name.endswith(".json"):
            continue
        jpath = os.path.join(pops_dir, name)
        data = load_json(jpath)
        if not data:
            continue
        # 兼容：有的版本是单个个体 dict，有的版本可能是 list[dict]
        if isinstance(data, list):
            if not data:
                continue
            indiv = data[0]
        else:
            indiv = data

        obj = indiv.get("objective")
        if obj is None:
            continue
        t = indiv.get("eval_time")
        cand = {
            "source": "pops_best",
            "json_path": jpath,
            "objective": float(obj),
            "eval_time": float(t) if t is not None else None,
            "gls_spec": indiv.get("gls_spec"),  # 大概率是 None
        }
        candidates.append(cand)

    return candidates


def collect_case_candidates(case_dir: str,
                            case_name: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    返回 (candidates, source_used)
    source_used in {"exports", "pops_best"}
    """
    cands = collect_from_exports(case_dir)
    source_used = "exports"
    if not cands:
        cands = collect_from_pops_best(case_dir)
        source_used = "pops_best"
    if not cands:
        print(f"[INFO] case={case_name}: no candidates found (no exports/pops_best)")
    return cands, source_used


def build_row_from_cand(case_name: str,
                        cand: Dict[str, Any],
                        source_used: str,
                        rank: Optional[int]) -> Dict[str, Any]:
    """
    把一个候选个体转成 CSV 行。
    """
    gls_spec = cand.get("gls_spec") or {}
    gls_flat = flatten_gls_spec(gls_spec)

    json_path = cand.get("json_path")
    py_path = ""
    update_src = ""
    if json_path and source_used == "exports":
        # 对 exports：JSON 和 PY 是同名不同后缀
        maybe_py = json_path[:-5] + ".py"
        if os.path.isfile(maybe_py):
            py_path = maybe_py
            update_src = extract_update_edge_distance(py_path)

    row = {
        "case": case_name,
        "rank": rank,
        "source": source_used,
        "json_path": json_path,
        "py_path": py_path or "",
        "objective": cand.get("objective"),
        "eval_time": cand.get("eval_time"),
        "update_edge_distance_source": update_src,
    }
    row.update(gls_flat)
    return row


def collect_case_topk(case_dir: str,
                      case_name: str,
                      topk: int) -> List[Dict[str, Any]]:
    """
    对单个 case：
    - 优先用 exports；
    - 如果没有 exports，就用 pops_best 做 fallback；
    - 取按 objective 升序的前 topk 个。
    返回的是 CSV row 列表。
    """
    cands, source_used = collect_case_candidates(case_dir, case_name)
    if not cands:
        return []

    # 按 objective 升序
    cands_sorted = sorted(cands, key=lambda x: x["objective"])
    cands_sorted = cands_sorted[: topk]

    rows: List[Dict[str, Any]] = []
    for rank, cand in enumerate(cands_sorted, start=1):
        row = build_row_from_cand(case_name, cand, source_used, rank=rank)
        rows.append(row)
    return rows


def collect_case_best(case_dir: str,
                      case_name: str) -> Tuple[
                          Optional[Dict[str, Any]],
                          Optional[Dict[str, Any]]
                      ]:
    """
    对单个 case 找出：
    - objective 最低的 cand -> row_gap
    - eval_time 最低的 cand -> row_time
    都使用同一批 candidates（exports 优先，否则 pops_best）。

    如果没有任何候选，则返回 (None, None)。
    """
    cands, source_used = collect_case_candidates(case_dir, case_name)
    if not cands:
        return None, None

    # 最小 gap（objective）
    best_gap_cand = min(cands, key=lambda x: x["objective"])
    row_gap = build_row_from_cand(case_name, best_gap_cand, source_used, rank=1)

    # 最小时间：忽略 eval_time 为 None 的；如果都为 None，则不生成 row_time
    cands_with_time = [c for c in cands if c.get("eval_time") is not None]
    if cands_with_time:
        best_time_cand = min(cands_with_time, key=lambda x: x["eval_time"])
        row_time = build_row_from_cand(case_name, best_time_cand, source_used, rank=1)
    else:
        row_time = None

    return row_gap, row_time


def main():
    parser = argparse.ArgumentParser(
        description="Summarize GLS solutions (top-k / best gap / best time) for a TSPGLS EoH batch run."
    )
    parser.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="批量实验的根目录，例如 results/TSPGLS_20251208_192933",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="每个 case 取多少个最优解（默认 5）",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="输出 CSV 前缀（默认写到 <run-root>/topk_report*.csv 等）",
    )
    args = parser.parse_args()

    run_root = os.path.abspath(args.run_root)
    topk = max(1, args.topk)

    if not os.path.isdir(run_root):
        print(f"[ERROR] run-root not found: {run_root}")
        return

    if args.output_prefix:
        prefix = args.output_prefix
    else:
        prefix = os.path.join(run_root, "topk_report")

    topk_csv = prefix + ".csv"
    best_gap_csv = prefix.replace("topk_report", "best_gap_report") + ".csv"
    best_time_csv = prefix.replace("topk_report", "best_time_report") + ".csv"

    # case 目录 = run_root 下的所有子目录
    all_entries = sorted(os.listdir(run_root))

    all_rows_topk: List[Dict[str, Any]] = []
    best_gap_rows: List[Dict[str, Any]] = []
    best_time_rows: List[Dict[str, Any]] = []

    for entry in all_entries:
        case_dir = os.path.join(run_root, entry)
        if not os.path.isdir(case_dir):
            continue
        # 粗略判断是不是某个 case：要求内部至少要有 results/ 目录
        if not os.path.isdir(os.path.join(case_dir, "results")):
            continue

        case_name = entry
        print(f"[INFO] processing case: {case_name}")

        # 1) top-k rows
        rows_topk = collect_case_topk(case_dir, case_name, topk)
        all_rows_topk.extend(rows_topk)

        # 2) best gap / best time
        row_gap, row_time = collect_case_best(case_dir, case_name)
        if row_gap is not None:
            best_gap_rows.append(row_gap)
        if row_time is not None:
            best_time_rows.append(row_time)

    if not all_rows_topk and not best_gap_rows and not best_time_rows:
        print("[WARN] no rows collected; nothing to write.")
        return

    fieldnames = [
        "case",
        "rank",
        "source",
        "json_path",
        "py_path",
        "objective",
        "eval_time",
        "candset_k",
        "guidance_top_k",
        "schedule_loop_max",
        "schedule_max_no_improve",
        "stopping_time_limit_s",
        "gls_spec_json",
        "update_edge_distance_source",
    ]

    os.makedirs(os.path.dirname(topk_csv), exist_ok=True)

    # 写 top-k 报表
    if all_rows_topk:
        with open(topk_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows_topk:
                for k in fieldnames:
                    row.setdefault(k, None)
                writer.writerow(row)
        print(f"[OK] wrote top-k report to: {topk_csv}")
    else:
        print("[INFO] no top-k rows to write.")

    # 写 best gap 报表
    if best_gap_rows:
        with open(best_gap_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in best_gap_rows:
                for k in fieldnames:
                    row.setdefault(k, None)
                writer.writerow(row)
        print(f"[OK] wrote best-gap report to: {best_gap_csv}")
    else:
        print("[INFO] no best-gap rows to write.")

    # 写 best time 报表
    if best_time_rows:
        with open(best_time_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in best_time_rows:
                for k in fieldnames:
                    row.setdefault(k, None)
                writer.writerow(row)
        print(f"[OK] wrote best-time report to: {best_time_csv}")
    else:
        print("[INFO] no best-time rows to write.")


if __name__ == "__main__":
    main()
