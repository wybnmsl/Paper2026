# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import csv
# import os
# import re
# import subprocess
# import time
# from pathlib import Path

# RE_COST = [
#     re.compile(r"Cost\.min\s*=\s*(\d+)"),
#     re.compile(r"Best tour length\s*[:=]\s*(\d+)"),
#     re.compile(r"Length\s*[:=]\s*(\d+)"),
# ]

# def read_opt_map(solutions_path: Path) -> dict[str, int]:
#     opt = {}
#     for line in solutions_path.read_text(encoding="utf-8", errors="ignore").splitlines():
#         line = line.strip()
#         if not line or line.startswith("#"):
#             continue
#         # e.g. "dsj1000 : 18660188 (CEIL_2D)"
#         if ":" not in line:
#             continue
#         name, val = line.split(":", 1)
#         name = name.strip()
#         m = re.search(r"(\d+)", val)
#         if m:
#             opt[name] = int(m.group(1))
#     return opt

# def write_par(par_path: Path, problem_file: Path, out_tour: Path, runs: int, seed: int,
#               time_limit: float | None, optimum: int | None, stop_at_optimum: bool):
#     lines = []
#     lines.append(f"PROBLEM_FILE = {problem_file}")
#     lines.append(f"OUTPUT_TOUR_FILE = {out_tour}")
#     lines.append(f"RUNS = {runs}")
#     lines.append(f"SEED = {seed}")
#     lines.append("TRACE_LEVEL = 0")
#     if time_limit is not None:
#         # LKH uses seconds
#         lines.append(f"TIME_LIMIT = {int(time_limit)}")
#     if optimum is not None:
#         lines.append(f"OPTIMUM = {optimum}")
#         if stop_at_optimum:
#             lines.append("STOP_AT_OPTIMUM = YES")
#     par_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# def parse_best_cost(log_text: str) -> int | None:
#     for rgx in RE_COST:
#         m = rgx.search(log_text)
#         if m:
#             return int(m.group(1))
#     # fallback: try find all integers after "Cost.min" occurrences
#     mins = []
#     for m in re.finditer(r"Cost\.min\s*=\s*(\d+)", log_text):
#         mins.append(int(m.group(1)))
#     return min(mins) if mins else None

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--tsplib_root", type=str, required=True, help="e.g. ~/zTSP_1208/tsplib-master")
#     ap.add_argument("--lkh", type=str, required=True, help="path to LKH executable")
#     ap.add_argument("--solutions", type=str, required=True, help="path to solutions mapping file")
#     ap.add_argument("--out_dir", type=str, default="lkh_runs_out")
#     ap.add_argument("--runs", type=int, default=10)
#     ap.add_argument("--seed", type=int, default=1)
#     ap.add_argument("--time_limit", type=float, default=60, help="per instance time limit (seconds), set <=0 to disable")
#     ap.add_argument("--stop_at_optimum", action="store_true", help="stop early when reach OPTIMUM")
#     ap.add_argument("--csv", type=str, default="lkh_tsplib_results.csv")
#     args = ap.parse_args()

#     tsplib_root = Path(os.path.expanduser(args.tsplib_root)).resolve()
#     lkh_path = Path(os.path.expanduser(args.lkh)).resolve()
#     sol_path = Path(os.path.expanduser(args.solutions)).resolve()
#     out_dir = Path(os.path.expanduser(args.out_dir)).resolve()

#     if not lkh_path.exists():
#         raise FileNotFoundError(f"LKH not found: {lkh_path}")
#     if not tsplib_root.exists():
#         raise FileNotFoundError(f"TSPLIB root not found: {tsplib_root}")
#     if not sol_path.exists():
#         raise FileNotFoundError(f"solutions file not found: {sol_path}")

#     opt_map = read_opt_map(sol_path)
#     tsp_files = sorted(tsplib_root.rglob("*.tsp"))

#     out_dir.mkdir(parents=True, exist_ok=True)

#     time_limit = None if args.time_limit <= 0 else args.time_limit

#     rows = []
#     for tsp in tsp_files:
#         name = tsp.stem
#         if name not in opt_map:
#             # solutions 里没给最优值就跳过（你也可以改成照样跑，只是不算 gap）
#             continue

#         inst_dir = out_dir / name
#         inst_dir.mkdir(parents=True, exist_ok=True)

#         par_file = inst_dir / f"{name}.par"
#         tour_file = inst_dir / f"{name}.lkh.tour"
#         log_file = inst_dir / f"{name}.log"

#         optimum = opt_map.get(name)

#         write_par(
#             par_path=par_file,
#             problem_file=tsp,
#             out_tour=tour_file,
#             runs=args.runs,
#             seed=args.seed,
#             time_limit=time_limit,
#             optimum=optimum,
#             stop_at_optimum=args.stop_at_optimum
#         )

#         t0 = time.perf_counter()
#         proc = subprocess.run(
#             [str(lkh_path), str(par_file)],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True
#         )
#         t1 = time.perf_counter()
#         wall_time = t1 - t0

#         log_text = proc.stdout or ""
#         log_file.write_text(log_text, encoding="utf-8", errors="ignore")

#         best = parse_best_cost(log_text)
#         if best is None:
#             # 记录失败
#             rows.append({
#                 "name": name,
#                 "opt": optimum,
#                 "best": "",
#                 "gap_percent": "",
#                 "time_sec": f"{wall_time:.6f}",
#                 "returncode": proc.returncode,
#                 "status": "NO_COST_PARSED",
#             })
#             continue

#         gap = (best - optimum) / float(optimum) * 100.0

#         rows.append({
#             "name": name,
#             "opt": optimum,
#             "best": best,
#             "gap_percent": f"{gap:.6f}",
#             "time_sec": f"{wall_time:.6f}",
#             "returncode": proc.returncode,
#             "status": "OK",
#         })

#         print(f"[{name}] best={best} opt={optimum} gap={gap:.4f}% time={wall_time:.2f}s")

#     # write csv
#     csv_path = Path(args.csv).resolve()
#     with csv_path.open("w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=["name", "opt", "best", "gap_percent", "time_sec", "returncode", "status"])
#         w.writeheader()
#         w.writerows(rows)

#     print(f"\nDone. CSV saved to: {csv_path}")
#     print(f"Logs & tours saved under: {out_dir}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any
import threading

RE_COST = [
    re.compile(r"Cost\.min\s*=\s*(\d+)"),
    re.compile(r"Best tour length\s*[:=]\s*(\d+)"),
    re.compile(r"Length\s*[:=]\s*(\d+)"),
]

RE_DIM = re.compile(r"^\s*DIMENSION\s*:?\s*(\d+)\s*$", re.IGNORECASE)

def read_opt_map(solutions_path: Path) -> dict[str, int]:
    opt = {}
    for line in solutions_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        name, val = line.split(":", 1)
        name = name.strip()
        m = re.search(r"(\d+)", val)
        if m:
            opt[name] = int(m.group(1))
    return opt

def parse_dimension(tsp_path: Path) -> Optional[int]:
    """
    Fast parse DIMENSION from TSPLIB .tsp header.
    """
    try:
        with tsp_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(300):  # header usually short; avoid scanning huge files
                line = f.readline()
                if not line:
                    break
                m = RE_DIM.match(line.strip())
                if m:
                    return int(m.group(1))
                # common format: "DIMENSION : 280" (already handled)
                # If reach coord section, stop
                if line.upper().startswith("NODE_COORD_SECTION") or line.upper().startswith("EDGE_WEIGHT_SECTION"):
                    break
    except Exception:
        return None
    return None

def write_par(
    par_path: Path,
    problem_file: Path,
    out_tour: Path,
    runs: int,
    seed: int,
    time_limit: Optional[float],
    optimum: Optional[int],
    stop_at_optimum: bool
):
    lines = []
    lines.append(f"PROBLEM_FILE = {problem_file}")
    lines.append(f"OUTPUT_TOUR_FILE = {out_tour}")
    lines.append(f"RUNS = {runs}")
    lines.append(f"SEED = {seed}")
    lines.append("TRACE_LEVEL = 0")
    if time_limit is not None:
        lines.append(f"TIME_LIMIT = {int(time_limit)}")
    if optimum is not None:
        lines.append(f"OPTIMUM = {optimum}")
        if stop_at_optimum:
            lines.append("STOP_AT_OPTIMUM = YES")
    par_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def parse_best_cost(log_text: str) -> Optional[int]:
    for rgx in RE_COST:
        m = rgx.search(log_text)
        if m:
            return int(m.group(1))
    mins = []
    for m in re.finditer(r"Cost\.min\s*=\s*(\d+)", log_text):
        mins.append(int(m.group(1)))
    return min(mins) if mins else None

def run_one_instance(
    tsp: Path,
    lkh_path: Path,
    out_dir: Path,
    opt_map: dict[str, int],
    runs: int,
    seed: int,
    time_limit: Optional[float],
    stop_at_optimum: bool,
) -> Dict[str, Any]:
    name = tsp.stem

    inst_dir = out_dir / name
    inst_dir.mkdir(parents=True, exist_ok=True)

    par_file = inst_dir / f"{name}.par"
    tour_file = inst_dir / f"{name}.lkh.tour"
    log_file = inst_dir / f"{name}.log"

    optimum = opt_map.get(name)

    # write par
    write_par(
        par_path=par_file,
        problem_file=tsp,
        out_tour=tour_file,
        runs=runs,
        seed=seed,
        time_limit=time_limit,
        optimum=optimum,
        stop_at_optimum=stop_at_optimum
    )

    # run
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [str(lkh_path), str(par_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        rc = proc.returncode
        log_text = proc.stdout or ""
    except Exception as e:
        t1 = time.perf_counter()
        log_file.write_text(f"EXCEPTION: {repr(e)}\n", encoding="utf-8", errors="ignore")
        return {
            "name": name,
            "opt": optimum if optimum is not None else "",
            "best": "",
            "gap_percent": "",
            "time_sec": f"{(t1 - t0):.6f}",
            "returncode": "",
            "status": f"EXCEPTION:{type(e).__name__}",
        }

    t1 = time.perf_counter()
    wall_time = t1 - t0

    # save log
    log_file.write_text(log_text, encoding="utf-8", errors="ignore")

    best = parse_best_cost(log_text)
    if best is None or optimum is None:
        return {
            "name": name,
            "opt": optimum if optimum is not None else "",
            "best": best if best is not None else "",
            "gap_percent": "",
            "time_sec": f"{wall_time:.6f}",
            "returncode": rc,
            "status": "NO_COST_PARSED" if best is None else "NO_OPTIMUM",
        }

    gap = (best - optimum) / float(optimum) * 100.0
    return {
        "name": name,
        "opt": optimum,
        "best": best,
        "gap_percent": f"{gap:.6f}",
        "time_sec": f"{wall_time:.6f}",
        "returncode": rc,
        "status": "OK",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsplib_root", type=str, required=True, help="e.g. ~/zTSP_1208/tsplib-master")
    ap.add_argument("--lkh", type=str, required=True, help="path to LKH executable")
    ap.add_argument("--solutions", type=str, required=True, help="path to solutions mapping file")
    ap.add_argument("--out_dir", type=str, default="lkh_runs_out")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time_limit", type=float, default=60, help="per instance time limit (seconds), set <=0 to disable")
    ap.add_argument("--stop_at_optimum", action="store_true", help="stop early when reach OPTIMUM")
    ap.add_argument("--csv", type=str, default="lkh_tsplib_results.csv")

    # 多线程参数
    ap.add_argument("--workers", type=int, default=0, help="num threads; 0 => min(8, cpu_count)")
    ap.add_argument("--max_n", type=int, default=1002, help="ONLY run instances with DIMENSION <= max_n (default 1002)")

    args = ap.parse_args()

    tsplib_root = Path(os.path.expanduser(args.tsplib_root)).resolve()
    lkh_path = Path(os.path.expanduser(args.lkh)).resolve()
    sol_path = Path(os.path.expanduser(args.solutions)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()

    if not lkh_path.exists():
        raise FileNotFoundError(f"LKH not found: {lkh_path}")
    if not tsplib_root.exists():
        raise FileNotFoundError(f"TSPLIB root not found: {tsplib_root}")
    if not sol_path.exists():
        raise FileNotFoundError(f"solutions file not found: {sol_path}")

    opt_map = read_opt_map(sol_path)
    tsp_files = sorted(tsplib_root.rglob("*.tsp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    time_limit = None if args.time_limit <= 0 else args.time_limit

    # 先筛选：有最优值 + DIMENSION <= 1002
    selected = []
    skipped_noopt = 0
    skipped_dim = 0
    skipped_nodim = 0
    for tsp in tsp_files:
        name = tsp.stem
        if name not in opt_map:
            skipped_noopt += 1
            continue
        dim = parse_dimension(tsp)
        if dim is None:
            skipped_nodim += 1
            continue
        if dim > args.max_n:
            skipped_dim += 1
            continue
        selected.append(tsp)

    cpu = os.cpu_count() or 4
    workers = args.workers if args.workers and args.workers > 0 else min(8, cpu)

    print(f"Found {len(tsp_files)} .tsp files")
    print(f"Selected (optimum+DIM<= {args.max_n}): {len(selected)}")
    print(f"Skipped: no_opt={skipped_noopt}, no_dim={skipped_nodim}, dim>{args.max_n}={skipped_dim}")
    print(f"Workers: {workers}")

    rows = []
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for tsp in selected:
            futs.append(ex.submit(
                run_one_instance,
                tsp, lkh_path, out_dir, opt_map,
                args.runs, args.seed, time_limit, args.stop_at_optimum
            ))

        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            # 统一在主线程打印，避免输出打架
            with print_lock:
                if row["status"] == "OK":
                    print(f"[{row['name']}] best={row['best']} opt={row['opt']} gap={row['gap_percent']}% time={float(row['time_sec']):.2f}s")
                else:
                    print(f"[{row['name']}] status={row['status']} time={float(row['time_sec']):.2f}s")

    # 稳定输出顺序
    rows.sort(key=lambda r: r["name"])

    csv_path = Path(args.csv).resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "opt", "best", "gap_percent", "time_sec", "returncode", "status"]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. CSV saved to: {csv_path}")
    print(f"Logs & tours saved under: {out_dir}")

if __name__ == "__main__":
    main()
