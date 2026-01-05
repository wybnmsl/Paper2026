import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


# 当前脚本所在目录：.../EoH-main/zTSP
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：.../EoH-main
PROJECT_ROOT = os.path.dirname(CUR_DIR)
# 本项目的 eoh/src：.../EoH-main/eoh/src
EOH_SRC = os.path.join(PROJECT_ROOT, "eoh", "src")

# 把本项目的 eoh/src 插到 sys.path 最前面
if EOH_SRC not in sys.path:
    sys.path.insert(0, EOH_SRC)

from eoh import eoh
from eoh.utils.getParas import Paras
from prob import TSPGLS

# TSPLIB 根目录及 solutions 文件（相对当前 zTSP 目录）
TSPLIB_ROOT = os.path.join(CUR_DIR, "tsplib-master")
TSPLIB_SOLUTIONS = os.path.join(TSPLIB_ROOT, "solutions")



# TSPLIB_CASES = [
#     "att48",
#     "bayg29",
#     "bays29",
#     "berlin52",
#     "bier127",
#     "brazil58",
#     "brg180",
#     "burma14",
#     "d198",
#     "dantzig42",
#     "eil51",
#     "eil76",
#     "fri26",
#     "gr17",
#     "gr21",
#     "gr24",
#     "gr48",
#     "gr96",
#     "hk48",
#     "kroA100",
#     "kroB100",
#     "kroC100",
#     "kroD100",
#     "kroE100",
#     "pr76",
#     "rat99",
#     "rd100",
#     "st70",
#     "swiss42",
#     "ulysses16",
#     "ulysses22",
# ]


# TSPLIB_CASES = [
#     "a280",
#     "ali535",
#     "att48",
#     "att532",
#     "bayg29",
#     "bays29",
#     "berlin52",
#     "bier127",
#     "brazil58",
#     "brg180",
#     "burma14",
#     "ch130",
#     "ch150",
#     "d198",
#     "d493",
#     "d657",
#     "dantzig42",
#     "eil51",
#     "eil76",
#     "eil101",
#     "fl417",
#     "fri26",
#     "gil262",
#     "gr17",
#     "gr21",
#     "gr24",
#     "gr48",
#     "gr96",
#     "gr120",
#     "gr137",
#     "gr202",
#     "gr229",
#     "gr431",
#     "gr666",
#     "hk48",
#     "kroA100",
#     "kroB100",
#     "kroC100",
#     "kroD100",
#     "kroE100",
#     "kroA150",
#     "kroB150",
#     "kroA200",
#     "kroB200",
#     "lin105",
#     "lin318",
#     "linhp318",
#     "p654",
#     "pa561",
#     "pcb442",
#     "pr76",
#     "pr107",
#     "pr124",
#     "pr136",
#     "pr144",
#     "pr152",
#     "pr226",
#     "pr264",
#     "pr299",
#     "pr439",
#     "rat99",
#     "rat195",
#     "rat575",
#     "rat783",
#     "rd100",
#     "rd400",
#     "si175",
#     "si535",
#     "st70",
#     "swiss42",
#     "ts225",
#     "tsp225",
#     "u159",
#     "u574",
#     "u724",
#     "ulysses16",
#     "ulysses22",
# ]


#>=100
# TSPLIB_CASES = [
#     "a280",
#     "ali535",
#     "bier127",
#     "brg180",
#     "ch130",
#     "ch150",
#     "d198",
#     "d493",
#     "d657",
#     "eil101",
#     "fl417",
#     "gil262",
#     "gr120",
#     "gr137",
#     "gr202",
#     "gr229",
#     "gr431",
#     "gr666",
#     "kroA100",
#     "kroB100",
#     "kroC100",
#     "kroD100",
#     "kroE100",
#     "kroA150",
#     "kroB150",
#     "kroA200",
#     "kroB200",
#     "lin105",
#     "lin318",
#     "linhp318",
#     "p654",
#     "pa561",
#     "pcb442",
#     "pr107",
#     "pr124",
#     "pr136",
#     "pr144",
#     "pr152",
#     "pr226",
#     "pr264",
#     "pr299",
#     "pr439",
#     "rat195",
#     "rat575",
#     "rat783",
#     "rd100",
#     "rd400",
#     "si175",
#     "si535",
#     "ts225",
#     "tsp225",
#     "u159",
#     "u574",
#     "u724",
# ]

#>=200
# TSPLIB_CASES = [
#     "a280",
#     "ali535",
#     "att532",
#     "d493",
#     "d657",
#     "fl417",
#     "gil262",
#     "gr202",
#     "gr229",
#     "gr431",
#     "gr666",
#     "kroA200",
#     "kroB200",
#     "lin318",
#     "linhp318",
#     "p654",
#     "pa561",
#     "pcb442",
#     "pr226",
#     "pr264",
#     "pr299",
#     "pr439",
#     "rat575",
#     "rat783",
#     "rd400",
#     "si535",
#     "ts225",
#     "tsp225",
#     "u574",
#     "u724",
# ]


#<200
TSPLIB_CASES = [
    "att48",
    "bayg29",
    "bays29",
    "berlin52",
    "bier127",
    "brazil58",
    "brg180",
    "burma14",
    "ch130",
    "ch150",
    "d198",
    "dantzig42",
    "eil51",
    "eil76",
    "eil101",
    "fri26",
    "gr17",
    "gr21",
    "gr24",
    "gr48",
    "gr96",
    "gr120",
    "gr137",
    "hk48",
    "kroA100",
    "kroB100",
    "kroC100",
    "kroD100",
    "kroE100",
    "kroA150",
    "kroB150",
    "kroA200",
    "kroB200",
    "lin105",
    "pr76",
    "pr107",
    "pr124",
    "pr136",
    "pr144",
    "pr152",
    "rat99",
    "rat195",
    "rd100",
    "si175",
    "st70",
    "swiss42",
    "u159",
    "ulysses16",
    "ulysses22",
]


def run_single_case(case: str, run_name: str):
    """
    在一个独立进程中跑单个 TSPLIB case 的 EoH 演化。
    - case: 例如 "att48"
    - run_name: 例如 "TSPGLS_20251208_190426"，用于统一 results 根目录
    """
    # 重新初始化 Paras（每个进程一套，避免跨进程共享）
    paras = Paras()

    # 和你原来 main() 里 set_paras 一样的配置
    paras.set_paras(
        method="eoh",
        problem=None,  # 下面再填
        llm_api_endpoint="api.holdai.top",
        llm_api_key="sk-C0IMvwTEfppOnFmHpi8Nzjw2hcg57rfrelOByZrY87OBFmvt",
        llm_model="gpt-5-mini-2025-08-07",
        ec_pop_size=6,
        ec_n_pop=6,
        exp_n_proc=2,           # 注意：case 并行时，可以考虑适当调小
        exp_debug_mode=False,
        eva_numba_decorator=False,
        eva_timeout=30,
    )

    # 把统一的 run_name 写进 paras，这样所有 case 共用一个 results 根目录
    setattr(paras, "exp_run_name", run_name)

    # 初始化单个 case 的问题
    try:
        problem_local = TSPGLS(
            tsplib_root=TSPLIB_ROOT,
            solutions_file=TSPLIB_SOLUTIONS,
            case_names=[case],        # 单 case
            n_inst_eva=None,
            time_limit=paras.eva_timeout,
            debug_mode=paras.exp_debug_mode,
        )
    except ValueError as e:
        print(f"[Batch][{case}] Skip case (ValueError): {e}")
        return
    except Exception as e:
        print(f"[Batch][{case}] ERROR when loading problem: {e}")
        return

    # 标记 case 名，EoH 里会用来建子目录
    setattr(problem_local, "case_tag", case)
    paras.problem = problem_local

    # 运行 EoH 演化
    print(f"\n----------------------------------------- ")
    print(f"---          Start EoH ({case})       ---")
    print(f"-----------------------------------------")

    evolution = eoh.EVOL(paras, prob=problem_local)
    evolution.run()

    print(f"[Batch][{case}] Finished.")



# def main():
#     # -------- 1) 统一 run_name（results 根目录） --------
#     prob_name = "TSPGLS"
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     run_name = f"{prob_name}_{timestamp}"

#     print(f"[Batch] Run name = {run_name}")
#     print(f"[Batch] Cases   = {TSPLIB_CASES}")

#     # -------- 2) 选择并行度：最多不超过 CPU 核数 --------
#     n_cases = len(TSPLIB_CASES)
#     n_cpu = os.cpu_count() or 1

#     # 这里取一个相对保守的并行度，你可以按实际机器情况调整：
#     #   - 如果 exp_n_proc 较大（比如 4/8），建议把 max_workers 设小一点；
#     #   - 如果 exp_n_proc=1，可以把它开到 n_cpu 或 n_cpu-1。
#     max_workers = min(n_cases, max(1, n_cpu // 2))

#     print(f"[Batch] Will run with up to {max_workers} case-workers in parallel.")

#     # -------- 3) 用进程池并行跑每个 case --------
#     futures = {}
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         for case in TSPLIB_CASES:
#             print(f"[Batch] Submit case: {case}")
#             fut = executor.submit(run_single_case, case, run_name)
#             futures[fut] = case

#         # 可选：如果你想看到每个 case 完成的顺序，可以用 as_completed
#         for fut in as_completed(futures):
#             case = futures[fut]
#             try:
#                 fut.result()
#                 print(f"[Batch] Case {case} done.")
#             except Exception as e:
#                 print(f"[Batch][ERROR] Case {case} failed with exception: {e}")

#     print("\n[Batch] All cases finished.")


def main():
    # -------- 1) 统一 run_name（results 根目录） --------
    prob_name = "TSPGLS"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{prob_name}_{timestamp}"

    print(f"[Batch] Run name = {run_name}")
    print(f"[Batch] Cases   = {TSPLIB_CASES}")

    n_cases = len(TSPLIB_CASES)
    print(f"[Batch] Total cases = {n_cases}")

    # -------- 2) 选择并行度：最多不超过 CPU 核数 --------
    n_cpu = os.cpu_count() or 1

    # 这里取一个相对保守的并行度，你可以按实际机器情况调整：
    max_workers = min(n_cases, max(1, n_cpu // 2))

    print(f"[Batch] Will run with up to {max_workers} case-workers in parallel.")

    # -------- 3) 用进程池并行跑每个 case --------
    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务时也打印一下“排队进度”
        for idx, case in enumerate(TSPLIB_CASES, start=1):
            print(f"[Batch] Queue case {idx}/{n_cases}: {case}")
            print(f"[Batch] Submit case: {case}")   # 原有打印，保持不变
            fut = executor.submit(run_single_case, case, run_name)
            futures[fut] = case

        finished = 0  # 已完成的 case 数

        # 可选：如果你想看到每个 case 完成的顺序，可以用 as_completed
        for fut in as_completed(futures):
            case = futures[fut]
            try:
                fut.result()
                print(f"[Batch] Case {case} done.")  # 原有打印，保持不变
                finished += 1
                print(f"[Batch] Progress: {finished}/{n_cases} cases finished.")
            except Exception as e:
                print(f"[Batch][ERROR] Case {case} failed with exception: {e}")
                finished += 1
                print(f"[Batch] Progress: {finished}/{n_cases} cases finished (including failed).")

            # 及时刷新缓冲，防止多进程输出挤在一起太久才出现
            sys.stdout.flush()

    print("\n[Batch] All cases finished.")


if __name__ == "__main__":
    main()
