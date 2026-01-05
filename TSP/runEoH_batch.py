# import os
# import sys

# # 当前脚本所在目录：.../EoH_1203/zTSP
# CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# # 项目根目录：.../EoH_1203
# PROJECT_ROOT = os.path.dirname(CUR_DIR)
# # 本项目的 eoh/src：.../EoH_1203/eoh/src
# EOH_SRC = os.path.join(PROJECT_ROOT, "eoh", "src")

# # 把本项目的 eoh/src 插到 sys.path 最前面
# if EOH_SRC not in sys.path:
#     sys.path.insert(0, EOH_SRC)

# from eoh import eoh
# from eoh.utils.getParas import Paras
# from prob import TSPGLS

# # TSPLIB 根目录及 solutions 文件（相对当前 zTSP 目录）
# TSPLIB_ROOT = os.path.join(CUR_DIR, 'tsplib-master')
# TSPLIB_SOLUTIONS = os.path.join(TSPLIB_ROOT, 'solutions')
# # 在这里列出你希望通过 runEoH 直接评估的 TSPLIB case 名称列表。
# # 例如：['kroA100']、['kroA100', 'pr1002'] 等；留空列表则回退到旧的 TSPAEL64.pkl 流程。
# TSPLIB_CASES = [
#     'kroA100',
# ]

# # Parameter initilization #
# paras = Paras() 

# # Set your local problem
# problem_local = TSPGLS()

# # Set parameters #
# paras.set_paras(method = "eoh",    # ['ael','eoh']
#                 problem = problem_local, # Set local problem, else use default problems
#                 llm_api_endpoint = "api.holdai.top", # set your LLM endpoint
#                 llm_api_key = "sk-obW0OmUVEeOlLfT7B944E2E2De3642F8B247C4252452DcE3",   # set your key
#                 llm_model = "gpt-5-mini-2025-08-07",
#                 ec_pop_size = 2, # number of samples in each population
#                 ec_n_pop = 2,  # number of populations
#                 exp_n_proc = 2,  # multi-core parallel
#                 exp_debug_mode = False,
#                 eva_numba_decorator = False,
#                 eva_timeout = 30    
#                 # Set the maximum evaluation time for each heuristic !
#                 # Increase it if more instances are used for evaluation !
#                 ) 

# # initilization: 根据 TSPLIB_CASES 决定是否直接从 TSPLIB 读取
# if TSPLIB_CASES:
#     prob_local = TSPGLS(
#         tsplib_root=TSPLIB_ROOT,
#         solutions_file=TSPLIB_SOLUTIONS,
#         case_names=TSPLIB_CASES,
#         n_inst_eva=None,  # 默认使用列表中的全部 case
#         time_limit=paras.eva_timeout,
#         debug_mode=paras.exp_debug_mode,
#     )
#     evolution = eoh.EVOL(paras, prob=prob_local)
# else:
#     evolution = eoh.EVOL(paras)

# # run 
# evolution.run()


import os
import sys
import time

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

# 你要批量评估的所有 case：
# TSPLIB_CASES = [
#     "kroA100",
#     "kroA150",
# ]

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


TSPLIB_CASES = [
    "a280",
]


def main():
    # -------- 1) 初始化全局参数 & 统一的 run_name --------
    paras = Paras()

    # 统一一个 run 名称，这样所有 case 都写到同一个 TSPGLS_时间戳 下面
    prob_name = "TSPGLS"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{prob_name}_{timestamp}"
    setattr(paras, "exp_run_name", run_name)

    # 这里写你的 EoH/EoM 相关参数（和原来的 runEoH 一样）
    paras.set_paras(
        method="eoh",
        problem=None,          # 先留空，下面每个 case 再填
        llm_api_endpoint="api.holdai.top",
        llm_api_key="sk-obW0OmUVEeOlLfT7B944E2E2De3642F8B247C4252452DcE3",
        llm_model="gpt-5-mini-2025-08-07",
        ec_pop_size=6, 
        ec_n_pop=6,
        exp_n_proc=6,
        exp_debug_mode=False,
        eva_numba_decorator=False,
        eva_timeout=30,
    )

    print(f"[Batch] Run name = {run_name}")
    print(f"[Batch] Cases   = {TSPLIB_CASES}")

    # -------- 2) 对每个 case 单独跑一遍 EoH --------
    for case in TSPLIB_CASES:
        print("\n" + "=" * 60)
        print(f"[Batch] Start case: {case}")
        print("=" * 60)

        # 1) 单 case 的 TSPGLS 初始化：可能因为缺少 .tsp 或 opt 解而失败
        try:
            problem_local = TSPGLS(
                tsplib_root=TSPLIB_ROOT,
                solutions_file=TSPLIB_SOLUTIONS,
                case_names=[case],        # 单个 case
                n_inst_eva=None,
                time_limit=paras.eva_timeout,
                debug_mode=paras.exp_debug_mode,
            )
        except ValueError as e:
            # 比如：No TSPLIB instances loaded from ...
            print(f"[Batch][WARN] Skip case {case}: {e}")
            continue
        except Exception as e:
            print(f"[Batch][ERROR] Unexpected error when loading case {case}: {e}")
            continue

        # 显式标记 case 名，用于目录名
        setattr(problem_local, "case_tag", case)

        paras.problem = problem_local

        # 2) 创建并运行 EoH
        evolution = eoh.EVOL(paras, prob=problem_local)
        evolution.run()


    print("\n[Batch] All cases finished.")


if __name__ == "__main__":
    main()
