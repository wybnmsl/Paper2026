import os
import sys

# 当前脚本所在目录：.../EoH_1203/zTSP
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：.../EoH_1203
PROJECT_ROOT = os.path.dirname(CUR_DIR)
# 本项目的 eoh/src：.../EoH_1203/eoh/src
EOH_SRC = os.path.join(PROJECT_ROOT, "eoh", "src")

# 把本项目的 eoh/src 插到 sys.path 最前面
if EOH_SRC not in sys.path:
    sys.path.insert(0, EOH_SRC)

from eoh import eoh
from eoh.utils.getParas import Paras
from prob import TSPGLS

# Parameter initilization #
paras = Paras() 

# Set your local problem
problem_local = TSPGLS()

# Set parameters #
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = problem_local, # Set local problem, else use default problems
                llm_api_endpoint = "api.holdai.top", # set your LLM endpoint
                llm_api_key = "sk-obW0OmUVEeOlLfT7B944E2E2De3642F8B247C4252452DcE3",   # set your key
                llm_model = "gpt-5-mini-2025-08-07",
                ec_pop_size = 6, # number of samples in each population
                ec_n_pop = 6,  # number of populations
                exp_n_proc = 6,  # multi-core parallel
                exp_debug_mode = False,
                eva_numba_decorator = False,
                eva_timeout = 30  
                # Set the maximum evaluation time for each heuristic !
                # Increase it if more instances are used for evaluation !
                ) 

# initilization
evolution = eoh.EVOL(paras)

# run 
evolution.run()

