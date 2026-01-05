# run_rule_eval.py
from prob import TSPGLS
from gls.gls_run import solve_instance
import importlib.util
import numpy as np

# 动态加载我们刚才存的模块
spec = importlib.util.spec_from_file_location("eoh_rule_best", "examples/user_tsp_gls/heuristic.py")
rule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule)

# 建问题与参数（按论文口径可调整）
prob = TSPGLS()
prob.time_limit = 60       # 论文是 60 秒/实例
prob.ite_max = 1000        # 论文是 1000 次 LS 迭代
prob.perturbation_moves = 1

gaps = []
for i in range(prob.n_inst_eva):
    gap = solve_instance(
        i,
        prob.opt_costs[i],     # ground truth 成本
        prob.instances[i],     # 你这边传的是距离矩阵/或实例矩阵（见下备注）
        prob.coords[i],        # 坐标
        prob.time_limit,
        prob.ite_max,
        prob.perturbation_moves,
        rule                   # ← 关键：把“包含 update_edge_distance 的模块”传进去
    )
    print(f"inst {i+1}: gap={gap:.3f}%")
    gaps.append(gap)

print("avg gap:", float(np.mean(gaps)))
