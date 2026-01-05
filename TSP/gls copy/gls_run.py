# zTSP/gls/gls_run.py
import time
import numpy as np

from utils import utils
from gls import gls_evol
from gls.spec import GLSSpec, default_gls_spec, from_json
from gls import gls_operators

def _preheat_numba_once():
    """
    子进程内做一次极小规模预热，触发 numba JIT 编译，降低后续评估的冷启动时间。
    """
    try:
        N = 6
        D = np.random.RandomState(0).rand(N, N).astype(np.float64)
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        tour = -1 * np.ones((N, 2), dtype=np.int64)
        for i in range(N):
            tour[i, 0] = (i - 1) % N
            tour[i, 1] = (i + 1) % N
        knn = 3
        NNs = np.argsort(D, axis=1)[:, 1:knn+1].astype(np.int64)
        # 触发两大算子的编译
        gls_operators.two_opt_a2a(tour, D, NNs, False)
        gls_operators.relocate_a2a(tour, D, NNs, False)
    except Exception:
        pass

def solve_instance(n, opt_cost, dis_matrix, coord, time_limit, ite_max, perturbation_moves, heuristic):
    """
    旧接口：保留不变（e/m 阶段依然使用）。
    """
    t = time.time()

    try:
        init_tour = gls_evol.nearest_neighbor_2End(dis_matrix, 0).astype(int)
        init_cost = utils.tour_cost_2End(dis_matrix, init_tour)
        nb = 100
        nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:nb+1].astype(int)

        best_tour, best_cost, iter_i = gls_evol.guided_local_search(
            coord, dis_matrix, nearest_indices, init_tour, init_cost,
            t + time_limit, ite_max, perturbation_moves,
            first_improvement=False, guide_algorithm=heuristic, spec=None
        )

        gap = (best_cost / opt_cost - 1) * 100.0

    except Exception:
        gap = 1E10

    return gap

def solve_instance_with_spec(n, opt_cost, dis_matrix, coord,
                             time_limit, ite_max, perturbation_moves,
                             heuristic, spec: GLSSpec | dict | None):
    """
    新接口：在“同一 GLS 骨架”下，依据 GLSSpec（结构/参数）运行。
    这里让 init / candset / operators / schedule / guidance / stopping 等字段真正生效。
    """
    # 宽松解析 + 默认值
    if isinstance(spec, GLSSpec):
        _spec = spec
    elif isinstance(spec, dict):
        _spec = from_json(spec)
    else:
        _spec = default_gls_spec()
    spec = _spec  # 后续统一使用 spec

    # 预热 numba（子进程第一次调用时）
    _preheat_numba_once()

    t0 = time.time()
    # time_limit_s 取两者较小（与 e/m 口径一致）
    tl = float(min(float(spec.stopping.get("time_limit_s", time_limit)), float(time_limit)))

    try:
        # 初始化解（支持不同 init 策略；目前主要是 nearest_neighbor）
        method = str(spec.init.get("method", "nearest_neighbor"))
        start = int(spec.init.get("start", 0))
        if method == "nearest_neighbor":
            init_tour = gls_evol.nearest_neighbor_2End(dis_matrix, start).astype(int)
        else:
            # 其它方法暂退化为 NN，保持骨架一致（后续可扩展）
            init_tour = gls_evol.nearest_neighbor_2End(dis_matrix, start).astype(int)

        init_cost = utils.tour_cost_2End(dis_matrix, init_tour)

        # 候选集：由 candset.type / k 控制
        n_nodes = dis_matrix.shape[0]
        k = int(spec.candset.get("k", 100))
        k = max(1, min(k, n_nodes - 1))
        ctype = str(spec.candset.get("type", "kNN")).lower()

        if ctype == "full":
            # 使用所有候选（除自身外）；按距离排序
            nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:].astype(int)
        else:
            # 默认：k 近邻
            nearest_indices = np.argsort(dis_matrix, axis=1)[:, 1:k+1].astype(int)

        # operators → first_improvement 策略（只取 two_opt 的设置作为整体策略开关）
        first_improve = False
        ops = getattr(spec, "operators", None) or []
        for op in ops:
            # 同时兼容 dataclass 与 dict
            name = getattr(op, "name", None)
            strategy = getattr(op, "strategy", None)
            if name is None or strategy is None:
                if isinstance(op, dict):
                    name = op.get("name")
                    strategy = op.get("strategy")
            if str(name) == "two_opt" and str(strategy) == "first":
                first_improve = True
                break

        # 调主循环（同一 GLS 骨架）
        best_tour, best_cost, iter_i = gls_evol.guided_local_search(
            coord, dis_matrix, nearest_indices, init_tour, init_cost,
            t0 + tl, int(spec.schedule.get("loop_max", ite_max)),
            perturbation_moves,
            first_improvement=first_improve,
            guide_algorithm=heuristic,
            spec=spec
        )

        gap = (best_cost / opt_cost - 1) * 100.0

    except Exception:
        gap = 1E10

    return gap
