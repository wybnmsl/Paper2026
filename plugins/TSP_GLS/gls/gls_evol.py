# zTSP/gls/gls_evol.py
"""
改进版 GLS 演化骨架

- 保留原始 two_opt / relocate 骨干（调用 gls_operators 中的 numba 加速算子）；
- 在此基础上引入：
    * 轻/重两档局部搜索（local_search_basic / local_search_strong_basic）；
    * builtin GLS 模式：D' = D + (lam * avg_edge_len) * penalty；
    * LLM 指导模式：由 guide_algorithm.update_edge_distance 生成 D'；
    * 扰动（random_relocate / double_bridge）；
    * 规格化参数来自 GLSSpec（candset / schedule / perturb / guidance / engine 等）。

注意：
- 该文件不依赖 LKH 风格的 gls_lk 模块；engine.type != "ls_basic" 时会退回到 basic 引擎。
"""

from __future__ import annotations

import time
import random
from typing import Tuple

import numpy as np

from . import gls_operators
from ..utils import utils
from .spec import GLSSpec


# ---------- 初始构造 ----------

def nearest_neighbor(dis_matrix: np.ndarray, depot: int) -> list:
    """简单的最近邻构造，返回 1D tour（以 depot 开头、最后回到 depot）。"""
    tour = [depot]
    n = dis_matrix.shape[0]
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, _ = min(neighbours, key=lambda e: e[1])
        tour.append(j)
    tour.append(depot)
    return tour


def nearest_neighbor_2End(dis_matrix: np.ndarray, depot: int) -> np.ndarray:
    """
    最近邻构造 + 2End 表示。
    2End route 的含义：route[i, 0]=prev(i), route[i, 1]=next(i)。
    """
    tour = [depot]
    n = dis_matrix.shape[0]
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, _ = min(neighbours, key=lambda e: e[1])
        tour.append(j)
    tour.append(depot)

    route2End = np.zeros((n, 2), dtype=np.int64)
    route2End[depot, 0] = tour[-2]
    route2End[depot, 1] = tour[1]
    for idx in range(1, n):
        node = tour[idx]
        route2End[node, 0] = tour[idx - 1]
        route2End[node, 1] = tour[idx + 1]
    return route2End


def route2tour(route: np.ndarray, start: int = 0) -> list:
    """从 2End 表示还原成顺序 tour（start, ...）。"""
    n = route.shape[0]
    tour = [start]
    cur = int(route[start, 1])
    while cur != start:
        tour.append(cur)
        cur = int(route[cur, 1])
        if len(tour) > n + 1:
            # 安全保护，防止异常结构导致死循环
            break
    return tour


def tour2route(tour: np.ndarray) -> np.ndarray:
    """从 1D tour（不含重复起点）构造 2End 表示。"""
    tour = np.asarray(tour, dtype=np.int64)
    n = len(tour)
    route2End = np.zeros((n, 2), dtype=np.int64)
    route2End[tour[0], 0] = tour[-1]
    route2End[tour[0], 1] = tour[1]
    for i in range(1, n - 1):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    route2End[tour[-1], 0] = tour[-2]
    route2End[tour[-1], 1] = tour[0]
    return route2End


# ---------- 局部搜索 ----------

def local_search_basic(
    init_route: np.ndarray,
    init_cost: float,
    D: np.ndarray,
    N: np.ndarray,
    first_improvement: bool,
) -> Tuple[np.ndarray, float]:
    """
    轻量版 LS：
      - 每轮依次执行 two_opt_a2a -> relocate_a2a；
      - 若两者都未改进，则结束；
      - 迭代轮数最多与规模线性相关（避免“跑到时间尽头”）。
    """
    n = D.shape[0]
    max_iters = max(4, n // 80)

    cur_route = init_route
    cur_cost = float(init_cost)

    for _ in range(max_iters):
        improved = False

        # Phase 1: 2-opt
        delta, new_route = gls_operators.two_opt_a2a(cur_route, D, N, first_improvement)
        if delta < -1e-12:
            cur_cost += float(delta)
            cur_route = new_route
            improved = True

        # Phase 2: relocate
        delta, new_route = gls_operators.relocate_a2a(cur_route, D, N, first_improvement, 0.0)
        if delta < -1e-12:
            cur_cost += float(delta)
            cur_route = new_route
            improved = True

        if not improved:
            break

    return cur_route, cur_cost


def local_search_strong_basic(
    init_route: np.ndarray,
    init_cost: float,
    D: np.ndarray,
    N: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    强化版 LS：
      - 使用 best-improvement；
      - 迭代轮数更大；
      - 在 two_opt_a2a / relocate_a2a 基础上额外叠加 Or-opt(chain_len=2)，
        用现有 gls_operators.or_opt_chain 实现，以避免引入新的低层算子。
    """
    n = D.shape[0]
    max_iters = max(8, n // 30)

    cur_route = init_route
    cur_cost = float(init_cost)

    for _ in range(max_iters):
        improved = False

        # Phase 1: 2-opt（best-improvement）
        delta, new_route = gls_operators.two_opt_a2a(cur_route, D, N, False)
        if delta < -1e-12:
            cur_cost += float(delta)
            cur_route = new_route
            improved = True

        # Phase 2: relocate（best-improvement）
        delta, new_route = gls_operators.relocate_a2a(cur_route, D, N, False, 0.0)
        if delta < -1e-12:
            cur_cost += float(delta)
            cur_route = new_route
            improved = True

        # Phase 3: Or-opt(chain_len=2) 作为附加算子
        if not improved:
            delta, new_route = gls_operators.or_opt_chain(
                cur_route, D, chain_len=2, first_improvement=True
            )
            if delta < -1e-12:
                cur_cost = utils.tour_cost_2End(D, new_route)
                cur_route = new_route
                improved = True

        if not improved:
            break

    return cur_route, cur_cost


# ---------- 扰动（kick） ----------

def _apply_perturbation(
    route2End: np.ndarray,
    edge_weight: np.ndarray,
    nearest_indices: np.ndarray,
    spec: GLSSpec | None,
) -> np.ndarray:
    """
    强扰动（kick）：
      - type: "none"            -> 不做扰动
      - type: "random_relocate" -> 随机 relocate 若干次
      - type: "double_bridge"   -> 对 tour 做 double-bridge 4-opt 扰动
    共同参数：
      - moves: 每次 kick 中执行多少次扰动（默认 3）
    """
    if spec is None or not isinstance(spec.perturb, dict):
        return route2End

    p_conf = spec.perturb
    p_type = str(p_conf.get("type", "none"))
    if p_type == "none":
        return route2End

    moves = int(p_conf.get("moves", 3))
    moves = max(1, moves)

    n = edge_weight.shape[0]
    route = route2End.copy()

    if p_type == "random_relocate":
        for _ in range(moves):
            i = random.randrange(0, n)
            j = random.randrange(0, n)
            if i == j:
                continue
            delta = gls_operators.relocate_cost(route, edge_weight, i, j)
            if delta < -1e-12:
                route = gls_operators.relocate(route, i, j)
        return route

    if p_type == "double_bridge":
        # 先转为 1D tour
        tour_list = route2tour(route)
        if len(tour_list) < 5:
            return route

        tour = tour_list[:-1]  # 去掉终点 depot（与起点相同）
        n1 = len(tour)
        if n1 < 8:
            return route

        # 选 4 个切点：0 < a < b < c < d < n1
        cuts = sorted(random.sample(range(1, n1), 4))
        a, b, c, d = cuts

        p0 = tour[:a]
        p1 = tour[a:b]
        p2 = tour[b:c]
        p3 = tour[c:d]
        p4 = tour[d:]

        new_tour = p0 + p2 + p3 + p1 + p4
        # 重新闭环
        new_tour.append(new_tour[0])
        new_route = tour2route(np.asarray(new_tour[:-1], dtype=np.int64))
        return new_route

    # 其它未知类型：忽略
    return route2End


# ---------- GLS 主循环 ----------

def guided_local_search(
    coords: np.ndarray,
    edge_weight: np.ndarray,
    nearest_indices: np.ndarray,
    init_tour: np.ndarray,
    init_cost: float,
    t_lim: float,
    ite_max: int,
    perturbation_moves: int,
    first_improvement: bool = False,
    guide_algorithm=None,
    spec: GLSSpec | None = None,
    trace_events: list | None = None,
    trace_t0: float | None = None,
):
    """
    在统一骨架上运行 GLS：

    - 若 guide_algorithm 为 None，则纯 LS；
    - 若 guide_algorithm 具有属性 lam，则视为“内置 GLS 模式”（BuiltinGLS）：
        * 用 D' = D + lam_eff * penalty 做 LS；
          其中 lam_eff = α * (初始平均边长)，α=guide_algorithm.lam；
        * 在当前局部最优 tour 上计算 utility(e) = w(e) / (1 + penalty(e))；
        * 对 utility 最大的若干条边加 penalty；
    - 否则视为“LLM 指导模式”：
        * 由 guide_algorithm.update_edge_distance(edge_weight, tour, penalty) 生成 D'；
        * 对 D' 做基本数值清洗（对称化 / 去 inf / clip）。
    - 支持 spec.perturb 配置的扰动。
    """
    random.seed(2024)

    n = edge_weight.shape[0]
    edge_penalty = np.zeros((n, n), dtype=np.float64)

    # ---------- 解析 engine 配置 ----------
    engine_type = "ls_basic"
    if isinstance(spec, GLSSpec):
        eng = getattr(spec, "engine", None)
        if isinstance(eng, dict):
            engine_type = str(eng.get("type", "ls_basic")) or "ls_basic"

    # 当前版本中，仅实现 ls_basic，其它类型统一回退
    if engine_type != "ls_basic":
        engine_type = "ls_basic"

    # [tLDR-trace] optional incumbent trace (event-based; record only improvements)
    if trace_events is not None and trace_t0 is None:
        trace_t0 = time.perf_counter()

    # ---------- 初始 LS ----------
    cur_route, cur_cost = local_search_basic(
        init_tour, init_cost, edge_weight, nearest_indices, first_improvement
    )
    best_route = cur_route.copy()
    best_cost = float(cur_cost)

    # [tLDR-trace] record initial incumbent at t=0 (after initial LS)
    if trace_events is not None and trace_t0 is not None:
        trace_events.append((0.0, float(best_cost)))

    # ---------- guidance / builtin GLS 模式解析 ----------
    top_k = 5
    if spec is not None and isinstance(spec.guidance, dict):
        try:
            tk = int(spec.guidance.get("top_k", 5))
            if tk >= 1:
                top_k = min(tk, n)
        except Exception:
            pass
    top_k = max(1, min(int(top_k), n))

    builtin_gls = False    # BuiltinGLS（lambda 在类属性 lam 中）
    gls_lam = 0.5
    if guide_algorithm is not None and hasattr(guide_algorithm, "lam"):
        builtin_gls = True
        try:
            gls_lam = float(getattr(guide_algorithm, "lam", 0.5))
        except Exception:
            gls_lam = 0.5

    # === 内置 GLS：预计算 lam_eff = α * (初始平均边长) ===
    gls_scale = 0.0
    if builtin_gls:
        alpha = gls_lam
        avg_cost = max(1e-9, cur_cost / float(n))
        gls_scale = alpha * avg_cost

    # ---------- 停止条件 / 扰动参数 ----------
    iter_i = 0
    no_improve = 0
    max_no_improve = 80
    if spec is not None and isinstance(spec.schedule, dict):
        try:
            max_no_improve = int(spec.schedule.get("max_no_improve", 80))
        except Exception:
            pass

    perturb_type = "none"
    perturb_interval = max_no_improve
    if spec is not None and isinstance(spec.perturb, dict):
        p_conf = spec.perturb
        perturb_type = str(p_conf.get("type", "none"))
        perturb_interval = int(p_conf.get("interval", max_no_improve))
        if perturb_interval <= 0:
            perturb_interval = max_no_improve

    # 强化 LS 调度
    strong_ls_used = 0
    max_strong_ls_calls = 2

    # 主循环
    while iter_i < ite_max and time.time() < t_lim:
        # === 1) 选择当前使用的距离矩阵 work_D ===
        if builtin_gls:
            # 经典 GLS：D' = D + lam_eff * penalty
            guided_D = edge_weight + gls_scale * edge_penalty
            work_D = guided_D
        elif guide_algorithm is not None and hasattr(guide_algorithm, "update_edge_distance"):
            # LLM 指导模式：让外部给一个 D'，我们做基本的数值清洗
            cur_tour_vec = np.array(route2tour(cur_route), dtype=np.int64)
            try:
                guided_raw = guide_algorithm.update_edge_distance(
                    edge_weight, cur_tour_vec, edge_penalty
                )
                guided_D = np.asarray(guided_raw, dtype=np.float64)
            except Exception:
                guided_D = edge_weight.copy()
            else:
                # 数值安全性处理
                if guided_D.shape != edge_weight.shape:
                    guided_D = edge_weight.copy()
                else:
                    guided_D = np.where(np.isfinite(guided_D), guided_D, edge_weight)
                    guided_D = 0.5 * (guided_D + guided_D.T)
                    np.fill_diagonal(guided_D, 0.0)

                    min_w = float(edge_weight.min())
                    max_w = float(edge_weight.max())
                    lo = max(1e-9, 0.5 * min_w)
                    hi = 2.0 * max_w
                    guided_D = np.clip(guided_D, lo, hi)

            work_D = guided_D
        else:
            # 无 guidance：直接用原始距离矩阵
            work_D = edge_weight

        # === 2) 在 work_D 上做一次 main LS ===
        cur_route, _ = local_search_basic(
            cur_route, cur_cost, work_D, nearest_indices, first_improvement
        )
        # 评估仍然使用真实距离
        cur_cost = utils.tour_cost_2End(edge_weight, cur_route)

        # === 3) 更新 best / no_improve ===
        if cur_cost < best_cost - 1e-12:
            best_route = cur_route.copy()
            best_cost = float(cur_cost)
            no_improve = 0
            # [tLDR-trace] incumbent improved
            if trace_events is not None and trace_t0 is not None:
                t_now = time.perf_counter() - trace_t0
                trace_events.append((float(t_now), float(best_cost)))
        else:
            no_improve += 1

        # === 3.5) 停滞后触发强化版 LS（仅 basic 引擎使用，最多若干次） ===
        if (
            engine_type == "ls_basic"
            and strong_ls_used < max_strong_ls_calls
            and no_improve >= max_no_improve // 2
            and time.time() < t_lim
        ):
            cur_route, _ = local_search_strong_basic(cur_route, cur_cost, work_D, nearest_indices)
            cur_cost = utils.tour_cost_2End(edge_weight, cur_route)
            strong_ls_used += 1

            if cur_cost < best_cost - 1e-12:
                best_route = cur_route.copy()
                best_cost = float(cur_cost)
                no_improve = 0

        # === 4) 按需触发 kick（仍然在真实距离上评估） ===
        if (
            perturb_type != "none"
            and perturb_interval > 0
            and (iter_i + 1) % perturb_interval == 0
            and time.time() < t_lim
        ):
            kicked_route = _apply_perturbation(cur_route, edge_weight, nearest_indices, spec)
            kicked_cost = utils.tour_cost_2End(edge_weight, kicked_route)
            if kicked_cost < cur_cost - 1e-12:
                cur_route = kicked_route
                cur_cost = kicked_cost
                no_improve = 0

        # === 5) 内置 GLS 模式：在当前局部最优 tour 上更新 penalty ===
        if builtin_gls:
            tour_vec = np.array(route2tour(cur_route), dtype=np.int64)
            m = len(tour_vec)
            if m > 1:
                utilities = np.empty(m, dtype=np.float64)
                for i in range(m):
                    u = int(tour_vec[i])
                    v = int(tour_vec[(i + 1) % m])
                    w = float(edge_weight[u, v])
                    p = float(edge_penalty[u, v])
                    utilities[i] = w / (1.0 + p)

                # 选 utility 最大的若干条边增加 penalty
                order = np.argsort(utilities)[::-1]
                num_update = max(1, min(m, top_k * max(1, int(perturbation_moves))))
                for idx in range(num_update):
                    e_idx = int(order[idx])
                    u = int(tour_vec[e_idx])
                    v = int(tour_vec[(e_idx + 1) % m])
                    edge_penalty[u, v] += 1.0
                    edge_penalty[v, u] += 1.0

        iter_i += 1

        # 软停止：如果没有 kick 机制，且停滞太久，就直接退出
        if no_improve >= max_no_improve and perturb_type == "none":
            break

    # 返回当前最优路线与成本
    return best_route, best_cost, iter_i
