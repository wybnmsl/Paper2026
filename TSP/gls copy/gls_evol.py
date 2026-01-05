# zTSP/gls/gls_evol.py
import time
import numpy as np
from numba import jit
from gls import gls_operators
from utils import utils
import random
from gls.spec import GLSSpec

# 原始 NN 与 2End 版本（保留）
def nearest_neighbor(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, _ = min(neighbours, key=lambda e: e[1])
        tour.append(j)
    tour.append(depot)
    return tour

def nearest_neighbor_2End(dis_matrix, depot):
    tour = [depot]
    n = len(dis_matrix)
    nodes = np.arange(n)
    while len(tour) < n:
        i = tour[-1]
        neighbours = [(j, dis_matrix[i, j]) for j in nodes if j not in tour]
        j, _ = min(neighbours, key=lambda e: e[1])
        tour.append(j)
    tour.append(depot)
    route2End = np.zeros((n, 2), dtype=np.int64)
    route2End[0, 0] = tour[-2]
    route2End[0, 1] = tour[1]
    for i in range(1, n):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    return route2End

@jit(nopython=True, cache=True)
def local_search(init_tour, init_cost, D, N, first_improvement=False):
    cur_route, cur_cost = init_tour, init_cost
    improved = True
    while improved:
        improved = False

        delta, new_tour = gls_operators.two_opt_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

        delta, new_tour = gls_operators.relocate_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

    return cur_route, cur_cost

@jit(nopython=True, cache=True)
def route2tour(route):
    s = 0
    tour = []
    for _ in range(len(route)):
        nxt = route[s, 1]
        tour.append(nxt)
        s = nxt
    return tour

@jit(nopython=True, cache=True)
def tour2route(tour):
    n = len(tour)
    route2End = np.zeros((n, 2), dtype=np.int64)
    route2End[tour[0], 0] = tour[-1]
    route2End[tour[0], 1] = tour[1]
    for i in range(1, n - 1):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    route2End[tour[n - 1], 0] = tour[n - 2]
    route2End[tour[n - 1], 1] = tour[0]
    return route2End

# def guided_local_search(coords, edge_weight, nearest_indices, init_tour, init_cost,
#                         t_lim, ite_max, perturbation_moves,
#                         first_improvement=False, guide_algorithm=None,
#                         spec: GLSSpec | None = None):
#     """
#     与 e/m 保持同一骨架；当 spec 提供时，使用其中的参数/top_k/算子等调整强度。
#     """
#     # 固定随机性（学术复现需要）
#     random.seed(2024)

#     # 基础局部搜索（numba 加速）
#     cur_route, cur_cost = local_search(init_tour, init_cost, edge_weight, nearest_indices, first_improvement)
#     best_route, best_cost = cur_route, cur_cost

#     n = edge_weight.shape[0]
#     edge_penalty = np.zeros((n, n), dtype=np.float64)

#     # 从 spec 读取引导强度
#     top_k = 5
#     if spec is not None and isinstance(spec.guidance, dict):
#         try:
#             _tk = int(spec.guidance.get("top_k", 5))
#             if _tk >= 1:
#                 top_k = min(_tk, n)
#         except Exception:
#             pass

#     # 从 spec 读取 schedule（max_no_improve）
#     iter_i = 0
#     no_improve = 0
#     max_no_improve = 80
#     if spec is not None and isinstance(spec.schedule, dict):
#         try:
#             max_no_improve = int(spec.schedule.get("max_no_improve", 80))
#         except Exception:
#             pass

#     # 从 spec.operators 解析算子开关：如果显式给出，则视为白名单
#     use_two_opt = True
#     use_relocate = True
#     if spec is not None:
#         try:
#             ops = getattr(spec, "operators", None) or []
#             names = set()
#             for op in ops:
#                 if op is None:
#                     continue
#                 name = getattr(op, "name", None)
#                 if name is None and isinstance(op, dict):
#                     name = op.get("name")
#                 if name is not None:
#                     names.add(str(name))
#             if names:
#                 use_two_opt = "two_opt" in names
#                 use_relocate = "relocate" in names
#         except Exception:
#             # 解析失败就退化为都开启
#             use_two_opt = True
#             use_relocate = True

#     # 主 GLS 循环
#     while iter_i < ite_max and time.time() < t_lim:
#         # 轻扰动（沿用旧接口；后续可用 spec.perturb 扩展）
#         for _ in range(perturbation_moves):
#             # 将 route2End 形式转为一维顺序
#             cur_tour = np.array(route2tour(cur_route), dtype=np.int64)

#             # 调用 e/m 产出的核心函数（黑盒）
#             edge_weight_guided = guide_algorithm.update_edge_distance(edge_weight, cur_tour, edge_penalty)
#             edge_weight_guided = np.asarray(edge_weight_guided, dtype=np.float64)

#             # 与原距离之差，用于挑选“最需处理的边”
#             edge_weight_gap = edge_weight_guided - edge_weight

#             # 逐步处理 top_k 条最显著边；中途检查时间
#             for _k in range(top_k):
#                 if time.time() >= t_lim:
#                     break
#                 idx = np.argmin(-edge_weight_gap, axis=None)
#                 rows, columns = np.unravel_index(idx, edge_weight_gap.shape)

#                 # 对称加罚（GLS 经典做法）
#                 edge_penalty[rows, columns] += 1.0
#                 edge_penalty[columns, rows] += 1.0

#                 # 已处理置零，避免重复
#                 edge_weight_gap[rows, columns] = 0.0
#                 edge_weight_gap[columns, rows] = 0.0

#                 # 使用“针对节点”的 all-variant 改善
#                 for id_ in (int(rows), int(columns)):
#                     if use_two_opt:
#                         delta, new_route = gls_operators.two_opt_o2a_all(
#                             cur_route, edge_weight_guided, nearest_indices, id_
#                         )
#                         if delta < 0:
#                             cur_cost = utils.tour_cost_2End(edge_weight, new_route)
#                             cur_route = new_route
#                         # 时间守卫：每次重操作后快速检查
#                         if time.time() >= t_lim:
#                             break

#                     if use_relocate and time.time() < t_lim:
#                         delta, new_route = gls_operators.relocate_o2a_all(
#                             cur_route, edge_weight_guided, nearest_indices, id_
#                         )
#                         if delta < 0:
#                             cur_cost = utils.tour_cost_2End(edge_weight, new_route)
#                             cur_route = new_route
#                         if time.time() >= t_lim:
#                             break

#                 if time.time() >= t_lim:
#                     break

#         # 回到 main LS 作整合提升
#         if time.time() >= t_lim:
#             break
#         cur_route, cur_cost = local_search(
#             cur_route, cur_cost, edge_weight, nearest_indices, first_improvement
#         )
#         cur_cost = utils.tour_cost_2End(edge_weight, cur_route)

#         if cur_cost < best_cost - 1e-12:
#             best_cost = cur_cost
#             best_route = cur_route
#             no_improve = 0
#         else:
#             no_improve += 1

#         iter_i += 1
#         # 软停止
#         if no_improve >= max_no_improve:
#             break

#         # 周期性快回退（原逻辑保留）
#         if iter_i % 50 == 0:
#             cur_route, cur_cost = best_route, best_cost

#     return best_route, best_cost, iter_i

def guided_local_search(coords, edge_weight, nearest_indices, init_tour, init_cost,
                        t_lim, ite_max, perturbation_moves,
                        first_improvement=False, guide_algorithm=None,
                        spec: GLSSpec | None = None):
    """
    与 e/m 保持同一骨架；当 spec 提供时，使用其中的参数/top_k 等调整强度。
    在原有 two_opt / relocate 骨干上，可选叠加 Or-opt(2/3) 段交换算子：
      - 两端仍然用真实距离 edge_weight 计算代价
      - 只改变当前 tour 的节点顺序，不改表示/接口
    """
    # 固定随机性（学术复现需要）
    random.seed(2024)

    # main LS（numba 加速）
    cur_route, cur_cost = local_search(
        init_tour, init_cost, edge_weight, nearest_indices, first_improvement
    )
    best_route, best_cost = cur_route, cur_cost

    n = edge_weight.shape[0]
    edge_penalty = np.zeros((n, n), dtype=np.float64)

    # 从 spec 读取引导强度
    top_k = 5
    if spec is not None and isinstance(spec.guidance, dict):
        try:
            _tk = int(spec.guidance.get("top_k", 5))
            if _tk >= 1:
                top_k = min(_tk, n)
        except Exception:
            pass
    top_k = max(1, min(int(top_k), n))

    # 从 spec 读取停止条件
    iter_i = 0
    no_improve = 0
    max_no_improve = 80
    if spec is not None and isinstance(spec.schedule, dict):
        try:
            max_no_improve = int(spec.schedule.get("max_no_improve", 80))
        except Exception:
            pass

    # 从 spec.operators 读取算子开关
    use_two_opt = True
    use_relocate = True
    use_or2 = False
    use_or3 = False
    if spec is not None:
        ops = getattr(spec, "operators", None) or []
        op_names = set()
        for op in ops:
            if op is None:
                continue
            name = getattr(op, "name", None)
            if name is None and isinstance(op, dict):
                name = op.get("name")
            if name is not None:
                op_names.add(str(name))
        if op_names:
            # 如果 LLM 显式给出 operators，则以其为准
            use_two_opt = "two_opt" in op_names
            use_relocate = "relocate" in op_names
            # 新增骨干算子
            use_or2 = "or_opt2" in op_names
            use_or3 = "or_opt3" in op_names

    while iter_i < ite_max and time.time() < t_lim:
        # 轻扰动（保留旧接口；spec.perturb 可用于未来扩展）
        for _ in range(perturbation_moves):
            # 将路由转为一维顺序（传给核心黑盒）
            cur_tour = np.array(route2tour(cur_route), dtype=np.int64)
            # 调用 e/m 产出的核心函数（黑盒）
            edge_weight_guided = guide_algorithm.update_edge_distance(
                edge_weight, cur_tour, edge_penalty
            )
            edge_weight_guided = np.asarray(edge_weight_guided, dtype=np.float64)
            # 与原距离之差，用于挑选“最需处理的边”
            edge_weight_gap = edge_weight_guided - edge_weight

            # 逐步处理 top_k 条最显著边；中途检查时间
            for _k in range(top_k):
                if time.time() >= t_lim:
                    break
                idx = np.argmin(-edge_weight_gap, axis=None)
                rows, columns = np.unravel_index(idx, edge_weight_gap.shape)

                # 对称加罚（GLS 经典做法）
                edge_penalty[rows, columns] += 1.0
                edge_penalty[columns, rows] += 1.0

                # 已处理置零，避免重复
                edge_weight_gap[rows, columns] = 0.0
                edge_weight_gap[columns, rows] = 0.0

                # 使用“针对节点”的 all-variant 改善
                for id_ in (int(rows), int(columns)):
                    if use_two_opt:
                        delta, new_route = gls_operators.two_opt_o2a_all(
                            cur_route, edge_weight_guided, nearest_indices, id_
                        )
                        if delta < 0:
                            # 接受与比较仍然使用真实距离 edge_weight
                            cur_cost = utils.tour_cost_2End(edge_weight, new_route)
                            cur_route = new_route
                    # 时间守卫：每次重操作后快速检查
                    if time.time() >= t_lim:
                        break
                    if use_relocate:
                        delta, new_route = gls_operators.relocate_o2a_all(
                            cur_route, edge_weight_guided, nearest_indices, id_
                        )
                        if delta < 0:
                            cur_cost = utils.tour_cost_2End(edge_weight, new_route)
                            cur_route = new_route
                    if time.time() >= t_lim:
                        break
                if time.time() >= t_lim:
                    break

        # 回到 main LS 作整合提升
        if time.time() >= t_lim:
            break
        cur_route, cur_cost = local_search(
            cur_route, cur_cost, edge_weight, nearest_indices, first_improvement
        )
        cur_cost = utils.tour_cost_2End(edge_weight, cur_route)

        if cur_cost < best_cost - 1e-12:
            best_route, best_cost = cur_route, cur_cost
            no_improve = 0
        else:
            no_improve += 1

        iter_i += 1
        # 软停止
        if no_improve >= max_no_improve:
            break

        # 周期性快回退（原逻辑保留）
        if iter_i % 50 == 0:
            cur_route, cur_cost = best_route, best_cost

    # === 结束后：可选 Or-opt(2/3) 段交换精修 ===
    if (use_or2 or use_or3) and time.time() < t_lim:
        improved = True
        while improved and time.time() < t_lim:
            improved = False
            if use_or2:
                delta, new_route = gls_operators.or_opt_chain(
                    best_route, edge_weight, chain_len=2, first_improvement=True
                )
                if delta < 0:
                    best_route = new_route
                    best_cost = utils.tour_cost_2End(edge_weight, best_route)
                    improved = True
                    continue  # 先把 2-chain 吃完，再看 3-chain
            if use_or3 and time.time() < t_lim:
                delta, new_route = gls_operators.or_opt_chain(
                    best_route, edge_weight, chain_len=3, first_improvement=True
                )
                if delta < 0:
                    best_route = new_route
                    best_cost = utils.tour_cost_2End(edge_weight, best_route)
                    improved = True

    return best_route, best_cost, iter_i
