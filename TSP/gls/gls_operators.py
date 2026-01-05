# zTSP/gls/gls_operators.py
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def two_opt(tour, i, j):
    if i == j:
        return tour
    a = tour[i,0]
    b = tour[j,0]
    tour[i,0] = tour[i,1]
    tour[i,1] = j
    tour[j,0] = i
    tour[a,1] = b
    tour[b,1] = tour[b,0]
    tour[b,0] = a
    c = tour[b,1]
    while tour[c,1] != j:
        d = tour[c,0]
        tour[c,0] = tour[c,1]
        tour[c,1] = d
        c = d
    return tour

@jit(nopython=True, cache=True)
def two_opt_cost(tour, D, i, j):
    if i == j:
        return 0.0
    a = tour[i,0]
    b = tour[j,0]
    delta = D[a, b] + D[i, j] - D[a, i] - D[b, j]
    return delta

@jit(nopython=True, cache=True)
def two_opt_a2a(tour, D, N, first_improvement=False, set_delta=0.0):
    best_move = None
    best_delta = set_delta
    for i in range(0, len(tour) - 1):
        for j in N[i]:
            if i in tour[j] or j in tour[i]:
                continue
            delta = two_opt_cost(tour, D, i, j)
            if delta < best_delta and not np.isclose(0.0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break
        if first_improvement and best_move is not None:
            break
    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0.0, tour

@jit(nopython=True, cache=True)
def two_opt_o2a(tour, D, i, first_improvement=False):
    assert i > 0 and i < len(tour) - 1
    best_move = None
    best_delta = 0.0
    for j in range(1, len(tour) - 1):
        if abs(i - j) < 2:
            continue
        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0.0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break
    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0.0, tour

@jit(nopython=True, cache=True)
def two_opt_o2a_all(tour, D, N, i):
    best_delta = 0.0
    for j in N[i]:
        if i in tour[j] or j in tour[i]:
            continue
        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0.0, delta):
            best_delta = delta
            tour = two_opt(tour, i, j)
    return best_delta, tour

@jit(nopython=True, cache=True)
def relocate(tour, i, j):
    a = tour[i,0]
    b = i
    c = tour[i,1]
    tour[a,1] = c
    tour[c,0] = a
    d = tour[j,1]
    tour[d,0] = i
    tour[i,0] = j
    tour[i,1] = d
    tour[j,1] = i
    return tour

@jit(nopython=True, cache=True)
def relocate_cost(tour, D, i, j):
    if i == j:
        return 0.0
    a = tour[i,0]
    b = i
    c = tour[i,1]
    d = j
    e = tour[j,1]
    delta = - D[a, b] - D[b, c] + D[a, c] - D[d, e] + D[d, b] + D[b, e]
    return delta

@jit(nopython=True, cache=True)
def relocate_o2a(tour, D, i, first_improvement=False):
    assert i > 0 and i < len(tour) - 1
    best_move = None
    best_delta = 0.0
    for j in range(1, len(tour) - 1):
        if i == j:
            continue
        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0.0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break
    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0.0, tour

@jit(nopython=True, cache=True)
def relocate_o2a_all(tour, D, N, i):
    best_delta = 0.0
    for j in N[i]:
        if tour[j,1] == i:
            continue
        delta = relocate_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0.0, delta):
            best_delta = delta
            tour = relocate(tour, i, j)
    return best_delta, tour

@jit(nopython=True, cache=True)
def relocate_a2a(tour, D, N, first_improvement=False, set_delta=0.0):
    best_move = None
    best_delta = set_delta
    for i in range(0, len(tour) - 1):
        for j in N[i]:
            if tour[j,1] == i:
                continue
            delta = relocate_cost(tour, D, i, j)
            if delta < best_delta and not np.isclose(0.0, delta):
                best_delta = delta
                best_move = i, j
                if first_improvement:
                    break
        if first_improvement and best_move is not None:
            break
    if best_move is not None:
        return best_delta, relocate(tour, *best_move)
    return 0.0, tour


# ====== 新增：纯 Python 版本的 route2End <-> tour 一维表示 ======

def route2tour_py(route):
    """
    route: (n, 2) prev/next 表示的 2End 路径
    返回：从 0 开始的一圈 tour（长度 n）
    """
    n = route.shape[0]
    tour = []
    s = 0
    for _ in range(n):
        nxt = int(route[s, 1])
        tour.append(nxt)
        s = nxt
    return tour

def tour2route_py(tour):
    """
    tour: 长度 n 的一圈（list / 1D array）
    返回：route2End 表示
    """
    import numpy as _np
    n = len(tour)
    route2End = _np.zeros((n, 2), dtype=_np.int64)
    route2End[tour[0], 0] = tour[-1]
    route2End[tour[0], 1] = tour[1]
    for i in range(1, n - 1):
        route2End[tour[i], 0] = tour[i - 1]
        route2End[tour[i], 1] = tour[i + 1]
    route2End[tour[-1], 0] = tour[-2]
    route2End[tour[-1], 1] = tour[0]
    return route2End


# ====== 新增：Or-opt 核算子（在 1D tour 上操作） ======

def _index_in_segment(pos, start, length, n):
    """
    环上判断 pos 是否落在 [start, start+length-1] (mod n) 这段 segment 内。
    """
    end = (start + length - 1) % n
    if start <= end:
        return start <= pos <= end
    else:
        # 包了尾 / 头两段
        return pos >= start or pos <= end

def _or_opt_best_move_1d(route, D, NNs, chain_len, first_improvement=False, set_delta=0.0):
    """
    在给定的 1D route 上枚举 Or-opt(chain_len)，返回最好的一步。
    route: 1D numpy array, shape (n, )
    D: 距离矩阵 (n, n)
    NNs: 最近邻索引矩阵 (n, k)
    chain_len: 2 or 3
    返回: (best_delta, new_route_1d)；若无改进，则 (0.0, route)
    """
    import numpy as _np

    n = route.shape[0]
    if chain_len < 2 or chain_len >= n:
        return 0.0, route

    # node -> position 映射
    pos = _np.empty(n, dtype=_np.int64)
    for idx in range(n):
        pos[int(route[idx])] = idx

    best_delta = float(set_delta)
    best_start = -1
    best_b = -1

    for start_idx in range(n):
        A = int(route[start_idx])
        # 借用 A 的近邻来选插入位置
        for j_node in NNs[A]:
            b_idx = int(pos[int(j_node)])

            # 不能把链条插到自己里面或直接相邻成 degenerate move
            if _index_in_segment(b_idx, start_idx, chain_len, n):
                continue
            if _index_in_segment((b_idx + 1) % n, start_idx, chain_len, n):
                continue

            P = int(route[(start_idx - 1) % n])
            B = int(route[(start_idx + chain_len - 1) % n])
            S = int(route[(start_idx + chain_len) % n])
            Q = int(route[b_idx])
            R = int(route[(b_idx + 1) % n])

            # Or-opt 经典 delta：删 (P,A),(B,S),(Q,R) 加 (P,S),(Q,A),(B,R)
            delta = (
                D[P, S] + D[Q, A] + D[B, R]
                - (D[P, A] + D[B, S] + D[Q, R])
            )

            if delta < best_delta - 1e-15:
                best_delta = float(delta)
                best_start = start_idx
                best_b = b_idx
                if first_improvement:
                    break
        if first_improvement and best_start >= 0:
            break

    if best_start < 0 or best_delta >= set_delta:
        return 0.0, route

    # 应用最佳 Or-opt move
    seg_nodes = []
    for k in range(chain_len):
        seg_nodes.append(int(route[(best_start + k) % n]))
    seg_set = set(seg_nodes)

    new_route = []
    for idx in range(n):
        node = int(route[idx])
        if node in seg_set:
            continue
        new_route.append(node)
        if idx == best_b:
            new_route.extend(seg_nodes)

    new_route = _np.asarray(new_route, dtype=_np.int64)
    return best_delta, new_route


def or_opt_chain_a2a(route2End, D, NNs, chain_len=2,
                     first_improvement=False, set_delta=0.0):
    """
    在 route2End 表示的 tour 上执行一次 Or-opt(chain_len) hill-climb 步。
    - route2End: (n, 2) prev/next
    - D: (n, n) 距离矩阵（真实距离）
    - NNs: (n, k) 最近邻索引
    返回: (delta, new_route2End)
      若无改进，则 (0.0, 原 route2End)
    """
    import numpy as _np

    # 转为 1D tour
    tour_list = route2tour_py(route2End)
    route_1d = _np.asarray(tour_list, dtype=_np.int64)

    delta, new_route_1d = _or_opt_best_move_1d(
        route_1d, D, NNs,
        chain_len=chain_len,
        first_improvement=first_improvement,
        set_delta=set_delta
    )

    if delta >= 0.0:
        return 0.0, route2End

    new_route2End = tour2route_py(list(new_route_1d))
    return float(delta), new_route2End





# ---------- Or-opt chain operators (2- / 3-node segments) ----------
def _route2tour_py(route):
    """将 2End 表示的 route 转成一维 tour（与 gls_evol.route2tour 语义一致）。

    只在 Python 端调用，不加 numba，避免与现有 jit 逻辑耦合。
    """
    n = len(route)
    tour = []
    s = 0
    for _ in range(n):
        nxt = int(route[s, 1])
        tour.append(nxt)
        s = nxt
    return tour


def _tour2route_py(tour):
    """与 gls_evol.tour2route 同构的 2End 构造。"""
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


def _apply_or_opt_move_1d(tour, s, e, j):
    """在一维 tour 上执行 Or-opt 段交换。

    - segment = tour[s:e]（假设 0 < s, e < n，且不跨越端点）
    - 插入位置：在原 tour 中的 j 之后。
    """
    seg = tour[s:e]
    base = tour[:s] + tour[e:]
    seg_len = e - s
    if j < s:
        ins_idx = j + 1
    else:
        # e 及之后的元素在 base 中左移 seg_len
        ins_idx = j - seg_len + 1
    new_tour = base[:ins_idx] + seg + base[ins_idx:]
    return new_tour


def _or_opt_chain_1d(tour, D, chain_len, first_improvement=False, set_delta=0.0):
    """在 1D tour 上做一次 Or-opt(chain_len) 的 best/first-improvement 搜索。

    复杂度 O(n^2)，但只在 T 阶段末尾少量调用，不进入 numba 主循环。
    返回 (delta, new_tour)，若无改进则 delta=0.0, new_tour=tour。
    """
    import math

    n = len(tour)
    if n < chain_len + 2:
        return 0.0, tour

    best_delta = set_delta
    best_move = None  # (s, e, j)

    # 只考虑不跨越端点的连续段：s in [1, n-chain_len-1]，e = s + chain_len < n
    for s in range(1, n - chain_len):
        e = s + chain_len
        pre = tour[s - 1]
        post = tour[e]
        head = tour[s]
        tail = tour[e - 1]

        for j in range(0, n - 1):
            # 插入位置在 (tour[j], tour[j+1]) 之间；不允许落在段内部或其紧前位置
            if j >= s - 1 and j < e:
                continue
            a = tour[j]
            b = tour[j + 1]

            # 只改三条边： (pre,head),(tail,post),(a,b)
            # 换成 (pre,post),(a,head),(tail,b)
            delta = (
                D[pre, post] + D[a, head] + D[tail, b]
                - (D[pre, head] + D[tail, post] + D[a, b])
            )
            if delta < best_delta and not math.isclose(delta, 0.0):
                best_delta = delta
                best_move = (s, e, j)
                if first_improvement:
                    new_tour = _apply_or_opt_move_1d(tour, s, e, j)
                    return best_delta, new_tour

    if best_move is None:
        return 0.0, tour

    s, e, j = best_move
    new_tour = _apply_or_opt_move_1d(tour, s, e, j)
    return best_delta, new_tour


def or_opt_chain(route, D, chain_len=2, first_improvement=False):
    """在 2End route 上封装一次 Or-opt(chain_len) 操作。

    - 输入/输出均为 2End 表示，兼容现有 two_opt / relocate 流程。
    - 代价计算一律基于真实距离矩阵 D，不依赖 update_edge_distance。
    """
    tour = _route2tour_py(route)
    delta, new_tour = _or_opt_chain_1d(
        tour, D, chain_len,
        first_improvement=first_improvement,
        set_delta=0.0
    )
    if delta < 0.0:
        return delta, _tour2route_py(new_tour)
    else:
        return 0.0, route


import numpy as np

def _route2tour_py(route, start=0):
    """从 2-end 表示还原成顺序 tour（0, i1, i2, ...）"""
    n = route.shape[0]
    tour = [start]
    cur = route[start, 1]
    while cur != start:
        tour.append(cur)
        cur = route[cur, 1]
        if len(tour) > n + 1:
            # 安全保护，避免死循环
            raise RuntimeError("invalid route: cycle longer than n")
    if len(tour) != n:
        raise RuntimeError(f"invalid route: visited {len(tour)} of {n}")
    return tour


def _tour2route_py(tour):
    """从顺序 tour 还原成 2-end 表示"""
    n = len(tour)
    route = np.empty((n, 2), dtype=np.int64)
    for i, u in enumerate(tour):
        prev_u = tour[i - 1]
        next_u = tour[(i + 1) % n]
        route[u, 0] = prev_u
        route[u, 1] = next_u
    return route


def or_opt_chain(route, edge_weight, chain_len=2, first_improvement=True):
    """
    Or-opt 链式挪动算子：
      - chain_len = 2 或 3
      - route: 2-end 表示 (n, 2)
      - edge_weight: 距离矩阵 (n, n)
    返回: (delta, new_route)
    """
    n = route.shape[0]
    tour = _route2tour_py(route, start=0)

    # 基础 cost，仅用于 best-improvement 时计算 Δ
    base_cost = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        base_cost += edge_weight[a, b]

    best_delta = 0.0
    best_move = None  # (i, j)

    # 遍历所有长度为 chain_len 的连续段 [i, i+len)
    for i in range(n):
        seg_idx = [(i + k) % n for k in range(chain_len)]
        prev_i = (i - 1) % n

        a = tour[prev_i]                # 段前一个点
        b = tour[i]                     # 段第一个点
        c = tour[seg_idx[-1]]           # 段最后一个点
        d = tour[(seg_idx[-1] + 1) % n] # 段后一个点

        for j in range(n):
            # 插入位置：插在 j 之后
            if j in seg_idx:
                continue

            e = tour[j]
            f = tour[(j + 1) % n]

            # 这些情况是 no-op，跳过
            if j == prev_i or f == b:
                continue

            # 旧边：a-b, c-d, e-f
            # 新边：a-d, e-b, c-f
            old_cost = edge_weight[a, b] + edge_weight[c, d] + edge_weight[e, f]
            new_cost = edge_weight[a, d] + edge_weight[e, b] + edge_weight[c, f]
            delta = new_cost - old_cost

            if delta < -1e-12:
                # 构造新的 tour（删除段再插入到 j 后）
                new_tour = []
                for idx in range(n):
                    if idx in seg_idx:
                        continue
                    new_tour.append(tour[idx])
                    if idx == j:
                        # 在 j 后插入段
                        for k in seg_idx:
                            new_tour.append(tour[k])

                if first_improvement:
                    # 立刻返回
                    return delta, _tour2route_py(new_tour)

                # best-improvement 模式下只记录最优 move
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)

        # first_improvement 模式下，上面已经 return 了；这里仅用于 best-improvement

    if best_move is None:
        # 没有任何改进
        return 0.0, route

    # best-improvement：应用最优 move 一次
    i, j = best_move
    seg_idx = [(i + k) % n for k in range(chain_len)]
    new_tour = []
    for idx in range(n):
        if idx in seg_idx:
            continue
        new_tour.append(tour[idx])
        if idx == j:
            for k in seg_idx:
                new_tour.append(tour[k])

    # best_delta 是 new_cost - base_cost
    return best_delta, _tour2route_py(new_tour)
