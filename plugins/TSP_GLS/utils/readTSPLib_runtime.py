"""
轻量级 TSPLIB 读取工具（运行时用，不再依赖中间的 TSPAEL64.pkl）。

主要功能：
- 在给定 root 目录下，根据 name 列表找到 {name}.tsp[.gz] / {name}.opt.tour[.gz]
- 使用 tsplib95 解析 tsp 文本，构造距离矩阵
- 读取 solutions 文件（若提供）以获得最优目标值
- 返回：
    names:          list[str]
    coords_list:    list[np.ndarray(N,2)]
    distmats_list:  list[np.ndarray(N,N)]
    opt_costs:      np.ndarray[len(names)]
"""
from __future__ import annotations

import os
import gzip
import tempfile
from typing import Iterable, List, Tuple, Dict, Optional
import math
import numpy as np

try:
    import tsplib95
except Exception:
    tsplib95 = None



def read_solutions_file(path: str) -> Dict[str, float]:
    """
    解析 mastqe/tsplib 仓库里的 solutions 文件，兼容几种常见格式：

    kroA100 : 21282
    kroB100: 22141
    kroC100  20749
    """
    mapping: Dict[str, float] = {}
    if not path or not os.path.exists(path):
        return mapping

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" in s:
                name_part, val_part = s.split(":", 1)
            else:
                parts = s.split()
                if len(parts) < 2:
                    continue
                name_part, val_part = parts[0], parts[1]
            name = name_part.strip()
            try:
                val = float(val_part.strip())
            except Exception:
                continue
            mapping[name] = val
    return mapping



def _load_text(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_tsplib_problem_from_text(text: str):
    """
    兼容 tsplib95 的 parse / load 两种接口。
    """
    if tsplib95 is None:
        raise RuntimeError("tsplib95 is not installed, but TSPLIB parsing is required.")

    parse_fn = getattr(tsplib95, "parse", None)
    if callable(parse_fn):
        return parse_fn(text)

    with tempfile.NamedTemporaryFile("w+", suffix=".tsp", delete=False) as f:
        tmp_path = f.name
        f.write(text)
    try:
        prob = tsplib95.load(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return prob





def coords_to_array(prob) -> np.ndarray:
    """
    从 tsplib95 的 problem 对象里提取 (N,2) 的坐标数组，优先级：
    1) node_coords（标准 TSPLIB 坐标）
    2) display_data（DISPLAY_DATA_SECTION）
    3) 若都没有，则用节点数 N 构造一圈“伪坐标”（均匀分布在单位圆上）

    这样可以兼容：
    - 经典带 NODE_COORD_SECTION 的实例（att48, bier127, gr229 等）
    - 只有 DISPLAY_DATA_SECTION 的实例（bayg29, bays29 等）
    - 只有距离矩阵（EXPLICIT）的实例
    """

    coords = getattr(prob, "node_coords", None)
    if coords:
        xs: list[tuple[float, float]] = []
        for i in range(1, len(coords) + 1):
            vals = coords[i]
            x, y = float(vals[0]), float(vals[1])
            xs.append((x, y))
        arr = np.asarray(xs, dtype=np.float64)
        if arr.shape[0] > 0:
            return arr

    disp = getattr(prob, "display_data", None)
    if disp:
        xs: list[tuple[float, float]] = []
        for i in range(1, len(disp) + 1):
            vals = disp[i]
            x, y = float(vals[0]), float(vals[1])
            xs.append((x, y))
        arr = np.asarray(xs, dtype=np.float64)
        if arr.shape[0] > 0:
            return arr

    try:
        n = int(getattr(prob, "dimension", 0))
    except Exception:
        n = 0

    if n <= 0:
        try:
            nodes = list(prob.get_nodes())
            n = len(nodes)
        except Exception:
            n = 0

    if n <= 0:
        raise ValueError("TSPLIB problem has neither node_coords/display_data nor valid dimension.")

    xs: list[tuple[float, float]] = []
    for i in range(n):
        theta = 2.0 * math.pi * i / max(n, 1)
        xs.append((math.cos(theta), math.sin(theta)))
    return np.asarray(xs, dtype=np.float64)


def build_distance_matrix(prob) -> np.ndarray:
    """
    使用 tsplib95.get_weight(i, j) 构造 (N,N) 的距离矩阵。
    """
    nodes = sorted(list(prob.get_nodes()))
    n = len(nodes)
    mat = np.zeros((n, n), dtype=np.float64)
    for ii, i in enumerate(nodes):
        for jj, j in enumerate(nodes):
            w = prob.get_weight(i, j)
            mat[ii, jj] = float(w)
    return mat



def _load_opt_tour_indices(root: str, name: str) -> Optional[np.ndarray]:
    """
    读取 {name}.opt.tour[.gz]，解析 TOUR_SECTION 里的城市序列，返回 0-based 序列。
    若找不到或解析失败，返回 None。
    """
    candidates = [
        os.path.join(root, f"{name}.opt.tour"),
        os.path.join(root, f"{name}.opt.tour.gz"),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        return None

    text = _load_text(path)
    lines = text.splitlines()
    in_section = False
    nodes: List[int] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("TOUR_SECTION"):
            in_section = True
            continue
        if up.startswith("EOF"):
            break
        if not in_section:
            continue
        toks = s.split()
        for tok in toks:
            try:
                v = int(tok)
            except Exception:
                continue
            if v == -1:
                in_section = False
                break
            nodes.append(v)
        if not in_section:
            break

    if not nodes:
        return None

    arr = np.asarray(nodes, dtype=np.int64) - 1
    return arr


def tour_cost(distmat: np.ndarray, order: np.ndarray) -> float:
    """
    在给定距离矩阵上计算封闭环路的总长度。
    order 为 0-based 节点序列。
    """
    if order is None or len(order) < 2:
        raise ValueError("Empty tour.")
    n = len(order)
    s = 0.0
    for i in range(n):
        a = int(order[i])
        b = int(order[(i + 1) % n])
        s += float(distmat[a, b])
    return float(s)



def find_instances(root: str, names_filter: Optional[Iterable[str]] = None) -> List[str]:
    """
    在 root 目录下查找所有 *.tsp / *.tsp.gz 的文件名（不带扩展名），
    再按 names_filter 进行过滤（若给定）。
    """
    all_names = set()
    for fn in os.listdir(root):
        low = fn.lower()
        if low.endswith(".tsp") or low.endswith(".tsp.gz"):
            base = fn.split(".", 1)[0]
            all_names.add(base)

    if names_filter is None:
        return sorted(all_names)

    out: List[str] = []
    for nm in names_filter:
        if nm in all_names:
            out.append(nm)
        else:
            nm_lower = nm.lower()
            for cand in all_names:
                if cand.lower() == nm_lower:
                    out.append(cand)
                    break
    seen = set()
    ordered = []
    for nm in out:
        if nm not in seen:
            ordered.append(nm)
            seen.add(nm)
    return ordered


def _load_tsplib_problem_from_root(root: str, name: str):
    """
    在 root 目录下，寻找 {name}.tsp 或 {name}.tsp.gz，解析为 tsplib95 problem。
    """
    candidates = [
        os.path.join(root, f"{name}.tsp"),
        os.path.join(root, f"{name}.tsp.gz"),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"Cannot find TSPLIB instance for name={name} under {root}")
    text = _load_text(path)
    return load_tsplib_problem_from_text(text)















def load_instances(
    root: str,
    solutions_file: Optional[str] = None,
    names: Optional[Iterable[str]] = None,
    min_nodes: int = 2,
    max_nodes: int = 999999,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    从 root 目录下按 names 读取一批 TSPLIB 实例，返回：
        names, coords_list, distmats_list, opt_costs
    若某个实例没有最优目标值信息（solutions 文件和 .opt.tour 都没有），则跳过。

    这里增加了详细的 debug 输出，以便定位某个实例（例如 bayg29）为什么被跳过。
    """
    if not os.path.isdir(root):
        raise NotADirectoryError(f"TSPLIB root not found: {root}")

    sol_map: Dict[str, float] = {}
    if solutions_file:
        sol_map = read_solutions_file(solutions_file)

    selected_names = find_instances(root, names)

    coords_list: List[np.ndarray] = []
    dist_list: List[np.ndarray] = []
    costs: List[float] = []
    kept_names: List[str] = []


    for nm in selected_names:
        try:
            prob = _load_tsplib_problem_from_root(root, nm)

            ew_type = getattr(prob, "edge_weight_type", None)
            if ew_type and str(ew_type).upper() == "SPECIAL":
                print(f"[TSPLIB] Skip {nm}: EDGE_WEIGHT_TYPE=SPECIAL (not supported in this loader).")
                continue

            coords = coords_to_array(prob)
            n = coords.shape[0]
            if n < min_nodes or n > max_nodes:
                print(f"[TSPLIB] Skip {nm}: node count {n} out of range [{min_nodes}, {max_nodes}].")
                continue

            distmat = build_distance_matrix(prob)

            opt_cost = None

            if nm in sol_map:
                opt_cost = float(sol_map[nm])

            if opt_cost is None:
                tour = _load_opt_tour_indices(root, nm)
                if tour is not None:
                    opt_cost = tour_cost(distmat, tour)

            if opt_cost is None:
                print(f"[TSPLIB] Skip {nm}: no optimal cost found (no solutions entry and no .opt.tour).")
                continue

            kept_names.append(nm)
            coords_list.append(coords)
            dist_list.append(distmat)
            costs.append(float(opt_cost))

        except Exception as ex:
            print(f"[TSPLIB] Error while loading {nm}: {ex}")
            continue

    if not kept_names:
        return [], [], [], np.asarray([], dtype=np.float64)

    opt_costs = np.asarray(costs, dtype=np.float64)
    return kept_names, coords_list, dist_list, opt_costs
