# DASH/plugins/tsp_gls/problem.py
import numpy as np
import time
import os
import types
import warnings
import sys
import json

from .utils import readTSPRandom
from .utils import readTSPLib_runtime as tsplib_rt
from .gls.gls_run import solve_instance as solve_instance_legacy
from .gls.gls_run import solve_instance_with_spec
from .gls.spec import GLSSpec, default_gls_spec, from_json

# DAG compatibility (kept for legacy DAG runners; not required by plain TSPGLS execution).
try:
    from DASH.dag.runtime import DagExecutor, REGISTRY
except Exception:
    DagExecutor = None
    REGISTRY = None

try:
    import DASH.dag.nodes_common as _nodes_common
    import DASH.dag.nodes_tsp as _nodes_tsp
    if REGISTRY is not None:
        REGISTRY.register_from_module(_nodes_common)
        REGISTRY.register_from_module(_nodes_tsp)
except Exception:
    pass


class TSPGLS:
    def __init__(
        self,
        tsplib_root: str | None = None,
        solutions_file: str | None = None,
        case_names: list[str] | None = None,
        n_inst_eva: int | None = None,
        time_limit: float = 10.0,
        ite_max: int = 1000,
        perturbation_moves: int = 1,
        debug_mode: bool = True,
    ) -> None:
        """
        TSP + GLS 的本地问题接口。

        两种数据来源：
        1）兼容旧流程：tsplib_root=None 时，从 TrainingData/TSPAEL64.pkl 读取；
        2）新流程：指定 tsplib_root / solutions_file / case_names，直接从原始 TSPLIB 目录读取。
        """
        self.time_limit = float(time_limit)
        self.ite_max = int(ite_max)
        self.perturbation_moves = int(perturbation_moves)
        self.debug_mode = bool(debug_mode)

        path = os.path.dirname(os.path.abspath(__file__))

        # ---------------- 数据来源：TSPLIB 原始目录 ----------------
        if tsplib_root is not None:
            try:
                from .utils import readTSPLib_runtime as tsplib_rt  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "Failed to import utils.readTSPLib_runtime. "
                    "Please create tsp_gls/utils/readTSPLib_runtime.py as designed "
                    "and make sure tsplib95 is installed."
                ) from e

            root_abs = tsplib_root
            if not os.path.isabs(root_abs):
                root_abs = os.path.join(path, tsplib_root)

            sol_abs = None
            if solutions_file:
                sol_abs = solutions_file
                if not os.path.isabs(sol_abs):
                    sol_abs = os.path.join(path, solutions_file)

            names, coords_list, inst_list, opt_costs = tsplib_rt.load_instances(
                root_abs,
                sol_abs,
                names=case_names,
            )
            if not names:
                raise ValueError(
                    f"No TSPLIB instances loaded from root={root_abs} with names={case_names}."
                )

            self.instance_names = names
            self.coords = coords_list
            self.instances = inst_list
            self.opt_costs = opt_costs

        # ---------------- 数据来源：旧版 TSPAEL64.pkl ----------------
        else:
            self.instance_path = os.path.join(path, "TrainingData", "TSPAEL64.pkl")
            self.coords, self.instances, self.opt_costs = readTSPRandom.read_instance_all(
                self.instance_path
            )
            self.instance_names = [f"inst_{i}" for i in range(len(self.coords))]

        # n_inst_eva：控制参与评估的实例数（默认全部）
        if n_inst_eva is None:
            self.n_inst_eva = len(self.coords)
        else:
            self.n_inst_eva = max(1, min(int(n_inst_eva), len(self.coords)))

        # 截断到前 n_inst_eva 个实例，保证后续索引一致
        if self.n_inst_eva < len(self.coords):
            self.coords = self.coords[: self.n_inst_eva]
            self.instances = self.instances[: self.n_inst_eva]
            self.opt_costs = self.opt_costs[: self.n_inst_eva]
            self.instance_names = self.instance_names[: self.n_inst_eva]

        from .prompts import GetPrompts
        self.prompts = GetPrompts()
        self.prelude = (
            "import numpy as np\nimport math\nimport random\n"
            f"from {__package__}.gls import gls_operators\n"
            f"from {__package__}.utils import utils\n"
            f"from {__package__}.gls import gls_evol\n"
        )

    def _compute_instance_features(self, coords, dis_matrix):
        """
        给当前 instance 提取一组简单但通用的特征，用于后续 LDRPO / 分析：
        - n_nodes：节点数
        - mean/std/min/max_dist：距离矩阵的统计量（只看上三角）
        - bbox_w/bbox_h：坐标的包围盒宽高
        """
        try:
            n = int(dis_matrix.shape[0]) if dis_matrix is not None else 0

            if n > 1 and dis_matrix is not None:
                dm = np.asarray(dis_matrix, dtype=float)
                iu = np.triu_indices(n, k=1)
                dvals = dm[iu]
                dvals = dvals[np.isfinite(dvals)]
                if dvals.size > 0:
                    mean_d = float(np.mean(dvals))
                    std_d = float(np.std(dvals))
                    min_d = float(np.min(dvals))
                    max_d = float(np.max(dvals))
                else:
                    mean_d = std_d = min_d = max_d = 0.0
            else:
                mean_d = std_d = min_d = max_d = 0.0

            bbox_w = bbox_h = 0.0
            if coords is not None:
                arr = np.asarray(coords, dtype=float)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    xs = arr[:, 0]
                    ys = arr[:, 1] if arr.shape[1] > 1 else xs
                    bbox_w = float(xs.max() - xs.min())
                    bbox_h = float(ys.max() - ys.min())

            return {
                "n_nodes": n,
                "mean_dist": mean_d,
                "std_dist": std_d,
                "min_dist": min_d,
                "max_dist": max_d,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    #  GLS 评估（旧骨架 + GLSSpec 骨架）
    # ------------------------------------------------------------------
    def evaluateGLS_legacy(self, heuristic_module):
        """
        沿用原先的 GLS 骨架评估接口（无 GLSSpec）。
        返回：
            mean_gap, [ {index, name, gap, eval_time}, ... ]
        """
        n_eval = self.n_inst_eva
        gaps = np.zeros(n_eval, dtype=np.float64)
        inst_details = []

        for i in range(n_eval):
            t0 = time.time()
            gap = solve_instance_legacy(
                i,
                self.opt_costs[i],
                self.instances[i],
                self.coords[i],
                self.time_limit,
                self.ite_max,
                self.perturbation_moves,
                heuristic_module,
            )
            t1 = time.time()

            if gap < 0 and abs(gap) < 1e-9:
                gap = 0.0
            gaps[i] = float(gap)

            name = None
            if hasattr(self, "instance_names"):
                try:
                    name = self.instance_names[i]
                except Exception:
                    name = None

            inst_details.append(
                {
                    "index": i,
                    "name": name,
                    "gap": float(gap),
                    "eval_time": float(t1 - t0),
                }
            )

        return float(np.mean(gaps)), inst_details

    def evaluateGLS_with_spec(
        self,
        heuristic_module,
        gls_spec: GLSSpec | dict | None,
        return_traj_meta: bool = True,
    ):
        """
        使用 GLSSpec（或等价 dict）来配置 GLS 骨干：
        - 若 gls_spec 是 GLSSpec，就直接用；
        - 若是 dict，用 from_json 解析；
        - 若为 None，则退回 default_gls_spec()。

        返回：
          - 若 return_traj_meta=True:
              mean_gap, inst_details, traj_meta
            其中 traj_meta 至少包含:
              {"tLDR_traj_mean": float|None, "tLDR_traj_list": list[float]}
          - 否则:
              mean_gap, inst_details
        """
        # 规范化 spec
        if isinstance(gls_spec, GLSSpec):
            spec = gls_spec
        elif isinstance(gls_spec, dict):
            spec = from_json(gls_spec)
        else:
            spec = default_gls_spec()

        n_eval = self.n_inst_eva
        gaps = np.zeros(n_eval, dtype=np.float64)
        inst_details = []

        tldr_traj_list: list[float] = []

        for i in range(n_eval):
            t0 = time.time()

            if return_traj_meta:
                gap, inst_meta = solve_instance_with_spec(
                    i,
                    self.opt_costs[i],
                    self.instances[i],
                    self.coords[i],
                    self.time_limit,
                    self.ite_max,
                    self.perturbation_moves,
                    heuristic_module,
                    spec,
                    return_meta=True,
                )
            else:
                gap = solve_instance_with_spec(
                    i,
                    self.opt_costs[i],
                    self.instances[i],
                    self.coords[i],
                    self.time_limit,
                    self.ite_max,
                    self.perturbation_moves,
                    heuristic_module,
                    spec,
                )
                inst_meta = None

            t1 = time.time()

            if gap < 0 and abs(gap) < 1e-9:
                gap = 0.0
            gaps[i] = float(gap)

            # 实例名字（沿用 instance_names）
            name = None
            try:
                if self.instance_names is not None and i < len(self.instance_names):
                    name = self.instance_names[i]
            except Exception:
                name = None

            feats = self._compute_instance_features(self.coords[i], self.instances[i])

            tldr_i = None
            if return_traj_meta and isinstance(inst_meta, dict):
                tldr_i = inst_meta.get("tLDR_traj", None)
                if tldr_i is not None:
                    try:
                        tldr_traj_list.append(float(tldr_i))
                    except Exception:
                        pass

            inst_details.append(
                {
                    "index": i,
                    "name": name,
                    "gap": float(gap),
                    "eval_time": float(t1 - t0),
                    "features": feats,
                    "tLDR_traj": (tldr_i if return_traj_meta else None),
                    "T_used": (inst_meta.get("T_used") if (return_traj_meta and isinstance(inst_meta, dict)) else None),
                    "n_events": (inst_meta.get("n_events") if (return_traj_meta and isinstance(inst_meta, dict)) else None),
                }
            )

        mean_gap = float(np.mean(gaps))

        if return_traj_meta:
            traj_meta = {
                "tLDR_traj_mean": (float(np.mean(tldr_traj_list)) if len(tldr_traj_list) > 0 else None),
                "tLDR_traj_list": tldr_traj_list,
            }
            return mean_gap, inst_details, traj_meta

        return mean_gap, inst_details

    # ------------------------------------------------------------------
    #  DAG 评估（保留兼容；TSP 下不推荐使用）
    # ------------------------------------------------------------------
    def evaluateDAG(self, emitted_code, budgets: dict | None = None):
        dag_module = types.ModuleType("dag_solver_module")
        try:
            dag_module.utils = __import__(f"{__package__}.utils.utils", fromlist=['utils'])
            dag_module.gls = __import__(f"{__package__}.gls", fromlist=['gls'])
            dag_module.gls.gls_evol = __import__(f"{__package__}.gls.gls_evol", fromlist=['gls_evol']).gls_evol
            dag_module.gls.gls_operators = __import__(f"{__package__}.gls.gls_operators", fromlist=['gls_operators']).gls_operators
        except Exception:
            pass

        time_limit_cap = None
        fast_eval_n = None
        if isinstance(budgets, dict):
            try:
                time_limit_cap = float(budgets.get("time_limit_s", None))
            except Exception:
                pass
            try:
                fast_eval_n = int(budgets.get("fast_eval_n", None))
            except Exception:
                pass

        try:
            local_env = dag_module.__dict__
            exec(emitted_code, local_env)
            if "solve" not in local_env:
                raise RuntimeError("DAG-emitted code has no solve() function.")

            solve_fn = local_env["solve"]

            n_eval = self.n_inst_eva if fast_eval_n is None else min(self.n_inst_eva, int(fast_eval_n))
            gaps = np.zeros(n_eval, dtype=np.float64)
            for i in range(n_eval):
                coords = self.coords[i]
                dist = self.instances[i]
                route = solve_fn(coords, dist)
                cost_closed = self._closed_tour_cost(dist, route)

                if cost_closed is None or not np.isfinite(cost_closed) or cost_closed <= 0:
                    gaps[i] = 1e10
                else:
                    gap = (cost_closed / self.opt_costs[i] - 1) * 100.0
                    if gap < 0 and abs(gap) < 1e-9:
                        gap = 0.0
                    gaps[i] = gap

            return float(np.mean(gaps))
        except Exception as e:
            if self.debug_mode:
                print(f"--- DAG Evaluation Error ---")
                print(f"Error: {e}")
                print(f"Code that failed:\n{emitted_code}")
                print(f"--------------------------")
            return 1e10

    # ------------------------------------------------------------------
    #  若 DAG 的 solve 返回的是“边列表”而非城市序列，这里做一个小工具转换
    # ------------------------------------------------------------------
    def _route_to_order_list(self, route):
        if not route:
            return None

        if isinstance(route[0], (int, np.integer)):
            return list(route)

        try:
            edges = list(route)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)

            start = None
            for node, neigh in adj.items():
                if len(neigh) == 2:
                    start = node
                    break
            if start is None:
                start = edges[0][0]

            order = [start]
            prev = None
            cur = start
            while True:
                neighs = adj.get(cur, [])
                nxt = None
                for v in neighs:
                    if v != prev:
                        nxt = v
                        break
                if nxt is None or nxt == start:
                    break
                order.append(nxt)
                prev, cur = cur, nxt

            return order
        except Exception:
            return None

    def _closed_tour_cost(self, distmat, route) -> float | None:
        if route is None:
            return None

        if isinstance(route, (list, tuple)) and route and isinstance(route[0], (tuple, list)):
            route = self._route_to_order_list(route)
            if route is None:
                return None

        if not route:
            return None

        arr = np.asarray(route, dtype=np.int64)
        if arr.min() == 1:
            arr = arr - 1

        n = len(arr)
        if n < 2:
            return None

        total = 0.0
        for i in range(n):
            a = int(arr[i])
            b = int(arr[(i + 1) % n])
            total += float(distmat[a, b])
        return total

    # ------------------------------------------------------------------
    #  顶层 evaluate：供 DASH 调用
    # ------------------------------------------------------------------
    def evaluate(self, individual):
        """
        顶层评估接口，兼容原有 DASH：
        - 优先走 GLSSpec 路径（个体带 gls_spec + code）；
        - 再退化到 DAG 路径（个体带 dag_spec）；
        - 最后退化到 legacy GLS（只有 code）。

        返回 dict：{"fitness": float, "meta": {...}}，
        其中 meta["instances"] 会包含 per-instance 的结果：
            [{"index": i, "name": case_name, "gap": ..., "eval_time": ...}, ...]
        """
        fitness = 1e10
        meta: dict = {}

        # ---------- 优先：GLS 规格（t 阶段） ----------
        if individual.get("gls_spec") and individual.get("code"):
            try:
                heuristic_module = types.ModuleType("heuristic_module")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(self.prelude + individual["code"], heuristic_module.__dict__)

                fitness, inst_details, traj_meta = self.evaluateGLS_with_spec(
                    heuristic_module, individual["gls_spec"], return_traj_meta=True
                )
                meta["path"] = "gls_spec"
                meta["instances"] = inst_details
                if isinstance(traj_meta, dict):
                    meta.update(traj_meta)
                return {"fitness": float(fitness), "meta": meta}
            except Exception as e:
                meta["error_gls_spec"] = f"GLS-spec evaluation failed: {e}"

        # ---------- 其次：DAG（保留兼容；TSP 不建议继续使用） ----------
        if individual.get("dag_spec"):
            dag_spec = individual["dag_spec"]
            meta["dag_plan"] = dag_spec
            try:
                executor = DagExecutor(REGISTRY)
                executor.load_spec(dag_spec)

                initial_ctx = {}
                if individual.get("code"):
                    initial_ctx["code"] = individual.get("code")
                if individual.get("rewrite_patches"):
                    initial_ctx["rewrite_patches"] = individual.get("rewrite_patches")

                executor.run(initial_context=initial_ctx)
                try:
                    meta["dag_trace"] = executor.profile_json()
                except Exception:
                    pass
                try:
                    meta["dag_budgets"] = executor.budgets
                except Exception:
                    pass

                emitted_code = executor.get_emitted_code()
                if emitted_code:
                    meta["emitted_code"] = emitted_code
                    fitness = self.evaluateDAG(emitted_code, executor.budgets)
                    meta["path"] = "dag"
                    return {"fitness": float(fitness), "meta": meta}
            except Exception as e:
                if self.debug_mode:
                    print("--- DAG Evaluation Error (outer) ---")
                    print(e)
                meta["error_dag"] = f"DAG evaluation failed: {e}"

        # ---------- 最后：仅 code → legacy GLS ----------
        if individual.get("code"):
            code_string = individual["code"]
            try:
                heuristic_module = types.ModuleType("heuristic_module")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(self.prelude + code_string, heuristic_module.__dict__)
                fitness, inst_details = self.evaluateGLS_legacy(heuristic_module)
                meta["path"] = "legacy_gls"
                meta["instances"] = inst_details
            except Exception as e:
                meta["error"] = f"Legacy evaluation failed: {e}"
            return {"fitness": float(fitness), "meta": meta}

        meta["error"] = "Individual has neither 'gls_spec' nor 'dag_spec' nor 'code'."
        return {"fitness": float(fitness), "meta": meta}
