import time
import warnings
from joblib import Parallel, delayed
from .evolution import Evolution
import re
import concurrent.futures
import math
import json
import os
import traceback
import signal
import multiprocessing as mp
from datetime import datetime  # NEW: for export timestamp
import hashlib


class InterfaceEC:
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,
                 llm_use_local, llm_local_url, debug_mode, interface_prob,
                 select, n_p, timeout, use_numba, **kwargs):
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        dialogue_path = kwargs.pop("dialogue_path", None)

        # ---- T thresholds / switches（先初始化，方便传给 Evolution）
        self.t_cfg = kwargs.get("t_cfg", {}) or {}
        self.t_cfg.setdefault("alpha", {"t1": 0.20, "t2": 0.10, "t3": 0.05})
        self.t_cfg.setdefault("beta_abs", {"t1": 1.0, "t2": 0.8, "t3": 0.5})
        self.t_cfg.setdefault("gamma_rel", 0.10)
        self.t_cfg.setdefault("gamma_abs", 0.5)
        self.t_cfg.setdefault("Omax", 20.0)
        self.t_cfg.setdefault("bypass_on_fail", True)   # 确保默认 True
        self.t_cfg.setdefault("diag_retry", False)
        self.t_cfg.setdefault("verbose", True)
        # 新增：T 阶段 history 控制
        self.t_cfg.setdefault("t_history_maxlen", 5)
        self.t_cfg.setdefault("t_history_in_prompt", True)


        # ---- Operator naming (DASH) ----
        # Implementation operators are Evolution method names; alias names are used in configs/logging.
        self.op_name_map = {
            "i1": "MDL-Init",
            "e1": "MDL-1",
            "e2": "MDL-2",
            "m1": "MCL-1",
            "m2": "MCL-2",
            "m3": "MCL-3",
            # SSL / T-phase (optional display names)
            "t1_gls_structure": "SSL-*/T1-Structure",
            "t2_gls_param": "SSL-*/T2-Param",
            "t3_gls_module": "SSL-*/T3-Module",
        }

        # alias -> impl (also accept legacy names for backward compatibility)
        self._op_alias_to_impl = {
            # init
            "MDL-Init": "i1",
            "i1": "i1",
            # MDL
            "MDL-1": "e1",
            "MDL-2": "e2",
            "e1": "e1",
            "e2": "e2",
            # MCL
            "MCL-1": "m1",
            "MCL-2": "m2",
            "MCL-3": "m3",
            "m1": "m1",
            "m2": "m2",
            "m3": "m3",
            # SSL / T-phase (keep impl names)
            "t1_gls_structure": "t1_gls_structure",
            "t2_gls_param": "t2_gls_param",
            "t3_gls_module": "t3_gls_module",
        }

        self._impl_i_operators = ["i1"]
        self._impl_mdl_operators = ["e1", "e2"]
        self._impl_mcl_operators = ["m1", "m2", "m3"]

        # User-facing (alias) operator lists
        self.i_operators = ["MDL-Init"]  # used only when falling back to LLM init
        self.e_operators = ["MDL-1", "MDL-2"]
        self.m_operators = ["MCL-1", "MCL-2", "MCL-3"]

        self.evol = Evolution(api_endpoint, api_key, llm_model,
                              llm_use_local, llm_local_url, debug_mode,
                              prompts,
                              dialogue_path=dialogue_path,
                              op_name_map=self.op_name_map,
                              t_history_maxlen=self.t_cfg["t_history_maxlen"],
                              t_history_in_prompt=self.t_cfg["t_history_in_prompt"])
        self.select = select
        self.debug = debug_mode
        self.m = m
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba
        # GLS t（同骨架）
        self.t_operators_gls = ['t1_gls_structure', 't2_gls_param', 't3_gls_module']

        # 记录 T 阶段最近一次 ACCEPT 的个体（用于“bypass”）
        self._last_t_accept = None

        # 输出目录，用于导出最终 accepted GLS solver / global top-k
        self.output_dir = kwargs.get("output_dir", None)
        if self.output_dir:
            # 先创建一个基础目录，后面 _persist_t_accept 还会细分到 results/exports
            os.makedirs(self.output_dir, exist_ok=True)

        # Global top-K archive (for final export across all generations)
        self.topk_archive = []
        # 关键：top-k 个数 = ec_pop_size
        self.topk_size = pop_size


    # ---------------- logging ----------------
    def log_print(self, content, *args, **kwargs):
        print(content, *args, **kwargs)

    # ---------------- baseline 种子构造 ----------------
    def _build_baseline_seed(self):
        """
        构造一个“强 baseline 种子”个体：

        - 使用与 test_gls_engine_perturb_grid 中 3.2_basic_reloc 相同的 BuiltinGLS 语义：
            D' = D + (lam * avg_edge_len) * penalty, lam=0.5
        - 骨架参数对齐你网格中最优配置：
            k=45, random_relocate, loop_max=400, max_no_improve=80, time_limit_s=10
        """

        # 这里不实现 update_edge_distance，只在模块级定义 lam：
        # guided_local_search 里只要发现 guide_algorithm 有 lam 属性，
        # 就会走 BuiltinGLS 分支，忽略 update_edge_distance 本身。
        code = (
            "# Builtin GLS baseline: lam is the alpha scaling factor\n"
            "# guided_local_search will run in builtin GLS mode whenever\n"
            "# the guide_algorithm object has attribute `lam`.\n"
            "lam = 0.5\n"
        )

        algorithm = (
            "BuiltinGLS baseline: lam=0.5, "
            "k=45, perturb=random_relocate, "
            "loop_max=400, max_no_improve=80, time_limit_s=10.0"
        )

        # 用框架统一的构造函数，自动带上 gls_spec 等字段
        ind = self._build_individual(code, algorithm)

        # 再显式把 gls_spec 调到你网格里的 best config 附近
        spec = ind.get("gls_spec") or self._default_gls_spec()

        # init：多起点 = 1
        spec.setdefault("init", {})
        spec["init"].setdefault("method", "nearest_neighbor")
        spec["init"].setdefault("start", 0)
        spec["init"]["multi_start"] = 1

        # candset：k=45（你 3.2_basic_reloc 的 sweet spot）
        spec.setdefault("candset", {})
        spec["candset"]["type"] = "kNN"
        spec["candset"]["k"] = 45

        # schedule：loop_max=400, max_no_improve=80
        spec.setdefault("schedule", {})
        spec["schedule"]["loop_max"] = 400
        spec["schedule"]["max_no_improve"] = 80

        # perturb：random_relocate + moves=1, interval=80
        spec.setdefault("perturb", {})
        spec["perturb"]["type"] = "random_relocate"
        spec["perturb"]["moves"] = 1
        spec["perturb"]["interval"] = 80

        # guidance：标记成 builtin（虽然后端只看 lam 属性，但标一标更直观）
        spec.setdefault("guidance", {})
        spec["guidance"]["where"] = "mid_ls"
        spec["guidance"]["weight"] = 1.0
        spec["guidance"]["top_k"] = 6
        spec["guidance"]["type"] = "builtin"

        # stopping：10 秒（与你 grid 里的 time_limit_s 一致）
        spec.setdefault("stopping", {})
        spec["stopping"]["time_limit_s"] = 10.0

        # engine：ls_basic（当前骨架只实现 basic，引擎字段主要用于未来扩展）
        spec.setdefault("engine", {})
        spec["engine"]["type"] = "ls_basic"

        ind["gls_spec"] = spec
        return ind




    # # ---------------- i阶 ----------------


    # ---------------- i阶 ----------------
    def population_generation(self):
        """
        新策略：i 阶段只保留一个强 baseline。
        - 用 _build_baseline_seed() 构造 baseline；
        - 评估一次 baseline；
        - 把 baseline 克隆成 pop_size 份作为初始种群。
        这样首个 MDL-e 阶段 (e1/e2) 的所有父代都是同一个 baseline。
        """
        self.log_print("creating initial population (baseline-only i-phase):")
        pop = []

        # 1) 构造 baseline 种子
        baseline = None
        try:
            baseline = self._build_baseline_seed()
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to build baseline seed: {e}")

        # 2) 如果 baseline 构造失败，就回退到旧逻辑（用 i1 生成），防止整个流程崩掉
        if baseline is None:
            if self.debug:
                print("[WARN] baseline seed is None, fallback to LLM i1 generation.")
            for _ in range(self.pop_size):
                parents, offsprings = self.get_offspring([], 'i1')
                if offsprings:
                    pop.extend(offsprings)

            self.log_print("initial population has been created! (fallback i1)")
            return pop

        # 3) 评估 baseline（走统一 _evaluate_individual 路径，兼容 GLS spec）
        try:
            res, eval_time = self._evaluate_individual(baseline)
            baseline["objective"] = res.get("fitness", 1e10)
            baseline["other_inf"] = res.get("meta", {})
            baseline["eval_time"] = eval_time
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to evaluate baseline seed: {e}")
            # 评估失败同样回退到旧逻辑
            for _ in range(self.pop_size):
                parents, offsprings = self.get_offspring([], 'i1')
                if offsprings:
                    pop.extend(offsprings)

            self.log_print("initial population has been created! (fallback i1)")
            return pop

        # 4) baseline 评估成功：克隆出 pop_size 份，作为初始种群
        for _ in range(self.pop_size):
            # 做一个浅拷贝，避免共享 same dict 引用（后面写 history 时不会互相干扰）
            ind = {
                "code": baseline.get("code"),
                "algorithm": baseline.get("algorithm"),
                "objective": baseline.get("objective"),
                "other_inf": (baseline.get("other_inf") or {}).copy(),
                "gls_spec": baseline.get("gls_spec"),
                "dag_spec": baseline.get("dag_spec"),
                "eval_time": baseline.get("eval_time"),
            }
            pop.append(ind)

        # 截断到 pop_size（理论上已经刚好等于 pop_size，这里只是防御性代码）
        if len(pop) > self.pop_size:
            pop = pop[: self.pop_size]

        self.log_print("initial population has been created! (baseline only)")
        return pop


    # ---------------- i阶 ----------------

    #     # 1) 先用 i1 正常让 LLM 生成一批个体
    #     for _ in range(self.pop_size):
    #         parents, offsprings = self.get_offspring([], 'i1')
    #         if offsprings:
    #             pop.extend(offsprings)

    #     # 2) 注入 baseline 种子（BuiltinGLS + k=45, random_relocate, tl=10）
    #     baseline = None
    #     try:
    #         baseline = self._build_baseline_seed()
    #     except Exception as e:
    #         if self.debug:
    #             print(f"[WARN] Failed to build baseline seed: {e}")

    #     if baseline is not None:
    #         try:
    #             # 用统一评估流程评估 baseline（会走 gls_spec 路径）
    #             res, eval_time = self._evaluate_individual(baseline)
    #             baseline["objective"] = res.get("fitness", 1e10)
    #             baseline["other_inf"] = res.get("meta", {})
    #             baseline["eval_time"] = eval_time
    #             pop.append(baseline)
    #         except Exception as e:
    #             if self.debug:
    #                 print(f"[WARN] Failed to evaluate baseline seed: {e}")

    #     # 3) 若个体数超过 pop_size，则按 objective 从小到大截断
    #     if len(pop) > self.pop_size:
    #         pop = sorted(pop, key=lambda x: x.get("objective", 1e10))[: self.pop_size]

    #     self.log_print("initial population has been created!")
    #     return pop



    # ---------------- i/e/m & 路由 ----------------
    def get_algorithm(self, pop, operator, **kwargs):
        """兼容 DASH 框架调用，并保留 exp_n_proc 的并行行为。"""
        results = []
        try:
            # 这里的 self.n_p 就是你配置里的 exp_n_proc
            results = Parallel(n_jobs=self.n_p)(
                delayed(self.get_offspring)(pop, operator, **kwargs)
                for _ in range(self.pop_size)
            )
        except Exception as e:
            if self.debug:
                print(f"Parallel generation failed for operator {operator}: {e}")

        out_parents, out_offs = [], []
        for res in results:
            if res and isinstance(res, tuple) and len(res) == 2:
                parents, off = res
                if parents is not None and off is not None:
                    out_parents.append(parents)
                    out_offs.append(off)
        return out_parents, out_offs

    def _update_topk_archive(self, individual, fitness, eval_time, meta):
        """
        维护一个全局 top-K 档案，用于在整个演化结束后导出若干最佳个体。
        按 fitness 从小到大排序，仅保留 self.topk_size 个。
        """
        try:
            fitness = float(fitness)
        except Exception:
            return

        # 过滤掉明显无效的解
        if not (0 <= fitness < 1e9):
            return

        code = individual.get("code", "")
        gls_spec = individual.get("gls_spec")
        dag_spec = individual.get("dag_spec")

        # gls_spec / dag_spec 可能包含不可 JSON 序列化的对象，因此这里兜底
        try:
            gls_key = json.dumps(gls_spec, sort_keys=True, ensure_ascii=False) if gls_spec is not None else None
        except TypeError:
            gls_key = str(gls_spec)

        try:
            dag_key = json.dumps(dag_spec, sort_keys=True, ensure_ascii=False) if dag_spec is not None else None
        except TypeError:
            dag_key = str(dag_spec)

        key = (self._normalize_code_for_topk(code), gls_key, dag_key)

        # 如果已经在档案里，则只在更优时更新
        for entry in getattr(self, "topk_archive", []):
            if entry.get("key") == key:
                if fitness < entry.get("fitness", 1e10):
                    entry["fitness"] = fitness
                    entry["eval_time"] = float(eval_time)
                    entry["meta"] = meta
                    entry["individual"] = individual.copy()
                return

        entry = {
            "key": key,
            "fitness": fitness,
            "eval_time": float(eval_time),
            "meta": meta,
            "individual": individual.copy(),
        }
        if not hasattr(self, "topk_archive"):
            self.topk_archive = []
        self.topk_archive.append(entry)
        self.topk_archive.sort(key=lambda e: e["fitness"])
        if len(self.topk_archive) > getattr(self, "topk_size", 3):
            self.topk_archive = self.topk_archive[: self.topk_size]

    @staticmethod
    def _normalize_code_for_topk(code):
        """用于 top-K 去重的轻量级 code 归一化（去空白）."""
        if not isinstance(code, str):
            return ""
        return re.sub(r"\s+", "", code)

    def export_topk_individuals(self, tag="global_topk"):
        """
        把全局 top-K 个体导出到 results/exports/ 目录。
        每个个体会复用 T 阶段的 _persist_t_accept 导出逻辑，
        区别仅在于 path_tags 和 success_count 的标记。
        """
        if not self.output_dir:
            return
        archive = getattr(self, "topk_archive", None)
        if not archive:
            return

        for rank, entry in enumerate(archive, start=1):
            indiv = (entry.get("individual") or {}).copy()
            attempt = {
                "topk_rank": rank,
                "tag": tag,
            }
            success = f"{tag}_r{rank}"
            path_tags = [tag, f"rank_{rank}"]
            self._persist_t_accept(indiv, attempt, success, path_tags=path_tags)



    def _evaluate_individual(self, individual):
        """
        调用本地问题接口进行评估，带“硬超时”：
        - 只限制 evaluate() 的执行时间，不限制 LLM 生成时间；
        - 一旦超过 self.timeout 秒，立即抛出异常，中断评估，
          返回 fitness = 1e10, meta["timeout"]=True。
        """
        start_time = time.time()
        result = {"fitness": 1e10, "meta": {}}

        # 如果没设 timeout 或者 <=0，就走无超时旧逻辑
        if not self.timeout or self.timeout <= 0:
            try:
                res = self.interface_eval.evaluate(individual)
                if isinstance(res, dict):
                    result["fitness"] = float(res.get("fitness", 1e10))
                    result["meta"] = res.get("meta", {})
                else:
                    result["fitness"] = float(res)
            except Exception as e:
                result["fitness"] = 1e10
                result["meta"] = {"error_eval": str(e)}
            eval_time = time.time() - start_time

            # 更新全局 top-K 档案
            try:
                self._update_topk_archive(
                    individual,
                    result.get("fitness", 1e10),
                    eval_time,
                    result.get("meta", {}),
                )
            except Exception:
                if self.debug:
                    print("[WARN] Failed to update top-k archive.", flush=True)

            return result, eval_time

        # ---------------- 带硬超时的路径 ----------------
        # 使用 signal.SIGALRM 实现硬超时
        def _timeout_handler(signum, frame):
            raise TimeoutError("Evaluation timeout")

        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
        except Exception:
            # 某些环境（例如 Windows）可能不支持 SIGALRM，此时退化为无超时模式
            try:
                res = self.interface_eval.evaluate(individual)
                if isinstance(res, dict):
                    result["fitness"] = float(res.get("fitness", 1e10))
                    result["meta"] = res.get("meta", {})
                else:
                    result["fitness"] = float(res)
            except Exception as e:
                result["fitness"] = 1e10
                result["meta"] = {"error_eval": str(e)}
            eval_time = time.time() - start_time

            try:
                self._update_topk_archive(
                    individual,
                    result.get("fitness", 1e10),
                    eval_time,
                    result.get("meta", {}),
                )
            except Exception:
                if self.debug:
                    print("[WARN] Failed to update top-k archive.", flush=True)

            return result, eval_time

        try:
            # 启动计时器
            signal.setitimer(signal.ITIMER_REAL, float(self.timeout))

            try:
                res = self.interface_eval.evaluate(individual)
                if isinstance(res, dict):
                    result["fitness"] = float(res.get("fitness", 1e10))
                    result["meta"] = res.get("meta", {})
                else:
                    result["fitness"] = float(res)
            except TimeoutError:
                result["fitness"] = 1e10
                result["meta"] = {"timeout": True}
            except Exception as e:
                result["fitness"] = 1e10
                result["meta"] = {"error_eval": str(e)}
        finally:
            # 关闭计时器 & 恢复旧 handler
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

        eval_time = time.time() - start_time

        # 更新全局 top-K 档案（按 fitness 从小到大保留前 self.topk_size 个）
        try:
            self._update_topk_archive(
                individual,
                result.get("fitness", 1e10),
                eval_time,
                result.get("meta", {}),
            )
        except Exception:
            if self.debug:
                print("[WARN] Failed to update top-k archive.", flush=True)

        return result, eval_time


    def _evaluate_with_timeout(self, individual, timeout=None):
        """兼容旧 T-phase 代码的接口。"""
        return self._evaluate_individual(individual)

    @staticmethod
    def _is_same_code(code_a, code_b):
        """粗略判断代码是否相同（去掉空白后比较）。"""
        if not code_a or not code_b:
            return False
        strip_a = re.sub(r'\s+', '', code_a)
        strip_b = re.sub(r'\s+', '', code_b)
        return strip_a == strip_b

    def check_duplicate(self, new_individual, pop):
        """检查新个体是否在当前种群中“等价”."""
        code = new_individual.get('code', '')
        gls_spec = new_individual.get('gls_spec')
        dag_spec = new_individual.get('dag_spec')

        for ind in pop:
            if not ind.get('code'):
                continue
            if not self._is_same_code(code, ind['code']):
                continue
            # code 相同，进一步比较 gls_spec/dag_spec
            same_gls = (gls_spec is None and ind.get('gls_spec') is None) or \
                       (gls_spec is not None and ind.get('gls_spec') is not None and
                        json.dumps(gls_spec, sort_keys=True) ==
                        json.dumps(ind.get('gls_spec'), sort_keys=True))
            same_dag = (dag_spec is None and ind.get('dag_spec') is None) or \
                       (dag_spec is not None and ind.get('dag_spec') is not None and
                        json.dumps(dag_spec, sort_keys=True) ==
                        json.dumps(ind.get('dag_spec'), sort_keys=True))
            if same_gls and same_dag:
                return True
        return False

    # ---------------- i/e/m offspring 生成 ----------------
    def _to_impl_operator(self, operator: str) -> str:
        """Translate user-facing operator name to internal implementation operator."""
        return self._op_alias_to_impl.get(operator, operator)

    # ---------------- i/e/m offspring 生成 ----------------
    def get_offspring(self, pop, operator, **kwargs):
        """Unified entry: dispatch to legacy i/MDL/MCL operators or SSL(T)-operators."""
        impl_op = self._to_impl_operator(operator)

        if impl_op in (self._impl_i_operators + self._impl_mdl_operators + self._impl_mcl_operators):
            return self._get_offspring_legacy(pop, impl_op, operator_alias=operator, **kwargs)

        if impl_op in self.t_operators_gls:
            # T operators keep their implementation names
            return self._get_offspring_gls(pop, impl_op, **kwargs)

        if self.debug:
            print(f"[WARN] Unknown operator: {operator} (impl={impl_op})")
        return [], []

    def _sample_parents(self, pop, impl_operator: str, k=4):
        """Select parents for legacy i/MDL/MCL operator prompts."""
        if impl_operator in self._impl_i_operators:
            return []
        if not pop:
            return []
        if impl_operator in self._impl_mdl_operators:
            k = min(k, len(pop))
            return self.select.parent_selection(pop, k)
        # MCL uses single-parent
        return self.select.parent_selection(pop, 1)

    def _get_llm_alg(self, impl_operator: str, parents):
        """Call Evolution to obtain (code, algorithm) for the given operator."""
        op_func = getattr(self.evol, impl_operator)
        if impl_operator == 'i1':
            code, algorithm = op_func()
        elif impl_operator in self._impl_mdl_operators:
            code, algorithm = op_func(parents)
        else:
            code, algorithm = op_func(parents[0])
        return code, algorithm

    def _build_individual(self, code, algorithm, base=None):
        """从 code + algorithm 构造个体字典，兼顾复用已有字段。"""
        ind = {
            'code': code,
            'algorithm': algorithm,
            'objective': 1e10,
            'other_inf': {},
            'gls_spec': None,
            'dag_spec': None,
            'eval_time': None
        }
        if base:
            # 例如 t 阶段需要继承旧个体的一些元信息
            for k in ['gls_spec', 'dag_spec', 'other_inf']:
                if k in base and base[k] is not None:
                    ind[k] = base[k]
        return ind

    def _get_offspring_legacy(self, pop, operator, operator_alias=None, **kwargs):
        """i/e/m 阶段：沿用原有 GLS 评估路径（无 DAG），并记录 EM LDR。"""
        parents = self._sample_parents(pop, operator)
        if operator != 'i1' and not parents:
            return [], []

        try:
            code, algorithm = self._get_llm_alg(operator, parents)
        except Exception as e:
            if self.debug:
                print(f"[ERROR] LLM generation failed for {operator}: {e}")
            return parents, [{
                'objective': 1e10,
                'other_inf': {'error': 'LLM_generation_failed'}
            }]

        if not code:
            return parents, [{
                'objective': 1e10,
                'other_inf': {'error': 'no_code_extracted'}
            }]

        cleaned = code.strip()
        # 简单检查是否包含至少一个函数定义
        if not re.search(r'^\s*def\s+\w+\s*\(.*\)\s*:', cleaned, flags=re.MULTILINE):
            return parents, [{
                'objective': 1e10,
                'other_inf': {'error': 'no_function_detected'}
            }]

        base = parents[0] if parents else None
        new_ind = self._build_individual(cleaned, algorithm, base=base)

        # 重复个体直接丢弃（objective 置为大数）
        if self.check_duplicate(new_ind, pop):
            if "other_inf" not in new_ind or not isinstance(new_ind["other_inf"], dict):
                new_ind["other_inf"] = {}
            new_ind['other_inf']['duplicate'] = True
            new_ind['objective'] = 1e10
            return parents, [new_ind]

        # 评估
        res, eval_time = self._evaluate_individual(new_ind)
        new_ind['objective'] = res.get('fitness', 1e10)
        new_ind['other_inf'] = res.get('meta', {})
        new_ind['eval_time'] = eval_time

        # 记录 EM LDR（仅在 base/new_ind 都是 dict 时生效）
        try:
            gen_index = kwargs.get("gen_index", None)
            if isinstance(base, dict) and isinstance(new_ind, dict):
                self._record_em_attempt(base, new_ind, operator, gen_index=gen_index)
        except Exception as e:
            if self.debug:
                print(f"[WARN] failed to record EM LDR: {e}")

        return parents, [new_ind]


    # ---------------- GLS T-phase offspring 生成 ----------------
    def _supports_gls_spec(self):
        """当前问题是否支持 GLSSpec 评估（t 阶段）"""
        return hasattr(self.interface_eval, "evaluateGLS_with_spec")


    def _default_gls_spec(self):
        """
        如果 parent 还没有 gls_spec，就用一份合理的默认配置。
        注意要与 eoh_evolution.Evolution._default_gls_spec_dict
        以及 zTSP/gls/spec.py 中的 GLSSpec 默认保持一致。
        """
        return {
            "init": {"method": "nearest_neighbor", "start": 0, "multi_start": 1},
            "candset": {"type": "kNN", "k": 45},
            "operators": [
                {"name": "two_opt", "strategy": "first"},
                {"name": "relocate", "strategy": "first"},
                # 这里先不显式写 or_opt2，底层骨架会处理
            ],
            "schedule": {
                "loop_max": 400,
                "max_no_improve": 80,
            },
            "accept": {"type": "improve_only", "temp0": 0.0},
            "perturb": {
                "type": "random_relocate",
                "moves": 1,
                "interval": 80,
            },
            "guidance": {
                "where": "mid_ls",
                "weight": 1.0,
                "top_k": 6,
                "type": "llm",
            },
            "stopping": {"time_limit_s": 10.0},
            # engine 字段不写也可以，from_json 会用 GLSSpec 默认的 ls_basic
        }


    def _get_offspring_gls(self, pop, operator, **kwargs):
        seed_pool = kwargs.get("seed_pool", None)
        rejection_reason = kwargs.get("rejection_reason", None)
        parents = self._choose_t_parent(pop, seed_pool=seed_pool)
        if not parents:
            return [], []
        parent = parents[0]
        if not parent.get('gls_spec'):
            parent['gls_spec'] = self._default_gls_spec()

        op_func = getattr(self.evol, operator)
        try:
            spec = op_func(parent, rejection_reason=rejection_reason)
        except Exception as e:
            if self.debug:
                print(f"[ERROR] GLS-spec LLM generation failed for {operator}: {e}")
            return parents, []

        if spec is None:
            return parents, []

        new_ind = self._build_individual(parent.get('code', ''), parent.get('algorithm', ''), base=parent)
        new_ind['gls_spec'] = spec
        return parents, [new_ind]

    # ---------------- GLS T-phase 评估与接受策略 ----------------
    def _choose_t_parent(self, pop, seed_pool=None):
        """
        保持与旧版本兼容的父代选择函数（仍用于 _get_offspring_gls）。
        """
        def _is_valid(ind):
            obj = ind.get('objective')
            return isinstance(obj, (int, float)) and obj < 1e9

        pool = [x for x in (seed_pool or []) if isinstance(x, dict) and _is_valid(x)]
        if not pool:
            pool = [x for x in pop if isinstance(x, dict) and _is_valid(x)]
        if not pool:
            return self.select.parent_selection(pop, 1)

        pool = sorted(pool, key=lambda x: x.get('objective', 1e10))
        k = max(1, int(math.ceil(0.2 * len(pool))))
        top = pool[:k]
        with_t = [x for x in top if x.get('eval_time') is not None]
        return [min(with_t, key=lambda x: x['eval_time'])] if with_t else [top[0]]

    def _accept_new(self, obj_c, time_c, best_obj, fastest_time, phase_tag: str, meta=None):
        # 旧的“相对全局 best/fastest” 接受逻辑，保留以兼容历史代码（当前不在新框架中使用）
        if best_obj is None or fastest_time is None:
            return False, "no_baseline"

        if obj_c is None or time_c is None:
            return False, "invalid_metrics"
        if obj_c >= 1e9:
            return False, "invalid_candidate"

        err = (meta or {}).get("error")
        if err:
            if "timeout" in err or "emit" in err or "exception" in err:
                return False, f"error:{err}"

        alpha     = self.t_cfg["alpha"].get(phase_tag, 0.0)     # 速度门槛
        beta_abs  = self.t_cfg["beta_abs"].get(phase_tag, 0.0)  # 速度赢时允许的质量变差
        gamma_rel = self.t_cfg["gamma_rel"]                     # 质量相对门槛
        gamma_abs = self.t_cfg["gamma_abs"]                     # 质量绝对门槛
        Omax      = self.t_cfg["Omax"]                          # 质量上限

        if obj_c > Omax:
            return False, f"quality_over_Omax({Omax})"

        if time_c <= fastest_time * (1.0 - alpha) and obj_c <= best_obj + beta_abs:
            return True, f"speed_win(alpha={alpha})"

        thr = min(best_obj * (1.0 + gamma_rel), best_obj + gamma_abs)
        if obj_c <= thr:
            return True, "quality_win"

        return False, "no_improve"



    def _accept_new_parent(self, obj_c, time_c, obj_p, time_p, phase_tag: str, meta=None):
        """
        新的 T 阶段**局部**接受准则（LDR 版）：
        - 针对“某一步 T 操作（t1/t2/t3）”以及它的 parent 做两两比较；
        - 做全局质量护栏；
        - 若无时间信息，退化为纯质量比较；
        - 有时间时按阶段 (t1/t2/t3) 基于 LDR + 质量/时间约束决策。
        注意：
        - 是否真正把某个候选作为“这一代的最终 parent”，由 run_t_pipeline_for_parent
        的“全局最终决策”（含时间单调约束）来控制，这里只做局部筛选。
        """
        # 基本合法性检查
        if obj_c is None or obj_c >= 1e9:
            return False, "invalid_candidate"

        # 全局质量护栏：过于糟糕的解直接拒绝
        Omax = float(self.t_cfg.get("Omax", 20.0))
        if obj_c > Omax:
            return False, f"quality_over_Omax({Omax})"

        # 若父代缺失，退化为无信息
        if obj_p is None or obj_p >= 1e9:
            return False, "invalid_parent"

        # 拿出通用阈值
        gamma_rel = float(self.t_cfg.get("gamma_rel", 0.10))
        gamma_abs = float(self.t_cfg.get("gamma_abs", 0.5))

        # 若没有时间信息，则退化为纯质量规则
        if time_p is None or time_c is None or time_p <= 0.0 or time_c <= 0.0:
            thr = min(obj_p * (1.0 + gamma_rel), obj_p + gamma_abs)
            if obj_c <= thr:
                return True, "quality_win(no_time)"
            return False, "no_improve(no_time)"

        # 有时间和 gap，计算 LDR 指标
        ldr = self._compute_ldr_fields(obj_p, obj_c, time_p, time_c)
        iLDR = float(ldr.get("ldr_i", 0.0))
        tLDR_delta = float(ldr.get("ldr_t", 0.0))

        # trajectory-aware tLDR(T) (paper definition) from task meta
        tLDR_traj = None
        if isinstance(meta, dict):
            tLDR_traj = meta.get("tLDR_traj_mean", None)
            if tLDR_traj is None:
                tLDR_traj = meta.get("tLDR_traj", None)

        if tLDR_traj is None:
            tLDR = tLDR_delta
            tLDR_source = "delta_fallback"
        else:
            tLDR = float(tLDR_traj)
            tLDR_source = "traj"


        # 速度提升（正数表示更快）
        speed_gain = (time_p - time_c) / max(time_p, 1e-9)
        rel_regret = (obj_c - obj_p) / max(obj_p, 1e-9)

        # 质量安全阈
        thr_quality = min(obj_p * (1.0 + gamma_rel), obj_p + gamma_abs)

        # --------- t1: 结构级（质量 + 动力学折中） ---------
        if phase_tag == "t1":
            # 质量直观提升，直接接受
            if obj_c <= thr_quality:
                return True, f"quality_win(iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

            # 允许一定 regret 换动力学提升
            max_rel_regret = float(self.t_cfg.get("t1_max_rel_regret", gamma_rel))
            min_score = float(self.t_cfg.get("t1_min_ldr_score", 0.0))
            alpha = float(self.t_cfg.get("t1_ldr_alpha", 1.0))   # tLDR 权重
            beta = float(self.t_cfg.get("t1_ldr_beta", 1.0))     # iLDR 权重
            score = alpha * tLDR + beta * iLDR

            if rel_regret <= max_rel_regret and score >= min_score:
                return True, f"ldr_win(score={score:.3f}, iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

            return False, "no_improve"

        # --------- t2: 时间 shrinker（效率优先） ---------
        if phase_tag == "t2":
            # 若 gap 明显变好，也可以接受
            if obj_c <= thr_quality:
                return True, f"quality_win(iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

            # 否则要求：时间明显缩短 + tLDR > 0 + 质量退化有限
            min_speed_gain = float(self.t_cfg.get("t2_min_speed_gain", 0.05))   # 至少 5% 提速
            max_rel_regret = float(self.t_cfg.get("t2_max_rel_regret", 0.02))   # gap 最多坏 2%

            if speed_gain >= min_speed_gain and tLDR > 0.0 and rel_regret <= max_rel_regret:
                return True, (
                    f"speed_shrink(Δt={speed_gain:.3f}, rel_regret={rel_regret:.3f}, "
                    f"iLDR={iLDR:.3f}, tLDR={tLDR:.3f})"
                )

            return False, "no_improve"

        # --------- t3: 质量抛光（少量时间换较大质量提升） ---------
        if phase_tag == "t3":
            # 要求：gap 有明显改善
            min_abs_improve = float(self.t_cfg.get("t3_min_abs_improve", 0.0))
            max_time_factor = float(self.t_cfg.get("t3_max_time_factor", 1.5))

            if obj_c >= obj_p - min_abs_improve:
                return False, "no_quality_gain"

            # 时间不能爆炸增长
            if time_c > time_p * max_time_factor:
                return False, "too_slow"

            # iLDR 为正，本身就意味着 log-gap 在下降
            if iLDR > 0.0:
                return True, f"polish_win(iLDR={iLDR:.3f}, tLDR={tLDR:.3f})"

            return False, "no_improve"

        # 其它未知阶段，保守拒绝
        return False, f"unknown_phase({phase_tag})"




    #     # 全局质量护栏：过于糟糕的解直接拒绝
    #     Omax = float(self.t_cfg.get("Omax", 20.0))
    #     if obj_c > Omax:
    #         return False, f"quality_over_Omax({Omax})"

    #     # 若父代缺失，退化为无信息
    #     if obj_p is None or obj_p >= 1e9:
    #         return False, "invalid_parent"

    #     # 拿出通用阈值
    #     gamma_rel = float(self.t_cfg.get("gamma_rel", 0.10))
    #     gamma_abs = float(self.t_cfg.get("gamma_abs", 0.5))

    #     # 若没有时间信息，则退化为纯质量规则
    #     if time_p is None or time_c is None or time_p <= 0.0 or time_c <= 0.0:
    #         thr = min(obj_p * (1.0 + gamma_rel), obj_p + gamma_abs)
    #         if obj_c <= thr:
    #             return True, "quality_win(no_time)"
    #         return False, "no_improve(no_time)"

    #     # 有时间和 gap，计算 LDR 指标
    #     ldr = self._compute_ldr_fields(obj_p, obj_c, time_p, time_c)
    #     iLDR = ldr.get("ldr_i", 0.0)
    #     tLDR = ldr.get("ldr_t", 0.0)

    #     # 速度提升（正数表示更快）
    #     speed_gain = (time_p - time_c) / max(time_p, 1e-9)
    #     rel_regret = (obj_c - obj_p) / max(obj_p, 1e-9)

    #     # 质量安全阈
    #     thr_quality = min(obj_p * (1.0 + gamma_rel), obj_p + gamma_abs)

    #     # --------- t1: 结构级（质量 + 动力学折中） ---------
    #     if phase_tag == "t1":
    #         # 质量直观提升，直接接受
    #         if obj_c <= thr_quality:
    #             return True, f"quality_win(iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

    #         # 允许一定 regret 换动力学提升
    #         max_rel_regret = float(self.t_cfg.get("t1_max_rel_regret", gamma_rel))
    #         min_score = float(self.t_cfg.get("t1_min_ldr_score", 0.0))
    #         alpha = float(self.t_cfg.get("t1_ldr_alpha", 1.0))   # tLDR 权重
    #         beta = float(self.t_cfg.get("t1_ldr_beta", 1.0))     # iLDR 权重
    #         score = alpha * tLDR + beta * iLDR

    #         if rel_regret <= max_rel_regret and score >= min_score:
    #             return True, f"ldr_win(score={score:.3f}, iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

    #         return False, "no_improve"

    #     # --------- t2: 时间 shrinker（效率优先） ---------
    #     if phase_tag == "t2":
    #         # 若 gap 明显变好，也可以接受
    #         if obj_c <= thr_quality:
    #             return True, f"quality_win(iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

    #         # 否则要求：时间明显缩短 + tLDR > 0 + 质量退化有限
    #         min_speed_gain = float(self.t_cfg.get("t2_min_speed_gain", 0.05))   # 至少 5% 提速
    #         max_rel_regret = float(self.t_cfg.get("t2_max_rel_regret", 0.02))   # gap 最多坏 2%

    #         if speed_gain >= min_speed_gain and tLDR > 0.0 and rel_regret <= max_rel_regret:
    #             return True, (
    #                 f"speed_shrink(Δt={speed_gain:.3f}, rel_regret={rel_regret:.3f}, "
    #                 f"iLDR={iLDR:.3f}, tLDR={tLDR:.3f})"
    #             )

    #         return False, "no_improve"

    #     # --------- t3: 质量抛光（少量时间换较大质量提升） ---------
    #     if phase_tag == "t3":
    #         # 要求：gap 有明显改善
    #         min_abs_improve = float(self.t_cfg.get("t3_min_abs_improve", 0.0))
    #         max_time_factor = float(self.t_cfg.get("t3_max_time_factor", 1.5))

    #         if obj_c >= obj_p - min_abs_improve:
    #             return False, "no_quality_gain"

    #         # 时间不能爆炸增长
    #         if time_c > time_p * max_time_factor:
    #             return False, "too_slow"

    #         # iLDR 为正，本身就意味着 log-gap 在下降
    #         if iLDR > 0.0:
    #             return True, f"polish_win(iLDR={iLDR:.3f}, tLDR={tLDR:.3f})"

    #         return False, "no_improve"

    #     # 其它未知阶段，保守拒绝
    #     return False, f"unknown_phase({phase_tag})"



    #     # 全局质量护栏：过于糟糕的解直接拒绝
    #     Omax = self.t_cfg["Omax"]
    #     if obj_c > Omax:
    #         return False, f"quality_over_Omax({Omax})"

    #     alpha     = self.t_cfg["alpha"].get(phase_tag, 0.0)
    #     beta_abs  = self.t_cfg["beta_abs"].get(phase_tag, 0.0)
    #     gamma_rel = self.t_cfg["gamma_rel"]
    #     gamma_abs = self.t_cfg["gamma_abs"]

    #     # 若父代 / 候选没有时间信息，则退化为纯质量比较
    #     if time_p is None or time_c is None or time_p <= 0:
    #         thr = min(obj_p * (1.0 + gamma_rel), obj_p + gamma_abs)
    #         if obj_c <= thr:
    #             return True, "quality_win(no_time)"
    #         return False, "no_improve(no_time)"

    #     # 情形 1：速度赢（在允许的质量损失范围内）
    #     speed_gain = (time_p - time_c) / max(time_p, 1e-9)
    #     if speed_gain >= alpha and obj_c <= obj_p + beta_abs:
    #         return True, f"speed_win(Δt={speed_gain:.3f}, Δobj={obj_c - obj_p:.3f})"

    #     # 情形 2：质量赢（允许时间略微劣化）
    #     thr = min(obj_p * (1.0 + gamma_rel), obj_p + gamma_abs)
    #     if obj_c <= thr:
    #         return True, "quality_win"

    #     return False, "no_improve"

    def _compute_ldr_fields(self, obj_parent, obj_child, time_parent, time_child):
        """
        计算 Lyapunov 风格的 LDR 指标：
          - V = max(gap, delta)
          - iLDR = log(V_parent) - log(V_child)
          - tLDR = iLDR / time_child
        返回一个 dict，包含:
          V_parent, V_child, ldr_i, ldr_t
        若信息不足或数值不合法，返回 0.
        """
        delta = float(self.t_cfg.get("ldr_delta", 1e-6))
        eps_t = float(self.t_cfg.get("ldr_eps_time", 1e-9))

        try:
            if obj_parent is None or obj_child is None:
                return {
                    "V_parent": None,
                    "V_child": None,
                    "ldr_i": 0.0,
                    "ldr_t": 0.0,
                }

            V_p = max(float(obj_parent), delta)
            V_c = max(float(obj_child), delta)
            if V_p <= 0.0 or V_c <= 0.0:
                return {
                    "V_parent": V_p,
                    "V_child": V_c,
                    "ldr_i": 0.0,
                    "ldr_t": 0.0,
                }

            logVp = math.log(V_p)
            logVc = math.log(V_c)
            ldr_i = logVp - logVc  # iLDR

            if time_child is None or time_child <= 0.0:
                ldr_t = 0.0
            else:
                ldr_t = ldr_i / max(float(time_child), eps_t)

            return {
                "V_parent": V_p,
                "V_child": V_c,
                "ldr_i": ldr_i,
                "ldr_t": ldr_t,
            }
        except Exception as e:
            if getattr(self, "debug", False):
                print(f"[WARN] _compute_ldr_fields failed: {e}")
            return {
                "V_parent": None,
                "V_child": None,
                "ldr_i": 0.0,
                "ldr_t": 0.0,
            }



    def _ensure_t_history(self, indiv):
        """
        确保 indiv['other_inf']['t_history'] 存在，并返回这个 list。
        """
        if indiv is None:
            return []
        other = indiv.get("other_inf")
        if other is None:
            other = {}
            indiv["other_inf"] = other
        hist = other.get("t_history")
        if hist is None:
            hist = []
            other["t_history"] = hist
        return hist

    def _compute_ldr_fields_legacy_order(self, obj_parent, time_parent, obj_child, time_child):
        """Backward-compatible wrapper for an older argument order.

        Old order: (obj_parent, time_parent, obj_child, time_child)
        Canonical order: (obj_parent, obj_child, time_parent, time_child)
        """
        return self._compute_ldr_fields(obj_parent, obj_child, time_parent, time_child)


    def _ensure_em_history(self, indiv):
        """
        确保 indiv['other_inf']['em_history'] 存在，并返回这个 list。
        em_history 用来记录 e/m 阶段的局部编辑（带 LDR）。
        """
        if indiv is None:
            return []
        other = indiv.get("other_inf")
        if other is None or not isinstance(other, dict):
            other = {}
            indiv["other_inf"] = other
        hist = other.get("em_history")
        if hist is None:
            hist = []
            other["em_history"] = hist
        return hist

    def _record_em_attempt(self, parent_indiv, child_indiv, operator, gen_index=None):
        """
        在 parent_indiv.other_inf['em_history'] 里追加一条记录：
        - gen：发生在哪一代（来自 DASH.run 传下来的 gen_index）
        - operator：使用的是哪一个 i/e/m 算子
        - obj/time：父子 objective 与 eval_time
        - V_parent/V_child/iLDR/tLDR：由 _compute_ldr_fields 计算
        """
        try:
            # 这里只在 parent/child 都是 dict 的情况下记录，避免和上层 Individual 包装冲突
            if not isinstance(parent_indiv, dict) or not isinstance(child_indiv, dict):
                return

            hist = self._ensure_em_history(parent_indiv)

            obj_p = parent_indiv.get("objective")
            t_p = parent_indiv.get("eval_time")
            obj_c = child_indiv.get("objective")
            t_c = child_indiv.get("eval_time")

            ldr = self._compute_ldr_fields(obj_p, obj_c, t_p, t_c)

            rec = {
                "gen": int(gen_index) if gen_index is not None else None,
                "operator": str(operator),
                "obj_parent": float(obj_p) if obj_p is not None else None,
                "time_parent": float(t_p) if t_p is not None else None,
                "obj_child": float(obj_c) if obj_c is not None else None,
                "time_child": float(t_c) if t_c is not None else None,
            }
            rec.update(ldr)

            hist.append(rec)
            maxlen = int(self.t_cfg.get("em_history_maxlen", 20) or 20)
            if len(hist) > maxlen:
                del hist[:-maxlen]
        except Exception as e:
            if self.debug:
                print(f"[WARN] failed to record EM history: {e}")

    def _record_t_attempt(self, parent_indiv, stage, gen_index,
                          obj_parent, time_parent, obj_child, time_child,
                          accepted, reason, meta_child=None):
        """
        在 parent_indiv.other_inf['t_history'] 里追加一条记录，并截断到 maxlen。
        现在会额外写入：
          - V_parent / V_child
          - ldr_i / ldr_t
          - ldr_score = ldr_alpha * ldr_t + ldr_beta * ldr_i
        """
        try:
            hist = self._ensure_t_history(parent_indiv)

            # 先填入基础信息
            rec = {
                "gen": int(gen_index) if gen_index is not None else None,
                "stage": stage,
                "accepted": bool(accepted),
                "reason": str(reason),
                "obj_parent": float(obj_parent) if obj_parent is not None else None,
                "time_parent": float(time_parent) if time_parent is not None else None,
                "obj_child": float(obj_child) if obj_child is not None else None,
                "time_child": float(time_child) if time_child is not None else None,
            }

            # 计算 LDR 指标
            ldr_fields = self._compute_ldr_fields(
                rec["obj_parent"],
                rec["obj_child"],
                rec["time_parent"],
                rec["time_child"],
            )
            rec.update(ldr_fields)

            # trajectory-aware tLDR(T) (paper) if provided
            tldr_traj = None
            if isinstance(meta_child, dict):
                tldr_traj = meta_child.get("tLDR_traj_mean", None)
                if tldr_traj is None:
                    tldr_traj = meta_child.get("tLDR_traj", None)
            rec["tLDR_traj"] = float(tldr_traj) if tldr_traj is not None else None

            # 统一的 LDR score（方便后续分析 / 策略）
            ldr_alpha = float(self.t_cfg.get("ldr_alpha", 0.5))
            ldr_beta = float(self.t_cfg.get("ldr_beta", 0.5))
            rec["ldr_score"] = (
                ldr_alpha * (rec.get("tLDR_traj") if rec.get("tLDR_traj") is not None else rec.get("ldr_t", 0.0))
                + ldr_beta * rec.get("ldr_i", 0.0)
            )

            hist.append(rec)
            maxlen = int(self.t_cfg.get("t_history_maxlen", 5) or 5)
            if len(hist) > maxlen:
                del hist[:-maxlen]
        except Exception as e:
            if self.debug:
                print(f"[WARN] failed to record T history: {e}")



    #         # 先计算 LDR 相关字段
    #         ldr = self._compute_ldr_fields(obj_parent, time_parent, obj_child, time_child)

    #         rec = {
    #             "gen": int(gen_index) if gen_index is not None else None,
    #             "stage": stage,
    #             "accepted": bool(accepted),
    #             "reason": str(reason),
    #             "obj_parent": float(obj_parent) if obj_parent is not None else None,
    #             "time_parent": float(time_parent) if time_parent is not None else None,
    #             "obj_child": float(obj_child) if obj_child is not None else None,
    #             "time_child": float(time_child) if time_child is not None else None,
    #         }
    #         rec.update(ldr)

    #         hist.append(rec)
    #         maxlen = int(self.t_cfg.get("t_history_maxlen", 5) or 5)
    #         if len(hist) > maxlen:
    #             del hist[:-maxlen]
    #     except Exception as e:
    #         if self.debug:
    #             print(f"[WARN] failed to record T history: {e}")

    def _persist_t_accept(self, indiv, attempt_count, success_count, path_tags=None):
        """
        当 T 阶段某个方案被 ACCEPT 时，把它的 code + gls_spec 导出为一个独立的 solver.py（GLS 版本）。
        现在同时把评估得到的 meta（other_inf，包括 per-instance 结果）一并写入 JSON，
        并增加 code_hash / gls_spec_hash 方便后续聚合分析。
        """
        try:
            if not self.output_dir:
                return

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 文件名中加上一点可读的 tag 信息，方便区分来源
            tag_suffix = ""
            if path_tags:
                safe_tags = [str(t).replace(" ", "_") for t in path_tags]
                tag_suffix = "_" + "_".join(safe_tags)

            fname = f"{ts}_seq{success_count}{tag_suffix}_accepted.py"
            fjson = f"{ts}_seq{success_count}{tag_suffix}_accepted.json"

            out_py = os.path.join(self.output_dir, "results", "exports", fname)
            out_js = os.path.join(self.output_dir, "results", "exports", fjson)
            os.makedirs(os.path.dirname(out_py), exist_ok=True)

            # 从个体中拿到评估元信息（在 _evaluate_individual / T-phase 中已经写入 other_inf）
            meta = indiv.get("other_inf") or {}
            if not isinstance(meta, dict):
                meta = {"raw_other_inf": meta}

            # 计算 code / spec 的哈希，便于后续聚合
            code = indiv.get("code") or ""
            gls_spec_obj = indiv.get("gls_spec")

            code_hash = None
            gls_spec_hash = None
            try:
                if code:
                    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
                if gls_spec_obj is not None:
                    spec_json = json.dumps(gls_spec_obj, sort_keys=True, ensure_ascii=False)
                    gls_spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
            except Exception:
                # 哈希失败不影响主流程
                code_hash = None
                gls_spec_hash = None

            payload = {
                "ts": ts,
                "path_tags": list(path_tags or []),
                "attempt_count": attempt_count,
                "success_count": success_count,
                "objective": indiv.get("objective"),
                "eval_time": indiv.get("eval_time"),
                "gls_spec": gls_spec_obj,
                "dag_spec": indiv.get("dag_spec"),
                "code_hash": code_hash,
                "gls_spec_hash": gls_spec_hash,
                "meta": meta,
            }

            with open(out_js, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            # 代码文件仍然通过 _write_export_py 生成
            self._write_export_py(out_py, indiv, payload)

            if self.debug:
                print(f"[export] saved: {out_js} & {out_py}")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to export accepted T-solution: {e}")



    def _write_export_py(self, path, individual, meta):
        """
        写出一个可直接 import 使用的 solver 文件（GLS 版本）。
        """
        code = individual.get("code", "")
        gls_spec = individual.get("gls_spec")
        dag_spec = individual.get("dag_spec")
        spec_json = None
        try:
            spec_json = json.dumps(gls_spec if gls_spec is not None else dag_spec, indent=2)
        except Exception:
            spec_json = "null"

        if gls_spec is not None:
            # ==== GLS 导出 ====
            content = f"""# Auto-generated by DASH — accepted GLS solver (framework + evolved heuristic)

import json
import numpy as _np
from gls.spec import GLSSpec, from_json
from gls.gls_run import solve_instance_with_spec as _solve_with_spec

# ---- Evolved heuristic (update_edge_distance) ----
{code}

# ---- GLSSpec recovered from T-phase ----
GLS_SPEC_JSON = {spec_json}

def get_gls_spec() -> GLSSpec:
    return from_json(json.loads(json.dumps(GLS_SPEC_JSON)))

def build_heuristic_module():
    import types, warnings
    mod = types.ModuleType("heuristic_module")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile({code!r}, "<heuristic_module>", "exec"), mod.__dict__)
    return mod

def solve_instance_export(n, opt_cost, dis_matrix, coords,
                          time_limit_s=10.0, ite_max=1000, perturbation_moves=1):
    spec = get_gls_spec()
    heuristic = build_heuristic_module()
    gap = _solve_with_spec(n, opt_cost, dis_matrix, coords,
                           time_limit_s, ite_max, perturbation_moves,
                           heuristic, spec)
    return gap

def evaluate_dataset(coords_list, instances_list, opt_costs_list,
                     time_limit_s=10.0, ite_max=1000, perturbation_moves=1):
    gaps = []
    for coords, inst, opt in zip(coords_list, instances_list, opt_costs_list):
        gaps.append(
            solve_instance_export(len(inst), opt, inst, coords,
                                  time_limit_s=time_limit_s,
                                  ite_max=ite_max,
                                  perturbation_moves=perturbation_moves)
        )
    return float(_np.mean(_np.array(gaps, dtype=_np.float64)))
"""
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    def run_t_pipeline_gls(self, pop, seed_pool=None, best_obj=None, fastest_time=None):
        """
        保留旧接口以兼容历史调用；当前实现会在内部选择一个父代并调用 run_t_pipeline_for_parent。
        """
        parents = self._choose_t_parent(pop, seed_pool=seed_pool)
        if not parents:
            return {
                "accepted": False,
                "reason": "no_parent",
                "individual": None,
                "path": [],
                "attempt_count": {},
                "success_count": {},
                "stage": None,
            }
        parent = parents[0]
        pop_objs = [p.get('objective') for p in pop if p.get('objective', 1e10) < 1e9]
        return self.run_t_pipeline_for_parent(parent=parent, pop_objs=pop_objs, gen_index=None)


    def run_t_pipeline_for_parent(self, parent, pop_objs=None, gen_index=None, ssl_stage=None, dsl_stage=None):
        """
        在给定父代上执行 GLS T-phase（t1→t2→t3 + bypass），
        接受准则相对该父代（及级联的中间候选），返回单个“最佳时间”的子代。

        关键修改：
        - 仍然允许 t1/t2/t3 各自用 LDR + 质量/时间规则做局部接受；
        - 但在“最终决策”阶段，增加一个全局时间约束：
            * 只在 eval_time 不超过 parent_time * t_final_max_time_factor 的候选中选
            * 若无任何候选满足，则视为本次 T 失败，保留父代（不更新 spec）
        """
        gen_idx = gen_index
        stage_tag = ssl_stage or dsl_stage
        dsl_tag = f"[{stage_tag}]" if stage_tag else ""

        def _print_stage(tag, cand, ok, reason):
            try:
                spec = cand.get('gls_spec') or {}
                st = (spec.get('stopping') or {}).get('time_limit_s')
                cs = (spec.get('candset') or {}).get('k')
                sch = (spec.get('schedule') or {})
                tk = (spec.get('guidance') or {}).get('top_k')
                msg = (
                    f"  >> {dsl_tag} {tag} result: "
                    f"Obj={cand.get('objective', 1e10):.6f}, "
                    f"t={cand.get('eval_time', 0.0):.3f}s | "
                    f"accept={ok} ({reason})"
                )
                msg += (
                    f" | spec: time_limit_s={st}, k={cs}, "
                    f"loop_max={sch.get('loop_max')}, "
                    f"max_no_improve={sch.get('max_no_improve')}, "
                    f"top_k={tk}"
                )
                err = (cand.get('other_inf') or {}).get('error')
                if err:
                    msg += f" | ERROR={err}"
                print(msg)
            except Exception:
                print(f"  >> {dsl_tag} {tag} result: accept={ok} ({reason})")

        if parent is None:
            return {
                "accepted": False,
                "reason": "no_parent",
                "individual": None,
                "path": [],
                "attempt_count": {},
                "success_count": {},
                "stage": None,
            }

        if not parent.get('gls_spec'):
            parent['gls_spec'] = self._default_gls_spec()

        parent_obj = parent.get('objective', 1e10)
        parent_time = parent.get('eval_time', None)

        attempt_count = {
            "t1_gls_structure": 0,
            "t2_gls_param": 0,
            "t3_gls_module": 0,
        }
        success_count = {
            "t1_gls_structure": 0,
            "t2_gls_param": 0,
            "t3_gls_module": 0,
        }
        path = []
        accepted_candidates = []

        # ---------- T1: 结构级别的 spec 调整 ----------
        if self.t_cfg.get("verbose", True):
            print(f"=== {dsl_tag} T1(GLS): Structure Swap ===")
        attempt_count["t1_gls_structure"] += 1

        # 注意：这里不要再传 pop_objs，Evolution.t1_gls_structure 只接受 (parent_indiv, rejection_reason)
        spec1 = self.evol.t1_gls_structure(parent, rejection_reason=None)

        if spec1:
            cand1 = {
                "algorithm": (parent.get("algorithm", "") + " [t1_gls_structure]"),
                "code": parent.get("code"),
                "gls_spec": spec1,
                "stage": "t1",
            }
            t0 = time.time()
            res1, _ = self._evaluate_with_timeout(cand1, self.timeout)
            t1_time = time.time()
            cand1["eval_time"] = t1_time - t0
            cand1["objective"] = res1.get("fitness", 1e10)

            parent_meta = dict(parent.get("other_inf") or {})
            meta1 = res1.get("meta", {}) or {}
            cand1["meta"] = meta1
            parent_meta.update(meta1)
            cand1["other_inf"] = parent_meta

            path.append("t1_gls")
            ok1, reason1 = self._accept_new_parent(
                cand1["objective"],
                cand1["eval_time"],
                parent_obj,
                parent_time,
                "t1",
                meta=cand1.get("meta"),
            )
            _print_stage("t1_gls", cand1, ok1, reason1)

            self._record_t_attempt(
                parent_indiv=parent,
                stage="t1",
                gen_index=gen_idx,
                obj_parent=parent_obj,
                time_parent=parent_time,
                obj_child=cand1["objective"],
                time_child=cand1["eval_time"],
                accepted=ok1,
                reason=reason1,
                meta_child=cand1.get("meta"),
            )
        else:
            cand1 = None
            ok1, reason1 = False, "t1_no_spec"

        current = None
        if ok1 and cand1 is not None:
            success_count["t1_gls_structure"] += 1
            current = cand1
            accepted_candidates.append(cand1)

        # ---------- 主路径：在 T1 基础上尝试 T2/T3 ----------
        if current is not None:
            # T2 正常
            if self.t_cfg.get("verbose", True):
                print(f"=== {dsl_tag} T2(GLS): Param Tune ===")
            attempt_count["t2_gls_param"] += 1
            spec2 = self.evol.t2_gls_param(current, rejection_reason=None)
            if spec2:
                cand2 = {
                    "algorithm": (current.get("algorithm", "") + " [t2_gls_param]"),
                    "code": current.get("code"),
                    "gls_spec": spec2,
                    "stage": "t2",
                }
                t0 = time.time()
                res2, _ = self._evaluate_with_timeout(cand2, self.timeout)
                t2_time = time.time()
                cand2["eval_time"] = t2_time - t0
                cand2["objective"] = res2.get("fitness", 1e10)

                parent_meta = dict(current.get("other_inf") or {})
                meta2 = res2.get("meta", {}) or {}
                cand2["meta"] = meta2
                parent_meta.update(meta2)
                cand2["other_inf"] = parent_meta

                path.append("t2_gls")
                ok2, reason2 = self._accept_new_parent(
                    cand2["objective"],
                    cand2["eval_time"],
                    current["objective"],
                    current.get("eval_time", None),
                    "t2",
                    meta=cand2.get("meta"),
                )
                _print_stage("t2_gls", cand2, ok2, reason2)

                self._record_t_attempt(
                    parent_indiv=current,
                    stage="t2",
                    gen_index=gen_idx,
                    obj_parent=current["objective"],
                    time_parent=current.get("eval_time", None),
                    obj_child=cand2["objective"],
                    time_child=cand2["eval_time"],
                    accepted=ok2,
                    reason=reason2,
                    meta_child=cand2.get("meta"),
                )

                if ok2:
                    success_count["t2_gls_param"] += 1
                    current = cand2
                    accepted_candidates.append(cand2)

            # T3 正常
            if self.t_cfg.get("verbose", True):
                print(f"=== {dsl_tag} T3(GLS): Single-Module Tweak ===")
            attempt_count["t3_gls_module"] += 1
            spec3 = self.evol.t3_gls_module(current, rejection_reason=None)
            if spec3:
                cand3 = {
                    "algorithm": (current.get("algorithm", "") + " [t3_gls_module]"),
                    "code": current.get("code"),
                    "gls_spec": spec3,
                    "stage": "t3",
                }
                t0 = time.time()
                res3, _ = self._evaluate_with_timeout(cand3, self.timeout)
                t3_time = time.time()
                cand3["eval_time"] = t3_time - t0
                cand3["objective"] = res3.get("fitness", 1e10)

                parent_meta = dict(current.get("other_inf") or {})
                meta3 = res3.get("meta", {}) or {}
                cand3["meta"] = meta3
                parent_meta.update(meta3)
                cand3["other_inf"] = parent_meta

                path.append("t3_gls")
                ok3, reason3 = self._accept_new_parent(
                    cand3["objective"],
                    cand3["eval_time"],
                    current["objective"],
                    current.get("eval_time", None),
                    "t3",
                    meta=cand3.get("meta"),
                )
                _print_stage("t3_gls", cand3, ok3, reason3)

                self._record_t_attempt(
                    parent_indiv=current,
                    stage="t3",
                    gen_index=gen_idx,
                    obj_parent=current["objective"],
                    time_parent=current.get("eval_time"),
                    obj_child=cand3["objective"],
                    time_child=cand3["eval_time"],
                    accepted=ok3,
                    reason=reason3,
                    meta_child=cand3.get("meta"),
                )

                if ok3:
                    success_count["t3_gls_module"] += 1
                    accepted_candidates.append(cand3)

        # ---------- bypass 路径：T1 不通过 ----------
        if (not ok1) and self.t_cfg.get("bypass_on_fail", True):
            # T2 bypass
            if self.t_cfg.get("verbose", True):
                print(f"=== {dsl_tag} T2(GLS): Param Tune (bypass) ===")
            attempt_count["t2_gls_param"] += 1
            spec2b = self.evol.t2_gls_param(parent, rejection_reason=reason1)
            if spec2b:
                cand2b = {
                    "algorithm": (parent.get("algorithm", "") + " [t2_gls_param_bypass]"),
                    "code": parent.get("code"),
                    "gls_spec": spec2b,
                    "stage": "t2_bypass",
                }
                t0 = time.time()
                res2b, _ = self._evaluate_with_timeout(cand2b, self.timeout)
                t2b_time = time.time()
                cand2b["eval_time"] = t2b_time - t0
                cand2b["objective"] = res2b.get("fitness", 1e10)

                parent_meta = dict(parent.get("other_inf") or {})
                meta2b = res2b.get("meta", {}) or {}
                parent_meta.update(meta2b)
                cand2b["other_inf"] = parent_meta

                path.append("t2_gls(bypass)")
                ok2b, reason2b = self._accept_new_parent(
                    cand2b["objective"],
                    cand2b["eval_time"],
                    parent_obj,
                    parent_time,
                    "t2",
                    meta=cand2b.get("other_inf"),
                )
                _print_stage("t2_gls(bypass)", cand2b, ok2b, reason2b)

                self._record_t_attempt(
                    parent_indiv=parent,
                    stage="t2_bypass",
                    gen_index=gen_idx,
                    obj_parent=parent_obj,
                    time_parent=parent_time,
                    obj_child=cand2b["objective"],
                    time_child=cand2b["eval_time"],
                    accepted=ok2b,
                    reason=reason2b,
                )

                if ok2b:
                    success_count["t2_gls_param"] += 1
                    accepted_candidates.append(cand2b)

                # T3 bypass
                if self.t_cfg.get("verbose", True):
                    print(f"=== {dsl_tag} T3(GLS): Single-Module (bypass) ===")
                attempt_count["t3_gls_module"] += 1
                spec3b = self.evol.t3_gls_module(parent, rejection_reason=reason2b)
                if spec3b:
                    cand3b = {
                        "algorithm": (parent.get("algorithm", "") + " [t3_gls_module_bypass]"),
                        "code": parent.get("code"),
                        "gls_spec": spec3b,
                        "stage": "t3_bypass",
                    }
                    t0 = time.time()
                    res3b, _ = self._evaluate_with_timeout(cand3b, self.timeout)
                    t3b_time = time.time()
                    cand3b["eval_time"] = t3b_time - t0
                    cand3b["objective"] = res3b.get("fitness", 1e10)

                    parent_meta = dict(parent.get("other_inf") or {})
                    meta3b = res3b.get("meta", {}) or {}
                    parent_meta.update(meta3b)
                    cand3b["other_inf"] = parent_meta

                    path.append("t3_gls(bypass)")
                    ok3b, reason3b = self._accept_new_parent(
                        cand3b["objective"],
                        cand3b["eval_time"],
                        parent_obj,
                        parent_time,
                        "t3",
                        meta=cand3b.get("other_inf"),
                    )
                    _print_stage("t3_gls(bypass)", cand3b, ok3b, reason3b)

                    self._record_t_attempt(
                        parent_indiv=parent,
                        stage="t3_bypass",
                        gen_index=gen_idx,
                        obj_parent=parent_obj,
                        time_parent=parent_time,
                        obj_child=cand3b["objective"],
                        time_child=cand3b["eval_time"],
                        accepted=ok3b,
                        reason=reason3b,
                    )

                    if ok3b:
                        success_count["t3_gls_module"] += 1
                        accepted_candidates.append(cand3b)

        # ---------- 最终决策（新增：全局时间单调约束） ----------
        if accepted_candidates:
            # 允许的最大时间放大倍数（默认 1.0：不允许比 parent 慢）
            max_time_factor = float(self.t_cfg.get("t_final_max_time_factor", 1.0))
            parent_time_cap = None
            if parent_time is not None and parent_time > 0.0 and max_time_factor > 0.0:
                parent_time_cap = parent_time * max_time_factor

            feasible = []
            for c in accepted_candidates:
                t_child = c.get("eval_time", None)
                if parent_time_cap is None or t_child is None:
                    # 没有 parent_time 信息时，退化为原始行为
                    feasible.append(c)
                else:
                    if t_child <= parent_time_cap + 1e-9:
                        feasible.append(c)

            if not feasible:
                # 所有候选都比父代慢太多：本次 T 视为失败，不更新 parent
                return {
                    "accepted": False,
                    "reason": "no_candidate_faster_than_parent",
                    "individual": None,
                    "path": path,
                    "attempt_count": attempt_count,
                    "success_count": success_count,
                    "stage": None,
                }

            chosen = min(
                feasible,
                key=lambda c: c.get("eval_time", float("inf")),
            )
            try:
                if chosen.get("gls_spec"):
                    self.best_gls_spec = chosen["gls_spec"]
                self._persist_t_accept(chosen, attempt_count, success_count, path_tags=path)
                self._last_t_accept = chosen
            except Exception:
                pass

            return {
                "accepted": True,
                "reason": "ok",
                "individual": chosen,
                "path": path,
                "attempt_count": attempt_count,
                "success_count": success_count,
                "stage": chosen.get("stage", None),
            }

        # 三个阶段都没产生可接受方案
        return {
            "accepted": False,
            "reason": f"t1_gls_reject:{reason1}",
            "individual": None,
            "path": path,
            "attempt_count": attempt_count,
            "success_count": success_count,
            "stage": None,
        }




    #     参数
    #     ----
    #     parent : dict
    #         当前要做 T 阶段优化的个体。
    #     pop_objs : list or None
    #         当前种群的 objective 列表（可选，暂时不传给 t1/t2/t3，只是保留接口）。
    #     gen_index : int or None
    #         当前 generation index，用于写日志。
    #     dsl_stage : str or None
    #         标记当前是 SSL-1 还是 SSL-2（例如 "dsl1" / "dsl2"），
    #         目前只用于打印前缀，不改变接受逻辑。
    #     """
    #     gen_idx = gen_index
    #     dsl_tag = f"[{dsl_stage}]" if dsl_stage else ""


    #     if parent is None:
    #         return {
    #             "accepted": False,
    #             "reason": "no_parent",
    #             "individual": None,
    #             "path": [],
    #             "attempt_count": {},
    #             "success_count": {},
    #             "stage": None,
    #         }

    #     if not parent.get('gls_spec'):
    #         parent['gls_spec'] = self._default_gls_spec()

    #     parent_obj = parent.get('objective', 1e10)
    #     parent_time = parent.get('eval_time', None)

    #     attempt_count = {
    #         "t1_gls_structure": 0,
    #         "t2_gls_param": 0,
    #         "t3_gls_module": 0,
    #     }
    #     success_count = {
    #         "t1_gls_structure": 0,
    #         "t2_gls_param": 0,
    #         "t3_gls_module": 0,
    #     }
    #     path = []
    #     accepted_candidates = []

    #     # ---------- T1: 结构级别的 spec 调整 ----------
    #     if self.t_cfg.get("verbose", True):
    #         print(f"=== {dsl_tag} T1(GLS): Structure Swap ===")
    #     attempt_count["t1_gls_structure"] += 1

    #     # 关键改动：这里不要再传 pop_objs，Evolution.t1_gls_structure 只接受 (parent_indiv, rejection_reason)
    #     spec1 = self.evol.t1_gls_structure(parent, rejection_reason=None)

    #     if spec1:
    #         cand1 = {
    #             "algorithm": (parent.get("algorithm", "") + " [t1_gls_structure]"),
    #             "code": parent.get("code"),
    #             "gls_spec": spec1,
    #             "stage": "t1",
    #         }
    #         t0 = time.time()
    #         res1, _ = self._evaluate_with_timeout(cand1, self.timeout)
    #         t1_time = time.time()
    #         cand1["eval_time"] = t1_time - t0
    #         cand1["objective"] = res1.get("fitness", 1e10)

    #         parent_meta = dict(parent.get("other_inf") or {})
    #         meta1 = res1.get("meta", {}) or {}
    #         parent_meta.update(meta1)
    #         cand1["other_inf"] = parent_meta

    #         path.append("t1_gls")
    #         ok1, reason1 = self._accept_new_parent(
    #             cand1["objective"],
    #             cand1["eval_time"],
    #             parent_obj,
    #             parent_time,
    #             "t1",
    #             meta=cand1.get("other_inf"),
    #         )
    #         _print_stage("t1_gls", cand1, ok1, reason1)

    #         self._record_t_attempt(
    #             parent_indiv=parent,
    #             stage="t1",
    #             gen_index=gen_idx,
    #             obj_parent=parent_obj,
    #             time_parent=parent_time,
    #             obj_child=cand1["objective"],
    #             time_child=cand1["eval_time"],
    #             accepted=ok1,
    #             reason=reason1,
    #         )
    #     else:
    #         cand1 = None
    #         ok1, reason1 = False, "t1_no_spec"

    #     current = None
    #     if ok1 and cand1 is not None:
    #         success_count["t1_gls_structure"] += 1
    #         current = cand1
    #         accepted_candidates.append(cand1)

    #     # ---------- 主路径：在 T1 基础上尝试 T2/T3 ----------
    #     if current is not None:
    #         # T2 正常
    #         if self.t_cfg.get("verbose", True):
    #             print(f"=== {dsl_tag} T2(GLS): Param Tune ===")
    #         attempt_count["t2_gls_param"] += 1
    #         spec2 = self.evol.t2_gls_param(current, rejection_reason=None)
    #         if spec2:
    #             cand2 = {
    #                 "algorithm": (current.get("algorithm", "") + " [t2_gls_param]"),
    #                 "code": current.get("code"),
    #                 "gls_spec": spec2,
    #                 "stage": "t2",
    #             }
    #             t0 = time.time()
    #             res2, _ = self._evaluate_with_timeout(cand2, self.timeout)
    #             t2_time = time.time()
    #             cand2["eval_time"] = t2_time - t0
    #             cand2["objective"] = res2.get("fitness", 1e10)

    #             parent_meta = dict(current.get("other_inf") or {})
    #             meta2 = res2.get("meta", {}) or {}
    #             parent_meta.update(meta2)
    #             cand2["other_inf"] = parent_meta

    #             path.append("t2_gls")
    #             ok2, reason2 = self._accept_new_parent(
    #                 cand2["objective"],
    #                 cand2["eval_time"],
    #                 current["objective"],
    #                 current.get("eval_time", None),
    #                 "t2",
    #                 meta=cand2.get("other_inf"),
    #             )
    #             _print_stage("t2_gls", cand2, ok2, reason2)

    #             self._record_t_attempt(
    #                 parent_indiv=current,
    #                 stage="t2",
    #                 gen_index=gen_idx,
    #                 obj_parent=current["objective"],
    #                 time_parent=current.get("eval_time", None),
    #                 obj_child=cand2["objective"],
    #                 time_child=cand2["eval_time"],
    #                 accepted=ok2,
    #                 reason=reason2,
    #             )

    #             if ok2:
    #                 success_count["t2_gls_param"] += 1
    #                 current = cand2
    #                 accepted_candidates.append(cand2)

    #         # T3 正常
    #         if self.t_cfg.get("verbose", True):
    #             print(f"=== {dsl_tag} T3(GLS): Single-Module Tweak ===")
    #         attempt_count["t3_gls_module"] += 1
    #         spec3 = self.evol.t3_gls_module(current, rejection_reason=None)
    #         if spec3:
    #             cand3 = {
    #                 "algorithm": (current.get("algorithm", "") + " [t3_gls_module]"),
    #                 "code": current.get("code"),
    #                 "gls_spec": spec3,
    #                 "stage": "t3",
    #             }
    #             t0 = time.time()
    #             res3, _ = self._evaluate_with_timeout(cand3, self.timeout)
    #             t3_time = time.time()
    #             cand3["eval_time"] = t3_time - t0
    #             cand3["objective"] = res3.get("fitness", 1e10)

    #             parent_meta = dict(current.get("other_inf") or {})
    #             meta3 = res3.get("meta", {}) or {}
    #             parent_meta.update(meta3)
    #             cand3["other_inf"] = parent_meta

    #             path.append("t3_gls")
    #             ok3, reason3 = self._accept_new_parent(
    #                 cand3["objective"],
    #                 cand3["eval_time"],
    #                 current["objective"],
    #                 current.get("eval_time", None),
    #                 "t3",
    #                 meta=cand3.get("other_inf"),
    #             )
    #             _print_stage("t3_gls", cand3, ok3, reason3)

    #             self._record_t_attempt(
    #                 parent_indiv=current,
    #                 stage="t3",
    #                 gen_index=gen_idx,
    #                 obj_parent=current["objective"],
    #                 time_parent=current.get("eval_time"),
    #                 obj_child=cand3["objective"],
    #                 time_child=cand3["eval_time"],
    #                 accepted=ok3,
    #                 reason=reason3,
    #             )

    #             if ok3:
    #                 success_count["t3_gls_module"] += 1
    #                 accepted_candidates.append(cand3)

    #     # ---------- bypass 路径：T1 不通过 ----------
    #     if (not ok1) and self.t_cfg.get("bypass_on_fail", True):
    #         # T2 bypass
    #         if self.t_cfg.get("verbose", True):
    #             print(f"=== {dsl_tag} T2(GLS): Param Tune (bypass) ===")
    #         attempt_count["t2_gls_param"] += 1
    #         spec2b = self.evol.t2_gls_param(parent, rejection_reason=reason1)
    #         if spec2b:
    #             cand2b = {
    #                 "algorithm": (parent.get("algorithm", "") + " [t2_gls_param_bypass]"),
    #                 "code": parent.get("code"),
    #                 "gls_spec": spec2b,
    #                 "stage": "t2_bypass",
    #             }
    #             t0 = time.time()
    #             res2b, _ = self._evaluate_with_timeout(cand2b, self.timeout)
    #             t2b_time = time.time()
    #             cand2b["eval_time"] = t2b_time - t0
    #             cand2b["objective"] = res2b.get("fitness", 1e10)

    #             parent_meta = dict(parent.get("other_inf") or {})
    #             meta2b = res2b.get("meta", {}) or {}
    #             parent_meta.update(meta2b)
    #             cand2b["other_inf"] = parent_meta

    #             path.append("t2_gls(bypass)")
    #             ok2b, reason2b = self._accept_new_parent(
    #                 cand2b["objective"],
    #                 cand2b["eval_time"],
    #                 parent_obj,
    #                 parent_time,
    #                 "t2",
    #                 meta=cand2b.get("other_inf"),
    #             )
    #             _print_stage("t2_gls(bypass)", cand2b, ok2b, reason2b)

    #             self._record_t_attempt(
    #                 parent_indiv=parent,
    #                 stage="t2_bypass",
    #                 gen_index=gen_idx,
    #                 obj_parent=parent_obj,
    #                 time_parent=parent_time,
    #                 obj_child=cand2b["objective"],
    #                 time_child=cand2b["eval_time"],
    #                 accepted=ok2b,
    #                 reason=reason2b,
    #             )

    #             if ok2b:
    #                 success_count["t2_gls_param"] += 1
    #                 accepted_candidates.append(cand2b)

    #             # T3 bypass
    #             if self.t_cfg.get("verbose", True):
    #                 print(f"=== {dsl_tag} T3(GLS): Single-Module (bypass) ===")
    #             attempt_count["t3_gls_module"] += 1
    #             spec3b = self.evol.t3_gls_module(parent, rejection_reason=reason2b)
    #             if spec3b:
    #                 cand3b = {
    #                     "algorithm": (parent.get("algorithm", "") + " [t3_gls_module_bypass]"),
    #                     "code": parent.get("code"),
    #                     "gls_spec": spec3b,
    #                     "stage": "t3_bypass",
    #                 }
    #                 t0 = time.time()
    #                 res3b, _ = self._evaluate_with_timeout(cand3b, self.timeout)
    #                 t3b_time = time.time()
    #                 cand3b["eval_time"] = t3b_time - t0
    #                 cand3b["objective"] = res3b.get("fitness", 1e10)

    #                 parent_meta = dict(parent.get("other_inf") or {})
    #                 meta3b = res3b.get("meta", {}) or {}
    #                 parent_meta.update(meta3b)
    #                 cand3b["other_inf"] = parent_meta

    #                 path.append("t3_gls(bypass)")
    #                 ok3b, reason3b = self._accept_new_parent(
    #                     cand3b["objective"],
    #                     cand3b["eval_time"],
    #                     parent_obj,
    #                     parent_time,
    #                     "t3",
    #                     meta=cand3b.get("other_inf"),
    #                 )
    #                 _print_stage("t3_gls(bypass)", cand3b, ok3b, reason3b)

    #                 self._record_t_attempt(
    #                     parent_indiv=parent,
    #                     stage="t3_bypass",
    #                     gen_index=gen_idx,
    #                     obj_parent=parent_obj,
    #                     time_parent=parent_time,
    #                     obj_child=cand3b["objective"],
    #                     time_child=cand3b["eval_time"],
    #                     accepted=ok3b,
    #                     reason=reason3b,
    #                 )

    #                 if ok3b:
    #                     success_count["t3_gls_module"] += 1
    #                     accepted_candidates.append(cand3b)

    #     # ---------- 最终决策 ----------
    #     if accepted_candidates:
    #         chosen = min(
    #             accepted_candidates,
    #             key=lambda c: c.get("eval_time", float("inf")),
    #         )
    #         try:
    #             if chosen.get("gls_spec"):
    #                 self.best_gls_spec = chosen["gls_spec"]
    #             self._persist_t_accept(chosen, attempt_count, success_count, path_tags=path)
    #             self._last_t_accept = chosen
    #         except Exception:
    #             pass

    #         return {
    #             "accepted": True,
    #             "reason": "ok",
    #             "individual": chosen,
    #             "path": path,
    #             "attempt_count": attempt_count,
    #             "success_count": success_count,
    #             "stage": chosen.get("stage", None),
    #         }

    #     # 三个阶段都没产生可接受方案
    #     return {
    #         "accepted": False,
    #         "reason": f"t1_gls_reject:{reason1}",
    #         "individual": None,
    #         "path": path,
    #         "attempt_count": attempt_count,
    #         "success_count": success_count,
    #         "stage": None,
    #     }


    # =================== T-phase entry (GLS only) ===================
    def run_t_pipeline(self, pop, seed_pool=None, best_obj=None, fastest_time=None):
        """
        保留旧入口，当前未在 DASH.run 中使用；内部转发到 run_t_pipeline_gls。
        """
        return self.run_t_pipeline_gls(pop, seed_pool=seed_pool, best_obj=best_obj, fastest_time=fastest_time)
