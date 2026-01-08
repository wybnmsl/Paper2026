import numpy as np
import json
import random
import time
import os

from .interface_ec import InterfaceEC


class DASH:

    @staticmethod
    def _flatten_any(x):
        out = []
        def push(z):
            if z is None:
                return
            if isinstance(z, dict):
                out.append(z)
            elif isinstance(z, (list, tuple)):
                for w in z:
                    push(w)
        push(x)
        return out

    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage

        # ---- LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ---- EC settings
        self.pop_size = paras.ec_pop_size
        self.n_pop = paras.ec_n_pop
        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights

        # ---- T settings
        self.t_cfg = {
            "alpha": getattr(paras, "t_alpha", {"t1": 0.20, "t2": 0.10, "t3": 0.05}),
            "beta_abs": getattr(paras, "t_beta_abs", {"t1": 1.0, "t2": 0.8, "t3": 0.5}),
            "gamma_rel": getattr(paras, "t_gamma_rel", 0.10),
            "gamma_abs": getattr(paras, "t_gamma_abs", 0.5),
            "Omax": getattr(paras, "t_Omax", 20.0),
            "bypass_on_fail": getattr(paras, "t_bypass_on_fail", True),
            "diag_retry": getattr(paras, "t_diag_retry", False),
            "verbose": getattr(paras, "t_verbose", True),
            "t_history_maxlen": getattr(paras, "t_history_maxlen", 5),
            "t_history_in_prompt": getattr(paras, "t_history_in_prompt", True),
        }
        self.t_elite = getattr(paras, "t_elite", self.pop_size)
        self.t_enable_gap_thres = getattr(paras, "t_enable_gap_thres", None)
        self.t_phase_verbose = getattr(paras, "t_verbose", True)

        self.t_op_stats = {
            "t1_structure_swap": {"success": 0, "attempts": 1},
            "t2_parameter_tune": {"success": 0, "attempts": 1},
            "t3_rewrite_node": {"success": 0, "attempts": 1},
        }
        self.stagnation_counter = 0
        self.quality_relax_threshold = 0.02

        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode
        self.ndelay = 1

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        # ---------------- 输出目录结构 ----------------
        base_out = paras.exp_output_path  # 通常是 "./"
        prob_name = self.prob.__class__.__name__
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 允许外部指定整个 batch 共用的 run 名称（例如 "TSPGLS_20251207_161810"）
        run_name = getattr(paras, "exp_run_name", None)
        if not run_name:
            run_name = f"{prob_name}_{timestamp}"
            # 顺手写回 paras，方便 batch 里复用同一个 Paras 实例
            setattr(paras, "exp_run_name", run_name)

        # run 根目录：.../results/<run_name>/
        self.output_root = os.path.join(base_out, "results", run_name)

        # 对于单 case：直接用 case 名；
        # 对于多实例问题：如果 problem.case_tag 存在，就用它；否则 fallback。
        case_tag = getattr(self.prob, "case_tag", None)
        if not case_tag:
            inst_names = getattr(self.prob, "instance_names", None)
            if isinstance(inst_names, (list, tuple)) and len(inst_names) == 1:
                case_tag = str(inst_names[0])
            elif isinstance(inst_names, (list, tuple)) and len(inst_names) > 1:
                case_tag = f"{len(inst_names)}cases"
            else:
                case_tag = "default"
        self.case_tag = case_tag

        # 当前 case 的输出目录：.../results/<run_name>/<case_tag>/
        self.output_path = os.path.join(self.output_root, self.case_tag)

        # 保持原有子目录结构（只是都挂在 case 目录下面）
        os.makedirs(os.path.join(self.output_path, "results", "pops"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "results", "pops_best"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "results", "history"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "results", "ops_log"), exist_ok=True)

        self.dialogue_path = os.path.join(self.output_path, "results", "llm_dialogue")
        os.makedirs(self.dialogue_path, exist_ok=True)
        os.makedirs(os.path.join(self.dialogue_path, "t_phase"), exist_ok=True)

        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        print("- DASH parameters loaded -")
        print(f"- Output path: {self.output_path} -")
        print(f"  (run_root={self.output_root}, case={self.case_tag})")
        random.seed(2024)


    def add2pop(self, population, offspring):
        flat = self._flatten_any(offspring)
        for off in flat:
            obj = off.get('objective', None)
            try:
                off['objective'] = float(obj) if obj is not None else 1e10
            except Exception:
                off['objective'] = 1e10
            population.append(off)

    def run(self):
        print("- Evolution Start (Hybrid Mode) -")
        time_start = time.time()
        interface_prob = self.prob

        interface_ec = InterfaceEC(
            self.pop_size,
            self.m,
            self.api_endpoint,
            self.api_key,
            self.llm_model,
            self.use_local_llm,
            self.llm_local_url,
            self.debug_mode,
            interface_prob,
            select=self.select,
            n_p=self.exp_n_proc,
            timeout=self.timeout,
            use_numba=self.use_numba,
            dialogue_path=self.dialogue_path,
            t_cfg=self.t_cfg,
            # 关键：告诉 InterfaceEC 把导出文件写到当前 run 的目录下
            output_dir=self.output_path,
        )

        # ---------- 初始种群 ----------
        population = []
        if self.load_pop:
            print("load initial population from " + self.load_pop_path)
            with open(self.load_pop_path) as file:
                data = json.load(file)
            for individual in data:
                population.append(individual)
            n_start = self.load_pop_id
        else:
            print("creating initial population:")
            population = interface_ec.population_generation()
            population = self.manage.population_management(population, self.pop_size)
            print("initial population has been created!")

            if population:
                print(f"Pop initial:", end="")
                for off in population:
                    print(f" Obj: {off.get('objective', 'N/A')}", end="|")
                print()

            filename = os.path.join(self.output_path, "results", "pops", "population_generation_0.json")
            with open(filename, 'w') as f:
                json.dump(population, f, indent=4)
            n_start = 0

        # 根据 InterfaceEC 里的定义，把算子划分为 MDL / MCL 两组
        mdl_ops = []
        mcl_ops = []
        if hasattr(interface_ec, "i_operators") and hasattr(interface_ec, "e_operators"):
            mdl_base = list(interface_ec.i_operators) + list(interface_ec.e_operators)
            mdl_ops = [op for op in self.operators if op in mdl_base]
        if hasattr(interface_ec, "m_operators"):
            mcl_ops = [op for op in self.operators if op in interface_ec.m_operators]

        n_mdl = len(mdl_ops)
        n_mcl = len(mcl_ops)

        for pop in range(n_start, self.n_pop):
            print(f"\n--- Generation {pop + 1} / {self.n_pop} ---")
            recent_valid_offs = []

            # ---------- Phase 1: MDL（机制发现：i1/e1/e2 等） ----------
            if n_mdl > 0:
                print("--- Phase 1: MDL (Mechanism Discovery Layer: i/e operators) ---")
                for i, op in enumerate(mdl_ops):
                    print(f" OP: {op}, [MDL {i + 1} / {n_mdl}] ", end="|")
                    op_idx = self.operators.index(op) if op in self.operators else i
                    op_w = self.operator_weights[op_idx]
                    _op_t0 = time.time()
                    parents, offsprings = [], []
                    if (np.random.rand() < op_w):
                        parents, offsprings = interface_ec.get_algorithm(
                        population,
                        op,
                        gen_index=pop,  # 这里把 generation index 传下去（0-based）
                    )
                    _op_t1 = time.time()
                    op_elapsed = float(_op_t1 - _op_t0)
                    print(f" Time: {op_elapsed:.3f}s ", end="|")
                    offs_flat = self._flatten_any(offsprings)
                    self.add2pop(population, offs_flat)
                    if offs_flat:
                        for off in offs_flat:
                            obj = off.get('objective')
                            t_eval = off.get('eval_time', None)
                            if t_eval is not None:
                                print(f" Obj: {obj} (t={t_eval:.3f}s)|", end="")
                            else:
                                print(f" Obj: {obj}|", end="")
                            if (obj is not None) and float(obj) < 1e9:
                                recent_valid_offs.append(off)
                    else:
                        print(" (no offspring)|", end="")
                    print()
                    # MDL 阶段同样记录 history，operator 用原始名字
                    try:
                        hist_payload = {
                            "pop_index": pop + 1,
                            "phase": "MDL",
                            "operator": op,
                            "op_time_sec": round(op_elapsed, 6),
                            "parents": parents,
                            "offsprings": offs_flat,
                        }
                        hist_name = os.path.join(
                            self.output_path,
                            "results",
                            "history",
                            f"pop_{pop + 1:03d}_MDL_{i + 1:02d}_{op}.json",
                        )
                        with open(hist_name, "w") as fh:
                            json.dump(hist_payload, fh, indent=2)
                    except Exception:
                        pass

                # MDL 后先做一次 population_management 把种群裁回 pop_size
                population = self.manage.population_management(population, min(len(population), self.pop_size))
            else:
                print("--- Phase 1: MDL skipped (no MDL operators configured) ---")

            # ---------- Phase 2: SSL-1（在 MDL 产物上做一次轻量 T-phase） ----------
            print("--- Phase 2: SSL-1 (Early Dynamics Shaping) ---")
            # 为 SSL-1 构造有效候选集合
            valid_pop = [
                p
                for p in population
                if p.get('objective', 1e10) < 1e9
            ]
            if not valid_pop:
                print("  >> SSL-1 skipped: no valid individuals in population.")
            else:
                best_obj = min(p['objective'] for p in valid_pop)
                fastest_time = min(
                    (p.get('eval_time') if p.get('eval_time') is not None else float("inf"))
                    for p in valid_pop
                )

                if self.t_enable_gap_thres is not None and best_obj > self.t_enable_gap_thres:
                    print(
                        f"  >> SSL-1 disabled: best_obj={best_obj:.6f} "
                        f"> t_enable_gap_thres={self.t_enable_gap_thres:.6f}"
                    )
                else:
                    # 选出按 objective 排序的 top-k 精英作为 SSL-1 的 T 父代
                    elite_sorted = sorted(valid_pop, key=lambda x: x['objective'])
                    t_parents = elite_sorted[: max(1, min(self.t_elite, len(elite_sorted)))]

                    if self.t_phase_verbose:
                        parent_objs = [p['objective'] for p in t_parents]
                        print(
                            f"  >> SSL-1 parents (top-{len(t_parents)} by gap): "
                            f"{parent_objs}"
                        )

                    pop_objs = [
                        p['objective']
                        for p in population
                        if p.get('objective', 1e10) < 1e9
                    ]

                    t_children = []
                    t_parent_logs = []
                    for idx, parent in enumerate(t_parents):
                        t_res = interface_ec.run_t_pipeline_for_parent(
                            parent=parent,
                            pop_objs=pop_objs,
                            gen_index=pop + 1,
                            ssl_stage="dsl1",
                        )
                        if not t_res:
                            continue

                        if t_res.get('accepted'):
                            child = t_res['individual']
                            t_children.append(child)
                            stage = t_res.get('stage', 't')
                            reason = t_res.get('reason', 'ok')
                            parent_obj = parent.get('objective')
                            parent_time = parent.get('eval_time')
                            child_obj = child.get('objective')
                            child_time = child.get('eval_time')
                            print(
                                f"  >> [SSL-1] T-parent[{idx}] {stage} ACCEPTED "
                                f"parent_obj={parent_obj:.6f} (t={parent_time:.3f}s) "
                                f"-> child_obj={child_obj:.6f} (t={child_time:.3f}s) "
                                f"| reason={reason}"
                            )
                            t_parent_logs.append(
                                f"parent{idx}: ACCEPT {stage} ({reason})"
                            )
                        else:
                            reason = t_res.get('reason', 'no_improve')
                            stage = t_res.get('stage', 't')
                            parent_obj = parent.get('objective')
                            parent_time = parent.get('eval_time')
                            print(
                                f"  >> [SSL-1] T-parent[{idx}] {stage} REJECTED "
                                f"parent_obj={parent_obj:.6f} (t={parent_time:.3f}s) "
                                f"| reason={reason}"
                            )
                            t_parent_logs.append(f"parent{idx}: REJECT ({reason})")

                    if t_children:
                        for child in t_children:
                            population.append(child)
                        population = self.manage.population_management(
                            population, min(len(population), self.pop_size)
                        )
                        if self.t_phase_verbose:
                            print(
                                f"  >> SSL-1 accepted {len(t_children)} new GLSSpec variants."
                            )
                    else:
                        if self.t_phase_verbose:
                            print(
                                "  >> SSL-1 accepted no candidates. Details: "
                                + "; ".join(t_parent_logs)
                            )

            # ---------- Phase 3: MCL（机制压缩：m1/m2/m3 等） ----------
            if n_mcl > 0:
                print("--- Phase 3: MCL (Mechanism Consolidation Layer: m operators) ---")
                for j, op in enumerate(mcl_ops):
                    print(f" OP: {op}, [MCL {j + 1} / {n_mcl}] ", end="|")
                    op_idx = self.operators.index(op) if op in self.operators else j
                    op_w = self.operator_weights[op_idx]
                    _op_t0 = time.time()
                    parents, offsprings = [], []
                    if (np.random.rand() < op_w):
                        parents, offsprings = interface_ec.get_algorithm(
                        population,
                        op,
                        gen_index=pop,  # 这里把 generation index 传下去（0-based）
                    )
                    _op_t1 = time.time()
                    op_elapsed = float(_op_t1 - _op_t0)
                    print(f" Time: {op_elapsed:.3f}s ", end="|")
                    offs_flat = self._flatten_any(offsprings)
                    self.add2pop(population, offs_flat)
                    if offs_flat:
                        for off in offs_flat:
                            obj = off.get('objective')
                            t_eval = off.get('eval_time', None)
                            if t_eval is not None:
                                print(f" Obj: {obj} (t={t_eval:.3f}s)|", end="")
                            else:
                                print(f" Obj: {obj}|", end="")
                            if (obj is not None) and float(obj) < 1e9:
                                recent_valid_offs.append(off)
                    else:
                        print(" (no offspring)|", end="")
                    print()
                    # MCL 阶段 history
                    try:
                        hist_payload = {
                            "pop_index": pop + 1,
                            "phase": "MCL",
                            "operator": op,
                            "op_time_sec": round(op_elapsed, 6),
                            "parents": parents,
                            "offsprings": offs_flat,
                        }
                        hist_name = os.path.join(
                            self.output_path,
                            "results",
                            "history",
                            f"pop_{pop + 1:03d}_MCL_{j + 1:02d}_{op}.json",
                        )
                        with open(hist_name, "w") as fh:
                            json.dump(hist_payload, fh, indent=2)
                    except Exception:
                        pass

                # MCL 后再裁剪一次
                population = self.manage.population_management(population, min(len(population), self.pop_size))
            else:
                print("--- Phase 3: MCL skipped (no MCL operators configured) ---")

            # ---------- Phase 4: SSL-2（沿用原有 T-phase 逻辑） ----------
            print("--- Phase 4: SSL-2 (Main T-phase: Structure & Efficiency) ---")
            if self.t_cfg.get("verbose", True):
                a = self.t_cfg["alpha"]
                b = self.t_cfg["beta_abs"]
                gr = self.t_cfg["gamma_rel"]
                ga = self.t_cfg["gamma_abs"]
                omax = self.t_cfg["Omax"]
                print(f" [Thresholds] α={a} β_abs={b} γ_rel={gr} γ_abs={ga} Omax={omax}")

            # 长期停滞时：直接跳过整个 SSL-2 T-phase（保留原逻辑）
            if self.stagnation_counter > 5:
                print("  >> SSL-2 T-Phase skipped due to prolonged stagnation. Focusing on i/e/m phases.")
                continue

            # 为 SSL-2 构造有效候选集合
            valid_pop = [
                p
                for p in population
                if p.get('objective', 1e10) < 1e9
            ]
            if not valid_pop:
                print("  >> SSL-2 skipped: no valid individuals in population.")
            else:
                best_obj = min(p['objective'] for p in valid_pop)
                fastest_time = min(
                    (p.get('eval_time') if p.get('eval_time') is not None else float("inf"))
                    for p in valid_pop
                )

                # 如果设置了启用阈值，则在 best_obj 较大时跳过 T-phase
                if self.t_enable_gap_thres is not None and best_obj > self.t_enable_gap_thres:
                    print(
                        f"  >> SSL-2 disabled: best_obj={best_obj:.6f} "
                        f"> t_enable_gap_thres={self.t_enable_gap_thres:.6f}"
                    )
                else:
                    # 选出按 objective 排序的 top-k 精英作为 T 父代
                    elite_sorted = sorted(valid_pop, key=lambda x: x['objective'])
                    t_parents = elite_sorted[: max(1, min(self.t_elite, len(elite_sorted)))]

                    if self.t_phase_verbose:
                        parent_objs = [p['objective'] for p in t_parents]
                        print(
                            f"  >> SSL-2 parents (top-{len(t_parents)} by gap): "
                            f"{parent_objs}"
                        )

                    pop_objs = [
                        p['objective']
                        for p in population
                        if p.get('objective', 1e10) < 1e9
                    ]

                    t_children = []
                    t_parent_logs = []
                    for idx, parent in enumerate(t_parents):
                        t_res = interface_ec.run_t_pipeline_for_parent(
                            parent=parent,
                            pop_objs=pop_objs,
                            gen_index=pop + 1,   # 把当前代数传入 T 阶段，用于记忆
                            ssl_stage="dsl2",
                        )
                        if not t_res:
                            continue

                        if t_res.get('accepted'):
                            child = t_res['individual']
                            t_children.append(child)
                            stage = t_res.get('stage', 't')
                            reason = t_res.get('reason', 'ok')
                            parent_obj = parent.get('objective')
                            parent_time = parent.get('eval_time')
                            child_obj = child.get('objective')
                            child_time = child.get('eval_time')
                            print(
                                f"  >> [SSL-2] T-parent[{idx}] {stage} ACCEPTED "
                                f"parent_obj={parent_obj:.6f} (t={parent_time:.3f}s) "
                                f"-> child_obj={child_obj:.6f} (t={child_time:.3f}s) "
                                f"| reason={reason}"
                            )
                            t_parent_logs.append(
                                f"parent{idx}: ACCEPT {stage} ({reason})"
                            )
                        else:
                            reason = t_res.get('reason', 'no_improve')
                            stage = t_res.get('stage', 't')
                            parent_obj = parent.get('objective')
                            parent_time = parent.get('eval_time')
                            print(
                                f"  >> [SSL-2] T-parent[{idx}] {stage} REJECTED "
                                f"parent_obj={parent_obj:.6f} (t={parent_time:.3f}s) "
                                f"| reason={reason}"
                            )
                            t_parent_logs.append(f"parent{idx}: REJECT ({reason})")

                    if t_children:
                        for child in t_children:
                            population.append(child)
                        population = self.manage.population_management(
                            population, min(len(population), self.pop_size)
                        )
                        self.stagnation_counter = 0
                        if self.t_phase_verbose:
                            print(
                                f"  >> SSL-2 accepted {len(t_children)} new GLSSpec variants."
                            )
                    else:
                        self.stagnation_counter += 1
                        if self.t_phase_verbose:
                            print(
                                "  >> SSL-2 accepted no candidates. Details: "
                                + "; ".join(t_parent_logs)
                            )

            # ---------- 保存种群 ----------
            filename = os.path.join(
                self.output_path,
                "results",
                "pops",
                f"population_generation_{pop + 1}.json",
            )
            with open(filename, 'w') as f:
                json.dump(population, f, indent=4)
            if population:
                filename_best = os.path.join(
                    self.output_path,
                    "results",
                    "pops_best",
                    f"population_generation_{pop + 1}.json",
                )
                with open(filename_best, 'w') as f:
                    json.dump(population[0], f, indent=4)

            # ---------- 汇总打印（每个个体：gap + eval_time） ----------
            pop_summaries = []
            for ind in population:
                obj = ind.get('objective')
                t_eval = ind.get('eval_time', None)
                if t_eval is not None:
                    pop_summaries.append(f"{obj} (t={t_eval:.3f}s)")
                else:
                    pop_summaries.append(f"{obj} (t=N/A)")
            elapsed_min = (time.time() - time_start) / 60.0
            print(
                f"--- Gen {pop + 1} finished. Time Cost: {elapsed_min:.1f} m | "
                f"Pop Objs: [{', '.join(pop_summaries)}]"
            )

        # ---------- After all generations: dump summary & export global top-K ----------
        total_time_sec = time.time() - time_start

        # 先写当前 case 的 summary.csv
        try:
            self._dump_run_summary(interface_ec, total_time_sec)
        except Exception as e:
            if self.debug_mode:
                print(f"[WARN] Failed to dump run summary: {e}")

        # 再导出 global top-k solver
        try:
            interface_ec.export_topk_individuals(tag="global_topk")
        except Exception as e:
            if self.debug_mode:
                print(f"[WARN] Failed to export global top-k solutions: {e}")


# Backward compatibility (legacy name)
EOH = DASH
