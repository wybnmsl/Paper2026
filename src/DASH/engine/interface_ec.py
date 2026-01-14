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
from datetime import datetime
import hashlib


class InterfaceEC:
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,
                 llm_use_local, llm_local_url, debug_mode, interface_prob,
                 select, n_p, timeout, use_numba, **kwargs):
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        dialogue_path = kwargs.pop("dialogue_path", None)

        # ---- T thresholds / switches (initialize first for passing into Evolution)
        self.t_cfg = kwargs.get("t_cfg", {}) or {}
        self.t_cfg.setdefault("alpha", {"t1": 0.20, "t2": 0.10, "t3": 0.05})
        self.t_cfg.setdefault("beta_abs", {"t1": 1.0, "t2": 0.8, "t3": 0.5})
        self.t_cfg.setdefault("gamma_rel", 0.10)
        self.t_cfg.setdefault("gamma_abs", 0.5)
        self.t_cfg.setdefault("Omax", 20.0)
        self.t_cfg.setdefault("bypass_on_fail", True)   # ensure default True
        self.t_cfg.setdefault("diag_retry", False)
        self.t_cfg.setdefault("verbose", True)
        # NEW: T-stage history controls
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
        # GLS T-operators (same skeleton)
        self.t_operators_gls = ['t1_gls_structure', 't2_gls_param', 't3_gls_module']

        # Track the most recent ACCEPTed individual in T-phase (for "bypass")
        self._last_t_accept = None

        # Output directory for exporting final accepted GLS solver / global top-k
        self.output_dir = kwargs.get("output_dir", None)
        if self.output_dir:
            # Create base directory; _persist_t_accept will further create results/exports
            os.makedirs(self.output_dir, exist_ok=True)

        # Global top-K archive (for final export across all generations)
        self.topk_archive = []
        # Key: top-k size = ec_pop_size
        self.topk_size = pop_size

    # ---------------- logging ----------------
    def log_print(self, content, *args, **kwargs):
        print(content, *args, **kwargs)

    # ---------------- baseline seed construction ----------------
    def _build_baseline_seed(self):
        """
        Build a "strong baseline seed" individual:

        - Use the same BuiltinGLS semantics as in test_gls_engine_perturb_grid 3.2_basic_reloc:
            D' = D + (lam * avg_edge_len) * penalty, lam=0.5
        - Align skeleton parameters to the best config from the grid:
            k=45, random_relocate, loop_max=400, max_no_improve=80, time_limit_s=10
        """

        # We do not implement update_edge_distance here; only define lam at module level:
        # guided_local_search will switch to BuiltinGLS mode whenever the guide_algorithm
        # object has attribute `lam`, ignoring update_edge_distance itself.
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

        # Use the unified constructor; it will attach fields like gls_spec
        ind = self._build_individual(code, algorithm)

        # Explicitly tune gls_spec close to the best config in the grid
        spec = ind.get("gls_spec") or self._default_gls_spec()

        # init: multi_start = 1
        spec.setdefault("init", {})
        spec["init"].setdefault("method", "nearest_neighbor")
        spec["init"].setdefault("start", 0)
        spec["init"]["multi_start"] = 1

        # candset: k=45 (sweet spot in 3.2_basic_reloc)
        spec.setdefault("candset", {})
        spec["candset"]["type"] = "kNN"
        spec["candset"]["k"] = 45

        # schedule: loop_max=400, max_no_improve=80
        spec.setdefault("schedule", {})
        spec["schedule"]["loop_max"] = 400
        spec["schedule"]["max_no_improve"] = 80

        # perturb: random_relocate + moves=1, interval=80
        spec.setdefault("perturb", {})
        spec["perturb"]["type"] = "random_relocate"
        spec["perturb"]["moves"] = 1
        spec["perturb"]["interval"] = 80

        # guidance: mark as builtin for clarity (backend uses lam attribute)
        spec.setdefault("guidance", {})
        spec["guidance"]["where"] = "mid_ls"
        spec["guidance"]["weight"] = 1.0
        spec["guidance"]["top_k"] = 6
        spec["guidance"]["type"] = "builtin"

        # stopping: 10 seconds (align with time_limit_s in grid)
        spec.setdefault("stopping", {})
        spec["stopping"]["time_limit_s"] = 10.0

        # engine: ls_basic (current skeleton only implements basic; engine field is for future extension)
        spec.setdefault("engine", {})
        spec["engine"]["type"] = "ls_basic"

        ind["gls_spec"] = spec
        return ind

    # ---------------- i-phase ----------------
    def population_generation(self):
        """
        New strategy: keep only one strong baseline in i-phase.
        - Build a baseline via _build_baseline_seed();
        - Evaluate the baseline once;
        - Clone the baseline into pop_size individuals as the initial population.

        This ensures that in the first MDL-e phase (e1/e2), all parents are the same baseline.
        """
        self.log_print("creating initial population (baseline-only i-phase):")
        pop = []

        # 1) Build the baseline seed
        baseline = None
        try:
            baseline = self._build_baseline_seed()
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to build baseline seed: {e}")

        # 2) If baseline construction fails, fallback to legacy logic (generate via i1)
        if baseline is None:
            if self.debug:
                print("[WARN] baseline seed is None, fallback to LLM i1 generation.")
            for _ in range(self.pop_size):
                parents, offsprings = self.get_offspring([], 'i1')
                if offsprings:
                    pop.extend(offsprings)

            self.log_print("initial population has been created! (fallback i1)")
            return pop

        # 3) Evaluate the baseline (unified _evaluate_individual path, compatible with GLS spec)
        try:
            res, eval_time = self._evaluate_individual(baseline)
            baseline["objective"] = res.get("fitness", 1e10)
            baseline["other_inf"] = res.get("meta", {})
            baseline["eval_time"] = eval_time
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to evaluate baseline seed: {e}")
            # If evaluation fails, also fallback to legacy logic
            for _ in range(self.pop_size):
                parents, offsprings = self.get_offspring([], 'i1')
                if offsprings:
                    pop.extend(offsprings)

            self.log_print("initial population has been created! (fallback i1)")
            return pop

        # 4) Baseline evaluation succeeded: clone pop_size copies as initial population
        for _ in range(self.pop_size):
            # Shallow copy to avoid sharing the same dict reference (history won't interfere)
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

        # Defensive truncation (should already be exactly pop_size)
        if len(pop) > self.pop_size:
            pop = pop[: self.pop_size]

        self.log_print("initial population has been created! (baseline only)")
        return pop

    # ---------------- i/e/m & routing ----------------
    def get_algorithm(self, pop, operator, **kwargs):
        """Compatible with DASH operator calls and keeps exp_n_proc parallel behavior."""
        results = []
        try:
            # self.n_p corresponds to exp_n_proc in config
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
        Maintain a global top-K archive for exporting the best individuals after evolution.
        Sort by fitness ascending and keep only self.topk_size entries.
        """
        try:
            fitness = float(fitness)
        except Exception:
            return

        # Filter out obviously invalid solutions
        if not (0 <= fitness < 1e9):
            return

        code = individual.get("code", "")
        gls_spec = individual.get("gls_spec")
        dag_spec = individual.get("dag_spec")

        # gls_spec / dag_spec may contain non-JSON-serializable objects; fallback safely
        try:
            gls_key = json.dumps(gls_spec, sort_keys=True, ensure_ascii=False) if gls_spec is not None else None
        except TypeError:
            gls_key = str(gls_spec)

        try:
            dag_key = json.dumps(dag_spec, sort_keys=True, ensure_ascii=False) if dag_spec is not None else None
        except TypeError:
            dag_key = str(dag_spec)

        key = (self._normalize_code_for_topk(code), gls_key, dag_key)

        # If already in archive, update only if better
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
        """Lightweight code normalization for top-K deduplication (strip whitespace)."""
        if not isinstance(code, str):
            return ""
        return re.sub(r"\s+", "", code)

    def export_topk_individuals(self, tag="global_topk"):
        """
        Export global top-K individuals to results/exports/.
        Each individual reuses _persist_t_accept export logic; only tags differ.
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
        Evaluate via the local problem interface with a "hard timeout":
        - Only limits evaluate() runtime (not LLM generation time).
        - If exceeding self.timeout seconds, raise an exception and stop evaluation,
          returning fitness=1e10 and meta["timeout"]=True.
        """
        start_time = time.time()
        result = {"fitness": 1e10, "meta": {}}

        # If timeout is not set or <= 0, use the legacy no-timeout path
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

            # Update global top-K archive
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

        # ---------------- Hard-timeout path ----------------
        # Use signal.SIGALRM to implement hard timeout
        def _timeout_handler(signum, frame):
            raise TimeoutError("Evaluation timeout")

        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
        except Exception:
            # Some environments (e.g., Windows) may not support SIGALRM; degrade to no-timeout
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
            # Start timer
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
            # Stop timer & restore old handler
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

        eval_time = time.time() - start_time

        # Update global top-K archive (keep best self.topk_size)
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
        """Compatibility wrapper for older T-phase code."""
        return self._evaluate_individual(individual)

    @staticmethod
    def _is_same_code(code_a, code_b):
        """Rough check for code equivalence (compare after stripping whitespace)."""
        if not code_a or not code_b:
            return False
        strip_a = re.sub(r'\s+', '', code_a)
        strip_b = re.sub(r'\s+', '', code_b)
        return strip_a == strip_b

    def check_duplicate(self, new_individual, pop):
        """Check whether a new individual is "equivalent" to one in the current population."""
        code = new_individual.get('code', '')
        gls_spec = new_individual.get('gls_spec')
        dag_spec = new_individual.get('dag_spec')

        for ind in pop:
            if not ind.get('code'):
                continue
            if not self._is_same_code(code, ind['code']):
                continue
            # Same code, further compare gls_spec/dag_spec
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

    # ---------------- i/e/m offspring generation ----------------
    def _to_impl_operator(self, operator: str) -> str:
        """Translate user-facing operator name to internal implementation operator."""
        return self._op_alias_to_impl.get(operator, operator)

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
        """Build an individual dict from (code, algorithm), optionally reusing existing fields."""
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
            # For example, T-phase may inherit some metadata from the parent
            for k in ['gls_spec', 'dag_spec', 'other_inf']:
                if k in base and base[k] is not None:
                    ind[k] = base[k]
        return ind

    def _get_offspring_legacy(self, pop, operator, operator_alias=None, **kwargs):
        """i/e/m phases: keep the original GLS evaluation path (no DAG) and record EM LDR."""
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
        # Basic check: must contain at least one function definition
        if not re.search(r'^\s*def\s+\w+\s*\(.*\)\s*:', cleaned, flags=re.MULTILINE):
            return parents, [{
                'objective': 1e10,
                'other_inf': {'error': 'no_function_detected'}
            }]

        base = parents[0] if parents else None
        new_ind = self._build_individual(cleaned, algorithm, base=base)

        # Drop duplicates (set objective to a large number)
        if self.check_duplicate(new_ind, pop):
            if "other_inf" not in new_ind or not isinstance(new_ind["other_inf"], dict):
                new_ind["other_inf"] = {}
            new_ind['other_inf']['duplicate'] = True
            new_ind['objective'] = 1e10
            return parents, [new_ind]

        # Evaluate
        res, eval_time = self._evaluate_individual(new_ind)
        new_ind['objective'] = res.get('fitness', 1e10)
        new_ind['other_inf'] = res.get('meta', {})
        new_ind['eval_time'] = eval_time

        # Record EM LDR (only works when both base/new_ind are dicts)
        try:
            gen_index = kwargs.get("gen_index", None)
            if isinstance(base, dict) and isinstance(new_ind, dict):
                self._record_em_attempt(base, new_ind, operator, gen_index=gen_index)
        except Exception as e:
            if self.debug:
                print(f"[WARN] failed to record EM LDR: {e}")

        return parents, [new_ind]

    # ---------------- GLS T-phase offspring generation ----------------
    def _supports_gls_spec(self):
        """Whether the current problem supports GLSSpec evaluation (T-phase)."""
        return hasattr(self.interface_eval, "evaluateGLS_with_spec")

    def _default_gls_spec(self):
        """
        If parent does not have gls_spec, use a reasonable default config.
        Note: keep consistent with eoh_evolution.Evolution._default_gls_spec_dict
        and GLSSpec defaults in zTSP/gls/spec.py.
        """
        return {
            "init": {"method": "nearest_neighbor", "start": 0, "multi_start": 1},
            "candset": {"type": "kNN", "k": 45},
            "operators": [
                {"name": "two_opt", "strategy": "first"},
                {"name": "relocate", "strategy": "first"},
                # Do not explicitly include or_opt2 here; the underlying skeleton will handle it
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
            # engine can be omitted; from_json will use GLSSpec default ls_basic
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

    # ---------------- GLS T-phase evaluation & acceptance ----------------
    def _choose_t_parent(self, pop, seed_pool=None):
        """
        Parent selection function kept for backward compatibility (still used in _get_offspring_gls).
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
        # Legacy "relative to global best/fastest" acceptance logic kept for historical compatibility
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

        alpha     = self.t_cfg["alpha"].get(phase_tag, 0.0)     # speed threshold
        beta_abs  = self.t_cfg["beta_abs"].get(phase_tag, 0.0)  # allowed quality degradation when speed wins
        gamma_rel = self.t_cfg["gamma_rel"]                     # relative quality threshold
        gamma_abs = self.t_cfg["gamma_abs"]                     # absolute quality threshold
        Omax      = self.t_cfg["Omax"]                          # quality upper bound

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
        New T-phase *local* acceptance criterion (LDR-based):
        - Compare candidate against its parent for a specific T-operator (t1/t2/t3).
        - Enforce a global quality guardrail.
        - If timing info is missing, fall back to pure quality comparison.
        - With timing info, decide using stage-specific rules based on LDR + constraints.

        Note:
        - The final decision (including monotonic time constraint) is handled in
          run_t_pipeline_for_parent; this function only performs local filtering.
        """
        # Basic validity checks
        if obj_c is None or obj_c >= 1e9:
            return False, "invalid_candidate"

        # Global quality guardrail: reject overly bad solutions
        Omax = float(self.t_cfg.get("Omax", 20.0))
        if obj_c > Omax:
            return False, f"quality_over_Omax({Omax})"

        # If parent is missing, treat as invalid
        if obj_p is None or obj_p >= 1e9:
            return False, "invalid_parent"

        # Shared thresholds
        gamma_rel = float(self.t_cfg.get("gamma_rel", 0.10))
        gamma_abs = float(self.t_cfg.get("gamma_abs", 0.5))

        # If timing info is missing, fall back to pure quality rule
        if time_p is None or time_c is None or time_p <= 0.0 or time_c <= 0.0:
            thr = min(obj_p * (1.0 + gamma_rel), obj_p + gamma_abs)
            if obj_c <= thr:
                return True, "quality_win(no_time)"
            return False, "no_improve(no_time)"

        # With time and gap, compute LDR metrics
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

        # Speed gain (positive means faster)
        speed_gain = (time_p - time_c) / max(time_p, 1e-9)
        rel_regret = (obj_c - obj_p) / max(obj_p, 1e-9)

        # Quality safety threshold
        thr_quality = min(obj_p * (1.0 + gamma_rel), obj_p + gamma_abs)

        # --------- t1: structure-level (quality + dynamics trade-off) ---------
        if phase_tag == "t1":
            # Direct quality improvement -> accept
            if obj_c <= thr_quality:
                return True, f"quality_win(iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

            # Allow limited regret in exchange for dynamics improvement
            max_rel_regret = float(self.t_cfg.get("t1_max_rel_regret", gamma_rel))
            min_score = float(self.t_cfg.get("t1_min_ldr_score", 0.0))
            alpha = float(self.t_cfg.get("t1_ldr_alpha", 1.0))   # tLDR weight
            beta = float(self.t_cfg.get("t1_ldr_beta", 1.0))     # iLDR weight
            score = alpha * tLDR + beta * iLDR

            if rel_regret <= max_rel_regret and score >= min_score:
                return True, f"ldr_win(score={score:.3f}, iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

            return False, "no_improve"

        # --------- t2: time shrinker (efficiency-first) ---------
        if phase_tag == "t2":
            # If gap improves significantly, accept
            if obj_c <= thr_quality:
                return True, f"quality_win(iLDR={iLDR:.3f}, tLDR_{tLDR_source}={tLDR:.3f})"

            # Otherwise require: clear time reduction + tLDR > 0 + limited quality degradation
            min_speed_gain = float(self.t_cfg.get("t2_min_speed_gain", 0.05))   # at least 5% speedup
            max_rel_regret = float(self.t_cfg.get("t2_max_rel_regret", 0.02))   # at most 2% gap worse

            if speed_gain >= min_speed_gain and tLDR > 0.0 and rel_regret <= max_rel_regret:
                return True, (
                    f"speed_shrink(Δt={speed_gain:.3f}, rel_regret={rel_regret:.3f}, "
                    f"iLDR={iLDR:.3f}, tLDR={tLDR:.3f})"
                )

            return False, "no_improve"

        # --------- t3: quality polishing (trade some time for larger quality gain) ---------
        if phase_tag == "t3":
            # Require: meaningful quality improvement
            min_abs_improve = float(self.t_cfg.get("t3_min_abs_improve", 0.0))
            max_time_factor = float(self.t_cfg.get("t3_max_time_factor", 1.5))

            if obj_c >= obj_p - min_abs_improve:
                return False, "no_quality_gain"

            # Prevent excessive time blow-up
            if time_c > time_p * max_time_factor:
                return False, "too_slow"

            # iLDR > 0 implies log-gap is decreasing
            if iLDR > 0.0:
                return True, f"polish_win(iLDR={iLDR:.3f}, tLDR={tLDR:.3f})"

            return False, "no_improve"

        # Unknown phase: conservative reject
        return False, f"unknown_phase({phase_tag})"

    def _compute_ldr_fields(self, obj_parent, obj_child, time_parent, time_child):
        """
        Compute Lyapunov-style LDR metrics:
          - V = max(gap, delta)
          - iLDR = log(V_parent) - log(V_child)
          - tLDR = iLDR / time_child

        Returns a dict containing:
          V_parent, V_child, ldr_i, ldr_t

        If info is insufficient or values are invalid, returns zeros.
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
        Ensure indiv['other_inf']['t_history'] exists and return this list.
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
        Ensure indiv['other_inf']['em_history'] exists and return this list.
        em_history records local edits in e/m phases (with LDR).
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
        Append a record into parent_indiv.other_inf['em_history']:
        - gen: which generation (from DASH.run via gen_index)
        - operator: which i/e/m operator
        - obj/time: parent/child objective and eval_time
        - V_parent/V_child/iLDR/tLDR: computed by _compute_ldr_fields
        """
        try:
            # Only record when both parent/child are dicts to avoid wrapper conflicts
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
        Append a record into parent_indiv.other_inf['t_history'] and truncate to maxlen.
        Additionally writes:
          - V_parent / V_child
          - ldr_i / ldr_t
          - ldr_score = ldr_alpha * ldr_t + ldr_beta * ldr_i
        """
        try:
            hist = self._ensure_t_history(parent_indiv)

            # Fill basic info
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

            # Compute LDR metrics
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

            # Unified LDR score (for analysis / policy)
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

    def _persist_t_accept(self, indiv, attempt_count, success_count, path_tags=None):
        """
        When a T-phase solution is ACCEPTed, export its code + gls_spec as a standalone solver.py (GLS version).
        Also write evaluation meta (other_inf, including per-instance results) into JSON,
        and add code_hash / gls_spec_hash for aggregation analysis.
        """
        try:
            if not self.output_dir:
                return

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Add readable tag info into filename for easier identification
            tag_suffix = ""
            if path_tags:
                safe_tags = [str(t).replace(" ", "_") for t in path_tags]
                tag_suffix = "_" + "_".join(safe_tags)

            fname = f"{ts}_seq{success_count}{tag_suffix}_accepted.py"
            fjson = f"{ts}_seq{success_count}{tag_suffix}_accepted.json"

            out_py = os.path.join(self.output_dir, "results", "exports", fname)
            out_js = os.path.join(self.output_dir, "results", "exports", fjson)
            os.makedirs(os.path.dirname(out_py), exist_ok=True)

            # Get evaluation metadata (written in _evaluate_individual / T-phase)
            meta = indiv.get("other_inf") or {}
            if not isinstance(meta, dict):
                meta = {"raw_other_inf": meta}

            # Hash code/spec for aggregation
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
                # Hash failures do not affect the main flow
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

            # Write code file via _write_export_py
            self._write_export_py(out_py, indiv, payload)

            if self.debug:
                print(f"[export] saved: {out_js} & {out_py}")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to export accepted T-solution: {e}")

    def _write_export_py(self, path, individual, meta):
        """
        Write an importable solver file (GLS version).
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
            # ==== GLS export ====
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
        Keep the legacy interface for historical callers.
        Current implementation selects one parent internally and calls run_t_pipeline_for_parent.
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
        Run GLS T-phase (t1→t2→t3 + bypass) on a given parent,
        using acceptance relative to the parent (and cascaded intermediate candidates),
        and return the single child with the best (fastest) time.

        Key change:
        - Still allow local acceptance in t1/t2/t3 using LDR + quality/time rules;
        - But in the final decision, add a global time constraint:
            * choose only among candidates with eval_time <= parent_time * t_final_max_time_factor
            * if none, treat this T attempt as failure and keep the parent (do not update spec)
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

        # ---------- T1: structure-level spec adjustment ----------
        if self.t_cfg.get("verbose", True):
            print(f"=== {dsl_tag} T1(GLS): Structure Swap ===")
        attempt_count["t1_gls_structure"] += 1

        # Do not pass pop_objs; Evolution.t1_gls_structure accepts only (parent_indiv, rejection_reason)
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

        # ---------- Main path: try T2/T3 on top of T1 ----------
        if current is not None:
            # T2 normal
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

            # T3 normal
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

        # ---------- Bypass path: if T1 fails ----------
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

        # ---------- Final decision (NEW: global monotonic time constraint) ----------
        if accepted_candidates:
            # Max time amplification factor (default 1.0: not slower than parent)
            max_time_factor = float(self.t_cfg.get("t_final_max_time_factor", 1.0))
            parent_time_cap = None
            if parent_time is not None and parent_time > 0.0 and max_time_factor > 0.0:
                parent_time_cap = parent_time * max_time_factor

            feasible = []
            for c in accepted_candidates:
                t_child = c.get("eval_time", None)
                if parent_time_cap is None or t_child is None:
                    # If parent_time is missing, fall back to original behavior
                    feasible.append(c)
                else:
                    if t_child <= parent_time_cap + 1e-9:
                        feasible.append(c)

            if not feasible:
                # All candidates are too slow compared to parent: treat this T as failure
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

        # No acceptable solution produced in all stages
        return {
            "accepted": False,
            "reason": f"t1_gls_reject:{reason1}",
            "individual": None,
            "path": path,
            "attempt_count": attempt_count,
            "success_count": success_count,
            "stage": None,
        }

    # =================== T-phase entry (GLS only) ===================
    def run_t_pipeline(self, pop, seed_pool=None, best_obj=None, fastest_time=None):
        """
        Keep the legacy entry point (not used in DASH.run currently);
        internally forwards to run_t_pipeline_gls.
        """
        return self.run_t_pipeline_gls(pop, seed_pool=seed_pool, best_obj=best_obj, fastest_time=fastest_time)
