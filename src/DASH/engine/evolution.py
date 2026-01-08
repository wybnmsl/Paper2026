import re
import time
import json
import textwrap
import os
from ..llm.interface_LLM import InterfaceLLM


class Evolution:
    def __init__(self, api_endpoint, api_key, model_LLM,
                 llm_use_local, llm_local_url, debug_mode, prompts, **kwargs):
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs  = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"
        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key,
                                          self.model_LLM, llm_use_local,
                                          llm_local_url, self.debug_mode)
        self.dialogue_path = kwargs.get("dialogue_path", None)
        if self.dialogue_path:
            os.makedirs(self.dialogue_path, exist_ok=True)
            os.makedirs(os.path.join(self.dialogue_path, "t_phase"), exist_ok=True)

        # T-phase history 控制（用于“带记忆的 top-k T 优化”）
        self.t_history_maxlen = kwargs.get("t_history_maxlen", 5)
        self.t_history_in_prompt = kwargs.get("t_history_in_prompt", True)
        # ---- Operator name mapping (DASH naming)
        # `operator_name` inside this class refers to the *implementation* method name
        # (e.g., e1/e2/m1...), while `op_name_map` provides a user-facing display name.
        self.op_name_map = kwargs.get(
            "op_name_map",
            {
                "i1": "MDL-Init",
                "e1": "MDL-1",
                "e2": "MDL-2",
                "m1": "MCL-1",
                "m2": "MCL-2",
                "m3": "MCL-3",
                # T-phase (SSL) labels are optional; InterfaceEC can override.
                "t1_gls_structure": "SSL/T1-Structure",
                "t2_gls_param": "SSL/T2-Param",
                "t3_gls_module": "SSL/T3-Module",
            },
        )



    # ---------------- logging ----------------
    def _log_dialogue(self, operator_name, prompt, response, extra=None, phase='e'):
        if not self.dialogue_path:
            return
        try:
            ts = int(time.time())
            safe_op = re.sub(r'[^a-zA-Z0-9_\-]', '_', operator_name)
            subdir = "t_phase" if phase == 't' else "legacy"
            base = os.path.join(self.dialogue_path, subdir)
            os.makedirs(base, exist_ok=True)
            with open(os.path.join(base, f"{ts}_{safe_op}_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(os.path.join(base, f"{ts}_{safe_op}_response.txt"), "w", encoding="utf-8") as f:
                f.write(response or "")
            payload = {
                "ts": ts,
                "operator": operator_name,
                "phase": phase,
                "prompt": prompt,
                "response": response or "",
                "extra": extra or {}
            }
            with open(os.path.join(base, f"{ts}_{safe_op}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if self.debug_mode:
                print(f"[WARN] Failed to log LLM dialogue: {e}")

    # ---------------- legacy i/e/m prompts ----------------

        # ---------------- legacy i/e/m prompts ----------------
    def get_prompt_i1(self):
        baseline_hint = (
            "We already have a strong classical Guided Local Search (GLS) baseline.\n"
            "In this baseline, the guided distance matrix is computed as:\n"
            "    D_base = edge_distance + lam * edge_n_used\n"
            "where 'lam' is a positive scalar penalty.\n"
            "In your design, you SHOULD treat D_base as a starting point: "
            "you may modify lam or add small corrections on top of D_base, "
            "but you MUST keep the function signature and input/output shapes unchanged.\n"
        )

        prompt_content = (
            self.prompt_task + "\n"
            + baseline_hint +
            f"First, describe your new algorithm and main steps in one sentence. "
            f"The description must be inside a brace. Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n"
            "You MUST return valid Python code. "
            "The code should either be inside a ```python ... ``` block, "
            "or start with 'def update_edge_distance('. "
            "Do not give additional explanations."
        )
        return prompt_content



    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i, ind in enumerate(indivs):
            prompt_indiv += f"No.{i+1} algorithm and its code:\n{ind['algorithm']}\n{ind['code']}\n"
        prompt_content = (
            self.prompt_task + "\n"
            f"I have {len(indivs)} existing algorithms with their codes as follows:\n"
            + prompt_indiv +
            "Please help me create a new algorithm that has a totally different form from the given ones.\n"
            f"First, describe your new algorithm and main steps in one sentence. The description must be inside a brace. "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n"
            "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i, ind in enumerate(indivs):
            prompt_indiv += f"No.{i+1} algorithm and its code:\n{ind['algorithm']}\n{ind['code']}\n"
        prompt_content = (
            self.prompt_task + "\n"
            f"I have {len(indivs)} existing algorithms with their codes as follows:\n"
            + prompt_indiv +
            "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.\n"
            "Firstly, identify the common backbone idea in the provided algorithms. "
            "Secondly, based on that backbone idea, describe your new algorithm in one sentence (inside a brace). "
            f"Thirdly, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n"
            "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m1(self, parent):
        prompt_content = (
            self.prompt_task + "\n"
            "I have one algorithm with its code as follows.\n"
            f"Algorithm description: {parent['algorithm']}\n"
            f"Code:\n{parent['code']}\n"
            "Please assist me in creating a new algorithm that has a different form but is a modified version of the provided algorithm.\n"
            f"First, describe your new algorithm and main steps in one sentence. The description must be inside a brace. "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n"
            "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m2(self, parent):
        prompt_content = (
            self.prompt_task + "\n"
            "I have one algorithm with its code as follows.\n"
            f"Algorithm description: {parent['algorithm']}\n"
            f"Code:\n{parent['code']}\n"
            "Please identify the main algorithm parameters and create a new algorithm that uses different parameter settings in the provided algorithm's score/function.\n"
            f"First, describe your new algorithm and main steps in one sentence. The description must be inside a brace. "
            f"Next, implement it in Python as a function named {self.prompt_func_name}. "
            f"This function should accept {len(self.prompt_func_inputs)} input(s): {self.joined_inputs}. "
            f"The function should return {len(self.prompt_func_outputs)} output(s): {self.joined_outputs}. "
            + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n"
            "Do not give additional explanations."
        )
        return prompt_content

    def get_prompt_m3(self, parent):
        prompt_content = (
            "First, identify the main components in the function below. "
            "Next, analyze whether any of these components could be overfit to the specific instances. "
            "Then, based on your analysis, simplify those components to enhance generalization to potential new instances. "
            "Finally, provide the revised code, keeping the function name, inputs, and outputs unchanged.\n"
            + parent['code'] + "\n"
            + self.prompt_inout_inf + "\n"
            "Do not give additional explanations."
        )
        return prompt_content

    def _get_alg(self, prompt_content, operator_name="unknown"):
        response = ""
        for _ in range(3):
            raw_response = self.interface_llm.get_response(prompt_content)
            if raw_response:
                response = raw_response
                break
            time.sleep(1)
        op_disp = self.op_name_map.get(operator_name, operator_name)
        self._log_dialogue(op_disp, prompt_content, response, phase='e')
        if not response:
            return None, None
        thought = ""
        thought_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        code = ""
        code_match = re.search(r'```python\s*([\s\S]+?)\s*```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            def_match = re.search(r'def\s+' + re.escape(self.prompt_func_name), response, re.DOTALL)
            if def_match:
                code = response[def_match.start():].strip()
        if not code:
            def_match_any = re.search(r'def\s+\w+\(.*\):', response, re.DOTALL)
            if def_match_any:
                code = def_match_any[def_match_any.start():].strip()
        if not thought and not code:
            if self.debug_mode:
                print(f"Could not extract thought or code from response for operator {op_disp}")
            return None, None
        return [code, thought]

    def i1(self):
        return self._get_alg(self.get_prompt_i1(), "i1")

    def e1(self, parents):
        return self._get_alg(self.get_prompt_e1(parents), "e1")

    def e2(self, parents):
        return self._get_alg(self.get_prompt_e2(parents), "e2")

    def m1(self, parent):
        return self._get_alg(self.get_prompt_m1(parent), "m1")

    def m2(self, parent):
        return self._get_alg(self.get_prompt_m2(parent), "m2")

    def m3(self, parent):
        return self._get_alg(self.get_prompt_m3(parent), "m3")

    # =========================================================
    # ===============  GLSSpec（同骨架）T阶段  =================
    # =========================================================

    @staticmethod
    def _default_gls_spec_dict():
        """
        与 zTSP/gls/spec.py 中 GLSSpec 的默认配置保持一致：
        - kNN 候选集以 k=45 为中心（你的 grid sweet spot）
        - loop_max / max_no_improve 稍微收紧：400 / 80
        - 默认开启 random_relocate 扰动，interval=80
        - mid_ls guidance，top_k=6，type="llm"
        - 默认 time_limit_s=10.0 秒
        """
        return {
            "init": {"method": "nearest_neighbor", "start": 0, "multi_start": 1},
            "candset": {"type": "kNN", "k": 45},
            "operators": [
                {"name": "two_opt", "strategy": "first"},
                {"name": "relocate", "strategy": "first"},
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
        }


    def _clip_gls_spec(self, d: dict) -> dict:
        # k, top_k, loop_max/max_no_improve, time_limit_s 做裁剪
        # 目标：把搜索空间收缩到一块“中等强度、时间受控”的区域，
        # 让 T 阶段别再跑到特别重或特别弱的 spec 上。

        # candset.k
        try:
            cs = d.get("candset", {}) or {}
            # 默认值靠近我们期望的中心（约 60）
            k = int(cs.get("k", 60))
            # 邻居数限制在 [16, 128] 之间，避免太小/太大
            k = max(16, min(k, 128))
            cs["k"] = k
            d["candset"] = cs
        except Exception:
            pass

        # guidance: top_k, weight
        try:
            gd = d.get("guidance", {}) or {}
            tk = int(gd.get("top_k", 6))
            # top_k 限制在 [2, 16]，既不过度稀疏也不过度密集
            tk = max(2, min(tk, 16))
            gd["top_k"] = tk

            w = float(gd.get("weight", 1.0))
            # guidance 权重限制在 [0.2, 2.0]，防止完全失效或极端放大
            w = max(0.2, min(w, 2.0))
            gd["weight"] = w

            d["guidance"] = gd
        except Exception:
            pass

        # schedule: loop_max, max_no_improve
        try:
            sch = d.get("schedule", {}) or {}
            loop_max = int(sch.get("loop_max", 800))
            # loop_max 限制在 [100, 2000]
            loop_max = max(100, min(loop_max, 2000))
            sch["loop_max"] = loop_max

            mni = int(sch.get("max_no_improve", 120))
            # max_no_improve 限制在 [20, min(loop_max, 400)]
            mni = max(20, min(mni, min(loop_max, 400)))
            sch["max_no_improve"] = mni

            d["schedule"] = sch
        except Exception:
            pass

        # stopping: time_limit_s
        try:
            st = d.get("stopping", {}) or {}
            tlim = float(st.get("time_limit_s", 8.0))
            # 单次 GLS 时间限制在 [4.0, 9.5] 秒
            tlim = max(4.0, min(tlim, 9.5))
            st["time_limit_s"] = tlim
            d["stopping"] = st
        except Exception:
            pass

        # operators 列表只做类型/必需字段检查，不强行裁剪 name，
        # 这样 "or_opt2" / "or_opt3" 等名字可以安全透传到底层。
        try:
            ops = d.get("operators", None)
            if isinstance(ops, list):
                cleaned = []
                for o in ops:
                    if not isinstance(o, dict):
                        continue
                    name = str(o.get("name", "two_opt"))
                    strategy = str(o.get("strategy", "first"))
                    cleaned.append({"name": name, "strategy": strategy})
                if cleaned:
                    d["operators"] = cleaned
        except Exception:
            pass

        return d

    def _format_t_history(self, parent_indiv):
        """
        把 parent_indiv.other_inf['t_history'] 压缩成几行文本，用在 T 阶段 prompt 里。
        现在会展示 iLDR / tLDR 这些动力学指标。
        """
        if not getattr(self, "t_history_in_prompt", True):
            return ""

        other = parent_indiv.get("other_inf") or {}
        hist = other.get("t_history") or []
        if not hist:
            return ""

        maxlen = getattr(self, "t_history_maxlen", 5) or 5
        recent = hist[-maxlen:]

        lines = ["Recent T-phase attempts on this parent (most recent last):"]
        for rec in recent:
            try:
                g  = rec.get("gen")
                st = rec.get("stage")
                ok = rec.get("accepted")
                rs = rec.get("reason")
                op = rec.get("obj_parent")
                oc = rec.get("obj_child")
                tp = rec.get("time_parent")
                tc = rec.get("time_child")
                li = rec.get("ldr_i")
                lt = rec.get("ldr_t")

                line = (
                    f"- [Gen {g}, {st}] "
                    f"parent_gap={op:.6f}, child_gap={oc:.6f}, "
                    f"parent_t={tp:.3f}s, child_t={tc:.3f}s"
                )
                if li is not None and lt is not None:
                    line += f", iLDR={li:+.3f}, tLDR={lt:+.3f}"
                line += f", accepted={ok}, reason={rs}"

                lines.append(line)
            except Exception:
                # 某条坏了就跳过，避免整个 prompt 崩掉
                continue

        if len(lines) == 1:
            return ""
        return "\n".join(lines)




    #     other = parent_indiv.get("other_inf") or {}
    #     hist = other.get("t_history") or []
    #     if not hist:
    #         return ""

    #     maxlen = getattr(self, "t_history_maxlen", 5) or 5
    #     recent = hist[-maxlen:]

    #     lines = ["Recent T-phase attempts on this parent (most recent last):"]
    #     for rec in recent:
    #         try:
    #             g  = rec.get("gen")
    #             st = rec.get("stage")
    #             ok = rec.get("accepted")
    #             rs = rec.get("reason")
    #             op = rec.get("obj_parent")
    #             oc = rec.get("obj_child")
    #             tp = rec.get("time_parent")
    #             tc = rec.get("time_child")
    #             lines.append(
    #                 f"- [Gen {g}, {st}] "
    #                 f"parent_gap={op:.6f}, child_gap={oc:.6f}, "
    #                 f"parent_t={tp:.3f}s, child_t={tc:.3f}s, "
    #                 f"accepted={ok}, reason={rs}"
    #             )
    #         except Exception:
    #             continue

    #     if len(lines) == 1:
    #         return ""
    #     return "\n".join(lines)



    # ---------------- t1_gls_structure ----------------
    def get_prompt_t1_gls_structure(self, parent_indiv, rejection_reason=None):
        par_spec = parent_indiv.get("gls_spec") or self._default_gls_spec_dict()
        par_obj  = parent_indiv.get("objective", "N/A")
        par_time = parent_indiv.get("eval_time", "N/A")

        feedback_lines = []
        if rejection_reason:
            feedback_lines.append(
                f"IMPORTANT: Previous attempt rejected: {rejection_reason}"
            )
        hist_txt = self._format_t_history(parent_indiv)
        if hist_txt:
            feedback_lines.append(hist_txt)
        feedback = "\n".join(feedback_lines)

        prompt = f"""
You are optimizing a Guided Local Search (GLS) **configuration** for a TSP-like
combinatorial optimization problem. The GLS solver skeleton and the evolved
heuristic (update_edge_distance) are already fixed; you are only allowed to
change the GLSSpec (init / candset / operators / schedule / stopping / guidance).

We treat the solver as a dynamical system. For each run we define a Lyapunov-style
energy:
  V = max(gap, 1e-6)

For a T-step editing, we use a *trajectory-aware* tLDR over the whole run:
  V(t)      = max(gap(t), 1e-6)
  V_best(t) = min_{{s<=t}} V(s)
  ell(t)    = log(V_best(t))

  J(T)         = (1/T) * ∫_0^T ell(t) dt
  tLDR_traj(T) = (2/T) * ( ell(0) - J(T) )

Higher tLDR_traj means **better anytime convergence** (faster decrease of incumbent gap over time). In this T1 step
(**Structure Swap**), your goal is:

- Propose a new GLSSpec that **improves tLDR and/or iLDR** on this parent,
- While keeping the final gap not significantly worse than the parent and
  below a global O_max.

Parent performance:
- objective (gap): {par_obj}
- eval_time (sec): {par_time}

{feedback}

HARD RULES:
- Output **only** a single JSON object, no extra text, no comments, no markdown.
- The JSON must follow this schema (keys are required):

{{
  "init": {{"method": "nearest_neighbor|greedy|random", "start": 0}},
  "candset": {{"type": "kNN|delaunay|hybrid", "k": <int>}},
  "operators": [{{"name": "two_opt|relocate", "strategy": "first|best"}}, ...],
  "schedule": {{"loop_max": <int>, "max_no_improve": <int>}},
  "accept": {{"type": "improve_only|sim_anneal", "temp0": <float>}},
  "stopping": {{"time_limit_s": <float>}},
  "guidance": {{"top_k": <int>, "lambda": <float>}}
}}

- You MUST keep the GLS solver skeleton unchanged (do not write any Python code).
- Prefer **smaller k**, shorter loops, and first-improvement if it helps tLDR.
- New time_limit_s should typically be ≤ parent time_limit_s, unless a slightly
  larger time budget yields much better iLDR/tLDR.

Parent GLSSpec:
{json.dumps(par_spec, indent=2)}
"""
        return textwrap.dedent(prompt)



    # ---------------- t1_gls_structure ----------------

#         feedback_lines = []
#         if rejection_reason:
#             feedback_lines.append(
#                 f"IMPORTANT: Previous attempt rejected: {rejection_reason}"
#             )
#         hist_txt = self._format_t_history(parent_indiv)
#         if hist_txt:
#             feedback_lines.append(hist_txt)
#         feedback = "\n".join(feedback_lines)

#         prompt = f"""
# You are optimizing a Guided Local Search (GLS) **policy specification** for TSP on the **same GLS framework** (do not change the solver skeleton).

# Your task in this step is **Structure Swap**: propose a new **GLSSpec JSON** that changes only the macro structure (candidate set type/size, operator sequence, high-level schedule, perturb placement), with the goal of **reducing runtime** while keeping solution quality acceptable.

# Parent performance:
# - objective (gap): {par_obj}
# - eval_time (sec): {par_time}

# {feedback}
# HARD RULES:
# - Output **only** a JSON object of the following schema (no Python code, no extra text):
# {{
#   "init": {{"method": "nearest_neighbor|greedy|random", "start": 0}},
#   "candset": {{"type": "kNN|delaunay|hybrid", "k": <int>}},
#   "operators": [{{"name": "two_opt|relocate", "strategy": "first|best"}}, ...],
#   "schedule": {{"loop_max": <int>, "max_no_improve": <int>}},
#   "accept": {{"type": "improve_only|sim_anneal", "temp0": <float>}},
#   "perturb": {{"type": "double_bridge|none", "interval": <int>}},
#   "guidance": {{"where": "pre_ls|mid_ls|post_ls", "weight": <float>, "top_k": <int>}},
#   "stopping": {{"time_limit_s": <float>}}
# }}
# - Keep **time_limit_s ≤ parent**; prefer **k smaller**, shorter loops, first-improvement, and cheap perturbation.
# - DO NOT change function signatures of the evolved heuristic (update_edge_distance).
# - This is structure-level change; do not micro-tune every numeric field.

# Parent GLSSpec:
# {json.dumps(par_spec, indent=2)}
# """
#         return textwrap.dedent(prompt)

    def t1_gls_structure(self, parent_indiv, rejection_reason=None):
        prompt = self.get_prompt_t1_gls_structure(parent_indiv, rejection_reason)
        response = self.interface_llm.get_response(prompt) or ""
        self._log_dialogue(self.op_name_map.get("t1_gls_structure", "t1_gls_structure"), prompt, response, phase='t')
        try:
            js = re.sub(r'```json\s*|\s*```', '', response).strip()
            spec = json.loads(js)
            if not isinstance(spec, dict):
                return None
            return self._clip_gls_spec(spec)
        except Exception:
            if self.debug_mode:
                print("[t1_gls_structure] Failed to parse JSON. Raw response:\n", response)
            return None


    def get_prompt_t2_gls_param(self, parent_indiv, rejection_reason=None):
        par_spec = parent_indiv.get("gls_spec") or self._default_gls_spec_dict()
        par_obj  = parent_indiv.get("objective", "N/A")
        par_time = parent_indiv.get("eval_time", "N/A")

        feedback_lines = []
        if rejection_reason:
            feedback_lines.append(
                f"IMPORTANT: Previous attempt rejected: {rejection_reason}"
            )
        hist_txt = self._format_t_history(parent_indiv)
        if hist_txt:
            feedback_lines.append(hist_txt)
        feedback = "\n".join(feedback_lines)

        prompt = f"""
Now you are a **Time Budget Shrinker** for GLS. The current GLSSpec is already a
reasonable structure (output of T1). In this T2 step, you should:

- Keep the overall structure (init / operators / guidance) mostly unchanged;
- **Shrink time-related parameters** (time_limit_s, loop_max, etc.) around the
  parent spec;
- Aim to **increase tLDR** (log-gap decay per second), while keeping the final
  gap similar to the parent (only very small degradation is allowed).

We use:
  V = max(gap, 1e-6)
  iLDR = log(V_before) - log(V_after)
  tLDR = iLDR / runtime

Good T2 proposals have:
- iLDR ≈ 0  (gap almost unchanged),
- tLDR > 0  (runtime shorter with similar gap).

Parent performance:
- objective (gap): {par_obj}
- eval_time (sec): {par_time}

{feedback}

Rules:
- Output **only** a GLSSpec JSON (same schema as in T1). No Python code.
- Keep "init", "operators" and "guidance" close to the parent; focus on:
  - smaller time_limit_s
  - smaller loop_max
  - reasonable max_no_improve
  - moderate/small candset.k
- Do NOT drastically increase k or loop_max.

Parent GLSSpec:
{json.dumps(par_spec, indent=2)}
"""
        return textwrap.dedent(prompt)



    # ---------------- t2_gls_param ----------------

#         feedback_lines = []
#         if rejection_reason:
#             feedback_lines.append(
#                 f"IMPORTANT: Previous attempt rejected: {rejection_reason}"
#             )
#         hist_txt = self._format_t_history(parent_indiv)
#         if hist_txt:
#             feedback_lines.append(hist_txt)
#         feedback = "\n".join(feedback_lines)

#         parent_gls = par_spec
#         prompt = f"""
# Now you are a **Parameter Tuning Specialist**. Keep the **structure unchanged**, and micro-tune **numeric fields** only (k, loop_max, max_no_improve, interval, top_k, weight, temp0, time_limit_s), targeting **faster runtime** with **no worse quality**.

# Parent performance:
# - objective (gap): {par_obj}
# - eval_time (sec): {par_time}

# {feedback}
# Rules:
# - Output **only** a GLSSpec JSON (same schema as before). No Python code.
# - Keep time_limit_s ≤ parent.
# - Prefer smaller k, smaller loop_max, moderate max_no_improve, and small top_k.

# Parent GLSSpec:
# {json.dumps(parent_gls, indent=2)}
# """
#         return textwrap.dedent(prompt)

    def t2_gls_param(self, parent_indiv, rejection_reason=None):
        prompt = self.get_prompt_t2_gls_param(parent_indiv, rejection_reason)
        response = self.interface_llm.get_response(prompt) or ""
        self._log_dialogue(self.op_name_map.get("t2_gls_param", "t2_gls_param"), prompt, response, phase='t')
        try:
            js = re.sub(r'```json\s*|\s*```', '', response).strip()
            spec = json.loads(js)
            if not isinstance(spec, dict):
                return None
            return self._clip_gls_spec(spec)
        except Exception:
            if self.debug_mode:
                print("[t2_gls_param] Failed to parse JSON. Raw response:\n", response)
            return None


    def get_prompt_t3_gls_module(self, parent_indiv, rejection_reason=None):
        par_spec = parent_indiv.get("gls_spec") or self._default_gls_spec_dict()
        par_obj  = parent_indiv.get("objective", "N/A")
        par_time = parent_indiv.get("eval_time", "N/A")

        feedback_lines = []
        if rejection_reason:
            feedback_lines.append(
                f"IMPORTANT: Previous attempt rejected: {rejection_reason}"
            )
        hist_txt = self._format_t_history(parent_indiv)
        if hist_txt:
            feedback_lines.append(hist_txt)
        feedback = "\n".join(feedback_lines)

        prompt = f"""
In T3 you are allowed to **modify only ONE module** in GLSSpec to perform a
small but effective **quality polishing** step.

We start from a configuration that is already relatively fast (after T1/T2).
You may slightly increase the time cost (e.g., slightly larger time_limit_s or
more careful local search) to gain **better gap**.

We still use the Lyapunov-style metric:
  V = max(gap, 1e-6)
  iLDR = log(V_before) - log(V_after)
  tLDR = iLDR / runtime

Here, we care more about **iLDR (pure log-gap drop)** than about speed, but the
runtime increase must remain modest.

Parent performance:
- objective (gap): {par_obj}
- eval_time (sec): {par_time}

{feedback}

Rules:
- Output **only** a GLSSpec JSON (same schema). No Python code.
- You may change only ONE of the following modules:
  - schedule (loop_max, max_no_improve)
  - stopping (time_limit_s)
  - guidance (top_k, lambda)
  - operators strategy (first/best) for two_opt/relocate
- Keep changes minimal: a small increase in time_limit_s or loop_max is allowed
  only if it yields a much better gap.
- Do NOT change the evolved heuristic function signature (update_edge_distance).

Parent GLSSpec:
{json.dumps(par_spec, indent=2)}
"""
        return textwrap.dedent(prompt)



    # ---------------- t3_gls_module ----------------

#         feedback_lines = []
#         if rejection_reason:
#             feedback_lines.append(
#                 f"IMPORTANT: Previous attempt rejected: {rejection_reason}"
#             )
#         hist_txt = self._format_t_history(parent_indiv)
#         if hist_txt:
#             feedback_lines.append(hist_txt)
#         feedback = "\n".join(feedback_lines)

#         prompt = f"""
# You may **modify only ONE module** in GLSSpec (e.g., switch two_opt strategy first↔best, or change perturb type), giving a **minimal yet effective** structural tweak.

# {feedback}
# Rules:
# - Output **only** a GLSSpec JSON (same schema). No Python code.
# - Keep time_limit_s ≤ parent; do not increase loop_max or k significantly.
# - Do not change the evolved heuristic function signature.

# Parent GLSSpec:
# {json.dumps(par_spec, indent=2)}
# """
#         return textwrap.dedent(prompt)

    def t3_gls_module(self, parent_indiv, rejection_reason=None):
        prompt = self.get_prompt_t3_gls_module(parent_indiv, rejection_reason)
        response = self.interface_llm.get_response(prompt) or ""
        self._log_dialogue(self.op_name_map.get("t3_gls_module", "t3_gls_module"), prompt, response, phase='t')
        try:
            js = re.sub(r'```json\s*|\s*```', '', response).strip()
            spec = json.loads(js)
            if not isinstance(spec, dict):
                return None
            return self._clip_gls_spec(spec)
        except Exception:
            if self.debug_mode:
                print("[t3_gls_module] Failed to parse JSON. Raw response:\n", response)
            return None
