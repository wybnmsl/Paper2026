# prompts_tsp.py
# Prompt spec for generating the guide(state) policy (deterministic) for TSP

class GetPrompts():
    def __init__(self):
        # 初次（无历史）必须输出 baseline；后续可在同一接口上改良
        self.prompt_task = (
            "Task: Design a deterministic policy function guide(state) for an exact "
            "Travelling Salesman Problem solver (branch-and-bound search). The solver builds tours "
            "incrementally and repeatedly calls this policy to choose the next city (pivot) and an ordered list of candidate cities to visit next.\n\n"
            "=== BASELINE POLICY (for the initial design with NO prior algorithms listed, "
            "you MUST output EXACTLY this function and nothing else) ===\n"
            "def guide(state):\n"
            "    dist = state['dist']; path = state['path']; visited = state['visited']\n"
            "    # ---- 选 pivot：使用当前路径的最后一个城市 ----\n"
            "    u = path[-1]\n"
            "    # ---- 候选排序：按距离升序排列未访问城市 ----\n"
            "    candidates = [v for v in range(len(dist)) if v not in visited]\n"
            "    order = sorted(candidates, key=lambda v: dist[u][v])\n"
            "    return {'pivot': u, 'order': order}\n"
            "=== END BASELINE ===\n"
        )
        self.prompt_func_name = "guide"
        self.prompt_func_inputs = ["state"]
        self.prompt_func_outputs = ["decision"]
        self.prompt_inout_inf = (
            "'state' is a dict with keys: 'dist' (NxN distance matrix), 'path' (list of visited city indices in order), "
            "'visited' (set of visited city indices), 'current_length' (float total distance so far), 'cache' (may include 'best' for current best tour length). "
            "Return exactly a dict: {'pivot': u, 'order': [c1, c2, ...]}."
        )
        self.prompt_other_inf = (
            "Output ONLY the Python function definition (no extra text/markdown). "
            "No randomness. Do not modify state or its contents (path/visited/cache). "
            "Final line must be: return {'pivot': u, 'order': order}."
        )
    def get_task(self): return self.prompt_task
    def get_func_name(self): return self.prompt_func_name
    def get_func_inputs(self): return self.prompt_func_inputs
    def get_func_outputs(self): return self.prompt_func_outputs
    def get_inout_inf(self): return self.prompt_inout_inf
    def get_other_inf(self): return self.prompt_other_inf
