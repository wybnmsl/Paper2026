from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class OperatorSpec:
    """Operator configuration inside GLSSpec."""
    name: str                      # e.g. "two_opt" | "relocate" | "or_opt2"
    strategy: str = "first"        # "first" | "best"


@dataclass
class GLSSpec:
    """
    SSL 阶段产物：在“同一 GLS 骨架”内的结构/参数配置。

    说明：
    - 该结构既被 EoH 框架使用（通过 from_json / default_gls_spec），
      也被独立测试脚本使用（直接 new + 手动修改字段）。
    """
    # 初始解相关
    # method: "nearest_neighbor"；start: 起始点；multi_start: 多起点数量
    init: Dict[str, Any] = field(default_factory=lambda: {
        "method": "nearest_neighbor",
        "start": 0,
        "multi_start": 1,
    })

    # 候选集配置
    candset: Dict[str, Any] = field(default_factory=lambda: {
        "type": "kNN",   # 目前仅支持 kNN
        "k": 60,
    })

    # 局部算子配置（目前主要影响日志 / 未来扩展）
    operators: List[OperatorSpec] = field(default_factory=lambda: [
        OperatorSpec(name="two_opt", strategy="first"),
        OperatorSpec(name="relocate", strategy="first"),
        # 预留 or_opt2 名字，方便未来扩展
        OperatorSpec(name="or_opt2", strategy="first"),
    ])

    # 调度 / 循环参数
    schedule: Dict[str, Any] = field(default_factory=lambda: {
        "loop_max": 800,
        "max_no_improve": 120,
    })

    # 接受准则（目前仅作占位）
    accept: Dict[str, Any] = field(default_factory=lambda: {
        "type": "gls",
        "temp0": 0.0,
    })

    # 扰动（kick）配置
    perturb: Dict[str, Any] = field(default_factory=lambda: {
        "type": "none",          # "none" | "random_relocate" | "double_bridge"
        "moves": 1,
        "interval": 80,
    })

    # guidance（LLM 或 builtin GLS）相关配置
    guidance: Dict[str, Any] = field(default_factory=lambda: {
        # 对 test_gls_engine_perturb_grid.py：
        #   - BuiltinGLS 时使用 "builtin"
        #   - 其它情况可以设置 "none" / "llm"
        "type": "none",          # "builtin" | "llm" | "none"
        "top_k": 6,
        # 兼容旧版字段（不强制使用）
        "where": "mid_ls",
        "weight": 1.0,
    })

    # 停止条件
    stopping: Dict[str, Any] = field(default_factory=lambda: {
        "time_limit_s": 8.0,
    })

    # 局部搜索引擎配置（SSL-2 新增）
    # 目前实现中：仅 "ls_basic" 生效，其它类型会回退到 ls_basic。
    engine: Dict[str, Any] = field(default_factory=lambda: {
        "type": "ls_basic",          # "ls_basic" | 其它（将被视作 ls_basic）
        # 下面字段主要为兼容测试脚本 / 未来扩展，当前骨架不会真正使用
        "lk_max_outer": 12,
        "lk_max_inner": 5,
        "lk_top_k": 10,
        "lk_first_improvement": False,
    })

    # 便于序列化 / 调试
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_gls_spec() -> GLSSpec:
    """返回一份默认的 GLSSpec，用于 baseline 或 ablation。"""
    return GLSSpec()


def to_dict(spec: GLSSpec) -> Dict[str, Any]:
    """
    兼容旧接口：module 级 to_dict(spec)。
    （部分早期脚本可能依赖该函数。）
    """
    return asdict(spec)


def from_json(js: Optional[Dict[str, Any]]) -> GLSSpec:
    """
    从 JSON/dict 构造 GLSSpec。
    未出现的字段使用 default_gls_spec 的默认值；未知字段忽略。
    """
    base = default_gls_spec()
    if js is None:
        return base
    if not isinstance(js, dict):
        # 不合法就直接退回默认配置
        return base

    def merge_dict(dst: Dict[str, Any], src_val: Any) -> Dict[str, Any]:
        if isinstance(src_val, dict):
            out = dict(dst)
            out.update(src_val)
            return out
        return dst

    # --- init ---
    init = merge_dict(base.init, js.get("init"))

    # --- candset ---
    candset = merge_dict(base.candset, js.get("candset"))

    # --- schedule / accept / perturb / guidance / stopping / engine ---
    schedule = merge_dict(base.schedule, js.get("schedule"))
    accept = merge_dict(base.accept, js.get("accept"))
    perturb = merge_dict(base.perturb, js.get("perturb"))
    guidance = merge_dict(base.guidance, js.get("guidance"))
    stopping = merge_dict(base.stopping, js.get("stopping"))
    engine = merge_dict(base.engine, js.get("engine"))

    # --- operators ---
    ops: List[OperatorSpec] = []
    ops_js = js.get("operators", [])
    if isinstance(ops_js, list) and ops_js:
        for o in ops_js:
            if not isinstance(o, dict):
                continue
            name = str(o.get("name", "two_opt"))
            strategy = str(o.get("strategy", "first"))
            ops.append(OperatorSpec(name=name, strategy=strategy))
    else:
        ops = list(base.operators)

    return GLSSpec(
        init=init,
        candset=candset,
        operators=ops,
        schedule=schedule,
        accept=accept,
        perturb=perturb,
        guidance=guidance,
        stopping=stopping,
        engine=engine,
    )
