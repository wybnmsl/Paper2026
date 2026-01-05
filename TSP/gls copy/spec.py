# zTSP/gls/spec.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class OperatorSpec:
    name: str                      # "two_opt" | "relocate"
    strategy: str = "first"        # "first" | "best"

@dataclass
class GLSSpec:
    """ t 阶段产物：在“同一 GLS 骨架”内的结构/参数配置 """
    init: Dict[str, Any] = field(default_factory=lambda: {"method": "nearest_neighbor", "start": 0})
    # candset: 默认 k 降到 60（更贴近大规模 TSP 常用的邻居数），避免一上来就扫太多边
    candset: Dict[str, Any] = field(default_factory=lambda: {"type": "kNN", "k": 60})
    operators: List[OperatorSpec] = field(default_factory=lambda: [
        OperatorSpec("two_opt", "first"),
        OperatorSpec("relocate", "first"),
    ])
    # schedule: loop_max 稍微下调到 800，max_no_improve 提高到 120，让“有改进时多走几步”
    schedule: Dict[str, Any] = field(default_factory=lambda: {"loop_max": 800, "max_no_improve": 120})
    accept: Dict[str, Any] = field(default_factory=lambda: {"type": "improve_only", "temp0": 0.0})
    perturb: Dict[str, Any] = field(default_factory=lambda: {"type": "none", "interval": 80})
    # guidance: top_k 调到 6，默认仍然 weight=1.0
    guidance: Dict[str, Any] = field(default_factory=lambda: {"where": "mid_ls", "weight": 1.0, "top_k": 6})
    # stopping: 默认给 8 秒，而不是 10 秒，让 baseline 就偏向“中等时间预算”
    stopping: Dict[str, Any] = field(default_factory=lambda: {"time_limit_s": 8.0})
    
def default_gls_spec() -> GLSSpec:
    return GLSSpec()

def to_dict(spec: GLSSpec) -> Dict[str, Any]:
    d = asdict(spec)
    # asdict 会把 OperatorSpec 也转成 dict；这里直接返回即可
    return d

def from_json(js: Dict[str, Any] | None) -> GLSSpec:
    """ 宽松解析：缺省字段用默认值填充；未知字段忽略 """
    base = default_gls_spec()
    if not isinstance(js, dict):
        return base

    def pick(d: Dict[str, Any], k: str, default):
        v = d.get(k, default)
        return v if isinstance(v, type(default)) else default

    init = pick(js, "init", base.init)
    candset = pick(js, "candset", base.candset)
    schedule = pick(js, "schedule", base.schedule)
    accept = pick(js, "accept", base.accept)
    perturb = pick(js, "perturb", base.perturb)
    guidance = pick(js, "guidance", base.guidance)
    stopping = pick(js, "stopping", base.stopping)

    ops = []
    ops_js = js.get("operators", [])
    if isinstance(ops_js, list) and ops_js:
        for o in ops_js:
            if not isinstance(o, dict): 
                continue
            name = str(o.get("name", "two_opt"))
            strategy = str(o.get("strategy", "first"))
            ops.append(OperatorSpec(name=name, strategy=strategy))
    else:
        ops = base.operators

    return GLSSpec(
        init=init, candset=candset, operators=ops,
        schedule=schedule, accept=accept, perturb=perturb,
        guidance=guidance, stopping=stopping
    )
