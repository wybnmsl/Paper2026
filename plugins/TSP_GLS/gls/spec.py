# plugins/TSP_GLS/gls/spec.py
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class OperatorSpec:
    """Operator configuration inside GLSSpec."""
    name: str                      # e.g. "two_opt" | "relocate" | "or_opt2"
    strategy: str = "first"        # "first" | "best"


@dataclass
class GLSSpec:
    """SSL product: structural/parametric configuration under one unified GLS backbone.

    This structure is used by:
      - the DASH engine (via from_json / default_gls_spec),
      - standalone test scripts (manual construction and edits).
    """
    init: Dict[str, Any] = field(default_factory=lambda: {
        "method": "nearest_neighbor",
        "start": 0,
        "multi_start": 1,
    })

    candset: Dict[str, Any] = field(default_factory=lambda: {
        "type": "kNN",
        "k": 60,
    })

    operators: List[OperatorSpec] = field(default_factory=lambda: [
        OperatorSpec(name="two_opt", strategy="first"),
        OperatorSpec(name="relocate", strategy="first"),
        OperatorSpec(name="or_opt2", strategy="first"),
    ])

    schedule: Dict[str, Any] = field(default_factory=lambda: {
        "loop_max": 800,
        "max_no_improve": 120,
    })

    accept: Dict[str, Any] = field(default_factory=lambda: {
        "type": "gls",
        "temp0": 0.0,
    })

    perturb: Dict[str, Any] = field(default_factory=lambda: {
        "type": "none",          # "none" | "random_relocate" | "double_bridge"
        "moves": 1,
        "interval": 80,
    })

    guidance: Dict[str, Any] = field(default_factory=lambda: {
        "type": "none",          # "builtin" | "llm" | "none"
        "top_k": 6,
        "where": "mid_ls",
        "weight": 1.0,
    })

    stopping: Dict[str, Any] = field(default_factory=lambda: {
        "time_limit_s": 8.0,
    })

    engine: Dict[str, Any] = field(default_factory=lambda: {
        "type": "ls_basic",          # only "ls_basic" is effective; others fall back to ls_basic
        "lk_max_outer": 12,
        "lk_max_inner": 5,
        "lk_top_k": 10,
        "lk_first_improvement": False,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_gls_spec() -> GLSSpec:
    """Return a default GLSSpec for baselines/ablations."""
    return GLSSpec()


def to_dict(spec: GLSSpec) -> Dict[str, Any]:
    """Backwards compatible module-level to_dict(spec)."""
    return asdict(spec)


def from_json(js: Optional[Dict[str, Any]]) -> GLSSpec:
    """Build GLSSpec from JSON/dict; missing fields use defaults; unknown fields ignored."""
    base = default_gls_spec()
    if js is None or not isinstance(js, dict):
        return base

    def merge_dict(dst: Dict[str, Any], src_val: Any) -> Dict[str, Any]:
        if isinstance(src_val, dict):
            out = dict(dst)
            out.update(src_val)
            return out
        return dst

    init = merge_dict(base.init, js.get("init"))
    candset = merge_dict(base.candset, js.get("candset"))
    schedule = merge_dict(base.schedule, js.get("schedule"))
    accept = merge_dict(base.accept, js.get("accept"))
    perturb = merge_dict(base.perturb, js.get("perturb"))
    guidance = merge_dict(base.guidance, js.get("guidance"))
    stopping = merge_dict(base.stopping, js.get("stopping"))
    engine = merge_dict(base.engine, js.get("engine"))

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
