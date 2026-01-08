# src/eoh/utils/validators.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Set

REQUIRED_NODE_KEYS = {"name", "fn_name", "inputs", "outputs"}
REQUIRED_TOP_KEYS  = {"nodes", "edges"}

def validate_dag_spec(spec: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(spec, dict):
        return False, "spec must be a dict"
    for k in REQUIRED_TOP_KEYS:
        if k not in spec:
            return False, f"spec missing top-level key '{k}'"
    nodes = spec["nodes"]
    edges = spec["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False, "nodes/edges must be lists"
    names: Set[str] = set()
    for nd in nodes:
        if not isinstance(nd, dict): return False, "node must be dict"
        if not REQUIRED_NODE_KEYS.issubset(nd.keys()):
            return False, f"node missing required keys: {REQUIRED_NODE_KEYS - set(nd.keys())}"
        if nd["name"] in names:
            return False, f"duplicate node name: {nd['name']}"
        names.add(nd["name"])
        if not isinstance(nd["inputs"], list) or not isinstance(nd["outputs"], list):
            return False, "inputs/outputs must be lists"
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            return False, "each edge must be (src, dst)"
        if e[0] not in names or e[1] not in names:
            return False, f"edge refers unknown node: {e}"
    return True, ""

def repair_dag_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """轻量修复：去重节点名、丢弃非法边、补齐缺字段。"""
    if not isinstance(spec, dict):
        return {"nodes": [], "edges": []}
    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])
    fixed_nodes = []
    seen = set()
    for i, nd in enumerate(nodes):
        if not isinstance(nd, dict):
            continue
        name = nd.get("name", f"node_{i}")
        if name in seen:
            name = f"{name}_{i}"
        seen.add(name)
        fn_name = nd.get("fn_name", "passthrough")
        ins = nd.get("inputs", []); outs = nd.get("outputs", [])
        if not isinstance(ins, list): ins = []
        if not isinstance(outs, list): outs = []
        fixed_nodes.append({
            "name": name, "fn_name": fn_name,
            "inputs": ins, "outputs": outs,
            "config": nd.get("config", {})
        })
    names = {n["name"] for n in fixed_nodes}
    fixed_edges = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2 and e[0] in names and e[1] in names:
            fixed_edges.append((e[0], e[1]))
    return {"nodes": fixed_nodes, "edges": fixed_edges, "config": spec.get("config", {})}
