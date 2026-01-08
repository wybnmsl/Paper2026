from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .utils.readMKP import MKPInstance, read_mknapcb
from .aco.spec import ACOSpec
from .aco.aco_run import solve_instance_with_spec_gap
from .plr import PLRConfig, PLRLibrary


@dataclass
class MKPACOProblem:
    """
    Minimal task wrapper to align with DASH paper-style usage:
      - load MKP instances
      - evaluate a spec on a subset
      - update / retrieve PLR library

    This wrapper intentionally does NOT depend on DASH engine code; it is a plugin-level component.
    """
    data_path: str
    res_path: Optional[str] = None
    plr_cfg: Optional[Dict[str, Any]] = None
    mode: str = "train"  # "train" | "eval"

    def __post_init__(self):
        self.instances: List[MKPInstance] = read_mknapcb(self.data_path, res_path=self.res_path)
        cfg = PLRConfig(**(self.plr_cfg or {"enabled": False}))
        self.plr = PLRLibrary(cfg)

        # fit groups on a sample of instances if enabled
        if cfg.enabled and self.instances:
            self.plr.fit_groups(self.instances[: min(200, len(self.instances))])

    def evaluate(
        self,
        spec: Union[ACOSpec, Dict[str, Any]],
        indices: Optional[Sequence[int]] = None,
        *,
        guide_module: Optional[Any] = None,
        update_plr_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(spec, dict):
            spec_obj = ACOSpec.from_json(spec, n_items=self.instances[0].n if self.instances else 1)
        else:
            spec_obj = spec

        if indices is None:
            indices = range(len(self.instances))

        gaps = []
        tldrs = []
        metas = []
        for idx in indices:
            inst = self.instances[int(idx)]
            gap, meta = solve_instance_with_spec_gap(inst, spec_obj, guide_module=guide_module, return_meta=True)
            gaps.append(float(gap))
            tldrs.append(float(meta.get("tLDR_traj", 0.0)))
            metas.append(meta)

            if self.plr.cfg.enabled and self.mode == "train" and update_plr_payload is not None:
                # update library with this candidate evaluated on this inst
                self.plr.update(inst, payload=update_plr_payload, meta=meta)

        out = {
            "mean_gap": sum(gaps) / max(len(gaps), 1),
            "mean_tLDR": sum(tldrs) / max(len(tldrs), 1),
            "num_eval": int(len(gaps)),
            "metas": metas,
        }
        return out

    def retrieve_warm_start(self, inst: MKPInstance, k: int = 1) -> Tuple[int, List[Dict[str, Any]]]:
        return self.plr.retrieve(inst, k=k)
