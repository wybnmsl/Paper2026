from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .utils.readCVRPLib import load_instances, CVRPInstance
from .aco.spec import ACOSpec, default_aco_spec, from_json
from .aco.aco_run import solve_instance_with_spec
from .utils.utils import profile_instance


class CVRP_ACO_Problem:
    """
    CVRP plugin wrapper for DASH.

    Responsibilities:
      - load instances from VRPLIB/CVRPLIB format
      - provide default spec schema (ACOSpec)
      - evaluate a spec on a set of instances
      - expose instance profile phi(x) for PLR
    """

    name: str = "CVRP_ACO"

    def __init__(self, data_root: Union[str, Path]):
        self.data_root = str(data_root)

    # ---------------------------- data ----------------------------

    def load(self, case_names: Sequence[str], require_sol: bool = False) -> List[CVRPInstance]:
        return load_instances(self.data_root, list(case_names), require_sol=require_sol)

    # ---------------------------- spec ----------------------------

    @staticmethod
    def default_spec() -> ACOSpec:
        return default_aco_spec()

    @staticmethod
    def parse_spec(spec: Union[ACOSpec, Dict[str, Any], None]) -> ACOSpec:
        if spec is None:
            return default_aco_spec()
        if isinstance(spec, ACOSpec):
            return spec
        return from_json(spec)

    # ---------------------------- evaluation ----------------------------

    def evaluate(
        self,
        instances: Sequence[CVRPInstance],
        spec: Union[ACOSpec, Dict[str, Any], None],
        *,
        time_limit_s: Optional[float] = None,
        return_trace: bool = True,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Evaluate a (possibly LLM-generated) spec on a batch of instances.
        Returns (gaps, metas).
        """
        spec_obj = self.parse_spec(spec)
        gaps: List[float] = []
        metas: List[Dict[str, Any]] = []
        for inst in instances:
            g, meta = solve_instance_with_spec(
                inst,
                spec_obj,
                time_limit_s=time_limit_s,
                return_trace=return_trace,
            )
            gaps.append(float(g))
            metas.append(meta)
        return gaps, metas

    # ---------------------------- PLR ----------------------------

    @staticmethod
    def profile(inst: CVRPInstance):
        return profile_instance(inst)


def make_problem(data_root: Union[str, Path]) -> CVRP_ACO_Problem:
    return CVRP_ACO_Problem(data_root=data_root)
