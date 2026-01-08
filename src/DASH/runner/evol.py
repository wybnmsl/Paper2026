import random

from ..engine.dash import DASH


class EVOL:
    """Thin runner wrapper for DASH engine.

    This file intentionally keeps the legacy class name EVOL to minimize changes
    in external scripts, while the underlying engine is renamed to DASH.
    """

    def __init__(self, paras, problem, select, manage):
        self.paras = paras
        self.problem = problem
        self.select = select
        self.manage = manage

    def run(self):
        # keep deterministic by default (matches the original runner behaviour)
        if getattr(self.paras, "exp_use_seed", False):
            seed = getattr(self.paras, "exp_seed", 2024)
            random.seed(seed)

        engine = DASH(self.paras, self.problem, self.select, self.manage)
        return engine.run()
