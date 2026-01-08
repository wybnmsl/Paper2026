"""BPP (Bin Packing Problem) plugin using GOA-style search.

This package is a DASH task plugin: it contains problem utilities, a GOA solver, and
optional Profiled Library Retrieval (PLR) for warm-starting under distribution shift.
"""

from .problem import BPPGOAProblem

__all__ = ["BPPGOAProblem"]
