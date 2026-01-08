# src/eoh/utils/profiler.py
from __future__ import annotations
from typing import Dict, Any, List, Optional

class Profiler:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def record(self, *, node: str, elapsed_sec: float, error: Optional[str]):
        self.records.append({
            "node": node,
            "elapsed_sec": float(elapsed_sec),
            "ok": error is None,
            "error": error or ""
        })

    def dumps(self) -> Dict[str, Any]:
        total = sum(r["elapsed_sec"] for r in self.records)
        slow = sorted(self.records, key=lambda x: x["elapsed_sec"], reverse=True)[:5]
        return {
            "total_elapsed_sec": float(total),
            "records": self.records,
            "top5_slowest": slow,
        }
