import os, re
from pathlib import Path
from collections import Counter

def get_hdr(p: Path):
    typ = ew_type = ew_fmt = dim = None
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(600):
            line = f.readline()
            if not line: break
            s = line.strip()
            u = s.upper()
            if u in ("NODE_COORD_SECTION","EDGE_WEIGHT_SECTION","EOF"):
                break
            if ":" in s:
                k,v = s.split(":",1)
                k = k.strip().upper()
                v = v.strip()
                if k=="TYPE": typ=v.upper()
                elif k=="DIMENSION":
                    m=re.search(r"\d+", v); dim=int(m.group()) if m else None
                elif k=="EDGE_WEIGHT_TYPE": ew_type=v.upper()
                elif k=="EDGE_WEIGHT_FORMAT": ew_fmt=v.upper()
    return typ, ew_type, ew_fmt, dim

root = Path(os.path.expanduser(input("tsplib_root: ").strip())).resolve()
tsp_files = sorted(root.rglob("*.tsp"))

c = Counter()
bad = 0
for p in tsp_files:
    typ, ew_type, ew_fmt, dim = get_hdr(p)
    if dim is None:
        bad += 1
        continue
    c[(typ or "", ew_type or "", ew_fmt or "")] += 1

print("=== FORMAT COUNTS (TYPE, EDGE_WEIGHT_TYPE, EDGE_WEIGHT_FORMAT) ===")
for k,v in c.most_common():
    print(v, k)
print("files:", len(tsp_files), "no_dim:", bad)
