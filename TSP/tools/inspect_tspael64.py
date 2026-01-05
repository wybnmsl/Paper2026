# tools/inspect_tspael64_v2.py
# -*- coding: utf-8 -*-
import argparse
import pickle
import numpy as np

def is_symmetric(a, tol=1e-9):
    return a.ndim == 2 and a.shape[0] == a.shape[1] and np.allclose(a, a.T, atol=tol)

def zero_diag(a, tol=1e-9):
    return a.ndim == 2 and np.allclose(np.diag(a), 0.0, atol=tol)

def integerish(a, tol=1e-9):
    if not np.issubdtype(a.dtype, np.number):
        return False
    return np.nanmax(np.abs(a - np.rint(a))) <= tol

def guess_tour_base(tour):
    arr = np.asarray(tour, dtype=int).ravel()
    if arr.size == 0: return "unknown"
    mn, mx = int(arr.min()), int(arr.max())
    # 常见：0-based => [0, n-1]; 1-based => [1, n]
    if mn == 0: return "0-based"
    if mn == 1: return "1-based"
    return f"min={mn}, max={mx}"

def brief_arr(a, k=3):
    return f"shape={a.shape}, dtype={a.dtype}, head=\n{np.array2string(a.reshape(-1)[:min(k,a.size)], precision=3, suppress_small=True)}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="TrainingData/TSPAEL64.pkl")
    ap.add_argument("--show", type=int, default=2, help="show first K instances per field")
    args = ap.parse_args()

    print(f"[load] {args.path}")
    with open(args.path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"[warn] top-level type is {type(data)} (not dict).")
        print("       Keys seen before: ", getattr(data, "keys", lambda: [])())
        return

    keys = list(data.keys())
    print(f"[info] dict keys: {keys}")

    coord = data.get("coordinate", None)
    distm = data.get("distance_matrix", None)
    opttour = data.get("optimal_tour", None)
    cost = data.get("cost", None)

    # coordinate
    if coord is not None:
        print("\n[coordinate]")
        print(" type:", type(coord))
        if isinstance(coord, (list, tuple)):
            print(" len:", len(coord))
            for i in range(min(args.show, len(coord))):
                a = np.asarray(coord[i])
                print(f"  - idx {i}: {brief_arr(a)}")
                if a.ndim == 2:
                    print(f"    looks_like_coords? cols={a.shape[1] if a.ndim==2 else 'NA'}")
        else:
            a = np.asarray(coord)
            print(" singleton:", brief_arr(a))

    # distance_matrix
    if distm is not None:
        print("\n[distance_matrix]")
        print(" type:", type(distm))
        if isinstance(distm, (list, tuple)):
            print(" len:", len(distm))
            for i in range(min(args.show, len(distm))):
                a = np.asarray(distm[i])
                print(f"  - idx {i}: {brief_arr(a)}")
                if a.ndim == 2:
                    print(f"    square={a.shape[0]==a.shape[1]}, symmetric={is_symmetric(a)}, zero_diag={zero_diag(a)}, integerish={integerish(a)}")
        else:
            a = np.asarray(distm)
            print(" singleton:", brief_arr(a))

    # optimal_tour
    if opttour is not None:
        print("\n[optimal_tour]")
        print(" type:", type(opttour))
        if isinstance(opttour, (list, tuple)):
            print(" len:", len(opttour))
            for i in range(min(args.show, len(opttour))):
                t = np.asarray(opttour[i])
                print(f"  - idx {i}: shape={t.shape}, dtype={t.dtype}, base={guess_tour_base(t)}")
                if t.size > 0:
                    print(f"    head: {t[:min(10,t.size)]}")
        else:
            t = np.asarray(opttour)
            print(" singleton:", f"shape={t.shape}, base={guess_tour_base(t)}")

    # cost
    if cost is not None:
        c = np.asarray(cost)
        print("\n[cost]")
        print(" type:", type(cost), "| shape:", c.shape, "| dtype:", c.dtype)
        print(" head:", c[:min(10, c.size)])

    # 简要签名
    sig = {
        "coordinate_type": type(coord).__name__ if coord is not None else None,
        "distance_matrix_type": type(distm).__name__ if distm is not None else None,
        "optimal_tour_type": type(opttour).__name__ if opttour is not None else None,
        "cost_type": type(cost).__name__ if cost is not None else None,
    }
    print("\n[signature guess]:", sig)

if __name__ == "__main__":
    main()
