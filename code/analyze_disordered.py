"""Analyze disordered_mi_results.json: fit speed and R^2 per (instance,
i0, arrival-time metric), same method as analyze.py, then summarize per N
and compare directly against the clean MI-graph result (R^2=0.85-1.00)
and the defect-sweep baseline (R^2~0.45-0.54, flat across N)."""
import json
import numpy as np

from analyze import fit_speed

results = json.load(open("disordered_mi_results.json"))
xkeys = ["t_halfmax", "t_arrival_th0.0001", "t_arrival_th1e-05", "t_arrival_th1e-06"]

per_run = []
for graph in results:
    for run in graph["runs"]:
        rows = run["rows"]
        for xkey in xkeys:
            r_vals, t_vals = [], []
            for row in rows:
                tv = row.get(xkey)
                if tv is None or tv <= 0:
                    continue
                r_vals.append(row["r"])
                t_vals.append(tv)
            if len(r_vals) < 3:
                continue
            v, c, r2 = fit_speed(np.array(r_vals), np.array(t_vals))
            per_run.append(dict(N=graph["N"], seed=graph["seed"], n_edges=graph["n_edges"],
                                 i0=run["i0"], degree_i0=run["degree_i0"], xkey=xkey,
                                 speed=v, r2=r2))

json.dump(per_run, open("disordered_speed_fits.json", "w"), indent=1)

print("=== per-N summary, half-max metric, all fits ===")
for N in sorted(set(r["N"] for r in per_run)):
    rows = [r for r in per_run if r["N"] == N and r["xkey"] == "t_halfmax"]
    r2s = np.array([r["r2"] for r in rows])
    print(f"N={N:2d}  n_fits={len(r2s):3d}  R2 mean={r2s.mean():.3f} median={np.median(r2s):.3f} "
          f"std={r2s.std():.3f}  min={r2s.min():.3f} max={r2s.max():.3f}  "
          f"frac>0.8={np.mean(r2s>0.8):.2f}")

print("\n=== per-N summary, threshold 1e-4 metric, all fits ===")
for N in sorted(set(r["N"] for r in per_run)):
    rows = [r for r in per_run if r["N"] == N and r["xkey"] == "t_arrival_th0.0001"]
    r2s = np.array([r["r2"] for r in rows])
    print(f"N={N:2d}  n_fits={len(r2s):3d}  R2 mean={r2s.mean():.3f} median={np.median(r2s):.3f} "
          f"std={r2s.std():.3f}  min={r2s.min():.3f} max={r2s.max():.3f}  "
          f"frac>0.8={np.mean(r2s>0.8):.2f}")

print("\n=== degree(i0) effect, half-max metric, pooled across N ===")
rows = [r for r in per_run if r["xkey"] == "t_halfmax"]
degs = sorted(set(r["degree_i0"] for r in rows))
for d in degs:
    sub = [r["r2"] for r in rows if r["degree_i0"] == d]
    print(f"degree_i0={d}: n={len(sub):3d}  R2 mean={np.mean(sub):.3f}")

print("\n=== n_edges (irregularity amount) vs R2, half-max metric ===")
rows = [r for r in per_run if r["xkey"] == "t_halfmax"]
extras = sorted(set(r["n_edges"] - r["N"] for r in rows))
for e in extras:
    sub = [r["r2"] for r in rows if (r["n_edges"] - r["N"]) == e]
    print(f"extra_edges_vs_ring={e:3d}: n={len(sub):3d}  R2 mean={np.mean(sub):.3f}")
