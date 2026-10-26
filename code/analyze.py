"""Analyze graph_quench_results.json: for each (graph instance, i0), fit a
ballistic model t_arrival = r/v + c (equivalently r = v*t + c) and extract
the speed v. Then compare v across different i0 within the same graph
(same realization) and across realizations/densities, and check whether v
correlates with the starting vertex's own degree -- the specific concern
the general-graph Lieb-Robinson literature raises.
"""
import json
import numpy as np


def fit_speed(r, t):
    """r = v*t + c, least squares; returns v, c, R^2."""
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    v, c = coef
    pred = A @ coef
    ss_res = np.sum((r - pred) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(v), float(c), float(r2)


def main():
    results = json.load(open("graph_quench_results.json"))
    xkeys = ["t_halfmax", "t_arrival_th0.0001", "t_arrival_th1e-05", "t_arrival_th1e-06"]

    per_run = []
    for graph in results:
        for run in graph["runs"]:
            rows = run["rows"]
            for xkey in xkeys:
                r_vals, t_vals = [], []
                for row in rows:
                    t = row.get(xkey)
                    if t is None or t <= 0:
                        continue
                    r_vals.append(row["r"])
                    t_vals.append(t)
                if len(r_vals) < 3:
                    continue
                v, c, r2 = fit_speed(np.array(r_vals), np.array(t_vals))
                per_run.append(dict(
                    N=graph["N"], graph_type=graph["graph_type"], graph_param=graph["graph_param"],
                    seed=graph["seed"], n_edges=graph["n_edges"], i0=run["i0"],
                    degree_i0=run["degree_i0"], xkey=xkey, speed=v, intercept=c, r2=r2,
                ))

    print(f"{'type':10s} {'param':6s} {'N':3s} {'seed':5s} {'i0':4s} {'deg':4s} {'xvar':14s} "
          f"{'speed':8s} {'R2':6s}")
    for row in per_run:
        print(f"{row['graph_type']:10s} {str(row['graph_param']):6s} {row['N']:<3d} "
              f"{row['seed']:<5d} {row['i0']:<4d} {row['degree_i0']:<4d} {row['xkey']:14s} "
              f"{row['speed']:<8.3f} {row['r2']:<6.3f}")

    json.dump(per_run, open("speed_fits.json", "w"), indent=1)
    print("\nwrote speed_fits.json")

    # Summary: spread of speed (half-max metric only, R^2 > 0.8) within each
    # (N, graph_type, graph_param) group, across all i0 and seeds.
    print("\n=== speed spread by group (half-max metric, R^2 > 0.8 fits only) ===")
    groups = {}
    for row in per_run:
        if row["xkey"] != "t_halfmax" or row["r2"] <= 0.8:
            continue
        key = (row["N"], row["graph_type"], row["graph_param"])
        groups.setdefault(key, []).append(row)
    for key, rows in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        speeds = np.array([r["speed"] for r in rows])
        degs = np.array([r["degree_i0"] for r in rows])
        if len(speeds) < 2:
            print(f"{key}: n={len(speeds)}, too few fits to assess spread")
            continue
        corr = np.corrcoef(degs, speeds)[0, 1] if np.std(degs) > 0 else float("nan")
        print(f"N={key[0]:2d} type={key[1]:9s} param={str(key[2]):6s} n_fits={len(speeds):3d} "
              f"speed_mean={speeds.mean():.3f} speed_std={speeds.std():.3f} "
              f"speed_cv={speeds.std()/speeds.mean():.3f} corr(degree,speed)={corr:.3f}")


if __name__ == "__main__":
    main()
