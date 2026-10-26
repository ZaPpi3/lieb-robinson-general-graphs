"""2026-10-26 follow-up: the round-2 result showed nominal rewiring
probability p is a poor independent variable at these small N (14-18),
since p only sets the *expected* number of rewired edges (~N*p, order 1-2
at p=0.1), and individual realizations vary hugely in how much they
actually differ from the ring at fixed p. This script measures the real
per-instance disruption directly instead: regenerates every (N, p, seed)
graph already used in graph_quench_results.json (ws_graph() is fully
deterministic given its seed, verified reproducible before writing this),
compares its edge set against the unrewired ring's, and joins that
disruption count onto the existing speed_fits.json rows by (N, p, seed).
No new quantum simulation needed -- graph generation alone is cheap.
"""
import json

import networkx as nx

import graph_quench_test as t


def disruption(N, k, p, seed):
    """Number of edges in the realized graph that are not present in the
    unrewired ring (same convention nx.watts_strogatz_graph uses: starts
    from the ring, replaces some edges). Returns (n_different_edges,
    total_edges, degree_min, degree_max)."""
    if p == 0.0:
        g = t.ring_graph(N)
    else:
        g = t.ws_graph(N, k, p, seed)
    ring_edges = set(frozenset(e) for e in t.ring_graph(N).edges())
    g_edges = set(frozenset(e) for e in g.edges())
    n_diff = len(g_edges - ring_edges)
    degs = [d for _, d in g.degree()]
    return n_diff, g.number_of_edges(), min(degs), max(degs)


def main():
    results = json.load(open("graph_quench_results.json"))
    fits = json.load(open("speed_fits.json"))

    # One disruption measurement per distinct (N, graph_type, k, p, seed)
    disruption_by_key = {}
    for g in results:
        if g["graph_type"] != "ws":
            continue
        N = g["N"]
        k, p = g["graph_param"]
        seed = g["seed"]
        key = (N, k, p, seed)
        if key not in disruption_by_key:
            disruption_by_key[key] = disruption(N, k, p, seed)

    # Join onto every fit row
    joined = []
    for r in fits:
        if r["graph_type"] != "ws":
            continue
        N = r["N"]
        k, p = r["graph_param"]
        seed = r["seed"]
        key = (N, k, p, seed)
        n_diff, n_edges, deg_min, deg_max = disruption_by_key[key]
        row = dict(r)
        row["n_edges_changed"] = n_diff
        row["deg_min"] = deg_min
        row["deg_max"] = deg_max
        joined.append(row)

    json.dump(joined, open("speed_fits_with_disruption.json", "w"), indent=1)
    print(f"wrote speed_fits_with_disruption.json ({len(joined)} rows, "
          f"{len(disruption_by_key)} distinct graph instances)")

    # Summary: R^2 grouped by actual edges-changed count, not nominal p
    import numpy as np
    xkey = "t_arrival_th0.0001"
    rows = [r for r in joined if r["xkey"] == xkey]
    by_ndiff = {}
    for r in rows:
        by_ndiff.setdefault(r["n_edges_changed"], []).append(r)
    print(f"\n{'n_edges_changed':16s} {'n_fits':7s} {'mean_R2':8s} {'median_R2':10s} {'n_neg_speed':12s}")
    for nd in sorted(by_ndiff):
        rs = by_ndiff[nd]
        r2s = np.array([r["r2"] for r in rs])
        speeds = np.array([r["speed"] for r in rs])
        n_neg = int(np.sum(speeds < 0))
        print(f"{nd:16d} {len(rs):7d} {r2s.mean():8.3f} {np.median(r2s):10.3f} {n_neg:12d}")


if __name__ == "__main__":
    main()
