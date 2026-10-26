"""2026-10-26: the disordered-chain MI test. The clean critical/gapped
chain's MI graph (run_mi_graph.py) turned out structurally incapable of
asymmetric irregularity (translational symmetry only allows adding whole
distance-shells). This breaks that symmetry for real: a random on-site
field h_i ~ Uniform(-W, W), i.i.d. per site (mi_graph.build_heisenberg_
disordered), giving a genuinely asymmetric ground-state MI matrix.

Scoping pass (disordered_explore.py, disordered_explore2.py; see NOTES.md)
found W=1.0, threshold=0.10 is a reliable window: connected at every N in
10..18 and every seed tried, with real degree variation (2-4, vs the
clean MI graph's uniform degree 4) -- genuine disorder-driven irregularity
in a real ground state, not a hand-built defect.

10 disorder realizations (seeds) per N, matching the defect_sweep's
statistics (NOTES.md: n=30 i0-level fits per N was enough to resolve the
N-dependence question cleanly). Pool(2) throughout, same safety margin as
defect_sweep (3 i0/job at N=18 needs Pool(2), not Pool(3), per the round-2
memory crisis)."""
import json
from multiprocessing import Pool

import networkx as nx

import mi_graph as m
import graph_quench_test as t

OUT = "disordered_mi_results.json"
thresholds = [1e-4, 1e-5, 1e-6]
T_max_by_N = {10: 14.0, 12: 16.0, 14: 18.0, 16: 20.0, 18: 22.0}
n_steps = 400
W = 1.0
MI_THRESH = 0.10
N_SEEDS = 10

jobs = []
for N in [10, 12, 14, 16, 18]:
    for seed in range(N_SEEDS):
        H, h = m.build_heisenberg_disordered(N, W, seed)
        E0, gs = m.ground_state(H)
        I = m.mi_matrix(gs, N)
        g = m.mi_threshold_graph(I, MI_THRESH)
        if not nx.is_connected(g):
            print(f"N={N} seed={seed}: disconnected at thresh={MI_THRESH}, skipping", flush=True)
            continue
        meta = dict(W=W, mi_thresh=MI_THRESH, seed=seed, E0=float(E0), n_edges=g.number_of_edges())
        i0_list = t.pick_i0_by_degree(g)
        jobs.append((N, "mi", (g, meta), seed, i0_list, T_max_by_N[N], n_steps, thresholds, "disordered_mi"))

if __name__ == "__main__":
    print(f"{len(jobs)} jobs total", flush=True)
    results = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(t._worker, jobs):
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, "total:", len(results), flush=True)
