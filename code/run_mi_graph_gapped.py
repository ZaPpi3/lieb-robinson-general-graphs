"""2026-10-29: the gapped-chain MI disconnection fix flagged as open in
NOTES.md ("whether a different observable, or a different graph-
construction rule entirely, would give a connected and testable graph
for the gapped phase is left open", also main.tex Sec. Limitations).

Diagnosis first (disordered_explore-style scoping, not guessed):
for the gapped/dimerized chain (delta=0.5), every site's mutual
information is dominated by a single, huge intra-dimer value (~1.3),
with everything else clustered in a narrow, near-degenerate band two
orders of magnitude smaller. There is no threshold window between
"only the dimer bonds" (the N/2-disconnected-pairs failure already
documented) and "essentially everything at once": a global threshold
is structurally the wrong tool here, not just mistuned.

Fix: a symmetric k-nearest-neighbour graph (`mi_graph.mi_knn_graph`,
edge (i,j) iff j in i's top-k MI partners or vice versa), the standard
manifold-learning fix for exactly this failure mode, since it guarantees
minimum degree k without needing a numeric gap anywhere in the
distribution. Verified before running any quench compute:
  - k=2 recovers the *exact* plain ring (identical edge set), already
    tested in Sec. 3 of the paper; not rerun here, no new information.
  - k=3 gives a new, genuinely different, still perfectly regular graph
    (ring plus one extra "skip-3" edge per dimer, degree 3 uniform).
  - k=4 is an ILL-POSED cut: sites 2 and 10 (for N=12) are tied to
    machine precision (0.011868637658161596, identical to 16 significant
    figures, a real consequence of the chain's own 2-site translational
    symmetry), so k=4 arbitrarily includes one and excludes the other via
    argsort's tie-breaking, producing a spurious degree-[4,5] asymmetry
    that is a tie-breaking artifact, not real physics. Confirmed by
    checking k=5 and k=6, which both restore exact regularity once the
    tied pair is included or excluded together. SKIPPED, not tested here;
    reporting it as a real result would have been a real mistake.
  - k=5 gives another new, perfectly regular graph (degree 5 uniform).

Both k=3 and k=5 are tested here, each predicted (by the same shell/
symmetry argument already established for the critical chain's MI-shell
graph) to stay robust and non-fragile, since the chain's own residual
2-site translational symmetry still forbids any single-site asymmetric
defect, exactly as it does for k=2's ring. This directly extends that
argument from the critical phase to the gapped phase, and from a
threshold-based construction to a k-NN-based one, for the first time.
"""
import json
from multiprocessing import Pool

import networkx as nx

import mi_graph as m
import graph_quench_test as t

OUT = "mi_graph_gapped_results.json"
thresholds = [1e-4, 1e-5, 1e-6]
T_max_by_N = {10: 14.0, 12: 16.0, 14: 18.0, 16: 20.0, 18: 22.0}
n_steps = 400
DELTA = 0.5  # gapped/dimerized chain control, matches mi_graph.py's existing convention
K_VALUES = [3, 5]

jobs = []
for N in [10, 12, 14, 16, 18]:
    H = m.build_heisenberg_dimerized(N, DELTA)
    E0, gs = m.ground_state(H)
    I = m.mi_matrix(gs, N)
    for k in K_VALUES:
        g = m.mi_knn_graph(I, k)
        assert nx.is_connected(g), f"N={N} k={k} gapped-chain kNN graph is disconnected, unexpected"
        degs = sorted(d for _, d in g.degree())
        assert degs[0] == degs[-1], f"N={N} k={k} not regular (degree {degs[0]}-{degs[-1]}), check for a tie at this k before trusting it"
        meta = dict(delta=DELTA, k=k, E0=float(E0), n_edges=g.number_of_edges(), degree=degs[0])
        print(f"N={N}: k={k}, {meta}, diameter={nx.diameter(g)}", flush=True)
        n_i0 = 3 if N >= 16 else 4
        i0_list = sorted(set([0, N // 4, N // 2, (3 * N) // 4][:n_i0]))
        jobs.append((N, "mi", (g, meta), 0, i0_list, T_max_by_N[N], n_steps, thresholds, f"mi_gapped_k{k}"))

if __name__ == "__main__":
    print(f"{len(jobs)} jobs total", flush=True)
    results = []
    with Pool(1) as pool:
        for r in pool.imap_unordered(t._worker, jobs):
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, "total:", len(results), flush=True)
