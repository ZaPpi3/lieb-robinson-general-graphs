"""2026-10-26: the MI-derived graph test. Everything tested before this
(ring, Watts-Strogatz, ring-plus-one-defect) was a synthetic proxy graph;
this builds the graph the companion paper's substrate actually specifies
-- connectivity set directly by a real ground state's mutual information,
not any assumed topology.

Exploration pass (mi_graph.py, run directly, see NOTES.md) found: the
critical chain's MI decays smoothly (power-law, Calabrese-Cardy, as
established throughout this whole portfolio) but *every pair at the same
chord distance has identical MI* by translational symmetry. So
thresholding an MI matrix from a translationally-symmetric ground state
can only ever capture whole distance-shells at once -- it is structurally
incapable of producing the kind of single, isolated, asymmetric defect
the synthetic ring_with_one_rewire() tests used. Two regimes exist: (1)
threshold between the nearest- and next-nearest-neighbour MI values gives
exactly the ring (already known: R^2~0.97); (2) threshold between the
next-nearest and third-neighbour MI values gives the ring plus a full
next-nearest-neighbour shell -- a denser but still perfectly regular
(every vertex degree 4, circulant) graph, not yet tested. This runs the
quench pipeline on regime (2) across several N, to test directly whether
translational symmetry alone (not sparsity) is what keeps a single speed
well-defined.

The gapped/dimerized chain's MI graph was found to be *disconnected* at
any reasonable threshold (only the strong dimer bonds survive; weak
bonds' MI is far too small) -- a distinct, already-complete negative
result in its own right, documented in NOTES.md without spending quench
compute on a graph where most arrival times would be undefined anyway.
"""
import json
from multiprocessing import Pool

import mi_graph as m
import graph_quench_test as t

OUT = "mi_graph_results.json"
thresholds = [1e-4, 1e-5, 1e-6]
T_max_by_N = {10: 14.0, 12: 16.0, 14: 18.0, 16: 20.0, 18: 22.0}
n_steps = 400


def build_ring_plus_nnn(N):
    """Threshold set at the midpoint between the 2nd- and 3rd-neighbour MI
    values for this specific N (computed fresh each time, not hardcoded,
    since the exact MI values depend weakly on N)."""
    H = m.build_heisenberg_dimerized(N, 0.0)
    E0, gs = m.ground_state(H)
    I = m.mi_matrix(gs, N)
    i2, i3 = I[0, 2], I[0, 3]
    thresh = 0.5 * (i2 + i3)
    g = m.mi_threshold_graph(I, thresh)
    return g, dict(E0=E0, threshold=float(thresh), I_nn=float(I[0, 1]),
                    I_2=float(i2), I_3=float(i3), n_edges=g.number_of_edges())


jobs = []
graph_cache = {}
for N in [10, 12, 14, 16, 18]:
    g, meta = build_ring_plus_nnn(N)
    import networkx as nx
    assert nx.is_connected(g), f"N={N} ring+nnn graph is disconnected, unexpected"
    print(f"N={N}: {meta}, diameter={nx.diameter(g)}", flush=True)
    # The graph is deterministic and fully regular (circulant, every
    # vertex degree 4) -- there is no "seed" to vary. Instead test several
    # explicitly different starting vertices directly, which is itself the
    # real check here: translational symmetry predicts they should all
    # give near-identical results, the same sanity check the plain ring
    # control used throughout this project, now applied to the denser
    # regular graph.
    # Keep i0 count modest at the largest N (memory lessons from earlier
    # today: 3 i0/job already needed Pool(2) at N=18; this uses only 5
    # total graph instances so there is little to gain from parallelism
    # anyway -- run sequentially below (Pool(1)) rather than add a second
    # risk axis on top of a new graph type.
    n_i0 = 3 if N >= 16 else 4
    i0_list = sorted(set([0, N // 4, N // 2, (3 * N) // 4][:n_i0]))
    jobs.append((N, "mi", (g, meta), 0, i0_list, T_max_by_N[N], n_steps, thresholds, "mi_ring_nnn"))

if __name__ == "__main__":
    print(f"{len(jobs)} jobs total", flush=True)
    results = []
    with Pool(1) as pool:
        for r in pool.imap_unordered(t._worker, jobs):
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, "total:", len(results), flush=True)
