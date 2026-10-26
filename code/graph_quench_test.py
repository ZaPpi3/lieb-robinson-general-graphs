"""Does a single, well-defined Lieb-Robinson speed survive on an irregular
graph, not just the regular 1D ring lieb-robinson-causal-test and its
published Test F tested? Full design rationale in ../NOTES.md.

Method: a local-defect quench, same trick as lieb-robinson-causal-test
(two exact time evolutions from nearby computational basis states, one a
fixed background pattern, one with a single extra bit flipped at i0;
differencing cancels the background and isolates the defect's own causal
signal). Unlike the ring case, no Neel-order reference is needed or used
here: a bipartite 2-coloring doesn't generally exist for an irregular
graph, but the differencing trick never actually required Neel order
specifically, only that both trajectories start from nearby computational
basis states (any fixed bit pattern works; Cheneau et al. 2012 use a
Mott-insulator product state for the same reason). Background = all-zero
bitstring here, for simplicity.

Arrival time t_arrival(r) is extracted exactly as in the ring case, but r
is now the graph-theoretic (BFS shortest-path) distance from i0, computed
once per graph with networkx, not a ring hop-count.

The diagnostic: for a REGULAR ring, extracting the slope of a straight-
line fit of graph-distance r against t_arrival gives the same speed
regardless of which site i0 the quench starts from (translational
symmetry guarantees this, and is used here only as a sanity check that
the pipeline recovers the known result). For an IRREGULAR graph there is
no such symmetry; the question is whether the extracted "speed" still
comes out consistent across different starting vertices (especially
min-degree vs max-degree ones) and across different random realizations
at the same density, or whether it varies in a way that tracks local
graph structure, which would mean no single Lieb-Robinson speed is well
defined for that graph at all.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import time

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

_PX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_PY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_PZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))


def _site_op(op, site, N):
    result = sp.eye(1, format="csr", dtype=complex)
    for j in range(N):
        result = sp.kron(result, op if j == site else sp.eye(2, format="csr", dtype=complex), format="csr")
    return result


def build_graph_heisenberg(N, graph, J=1.0):
    """H = J * sum_{(i,j) in graph.edges} (Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j),
    genuine (eigenvalue +-1) Pauli operators, uniform coupling on every
    edge. Matches the convention already verified in
    heisenberg-substrate-numerical-tests/code/test_f_causal_structure.py,
    generalized from a ring's fixed bond list to an arbitrary edge set."""
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    ops = {}
    for i in range(N):
        ops[i] = (_site_op(_PX, i, N), _site_op(_PY, i, N), _site_op(_PZ, i, N))
    for i, j in graph.edges():
        sxi, syi, szi = ops[i]
        sxj, syj, szj = ops[j]
        H = H + J * (sxi @ sxj + syi @ syj + szi @ szj)
    return H


def ring_graph(N):
    g = nx.cycle_graph(N)
    return g


def er_graph(N, mean_degree, seed):
    p = min(mean_degree / (N - 1), 1.0)
    rng = np.random.default_rng(seed)
    for attempt in range(50):
        g = nx.gnp_random_graph(N, p, seed=int(rng.integers(1_000_000)))
        if nx.is_connected(g):
            return g
    raise RuntimeError(f"could not generate a connected ER graph at N={N}, mean_degree={mean_degree}")


def ws_graph(N, k, p, seed):
    """Watts-Strogatz: start from a ring where each node connects to its k
    nearest neighbours, then rewire each edge with probability p. p=0 is
    exactly the ring (plus extra neighbour shells if k>2); p=1 is close to
    a random graph. Chosen over raw Erdos-Renyi so diameter/irregularity
    can be dialed in gradually instead of collapsing to a tiny diameter as
    soon as density rises (see NOTES.md, 2026-10-25 entry: ER graphs at
    N=10-14 had only 2-4 distinct hop-distances, too few to fit a speed
    from at all)."""
    rng = np.random.default_rng(seed)
    for attempt in range(50):
        g = nx.watts_strogatz_graph(N, k, p, seed=int(rng.integers(1_000_000)))
        if nx.is_connected(g):
            return g
    raise RuntimeError(f"could not generate a connected WS graph at N={N}, k={k}, p={p}")


def site_sz_signs(N):
    dim = 2 ** N
    idx = np.arange(dim)
    bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
    return 1 - 2 * bits


def one_hot(idx, dim):
    v = np.zeros(dim, dtype=complex)
    v[idx] = 1.0
    return v


def arrival_times(absD, ts, thresholds):
    n_steps, N = absD.shape
    out = {th: np.full(N, np.nan) for th in thresholds}
    for th in thresholds:
        crossed = absD > th
        for j in range(N):
            hits = np.where(crossed[:, j])[0]
            if hits.size:
                out[th][j] = ts[hits[0]]
    return out


def halfmax_times(absD, ts):
    n_steps, N = absD.shape
    peak = absD.max(axis=0)
    out = np.full(N, np.nan)
    for j in range(N):
        if peak[j] <= 0:
            continue
        hits = np.where(absD[:, j] >= 0.5 * peak[j])[0]
        if hits.size:
            out[j] = ts[hits[0]]
    return out


def ring_with_one_rewire(N, seed):
    """Exactly one edge changed from the ring, by construction, not by
    chance. 2026-10-26: added to directly test whether a single defect's
    disruptiveness (found to cut mean R^2 roughly in half, N=14-18) grows,
    shrinks, or saturates with N, which the random WS draws couldn't
    answer cleanly since landing on exactly 1 changed edge was a matter
    of luck, not something that could be requested at a chosen N.

    Removes one ring edge (u, u+1), adds a new edge from u to a random
    other non-adjacent vertex w (half-edge rewiring, same convention
    Watts-Strogatz itself uses). Degree sequence is always exactly one
    N-1 the same, one vertex with degree 1 (the stranded endpoint v),
    one with degree 3 (w), identically reproducing the "one real defect"
    structure found in the round-2 data, but guaranteed rather than
    hoped-for."""
    rng = np.random.default_rng(seed)
    g = ring_graph(N)
    u = int(rng.integers(N))
    v = (u + 1) % N
    g.remove_edge(u, v)
    candidates = [w for w in range(N) if w != u and not g.has_edge(u, w)]
    w = candidates[int(rng.integers(len(candidates)))]
    g.add_edge(u, w)
    return g


def pick_i0_by_degree(g):
    """Return the min-degree, a median-degree, and the max-degree vertex.
    Added 2026-10-26 after finding (NOTES.md) that starting-vertex degree
    has a secondary but real effect on fit quality (leaf vertices give
    worse fits than degree-2 ones, on average) -- controls for this
    directly instead of picking i0 by fixed index position, which leaves
    the realized degree to chance."""
    degs = dict(g.degree())
    ordered = sorted(degs, key=lambda v: degs[v])
    lo = ordered[0]
    hi = ordered[-1]
    mid = ordered[len(ordered) // 2]
    return sorted(set([lo, mid, hi]))


def run_graph(N, graph_type, graph_param, seed, i0_list, T_max, n_steps, thresholds, label):
    t_start = time.time()
    if graph_type == "ring":
        g = ring_graph(N)
    elif graph_type == "er":
        g = er_graph(N, graph_param, seed)
    elif graph_type == "ws":
        k, p = graph_param
        g = ws_graph(N, k, p, seed)
    elif graph_type == "ring1":
        g = ring_with_one_rewire(N, seed)
    elif graph_type == "mi":
        g, mi_meta = graph_param  # (pre-built networkx graph, JSON-safe metadata) passed in directly
        graph_param = mi_meta  # don't try to JSON-serialize the graph object itself below
    else:
        raise ValueError(graph_type)

    if i0_list == "by_degree":
        i0_list = pick_i0_by_degree(g)

    degrees = dict(g.degree())
    H = build_graph_heisenberg(N, g)
    Hc = H.astype(complex)
    dim = 2 ** N
    ts = np.linspace(0.0, T_max, n_steps, endpoint=True)

    runs = []
    for i0 in i0_list:
        idx_bg = 0
        idx_def = idx_bg ^ (1 << i0)
        psi_bg = one_hot(idx_bg, dim)
        psi_def = one_hot(idx_def, dim)
        traj_bg = expm_multiply(-1j * Hc, psi_bg, start=0.0, stop=T_max, num=n_steps, endpoint=True)
        traj_def = expm_multiply(-1j * Hc, psi_def, start=0.0, stop=T_max, num=n_steps, endpoint=True)
        signs = site_sz_signs(N)
        Mbg = (np.abs(traj_bg) ** 2) @ signs
        Mdef = (np.abs(traj_def) ** 2) @ signs
        D = Mdef - Mbg
        absD = np.abs(D)

        arr = arrival_times(absD, ts, thresholds)
        t_half = halfmax_times(absD, ts)
        dist = nx.single_source_shortest_path_length(g, i0)

        rows = []
        for j in range(N):
            if j == i0:
                continue
            r = dist.get(j)
            if r is None:
                continue
            v_half = t_half[j]
            row = dict(j=int(j), r=int(r), degree_j=int(degrees[j]),
                       t_halfmax=None if np.isnan(v_half) else float(v_half))
            for th in thresholds:
                v = arr[th][j]
                row[f"t_arrival_th{th:g}"] = None if np.isnan(v) else float(v)
            rows.append(row)
        runs.append(dict(i0=int(i0), degree_i0=int(degrees[i0]), rows=rows))

    elapsed = time.time() - t_start
    print(f"{label} N={N:2d} type={graph_type} param={graph_param} seed={seed} "
          f"n_edges={g.number_of_edges()} done in {elapsed:.1f}s", flush=True)
    return dict(label=label, N=N, graph_type=graph_type, graph_param=graph_param, seed=seed,
                n_edges=g.number_of_edges(), degrees=degrees, T_max=T_max, n_steps=n_steps,
                thresholds=thresholds, runs=runs)


def _worker(job):
    return run_graph(*job)


if __name__ == "__main__":
    from multiprocessing import Pool

    OUT = "graph_quench_results.json"
    thresholds = [1e-4, 1e-5, 1e-6]
    jobs = []
    # Round 1 (ER at mean degree 3-7) is abandoned: at N=10-14 those graphs
    # had only 2-4 distinct hop-distances, too few to fit a speed from (see
    # NOTES.md). Round 2: Watts-Strogatz with k=2 (start from a ring, each
    # edge independently rewired with probability p), which keeps enough
    # path length to resolve a speed (diameter ~8-15 at N=16-18, checked
    # directly before committing to this sweep) while still letting
    # irregularity be dialed in gradually via p, rather than jumping
    # straight to a small-diameter regime the way raw ER did.
    # Timing calibration before committing (see NOTES.md): N=18 with 4-5 i0
    # and T_max=2N per job did not finish a single job in 5+ minutes and was
    # killed. N=16 with 2 i0, T_max=20 took 45s; N=18 with 2 i0, T_max=22
    # took 254s. Scaled down to 2 i0 per job and a tighter time window
    # (enough to cross the measured diameter, not 2x the system size).
    T_max_by_N = {14: 18.0, 16: 20.0, 18: 22.0}
    n_steps = 400
    for N in [14, 16, 18]:
        T_max = T_max_by_N[N]
        i0_list = [0, N // 2]
        for p in [0.0, 0.1, 0.3, 0.6, 1.0]:
            tag = "ws_ring" if p == 0.0 else f"ws_p{p:g}"
            for seed in [1, 2]:
                jobs.append((N, "ws", (2, p), seed, i0_list, T_max, n_steps, thresholds, tag))

    results = []
    with Pool(12) as p:
        for r in p.imap_unordered(_worker, jobs):
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, flush=True)
