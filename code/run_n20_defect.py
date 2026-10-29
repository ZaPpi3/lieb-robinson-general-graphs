"""2026-10-29: pushes the single-defect N-dependence result past N=18,
the specific gap main.tex's Limitations names explicitly ("it remains
possible that qualitatively new behaviour appears beyond the sizes
reached here"). N=20 was abandoned once already (a single CPU job used
8GB RSS and did not finish in 5+ minutes); made tractable this time by
`gpu_propagator.py`'s GPU Lanczos/Krylov trajectory, validated against
the trusted `scipy.sparse.linalg.expm_multiply` baseline at N=14 and
N=16 (machine-precision agreement, `validate_gpu_propagator.py`) before
being used here for anything.

Reuses `graph_quench_test.py`'s established, unmodified utility
functions (ring_with_one_rewire, pick_i0_by_degree, build_graph_heisenberg,
site_sz_signs, arrival_times, halfmax_times) for everything except the
expensive trajectory step itself, which is the only part that changes.
10 disorder realizations (same statistics convention as the original
N=10-18 defect_sweep_results.json), 3 starting vertices each via
pick_i0_by_degree, matching that sweep's own design exactly so the two
are directly comparable.
"""
import json
import time

import numpy as np

import graph_quench_test as t
from gpu_propagator import gpu_krylov_trajectory

OUT = "defect_sweep_n20_results.json"
N = 20
T_max = 24.0
n_steps = 400
thresholds = [1e-4, 1e-5, 1e-6]
N_SEEDS = 10


def run_one(seed):
    t_start = time.time()
    g = t.ring_with_one_rewire(N, seed)
    i0_list = t.pick_i0_by_degree(g)
    degrees = dict(g.degree())
    H = t.build_graph_heisenberg(N, g).astype(complex)
    dim = 2 ** N
    ts = np.linspace(0.0, T_max, n_steps, endpoint=True)

    runs = []
    for i0 in i0_list:
        idx_bg = 0
        idx_def = idx_bg ^ (1 << i0)
        psi_bg = t.one_hot(idx_bg, dim)
        psi_def = t.one_hot(idx_def, dim)
        traj_bg = gpu_krylov_trajectory(H, psi_bg, ts, krylov_dim=30)
        traj_def = gpu_krylov_trajectory(H, psi_def, ts, krylov_dim=30)
        signs = t.site_sz_signs(N)
        Mbg = (np.abs(traj_bg) ** 2) @ signs
        Mdef = (np.abs(traj_def) ** 2) @ signs
        D = Mdef - Mbg
        absD = np.abs(D)

        arr = t.arrival_times(absD, ts, thresholds)
        t_half = t.halfmax_times(absD, ts)
        import networkx as nx
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
    print(f"N={N} seed={seed} n_edges={g.number_of_edges()} done in {elapsed:.1f}s", flush=True)
    return dict(label="defect1_n20", N=N, graph_type="ring1", graph_param=None,
                seed=seed, n_edges=g.number_of_edges(), degrees=degrees,
                T_max=T_max, n_steps=n_steps, thresholds=thresholds, runs=runs)


if __name__ == "__main__":
    results = []
    for seed in range(N_SEEDS):
        r = run_one(seed)
        results.append(r)
        json.dump(results, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, "total:", len(results), flush=True)
