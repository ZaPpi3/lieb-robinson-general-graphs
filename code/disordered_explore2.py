"""Follow-up scoping: N=12,14 at W=1.0 found a window (thresh~0.10-0.12)
giving connected, genuinely irregular (degree 2-4) graphs -- not just the
exact ring, not disconnected. Check this holds at N=10,16,18 (the sizes
the production sweep needs) and over more seeds, before committing any
real quench compute."""
import numpy as np
import networkx as nx

import mi_graph as m

for N in [10, 16, 18]:
    for W in [1.0]:
        for seed in range(4):
            H, h = m.build_heisenberg_disordered(N, W, seed)
            E0, gs = m.ground_state(H)
            I = m.mi_matrix(gs, N)
            for thresh in [0.08, 0.10, 0.12, 0.15]:
                g = m.mi_threshold_graph(I, thresh)
                connected = nx.is_connected(g)
                diam = nx.diameter(g) if connected else None
                degs = sorted(d for _, d in g.degree())
                extra = g.number_of_edges() - N
                print(f"N={N:2d} seed={seed} thresh={thresh:5.2f} n_edges={g.number_of_edges():3d} "
                      f"extra_vs_ring={extra:2d} connected={connected!s:5s} diam={diam} "
                      f"degree_range=[{degs[0]},{degs[-1]}]", flush=True)
