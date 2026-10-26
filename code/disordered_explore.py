"""2026-10-26: scoping pass for the disordered-chain MI test, before
spending any quench compute. Question: does a random on-site field (which
breaks translational symmetry completely, unlike the dimerized chain's
2-site-periodic staggering) produce a genuinely asymmetric MI matrix, and
is there a threshold that gives a connected, irregular (not uniform-degree)
graph out of it -- the thing the clean critical/gapped chain structurally
could not produce?
"""
import numpy as np
import networkx as nx

import mi_graph as m

for N in [12, 14]:
    for W in [0.5, 1.0, 2.0, 4.0]:
        for seed in [0, 1]:
            H, h = m.build_heisenberg_disordered(N, W, seed)
            E0, gs = m.ground_state(H)
            I = m.mi_matrix(gs, N)
            nn = np.array([I[i, (i + 1) % N] for i in range(N)])
            print(f"N={N} W={W} seed={seed}: nn_I mean={nn.mean():.4f} std={nn.std():.4f} "
                  f"range=[{nn.min():.4f},{nn.max():.4f}]  (std/mean={nn.std()/nn.mean():.3f})", flush=True)
            for thresh in [0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02]:
                g = m.mi_threshold_graph(I, thresh)
                connected = nx.is_connected(g)
                diam = nx.diameter(g) if connected else None
                degs = sorted(d for _, d in g.degree())
                print(f"    thresh={thresh:5.2f} n_edges={g.number_of_edges():3d} "
                      f"connected={connected!s:5s} diam={diam} degree_range=[{degs[0]},{degs[-1]}]", flush=True)
