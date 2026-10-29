"""2026-10-26: the MI-derived graph test, the gap flagged as most
important after the single-defect result (NOTES.md). Everything tested
so far (ring, Watts-Strogatz, ring-plus-one-defect) is a synthetic proxy
graph. The substrate the companion paper actually defines
(`Papers\\Baseline\\companion.tex`, `GitHub\\gravitational-baseline-
principle\\gravity.tex`) has its connectivity A_ij set directly by the
ground state's own mutual information, not by any assumed topology. This
builds that graph for real: compute the exact ground-state MI matrix of
the critical (and, as a control, gapped/dimerized) Heisenberg chain,
threshold it to a sparse adjacency, and hand the result to the same
quench-propagation pipeline already used throughout this project.

Self-contained (own Hamiltonian/ground-state/MI code), following the
project's own established lesson (test_f_causal_structure.py's
docstring, heisenberg-substrate-numerical-tests) that genuine complex
Pauli operators, not any gauge-traded real-matrix trick, are needed for
results to be trustworthy and portable across files.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx

_PX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_PY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_PZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))


def _site_op(op, site, N):
    result = sp.eye(1, format="csr", dtype=complex)
    for j in range(N):
        result = sp.kron(result, op if j == site else sp.eye(2, format="csr", dtype=complex), format="csr")
    return result


def build_heisenberg_dimerized(N, delta):
    """H = sum_i J_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1}),
    J_i = 1 + delta*(-1)^i, periodic. delta=0: critical (uniform)
    chain. delta>0: gapped dimerized control."""
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        Ji = 1.0 + delta * ((-1) ** i)
        H = H + Ji * (_site_op(_PX, i, N) @ _site_op(_PX, j, N)
                       + _site_op(_PY, i, N) @ _site_op(_PY, j, N)
                       + _site_op(_PZ, i, N) @ _site_op(_PZ, j, N))
    return H


def build_heisenberg_disordered(N, W, seed):
    """H = sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1}) + sum_i h_i Sz_i,
    uniform J=1 ring coupling, h_i ~ Uniform(-W, W) i.i.d. per site. Unlike
    build_heisenberg_dimerized's staggered J_i (which still has a 2-site
    translation symmetry), a per-site random field breaks translational
    symmetry completely and independently at every site: the real test
    of whether a genuinely asymmetric MI structure (not just a shell
    pattern) ever shows up in a real ground state."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(-W, W, size=N)
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        H = H + (_site_op(_PX, i, N) @ _site_op(_PX, j, N)
                  + _site_op(_PY, i, N) @ _site_op(_PY, j, N)
                  + _site_op(_PZ, i, N) @ _site_op(_PZ, j, N))
    for i in range(N):
        H = H + h[i] * _site_op(_PZ, i, N)
    return H, h


def ground_state(H):
    evals, evecs = eigsh(H, k=1, which="SA")
    return float(evals[0]), evecs[:, 0]


def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    return float(-np.sum(evals * np.log(evals)))


def partial_trace(psi, keep, N):
    psi = psi.reshape([2] * N)
    trace_over = [i for i in range(N) if i not in keep]
    psi = np.transpose(psi, keep + trace_over)
    psi = psi.reshape(2 ** len(keep), 2 ** len(trace_over))
    return psi @ psi.conj().T


def mi_matrix(psi, N):
    """All-pairs mutual information matrix for a pure state psi."""
    S = [von_neumann_entropy(partial_trace(psi, [i], N)) for i in range(N)]
    I = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            Sij = von_neumann_entropy(partial_trace(psi, [i, j], N))
            I[i, j] = I[j, i] = S[i] + S[j] - Sij
    return I


def mi_threshold_graph(I, threshold):
    """Graph with an edge (i,j) iff I[i,j] > threshold. No topology
    assumed in advance: whatever structure results is whatever the
    ground state's own correlations actually produce."""
    N = I.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if I[i, j] > threshold:
                g.add_edge(i, j)
    return g


if __name__ == "__main__":
    # Exploration pass: compute the real MI matrix for a few N, look at
    # its decay structure, and scan thresholds to find one that gives a
    # connected, reasonably sparse graph (not the ER sweep's mistake of
    # collapsing to diameter 2-3; not so sparse it disconnects).
    for N in [12, 14, 16]:
        for delta, label in [(0.0, "critical"), (0.5, "gapped (delta=0.5)")]:
            H = build_heisenberg_dimerized(N, delta)
            E0, gs = ground_state(H)
            I = mi_matrix(gs, N)
            offdiag = I[np.triu_indices(N, k=1)]
            print(f"N={N} {label}: E0={E0:.4f}  I range [{offdiag.min():.4f}, {offdiag.max():.4f}]  "
                  f"nn_I={I[0,1]:.4f}  nnn_I={I[0,2]:.4f}  far_I={I[0, N//2]:.4f}")
            for thresh in [0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]:
                g = mi_threshold_graph(I, thresh)
                connected = nx.is_connected(g)
                diam = nx.diameter(g) if connected else None
                extra = g.number_of_edges() - N  # edges beyond a ring's N
                print(f"    thresh={thresh:5.2f}  n_edges={g.number_of_edges():3d}  "
                      f"connected={connected!s:5s}  diam={diam}  extra_vs_ring={extra}")


def mi_knn_graph(I, k):
    """Graph with an edge (i,j) iff j is among i's k strongest MI
    partners OR i is among j's k strongest MI partners (standard
    symmetric k-nearest-neighbour graph construction from manifold
    learning). Guarantees minimum degree k by construction, unlike a
    global threshold, which is exactly the property a global threshold
    lacks when the MI distribution has a large gap after the single
    strongest bond per site but near-degenerate values below it (the
    gapped/dimerized chain's own case, NOTES.md 2026-10-26)."""
    N = I.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(N))
    for i in range(N):
        row = I[i, :].copy()
        row[i] = -np.inf
        top_k = np.argsort(-row)[:k]
        for j in top_k:
            g.add_edge(i, int(j))
    return g
