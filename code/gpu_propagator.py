"""2026-10-29: GPU-accelerated time evolution for the N>18 push flagged
in main.tex's Limitations ("N=20 was attempted and abandoned as too
computationally expensive"). Diagnosed first (not guessed): profiling
showed Hamiltonian construction is fast (~7s at N=20) and NOT the
bottleneck; `scipy.sparse.linalg.expm_multiply` itself is (70s for just
20 of the needed 400 output steps over a fifth of the needed T_max).

cupy does not provide an `expm_multiply` equivalent (checked directly:
`cupyx.scipy.sparse.linalg` has no such function), so this implements a
Lanczos/Krylov-subspace propagator on the GPU instead, the standard
technique large-scale exact-diagonalization codes use for exactly this
problem (time-evolving a large sparse Hermitian system). This is not a
new method, just a new implementation of one, which is exactly why it
must be validated against the already-trusted `scipy.sparse.linalg.
expm_multiply` before being trusted for anything, not assumed correct
because the underlying algorithm is standard.
"""
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp
from scipy.linalg import expm as scipy_expm


def gpu_krylov_trajectory(H_cpu_sparse, psi0, ts, krylov_dim=40, reorth_tol=1e-12):
    """Propagate psi0 under exp(-i H t) for every t in ts (must start at
    0 and increase monotonically), using a GPU Lanczos/Krylov method.
    H must be Hermitian. Returns a (len(ts), dim) complex128 numpy array,
    matching scipy.sparse.linalg.expm_multiply's own (num, dim) output
    convention for start=0, stop=ts[-1], num=len(ts), endpoint=True."""
    H_gpu = cpsp.csr_matrix(H_cpu_sparse.astype(complex))
    dim = H_gpu.shape[0]
    psi = cp.asarray(psi0, dtype=cp.complex128)
    traj = np.zeros((len(ts), dim), dtype=complex)
    traj[0] = cp.asnumpy(psi)
    for idx in range(1, len(ts)):
        dt = float(ts[idx] - ts[idx - 1])
        psi = _lanczos_step(H_gpu, psi, dt, krylov_dim, reorth_tol)
        traj[idx] = cp.asnumpy(psi)
    return traj


def _lanczos_step(H, psi, dt, m, reorth_tol):
    dim = psi.shape[0]
    beta0 = cp.linalg.norm(psi)
    V = cp.zeros((m + 1, dim), dtype=cp.complex128)
    V[0] = psi / beta0
    alphas = np.zeros(m)
    betas = np.zeros(m)
    m_eff = m
    for j in range(m):
        w = H @ V[j]
        alpha = float(cp.real(cp.vdot(V[j], w)))
        alphas[j] = alpha
        w = w - alpha * V[j]
        if j > 0:
            w = w - betas[j - 1] * V[j - 1]
        # full reorthogonalization against every prior Krylov vector:
        # the small, cheap, numerically safe choice at this Krylov
        # dimension (m~40), standard practice to prevent the basis
        # from losing orthogonality after a handful of steps.
        for i in range(j + 1):
            w = w - cp.vdot(V[i], w) * V[i]
        beta = float(cp.linalg.norm(w))
        if beta < reorth_tol:
            m_eff = j + 1
            break
        betas[j] = beta
        V[j + 1] = w / beta
    T = (np.diag(alphas[:m_eff])
         + np.diag(betas[:m_eff - 1], 1)
         + np.diag(betas[:m_eff - 1], -1))
    expT_col0 = scipy_expm(-1j * dt * T)[:, 0]
    coeffs = cp.asarray(expT_col0, dtype=cp.complex128)
    psi_new = beta0 * (V[:m_eff].T @ coeffs)
    return psi_new
