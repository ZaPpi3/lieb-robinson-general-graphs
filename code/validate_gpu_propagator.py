"""2026-10-29: validates gpu_propagator.py's Lanczos/Krylov trajectory
against scipy.sparse.linalg.expm_multiply (already trusted throughout
this whole project) at a size where both are fast, before the GPU
version is used for anything at N>18. Must be run before trusting any
N>18 result from gpu_propagator.py.
"""
import time
import numpy as np
from scipy.sparse.linalg import expm_multiply

from graph_quench_test import ring_graph, build_graph_heisenberg, one_hot
from gpu_propagator import gpu_krylov_trajectory

N = 14
T_max = 18.0
n_steps = 400

g = ring_graph(N)
H = build_graph_heisenberg(N, g).astype(complex)
dim = 2 ** N
ts = np.linspace(0.0, T_max, n_steps, endpoint=True)

idx_bg = 0
idx_def = idx_bg ^ (1 << 0)
psi_bg = one_hot(idx_bg, dim)
psi_def = one_hot(idx_def, dim)

print(f"N={N}, dim={dim}, n_steps={n_steps}, T_max={T_max}")

t0 = time.time()
traj_bg_cpu = expm_multiply(-1j * H, psi_bg, start=0.0, stop=T_max, num=n_steps, endpoint=True)
traj_def_cpu = expm_multiply(-1j * H, psi_def, start=0.0, stop=T_max, num=n_steps, endpoint=True)
print(f"CPU (scipy) baseline: {time.time() - t0:.1f}s")

for krylov_dim in [20, 30, 40]:
    t0 = time.time()
    traj_bg_gpu = gpu_krylov_trajectory(H, psi_bg, ts, krylov_dim=krylov_dim)
    traj_def_gpu = gpu_krylov_trajectory(H, psi_def, ts, krylov_dim=krylov_dim)
    gpu_time = time.time() - t0

    err_bg = np.max(np.abs(traj_bg_gpu - traj_bg_cpu))
    err_def = np.max(np.abs(traj_def_gpu - traj_def_cpu))
    # also compare the actual physical observable this project uses
    # (the Sz-difference signal), not just the raw wavefunction, since
    # that is what every quench result in this project is built from
    idx = np.arange(dim)
    bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
    signs = 1 - 2 * bits
    M_bg_cpu = (np.abs(traj_bg_cpu) ** 2) @ signs
    M_def_cpu = (np.abs(traj_def_cpu) ** 2) @ signs
    M_bg_gpu = (np.abs(traj_bg_gpu) ** 2) @ signs
    M_def_gpu = (np.abs(traj_def_gpu) ** 2) @ signs
    D_cpu = M_def_cpu - M_bg_cpu
    D_gpu = M_def_gpu - M_bg_gpu
    err_D = np.max(np.abs(D_gpu - D_cpu))

    print(f"krylov_dim={krylov_dim}: GPU time={gpu_time:.1f}s  "
          f"max|psi_gpu-psi_cpu|={max(err_bg, err_def):.3e}  "
          f"max|D_gpu-D_cpu|={err_D:.3e}  speedup={(time.time()-t0)/1:.1f}")
