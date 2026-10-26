# A Single Causal Speed on Irregular Graphs

**Synthetic Defects, Symmetric Ground States, and Real Disorder**

Paul William Jarvis · Independent Researcher, United Kingdom
[mrpaulwjarvis@gmail.com](mailto:mrpaulwjarvis@gmail.com) · ORCID: [0009-0009-8933-857X](https://orcid.org/0009-0009-8933-857X)

Draft: [main.pdf](main.pdf) · LaTeX source: [main.tex](main.tex)

---

## Summary

The Lieb-Robinson bound guarantees a finite propagation speed for local quantum lattice systems, and this has been proven to hold on general, irregular graphs, not just regular lattices. It does not guarantee the stronger thing most physical reasoning assumes: that a *single, sharply-defined* speed, not just a finite one, describes the causal front. This repository numerically tests that stronger claim by exact diagonalization of the Heisenberg model, using a real-time local-defect quench (the same protocol used to measure causal fronts in quantum gas microscope experiments), on three qualitatively different sources of graph irregularity.

**The finding is a spectrum, not a binary answer.** A single hand-engineered asymmetric defect is the worst case: a near-immediate cliff in fit quality from the ring's own R²≈0.97-0.98 down to R²≈0.41-0.50, present already at the smallest possible disruption and flat across every system size tested (N=10-18). A graph built directly from a real, translationally-symmetric ground state's own mutual information is, by contrast, not fragile at all (R²=0.85-1.00): symmetry forces any such graph to add or remove whole distance-shells, never a single isolated defect. Breaking that symmetry for real, with physically ordinary disorder (a random on-site field), produces a third, genuinely intermediate result (R²=0.66-0.80) that degrades smoothly with the amount of disorder rather than collapsing in one step.

## Status

This is a working draft, not yet submitted or peer-reviewed. Code, data, and reproduction instructions are in `code/`.

## Repository structure

- `main.tex` / `main.pdf`: the manuscript.
- `code/`: all simulation and analysis scripts.
  - `graph_quench_test.py`: core simulation (Hamiltonian construction, quench protocol, graph generators including the ring and the single-rewire defect).
  - `mi_graph.py`: mutual-information computation and MI-threshold graph construction (critical, gapped, and disordered chains).
  - `run_defect_sweep.py`, `run_mi_graph.py`, `run_disordered_mi.py`: production sweeps for the three main results.
  - `analyze.py`, `analyze_disordered.py`, `edge_disruption.py`: fit extraction and analysis.
  - `*.json`: raw results from every production run.

## Reproducing

```bash
cd code
python3 run_defect_sweep.py      # single-defect N-dependence sweep (Sec. 4)
python3 run_mi_graph.py          # symmetric MI-graph test (Sec. 5)
python3 run_disordered_mi.py     # disordered MI-graph test (Sec. 6)
python3 analyze.py               # fits for defect sweep / ring control
python3 analyze_disordered.py    # fits for the disordered sweep
python3 edge_disruption.py       # edge-disruption cliff measurement
```

Requires `numpy`, `scipy`, `networkx`. N=18 jobs use the full 2^18-dimensional Hilbert space and are the most expensive; see in-script comments for memory-safe parallelism settings.

## License

This work is licensed under CC BY 4.0: see the [`LICENSE`](LICENSE) file for details. In short: share and adapt freely, including commercially, provided appropriate credit is given.
