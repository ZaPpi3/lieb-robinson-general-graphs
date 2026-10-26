"""2026-10-26: direct attack on the N-dependence question left open by
the edge-disruption analysis (NOTES.md): is a single defect's
disruptiveness (mean R^2 roughly: 0.25 at N=14, 0.60 at N=16, 0.89 at
N=18, but those N=18 numbers came from only n=4 lucky WS draws) a real,
N-dependent trend, or just thin-statistics noise?

Uses ring_with_one_rewire() (added to graph_quench_test.py) instead of
Watts-Strogatz: exactly one edge changed from the ring, by construction,
not by chance, so seeds can be spent directly on getting many independent
single-defect instances per N rather than hoping a random p lands there.

N=20 was tried and abandoned: a single N=20 job (3 i0, dim=2^20) used
8GB RSS and had not finished after 5+ minutes CPU time -- too expensive
for the statistics it would buy at this stage. Scoped instead to
N=10..18, where good statistics (10 seeds/N = 30 i0-level fits per N,
3x the N=18 sample the open question was flagged on) are affordable.
"""
import json
from multiprocessing import Pool

import graph_quench_test as t

OUT = "defect_sweep_results.json"
thresholds = [1e-4, 1e-5, 1e-6]
# T_max scaled to roughly cross the ring's own diameter at each N (same
# logic as T_max_by_N elsewhere); n_steps held at 400 throughout.
T_max_by_N = {10: 14.0, 12: 16.0, 14: 18.0, 16: 20.0, 18: 22.0}
n_steps = 400
seeds_by_N = {10: 10, 12: 10, 14: 10, 16: 10, 18: 10}

jobs = []
for N in [10, 12, 14, 16, 18]:
    for seed in range(seeds_by_N[N]):
        jobs.append((N, "ring1", None, seed, "by_degree", T_max_by_N[N], n_steps, thresholds, "defect1"))

if __name__ == "__main__":
    print(f"{len(jobs)} jobs total", flush=True)
    results = []
    # N=18 needs Pool(2) with 3 i0/job (NOTES.md compute notes). jobs are
    # listed N-ascending per seed, so by the time N=18 jobs are reached,
    # imap_unordered could run several concurrently -- stay at the
    # confirmed-safe Pool(2) throughout rather than risk Pool(3) across
    # multiple simultaneous N=18 jobs.
    with Pool(2) as pool:
        for r in pool.imap_unordered(t._worker, jobs):
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, "total:", len(results), flush=True)
