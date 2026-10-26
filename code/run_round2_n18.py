"""Remaining N=18 portion of run_round2.py (15 jobs: 5 p-values x 3 seeds),
proactively killed at Pool(4) when available memory dropped steadily from
7.6GB to 4.3GB over a few minutes with no sign of plateauing (heavier than
round 1's N=18 jobs: 3 starting vertices per job now instead of 2, from
the degree-aware i0 selection added 2026-10-26). Rerun at Pool(2) instead
of Pool(4) or round 1's Pool(3), since 3 i0 per job is 50% more work per
job than round 1's successful Pool(3) run. Appends to the existing
graph_quench_results.json; all N=14/16 round-2 jobs already completed and
are untouched.
"""
import json
from multiprocessing import Pool

import graph_quench_test as t

OUT = "graph_quench_results.json"
thresholds = [1e-4, 1e-5, 1e-6]
T_max = 22.0
n_steps = 400
N = 18

jobs = []
for p in [0.0, 0.1, 0.15, 0.2, 0.3]:
    tag = "r2_ws_ring" if p == 0.0 else f"r2_ws_p{p:g}"
    for seed in [11, 12, 13]:
        jobs.append((N, "ws", (2, p), seed, "by_degree", T_max, n_steps, thresholds, tag))

if __name__ == "__main__":
    existing = json.load(open(OUT))
    print(f"appending to {len(existing)} existing entries, {len(jobs)} new N=18 jobs", flush=True)
    with Pool(2) as pool:
        for r in pool.imap_unordered(t._worker, jobs):
            existing.append(r)
            json.dump(existing, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, "total entries:", len(existing), flush=True)
