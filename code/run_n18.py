"""Run just the N=18 portion of graph_quench_test.py's sweep (10 jobs: 5
p-values x 2 seeds), lost to a harness memory-pressure kill before they
started in the first run. Appends to the existing graph_quench_results.json
rather than overwriting the N=14/16 results already in it.
"""
import json
from multiprocessing import Pool

import graph_quench_test as t

OUT = "graph_quench_results.json"
thresholds = [1e-4, 1e-5, 1e-6]

N = 18
T_max = 22.0
n_steps = 400
i0_list = [0, N // 2]
jobs = []
for p in [0.0, 0.1, 0.3, 0.6, 1.0]:
    tag = "ws_ring" if p == 0.0 else f"ws_p{p:g}"
    for seed in [1, 2]:
        jobs.append((N, "ws", (2, p), seed, i0_list, T_max, n_steps, thresholds, tag))

if __name__ == "__main__":
    existing = json.load(open(OUT))
    print(f"appending to {len(existing)} existing entries", flush=True)
    with Pool(3) as pool:
        for r in pool.imap_unordered(t._worker, jobs):
            existing.append(r)
            json.dump(existing, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, "total entries:", len(existing), flush=True)
