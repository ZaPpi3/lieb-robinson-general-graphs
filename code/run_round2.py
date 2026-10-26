"""Round 2 (2026-10-26, see NOTES.md): narrower p grid around the
transition zone found in round 1 (p=0.1 held up well, p=0.3 had already
collapsed), degree-aware i0 selection (min/median/max degree vertex per
graph, controlling for the secondary degree effect found in the
2026-10-26 mechanism check), and more seeds. Scoped to the transition
zone (p in {0.0, 0.1, 0.15, 0.2, 0.3}) rather than re-running the
already-clearly-collapsed p=0.6/1.0 points, which add little at this
compute cost. Appends to the existing graph_quench_results.json.
"""
import json
from multiprocessing import Pool

import graph_quench_test as t

OUT = "graph_quench_results.json"
thresholds = [1e-4, 1e-5, 1e-6]
T_max_by_N = {14: 18.0, 16: 20.0, 18: 22.0}
n_steps = 400

jobs = []
for N in [14, 16, 18]:
    T_max = T_max_by_N[N]
    for p in [0.0, 0.1, 0.15, 0.2, 0.3]:
        tag = "r2_ws_ring" if p == 0.0 else f"r2_ws_p{p:g}"
        for seed in [11, 12, 13]:
            jobs.append((N, "ws", (2, p), seed, "by_degree", T_max, n_steps, thresholds, tag))

if __name__ == "__main__":
    existing = json.load(open(OUT))
    print(f"appending to {len(existing)} existing entries, {len(jobs)} new jobs", flush=True)
    # N=18 jobs need low parallelism (see NOTES.md compute notes: Pool(8)
    # thrashed the WSL VM's 23GB RAM; Pool(3) stayed healthy). N=14/16 are
    # cheap enough that this only costs a little time, not worth a separate
    # pool size per N for a 45-job run.
    with Pool(4) as pool:
        for r in pool.imap_unordered(t._worker, jobs):
            existing.append(r)
            json.dump(existing, open(OUT, "w"), indent=1)
    print("done, wrote", OUT, "total entries:", len(existing), flush=True)
