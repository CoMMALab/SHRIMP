# Planner result files (`*.npz`) for validating paths in VAMP

Written by `ik_benchmark/solve_prrax.py` (prrax) and `ik_benchmark/crab_rrtc.py` (CRAB), same layout:

- `ik_benchmark/results/<chain>_prrax.npz`
- `ik_benchmark/results/<chain>_crab_rrtc.npz`  (paths are saved by the current version; rerun if missing)

`<chain>`: `mbm` = Panda (7 DOF, tool `panda_grasptarget`), `mbm_baxter` = Baxter (14 DOF).

## Keys (P = problems, F = families)

| key | shape / type | meaning |
|---|---|---|
| `solved` | (P,) bool | planner reported a connection |
| `path_len` | (P,) int32 | waypoints in the path; 0 if unsolved |
| `paths` | (P, 512, dof) float32 | row `i` = path of problem `i`; rows `>= path_len[i]` are zero padding. Radians. |
| `cost` | (P,) float | L2 length of the path (sum of norms of consecutive differences); NaN if unsolved |
| `category` | (P,) int | index into `category_names` |
| `category_names` | (F,) str | MBM family names |
| `lat_ms` | (30,) | single-problem latency samples (ms) |
| `batch_ms` | scalar | total time of the batched run (ms) |
| `kernel_ms` | (30,) | prrax only: its reported kernel time |

## Path conventions

- `paths[i, :path_len[i]]` goes **start -> goal**: first row = `start`, last row = goal (`goals[0]`), within ~1e-3.
- Joint order = VAMP `problems.pkl` order. Panda: 7 joints. Baxter: `left_s0 .. left_w2, right_s0 .. right_w2` (14).
- Rows are the planner's **tree nodes**, not an interpolated path. Validate the straight segments between consecutive
  rows (e.g. with VAMP's `validate` at its resolution, or interpolate at ~1/32 rad).
- Only problems that passed the CRAB goal filter appear (Baxter: 1470 problems), so P is smaller than the full VAMP set.

## Matching a row to a VAMP problem

Row `i` of the results file = row `i` of the targets file
`ik_benchmark/targets/<robot>_<chain>.npz` (`<robot>` = `panda` or `baxter`; e.g. `baxter_mbm_baxter.npz`).
Its keys `category` (index into `category_names`) and `problem_index` (the pickle's `index` field) identify the problem:

```python
import numpy as np, pickle

t = np.load("ik_benchmark/targets/baxter_mbm_baxter.npz")
r = np.load("ik_benchmark/results/mbm_baxter_prrax.npz")        # or ..._crab_rrtc.npz
data = pickle.load(open("ik_benchmark/targets/baxter_problems.pkl", "rb"))["problems"]   # VAMP's problems.pkl
names = [str(n) for n in t["category_names"]]

for i in range(len(r["solved"])):
    if not (r["solved"][i] and r["path_len"][i] > 0):
        continue
    fam, idx = names[t["category"][i]], int(t["problem_index"][i])
    prob = next(p for p in data[fam] if p.get("index") == idx)    # dict: start, goals, sphere, box, cylinder, valid, ...
    path = r["paths"][i, :r["path_len"][i]].astype(float)         # (L, dof), start -> goal
    # sanity: np.allclose(path[0], prob["start"], atol=1e-3); np.allclose(path[-1], prob["goals"][0], atol=1e-3)
    # then build the VAMP environment from prob (vamp.problem_dict_to_vamp) and validate every segment path[k] -> path[k+1]
```

The targets file also stores per problem `start`, `q_star` (= `goals[0]`), and `lo`/`hi` joint limits, so the pickle is
only needed for the obstacles.

## What to report

Per family and overall: number of solved problems, number whose whole path VAMP accepts (all segments collision-free,
within limits, endpoints match), and the first failing segment (self vs environment) for the invalid ones.
Do the same for both files to compare planners under VAMP's collision model.
