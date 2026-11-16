"""Validate planner paths (results/<chain>_{prrax,crab_rrtc}.npz, see PATHS_NPZ.md) under VAMP's collision model.

    python validate_in_vamp.py --robot panda  --file results/mbm_prrax.npz     --targets targets
    python validate_in_vamp.py --robot baxter --file results/mbm_baxter_crab_rrtc.npz --targets targets [--n 200] [--out report.json]

Per solved problem: endpoints match start/goal (1e-3), every row within the joint limits, and every segment between
consecutive rows passes vamp.<robot>.validate_motion. For the first failing segment it walks the segment at 1/32 rad
and reports whether the first bad configuration is a SELF collision (invalid with an empty environment) or an
ENVIRONMENT one. Needs a vamp build with the robot (panda / baxter)."""
import argparse, json, pickle
from collections import defaultdict
import numpy as np
import vamp

ap = argparse.ArgumentParser()
ap.add_argument("--robot", choices=("panda", "baxter"), required=True)
ap.add_argument("--file", required=True)
ap.add_argument("--targets", default="targets", help="dir with <robot>_<chain>.npz and <robot>_problems.pkl")
ap.add_argument("--n", type=int, default=0, help="only the first n solved problems (0 = all)")
ap.add_argument("--out", default=None, help="write per-problem json here")
a = ap.parse_args()

chain = "mbm" if a.robot == "panda" else "mbm_baxter"
t = np.load(f"{a.targets}/{a.robot}_{chain}.npz"); r = np.load(a.file)
data = pickle.load(open(f"{a.targets}/{a.robot}_problems.pkl", "rb"))["problems"]
names = [str(x) for x in t["category_names"]]
mod = getattr(vamp, a.robot); lo = np.array(mod.lower_bounds()); hi = np.array(mod.upper_bounds()); empty = vamp.Environment()
STEP = 1 / 32

def first_bad(path, env):
    """First configuration along the path (1/32 rad steps) that is invalid in env, with its segment index."""
    for k in range(len(path) - 1):
        d = path[k + 1] - path[k]; n = max(1, int(np.ceil(np.abs(d).max() / STEP)))
        for s in range(n + 1):
            q = path[k] + d * s / n
            if not mod.validate(q.tolist(), env, True): return k, q
    return None

rows, per = [], defaultdict(lambda: defaultdict(int))
todo = [i for i in range(len(r["solved"])) if r["solved"][i] and r["path_len"][i] > 0]
if a.n: todo = todo[:a.n]
for i in todo:
    fam, idx = names[int(t["category"][i])], int(t["problem_index"][i])
    prob = next(p for p in data[fam] if p.get("index") == idx)
    env = vamp.problem_dict_to_vamp(prob)
    path = r["paths"][i, :r["path_len"][i]].astype(np.float64)
    why, seg, kind = "valid", None, None
    if not (np.allclose(path[0], prob["start"], atol=1e-3) and np.allclose(path[-1], prob["goals"][0], atol=1e-3)):
        why = "endpoints"
    elif (path < lo - 1e-6).any() or (path > hi + 1e-6).any():
        why = "limits"
    else:
        for k in range(len(path) - 1):
            if not mod.validate_motion(path[k].tolist(), path[k + 1].tolist(), env, True):
                why, seg = "invalid_segment", k; break
        if why != "valid":
            fb = first_bad(path, env)
            if fb is not None: seg = fb[0]; kind = "self" if first_bad(path[fb[0]:fb[0] + 2], empty) is not None else "environment"
            else: kind = "resolution_only"   # validate_motion rejects it but 1/32 rad sampling finds nothing
    rows.append(dict(i=i, family=fam, problem=idx, result=why, segment=seg, n_waypoints=len(path), first_bad=kind))
    per[fam][why if kind is None else f"{why}:{kind}"] += 1
    per["ALL"][why if kind is None else f"{why}:{kind}"] += 1

print(f"{a.file}: {len(todo)} solved problems checked")
for fam in sorted(per, key=lambda f: (f == "ALL", f)):
    c = per[fam]; n = sum(c.values()); print(f"  {fam:<36}{c.get('valid', 0)}/{n} valid   " + "  ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "valid"))
if a.out: json.dump(rows, open(a.out, "w"))
