"""Re-check another planner's paths with OUR float64 checker (crab env). Run:
    python ik_benchmark/validate_paths.py --chain mbm_baxter [--file ik_benchmark/results/mbm_baxter_prrax.npz]
        [--n 100] [--res 0.05]

Takes `paths` / `path_len` / `solved` from a results file (solve_prrax.py writes them), evenly spaced over the
solved problems, and runs crab_rrtc.valid_path on each: endpoints, joint limits, and self + obstacle
collision (generate_mbm.py's checker: URDF spheres, SRDF-disabled and rigid pairs skipped, MBM obstacles) at
`--res` rad along every edge. Prints how many are valid, why the rest are not, and the cost of valid vs
invalid paths -- if a planner's paths fail here, it is checking a weaker collision model than ours.
"""

import argparse
from pathlib import Path

import numpy as np

import problem
from crab_rrtc import valid_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", choices=("mbm", "mbm_baxter"), default="mbm_baxter")
    ap.add_argument("--file", default=None, help="default: results/<chain>_prrax.npz")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--res", type=float, default=0.05)
    args = ap.parse_args()

    t = np.load(problem.target_path(args.chain))
    r = np.load(args.file or problem.REPO / "ik_benchmark" / "results" / f"{args.chain}_prrax.npz")
    lo, hi = t["lo"], t["hi"]
    checker = problem.SelfCollision(problem.ChainFk(args.chain))
    solved = np.flatnonzero(r["solved"] & (r["path_len"] > 0))
    pick = solved[np.arange(min(args.n, len(solved))) * len(solved) // min(args.n, len(solved))]
    why, cost_ok, cost_bad = {}, [], []
    names = list(t["category_names"])
    per_fam = {}
    for i in pick:
        p = r["paths"][i, :r["path_len"][i]].astype(np.float64)
        w = valid_path(p, t["start"][i], t["q_star"][i], lo, hi, checker, problem.obstacles_of(t, i), args.res)
        why[w or "valid"] = why.get(w or "valid", 0) + 1
        if w == "collision" and why[w] <= 5:                       # what collides, at the first bad waypoint
            obs = problem.obstacles_of(t, i)
            bad = next(q for a, b in zip(p[:-1], p[1:])
                       for q in (a + (b - a) * s / max(1, int(np.ceil(np.abs(b - a).max() / args.res)))
                                 for s in range(max(1, int(np.ceil(np.abs(b - a).max() / args.res))) + 1))
                       if checker.clearance(q, obs) < 0)
            kind = "SELF" if checker.clearance(bad) < 0 else "OBSTACLE"
            print(f"  problem {i}: first collision is {kind}: {checker.diagnose(bad, obs)}")
        (cost_ok if not w else cost_bad).append(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        f = per_fam.setdefault(names[int(t["category"][i])], [0, 0])
        f[0] += 1
        f[1] += not w
    print(f"{Path(args.file or 'prrax').name}: {len(pick)} of {len(solved)} solved paths re-checked at {args.res} rad")
    print(f"  results: {why}")
    for k, (n, ok) in per_fam.items():
        print(f"  {k:<34}{ok}/{n} valid")
    if cost_ok:
        print(f"  median cost valid {np.median(cost_ok):.2f}", end="")
    if cost_bad:
        print(f"  invalid {np.median(cost_bad):.2f}", end="")
    print()


if __name__ == "__main__":
    main()
