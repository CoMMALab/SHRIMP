"""Score IK results against the hard RBY1 targets, with CRAB's float64 FK as the judge.
Run in the crab env: python ik_benchmark/report.py [--chain arm|full|bimanual] [--set coll]

A returned q solves a target when it is finite, within the joint limits (1e-4 slack), and EVERY
tool's pose is within pos_tol / rot_tol of its target ("pose"). It must also be self-collision-free
(sphere check, problem.SelfCollision; plus the problem's obstacles for `--chain mbm`) to count as
"pose+free". It does not have to equal q_star.
A result file holds one solution per target (`q`) or all S seeds' solutions (`q_all`); with S seeds a
target is solved if ANY seed passes (equivalent to returning the best valid seed).

Columns: success % per category for pose+free; 'pose' = overall pose-only success (ignoring
collisions) so the gap shows how much the solver's collisions matter; 'ms' = median time of one
target's solve (pyroffi: wall-clock with sync; crocket: CUDA events over back-to-back launches of that
target's S seeds); 'batch us' = all targets in ONE launch / targets (crocket only)."""

import argparse

import numpy as np

import problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", choices=sorted(problem.CHAINS), default="arm")
    parser.add_argument("--set", default="", choices=("", "coll"), dest="tset")
    args = parser.parse_args()

    root = problem.REPO / "ik_benchmark"
    t = np.load(problem.target_path(args.chain, args.tset))
    fk = problem.ChainFk(args.chain)
    checker = problem.SelfCollision(fk)
    lo, hi = t["lo"], t["hi"]
    target = t["target"]                                   # (N, tools, 7)
    pos_tol, rot_tol = float(t["pos_tol"]), float(t["rot_tol"])
    names, category = [str(c) for c in t["category_names"]], t["category"]

    results = [np.load(p) for p in sorted((root / "results").glob("*.npz"))]
    results = [r for r in results if str(r["chain"]) == args.chain
               and (str(r["tset"]) if "tset" in r.files else "") == args.tset]
    if not results:
        raise SystemExit(f"no results for chain {args.chain} set '{args.tset}'; run a solve/crocket script first")

    print(f"rby1 {args.chain} ({fk.dof} DOF, tools {list(fk.tools)}), set '{args.tset}', "
          f"{len(category)} targets, tol {pos_tol} m / {rot_tol} rad, "
          f"{checker.n_spheres} spheres / {checker.n_pairs} pairs")
    print("success % per category for pose + self-collision-free (a target is solved if any seed passes)\n")
    short = [c.replace("_both_arms", "").replace("bookshelf", "bs") for c in names]   # Baxter family names are long
    w = max(len(c) for c in short) + 2
    header = (f"{'solver':<30}" + "".join(f"{c:>{w}}" for c in short)
              + f"{'all':>9}{'pose':>8}{'ms':>9}{'batch us':>10}")
    print(header)
    for r in results:
        q_all = r["q_all"] if "q_all" in r.files else r["q"][:, None, :]   # (N, S, dof)
        n, S = q_all.shape[:2]
        pose_ok = np.zeros((n, S), dtype=bool)    # pose within tolerance and q within limits
        free_ok = np.zeros((n, S), dtype=bool)    # ... and self-collision-free
        for i in range(n):
            obs = problem.obstacles_of(t, i)     # MBM problems carry their own obstacles
            for s in range(S):
                q = q_all[i, s]
                if not np.all(np.isfinite(q)) or np.any(q < lo - 1e-4) or np.any(q > hi + 1e-4):
                    continue
                poses = fk.pose(q)
                pos_ok = np.linalg.norm(poses[:, :3] - target[i][:, :3], axis=1) < pos_tol
                rot_ok = problem.rot_angle(poses[:, 3:], target[i][:, 3:]) < rot_tol
                if pos_ok.all() and rot_ok.all():
                    pose_ok[i, s] = True
                    free_ok[i, s] = not checker.collides(q, obs)
        ok = free_ok.any(axis=1)
        pose_any = pose_ok.any(axis=1)
        cols = "".join(f"{100.0 * ok[category == c].mean():>{w - 1}.1f}%" for c in range(len(names)))
        batch = (f"{float(r['batch_seconds_per_problem']) * 1e6:>10.2f}"
                 if "batch_seconds_per_problem" in r.files else f"{'-':>10}")
        print(f"{str(r['name']):<30}{cols}{100.0 * ok.mean():>8.1f}%{100.0 * pose_any.mean():>7.1f}%"
              f"{np.median(r['seconds']) * 1e3:>9.3f}{batch}")


if __name__ == "__main__":
    main()
