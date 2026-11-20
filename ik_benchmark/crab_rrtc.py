"""CRAB's GPU RRT-Connect on the MBM problems (crab env). Run:
    python ik_benchmark/crab_rrtc.py [--chain mbm|mbm_baxter] [--blocks 64] [--max-iters 3000]
        [--max-samples 8192] [--range 0.5] [--limit N] [--validate 50]

One CUDA kernel (src/cuda/rrtc.cuh, on CRAB's crab_codegen/robot_header.py) plans every problem of ik_benchmark/targets/<robot>_mbm*.npz:
start = the problem's start config, goal = its goal config (q_star), world = its obstacles. Reports, per
family: success, median latency of one problem alone on the GPU, batched time per problem, path cost; and
re-checks `--validate` solved paths in float64 (limits, endpoints, self + obstacle collision along every
edge at --res rad) with the same checker generate_mbm.py uses.
"""

import argparse
import os
import tempfile
from pathlib import Path

os.environ["CRAB_PRECISION"] = "float32"       # before crab_model is imported anywhere

import numpy as np  # noqa: E402

import problem  # noqa: E402


def build(chain: str, lo, hi, checker, cpu: bool = False, interleave: bool = False):
    import crab_model  # noqa: F401
    from crab_codegen.robot_header import build_cpu_robot_header, build_robot_header
    from crab_io.urdf import load_urdf, urdf_to_robot
    from src.rrtc import build_cpu_rrtc_source, build_rrtc_source

    c = problem.CHAINS[chain]
    urdf = load_urdf(str(problem.common.ROBOTS[c.robot].urdf_path))
    robot = urdf_to_robot(urdf, [c.tools[0]]).finalize()
    q_index = [robot.joints[n].id for n in c.joints]
    args = (robot, q_index, lo, hi, checker.pair_i.tolist(), checker.pair_j.tolist())
    if cpu:
        return build_cpu_rrtc_source(build_cpu_robot_header(*args, interleave=interleave))
    return build_rrtc_source(build_robot_header(*args))


def valid_path(path, start, goal, lo, hi, checker, obs, res: float) -> str:
    """"" if the path is a valid collision-free solution, else why not."""
    if np.abs(path[0] - start).max() > 1e-3 or np.abs(path[-1] - goal).max() > 1e-3:
        return "endpoints"
    if (path < lo - 1e-4).any() or (path > hi + 1e-4).any():
        return "limits"
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / res)))
        for s in range(n + 1):
            if checker.clearance(a + (b - a) * s / n, obs) < 0:
                return "collision"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", choices=("mbm", "mbm_baxter"), default="mbm")
    parser.add_argument("--blocks", type=int, default=64, help="thread blocks (x 32 threads) per problem")
    parser.add_argument("--max-iters", type=int, default=3000, help="expansion iterations per block")
    parser.add_argument("--max-samples", type=int, default=8192, help="nodes per tree")
    parser.add_argument("--range", type=float, default=0.5, help="max extension (rad, max-norm-free L2)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N problems (0 = all)")
    parser.add_argument("--lat-n", type=int, default=30, help="problems timed alone")
    parser.add_argument("--validate", type=int, default=50, help="solved paths re-checked in float64")
    parser.add_argument("--res", type=float, default=0.05, help="validation resolution (rad)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--profile", action="store_true",
                        help="print cycles per phase (FK / obstacles / self pairs; nearest / edge_ok / add_node)")
    args = parser.parse_args()

    from cuda_host import compile_cu

    t = np.load(problem.target_path(args.chain))
    P = len(t["q_star"]) if not args.limit else min(args.limit, len(t["q_star"]))
    lo, hi = t["lo"], t["hi"]
    dof = len(lo)
    fk = problem.ChainFk(args.chain)
    checker = problem.SelfCollision(fk)

    eigen = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")
    exe = compile_cu(build(args.chain, lo, hi, checker), f"crab_rrtc_{args.chain}{'_prof' if args.profile else ''}",
                     problem.REPO / "ik_benchmark" / "build",
                     flags=("--expt-relaxed-constexpr", f"-I{eigen}", *(("-DSHRIMP_PROF",) if args.profile else ())))

    starts, goals = t["start"][:P], t["q_star"][:P]
    MS, MB, MC = t["sph"].shape[1], t["box"].shape[1], t["cap"].shape[1]
    env = np.concatenate([t["sph"][:P].reshape(P, -1), t["box"][:P].reshape(P, -1), t["cap"][:P].reshape(P, -1)], 1)
    env_n = np.stack([t["n_sph"][:P], t["n_box"][:P], t["n_cap"][:P]], 1)
    with tempfile.TemporaryDirectory() as d:
        f = {n: Path(d) / n for n in ("starts", "goals", "env", "env_n", "path", "len", "stat", "lat")}
        np.ascontiguousarray(starts, np.float32).tofile(f["starts"])
        np.ascontiguousarray(goals, np.float32).tofile(f["goals"])
        np.ascontiguousarray(env, np.float32).tofile(f["env"])
        np.ascontiguousarray(env_n, np.int32).tofile(f["env_n"])
        import subprocess
        lat_n = min(P, args.lat_n)
        out = subprocess.check_output(
            [str(exe), str(P), str(args.blocks), str(args.max_iters), str(args.max_samples), str(args.range),
             str(args.seed), str(MS), str(MB), str(MC), str(lat_n), *(str(f[n]) for n in
             ("starts", "goals", "env", "env_n", "path", "len", "stat", "lat"))], text=True)
        batch_ms = float(out)
        maxp = 512
        path = np.fromfile(f["path"], np.float32).reshape(P, maxp, dof).astype(np.float64)
        plen = np.fromfile(f["len"], np.int32)
        stat = np.fromfile(f["stat"], np.int32).reshape(P, 2)
        lat = np.fromfile(f["lat"], np.float32)

    solved = (stat[:, 0] == 1) & (plen > 0)
    cost = np.full(P, np.nan)
    for i in np.flatnonzero(solved):
        p = path[i, :plen[i]]
        cost[i] = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()

    checked = np.flatnonzero(solved)[:args.validate]
    ok = {}
    why = {}
    for i in checked:
        r = valid_path(path[i, :plen[i]], starts[i], goals[i], lo, hi, checker, problem.obstacles_of(t, i), args.res)
        ok[i] = r == ""
        why[r] = why.get(r, 0) + 1

    cat, names = t["category"][:P], list(t["category_names"])
    print(f"{args.chain}: {P} problems, {dof} DOF, {args.blocks} blocks/problem, batch {batch_ms:.2f} ms "
          f"({batch_ms / P * 1e3:.1f} us/problem), alone median {np.median(lat):.2f} ms (first {lat_n})")
    line = np.linalg.norm(goals - starts, axis=1)              # straight start -> goal length = the cost floor
    first = solved & (stat[:, 1] == 1)                         # connected on iteration 1 (the direct attempt)
    print(f"{'family':<34}{'n':>5}{'solved':>8}{'lat ms':>9}{'cost med':>10}{'line med':>10}{'iter 1':>8}")
    for ci, name in enumerate(names):
        m = cat == ci
        if not m.any():
            continue
        timed = np.arange(lat_n) * P // lat_n                  # problem index of each latency sample
        li_lat = lat[m[timed]]
        print(f"{name:<34}{m.sum():>5}{solved[m].mean() * 100:>7.1f}%"
              f"{(np.median(li_lat) if len(li_lat) else float('nan')):>9.2f}"
              f"{np.nanmedian(cost[m]) if solved[m].any() else float('nan'):>10.2f}"
              f"{np.median(line[m]):>10.2f}{first[m].mean() * 100:>7.1f}%")
    print(f"{'all':<34}{P:>5}{solved.mean() * 100:>7.1f}%")
    if (~solved).any():                      # unsolved: iteration budget used up (iters == max) or a tree filled up
        out_of_iters = int(((~solved) & (stat[:, 1] == args.max_iters)).sum())
        print(f"unsolved {int((~solved).sum())}: {out_of_iters} hit --max-iters, "
              f"{int((~solved).sum()) - out_of_iters} filled a tree (--max-samples)")
    if len(checked):
        print(f"float64 validation of {len(checked)} solved paths: {sum(ok.values())} valid; "
              f"failures: { {k: v for k, v in why.items() if k} or 'none' }")

    out_dir = problem.REPO / "ik_benchmark" / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f"{args.chain}_crab_rrtc.npz", solved=solved, iters=stat[:, 1], cost=cost, lat_ms=lat,
             batch_ms=batch_ms, category=cat, category_names=np.array(names),
             paths=path.astype(np.float32), path_len=np.where(solved, plen, 0).astype(np.int32))


if __name__ == "__main__":
    main()
