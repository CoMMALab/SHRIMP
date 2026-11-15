"""Baseline: pyroffi's IK on the hard RBY1 targets (pyroffi env; GPU for the CUDA solvers).
Run: python ik_benchmark/solve_pyroffi.py [--chain arm|full|bimanual|mbm] [--warm]
                                          [--configs hjcd:32,hjcd:256,ls:32,ls:256]
Each config is solver:num_seeds, i.e. the "blast more seeds" axis. One solve per target,
sequentially, timed wall-clock with sync (compile excluded by a warm-up). With several tools
(bimanual) pyroffi's multi-end-effector solve is used. Results go to ik_benchmark/results/<name>.npz
and are scored by report.py (crab env, CRAB's float64 FK) so the verification doesn't use
pyroffi's own FK."""

import argparse
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroffi as pk
import yourdfpy

import problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", choices=sorted(problem.CHAINS), default="arm")
    parser.add_argument("--configs", default="hjcd:32,hjcd:256,ls:32,ls:256")
    parser.add_argument("--set", default="", choices=("", "coll"), dest="tset")
    parser.add_argument("--warm", action="store_true", help="mbm: pass the problem's start config as previous_cfg")
    args = parser.parse_args()

    on_gpu = jax.default_backend() == "gpu"
    t = np.load(problem.target_path(args.chain, args.tset))
    joints = [str(j) for j in t["joints"]]
    tools = [str(x) for x in t["tools"]]
    multi = len(tools) > 1
    lo, hi = t["lo"], t["hi"]
    target = t["target"]                                  # (N, tools, 7)

    spec = problem.common.ROBOTS[problem.CHAINS[args.chain].robot]
    robot = pk.Robot.from_urdf(yourdfpy.URDF.load(str(spec.urdf_path)))
    names = list(robot.joints.actuated_names)
    chain_idx = [names.index(j) for j in joints]
    fixed = np.ones(len(names), dtype=bool)
    fixed[chain_idx] = False                      # everything off the chain stays at 0
    prev = np.zeros(len(names), dtype=np.float32)
    prev[chain_idx] = (lo + hi) / 2               # blind start: middle of the limits
    prevs = np.tile(prev, (len(target), 1))
    if args.warm:                                 # MBM: the problem's start configuration
        prevs[:, chain_idx] = t["start"]
    fixed_mask = jnp.asarray(fixed)

    out_dir = problem.REPO / "ik_benchmark" / "results"
    out_dir.mkdir(exist_ok=True)
    n = len(target)

    def pose_of(row):  # [x y z qw qx qy qz] -> SE3(wxyz_xyz)
        return jaxlie.SE3(wxyz_xyz=jnp.asarray(np.concatenate([row[3:], row[:3]]), dtype=jnp.float32))

    def poses_of(i):
        poses = [pose_of(target[i][k]) for k in range(len(tools))]
        return poses if multi else poses[0]

    for item in args.configs.split(","):
        solver, seeds = item.split(":")
        seeds = int(seeds)
        solve = jax.jit(lambda poses, key, prev0: robot.inverse_kinematics(
            target_link_name=tools if multi else tools[0], target_pose=poses, rng_key=key,
            previous_cfg=prev0, fixed_joint_mask=fixed_mask, solver=solver, num_seeds=seeds,
            use_cuda=on_gpu))

        jax.block_until_ready(solve(poses_of(0), jax.random.PRNGKey(0), jnp.asarray(prevs[0])))  # compile
        q_out = np.zeros((n, len(joints)))
        seconds = np.zeros(n)
        for i in range(n):
            poses, key, prev0 = poses_of(i), jax.random.PRNGKey(i), jnp.asarray(prevs[i])
            start = time.perf_counter()
            cfg = jax.block_until_ready(solve(poses, key, prev0))
            seconds[i] = time.perf_counter() - start
            q_out[i] = np.asarray(cfg)[chain_idx]

        name = f"pyroffi_{solver}_{seeds}{'_cuda' if on_gpu else '_cpu'}{'_warm' if args.warm else ''}"
        fname = f"{args.chain}_{name}{'_' + args.tset if args.tset else ''}"   # chains/sets don't overwrite each other
        np.savez(out_dir / f"{fname}.npz", q=q_out, seconds=seconds, chain=args.chain, tset=args.tset,
                 name=name)
        print(f"{name}: median {np.median(seconds) * 1e3:.2f} ms/solve")


if __name__ == "__main__":
    main()
