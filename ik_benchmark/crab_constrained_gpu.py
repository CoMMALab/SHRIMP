"""CRAB's GPU constrained RRT-Connect (src/cuda/crrtc.cuh) on McVAMP's Panda line / plane problem sets (crab env). Run:
    python ik_benchmark/crab_constrained_gpu.py --problems plane [--limit N] [--blocks 64]
    python ik_benchmark/crab_constrained_gpu.py --problems line --edge-mode 1 --smooth --freeze --closer   # ablations

Same problems, region and validation as crab_constrained_cpu.py (read its docstring for the file format and for what is
checked). The GPU planner runs every problem as `--blocks` warps sharing the problem's trees; all problems are one launch.
Each lane of a warp projects its own candidate / waypoint (see crrtc.cuh), so --lanes is the number of steer candidates
(<= 32) and the chain has up to 32 waypoints spaced --spacing apart. Reports the batched time per problem and the time
of single problems alone (CUDA events, kernels only). Saves results/constrained_<problems>_crab_gpu.npz.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ["CRAB_PRECISION"] = "float32"       # before crab_model is imported anywhere

import numpy as np  # noqa: E402

import problem  # noqa: E402
from crab_constrained_cpu import MAXP, urdf_limits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", choices=("plane", "line"), default="plane")
    ap.add_argument("--file", default=None)
    ap.add_argument("--keep-orientation", action="store_true")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=64, help="warps per problem")
    ap.add_argument("--max-iters", type=int, default=5000, help="iterations per block")
    ap.add_argument("--max-samples", type=int, default=16384, help="nodes per tree")
    ap.add_argument("--range", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lat-n", type=int, default=30, help="problems timed alone")
    g = ap.add_argument_group("region")
    g.add_argument("--tol", type=float, default=1e-3, help="per-row slack at which a row counts as satisfied")
    g = ap.add_argument_group("projection")
    g.add_argument("--proj-iters", type=int, default=15)
    g.add_argument("--lambda", dest="lam", type=float, default=1e-2, help="LM damping")
    g.add_argument("--rate", type=float, default=1.0, help="step scale")
    g.add_argument("--stall2", type=float, default=1e-12)
    g.add_argument("--no-hold", action="store_true", help="drop rows already inside their bounds from the Jacobian")
    g = ap.add_argument_group("local planner")
    g.add_argument("--lanes", type=int, default=8, help="steer candidates and emitted chain waypoints (<= 32)")
    g.add_argument("--perturb", type=float, default=0.1)
    g.add_argument("--slack", type=float, default=2.0)
    g.add_argument("--reached2", type=float, default=1e-2)
    g.add_argument("--gap-slack", type=float, default=2.0, help="adjacent chain waypoints within this x the chord spacing")
    g.add_argument("--spacing", type=float, default=1.0 / 32.0, help="chain waypoint spacing (rad), at most 32 waypoints")
    g.add_argument("--endpoint-only", action="store_true", help="add only the endpoint of each extension to the tree")
    g = ap.add_argument_group("cpRRTC ablations (all off by default)")
    g.add_argument("--edge-mode", type=int, choices=(0, 1), default=0, help="0 per-lane projection, 1 joint chain projection")
    g.add_argument("--smooth", action="store_true", help="edge-mode 1: pull adjacent waypoints to the chord spacing")
    g.add_argument("--freeze", action="store_true", help="edge-mode 1: stop updating the satisfied prefix")
    g.add_argument("--closer", action="store_true", help="connect: a hop must get closer to its target")
    g.add_argument("--closer2", type=float, default=1.0 / 1024.0, help="--closer: squared progress a hop must make")
    g.add_argument("--smooth-rate", type=float, default=0.99)
    ap.add_argument("--validate", type=int, default=50)
    ap.add_argument("--res", type=float, default=0.05, help="validation resolution (rad)")
    ap.add_argument("--endpoint-tol", type=float, default=0.5, help="validation: sanity bound on path end vs file (rad)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.file) if args.file else (problem.REPO / "ik_benchmark" / "targets" / "constrained" /
                                             f"{args.problems}_problems.json")
    if not path.exists():
        ap.error(f"{path} not found: copy McVAMP's experiments/assets/problems/benchmark/{args.problems}_problems.json "
                 f"there (or pass --file)")
    data = json.load(open(path))
    sel = data[args.first:args.first + args.limit] if args.limit else data[args.first:]
    n = len(sel)

    import crab_model  # noqa: F401
    from crab_codegen.constraint_header import build_tsr_header, pack_tsr
    from crab_codegen.robot_header import build_robot_header
    from crab_io.urdf import load_urdf, urdf_to_robot
    from crab_kinematics.constraints import tsr_error_and_jacobian
    from crusty_jit import build_simple_jit_function
    from cuda_host import compile_cu
    from src.rrtc import build_crrtc_source

    c = problem.CHAINS["mbm"]
    lo, hi = urdf_limits(c.joints, c.robot)
    dof = len(lo)
    checker = problem.SelfCollision(problem.ChainFk("mbm"))
    robot = urdf_to_robot(load_urdf(str(problem.common.ROBOTS[c.robot].urdf_path)), [c.tools[0]]).finalize()
    q_index = [robot.joints[nm].id for nm in c.joints]
    header = build_robot_header(robot, q_index, lo, hi, checker.pair_i.tolist(), checker.pair_j.tolist())
    eigen = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")
    exe = compile_cu(build_crrtc_source(header, build_tsr_header(robot, gpu=True)), "crab_crrtc_gpu_panda",
                     problem.REPO / "ik_benchmark" / "build", flags=("--expt-relaxed-constexpr", f"-I{eigen}"))

    starts = np.array([p["problem_start"] for p in sel], np.float32)
    goals = np.array([p["problem_end"] for p in sel], np.float32)
    MS, MC = 1, 1
    MB = max(1, max(len(p["cuboid_obstacles"]) for p in sel))
    env = np.zeros((n, MS * 4 + MB * 15 + MC * 7), np.float32)
    env_n = np.zeros((n, 3), np.int32)
    tsr, bounds = [], []
    for i, p in enumerate(sel):
        for j, cb in enumerate(p["cuboid_obstacles"]):            # center xyz, half extents xyz -> axis-aligned box
            env[i, 4 + 15 * j:4 + 15 * (j + 1)] = np.concatenate([cb[:3], np.eye(3).reshape(-1), cb[3:6]])
        env_n[i, 1] = len(p["cuboid_obstacles"])
        lb, ub = list(p["tsr_lower_bound"]), list(p["tsr_upper_bound"])
        if not args.keep_orientation:
            lb[3:], ub[3:] = [-1000.0] * 3, [1000.0] * 3
        r_te, w_tr = np.array([p["eef_transforms_ref_frame_w_world"]]), np.array([p["eef_transforms"]])
        tsr.append(pack_tsr(r_te, w_tr, lb, ub, args.tol))
        bounds.append((r_te[0], w_tr[0], np.array(lb), np.array(ub)))
    tsr = np.stack(tsr)

    cparams = np.array([args.lanes, args.perturb, args.slack, args.reached2, 0 if args.endpoint_only else 1,
                        args.edge_mode, int(args.smooth), int(args.freeze), int(args.closer), args.closer2,
                        args.smooth_rate, args.spacing, args.gap_slack, args.proj_iters, args.lam, args.rate,
                        args.stall2, 0 if args.no_hold else 1], np.float32)
    lat_n = min(n, args.lat_n)
    print(f"crab gpu constrained rrtc, panda {args.problems}: {n} problems, {args.blocks} blocks/problem, max "
          f"{args.max_iters} iterations/block, range {args.range}, edge-mode {args.edge_mode}"
          f"{' smooth' if args.smooth else ''}{' freeze' if args.freeze else ''}{' closer' if args.closer else ''}")
    with tempfile.TemporaryDirectory() as d:
        f = {k: Path(d) / k for k in ("cp", "starts", "goals", "env", "env_n", "tsr", "path", "len", "stat", "lat")}
        for k, v, dt in (("cp", cparams, np.float32), ("starts", starts, np.float32), ("goals", goals, np.float32),
                         ("env", env, np.float32), ("env_n", env_n, np.int32), ("tsr", tsr, np.float32)):
            np.ascontiguousarray(v, dt).tofile(f[k])
        out = subprocess.check_output(
            [str(exe), str(n), str(args.blocks), str(args.max_iters), str(args.max_samples), str(args.range),
             str(args.seed), str(MS), str(MB), str(MC), str(lat_n), *(str(f[k]) for k in
             ("cp", "starts", "goals", "env", "env_n", "tsr", "path", "len", "stat", "lat"))], text=True)
        batch_ms = float(out)
        paths = np.fromfile(f["path"], np.float32).reshape(n, MAXP, dof)
        plen = np.fromfile(f["len"], np.int32)
        stat = np.fromfile(f["stat"], np.int32).reshape(n, 2)
        lat = np.fromfile(f["lat"], np.float32).astype(np.float64)

    status = stat[:, 0]
    ok = (status == 1) & (plen > 0)

    # float64 validation: endpoints, limits, every node on the region, collision between nodes (as the CPU harness)
    kern = build_simple_jit_function(robot, tsr_error_and_jacobian, name="crrtc_val_tsr_gpu",
                                     extra_inputs=[("r_te", 7), ("w_tr", 7)])

    def raw_error(q, r_te, w_tr):
        full = np.zeros(robot.nq)
        full[q_index] = q
        return np.asarray(kern(full, r_te, w_tr), dtype=np.float64).reshape(-1)[:6]

    def region_error(q, r_te, w_tr, lb, ub):
        e = raw_error(q, r_te, w_tr)
        return np.where(e < lb, e - lb, np.where(e > ub, e - ub, 0.0))

    def validate(i):
        pth = paths[i, :plen[i]].astype(np.float64)
        r_te, w_tr, lb, ub = bounds[i]
        obs = {"sph": np.zeros((0, 4)), "cap": np.zeros((0, 7)),
               "box": env[i, 4:4 + 15 * env_n[i, 1]].reshape(-1, 15).astype(np.float64)}
        for name, end, ref in (("start", pth[0], starts[i].astype(np.float64)), ("goal", pth[-1], goals[i].astype(np.float64))):
            off = np.abs(region_error(ref, r_te, w_tr, lb, ub)).max()
            moved = np.abs(raw_error(end, r_te, w_tr) - raw_error(ref, r_te, w_tr)).max()
            if moved > 2 * (off + args.tol) or np.abs(end - ref).max() > args.endpoint_tol:
                details.append(f"problem {args.first + i}: {name} moved {np.abs(end - ref).max():.2e} rad in joint space, "
                               f"{moved:.2e} in region error; the file's {name} is off the region by {off:.2e}")
                return "endpoints"
        if (pth < lo - 1e-4).any() or (pth > hi + 1e-4).any():
            return "limits"
        if max(np.abs(region_error(q, r_te, w_tr, lb, ub)).max() for q in pth) > 2 * args.tol:
            return "off the region"
        for e, (a, b) in enumerate(zip(pth[:-1], pth[1:])):
            m = max(1, int(np.ceil(np.abs(b - a).max() / args.res)))
            for s in range(m + 1):
                gap = checker.clearance(a + (b - a) * s / m, obs)
                if gap < 0:
                    details.append(f"problem {args.first + i}: collision (gap {gap:.2e}) on edge {e} of {len(pth) - 1} at "
                                   f"{s}/{m}, {len(obs['box'])} obstacles: {checker.diagnose(a + (b - a) * s / m, obs)}")
                    return "collision"
        return ""

    valid, why, details = 0, {}, []
    checked = np.flatnonzero(ok)[:args.validate]
    for i in checked:
        res = validate(i)
        valid += res == ""
        why[res] = why.get(res, 0) + 1

    print(f"solved {ok.sum()}/{n} ({ok.mean() * 100:.1f}%); batch {batch_ms:.2f} ms ({batch_ms / n * 1e3:.1f} us/problem), "
          f"alone median {np.median(lat):.2f} ms (first {lat_n} spread over the set)")
    print(f"unsolved: {int((status == 0).sum() + (status == -1).sum())} out of iterations or a tree full, "
          f"{int((status == -2).sum())} start/goal not on the region, {int(((status == 1) & (plen <= 0)).sum())} path too long")
    if ok.any():
        print(f"per solved problem (median): {np.median(stat[ok, 1]):.0f} iterations (of the winning block), "
              f"path {np.median(plen[ok]):.0f} nodes")
    if len(checked):
        print(f"float64 validation of {len(checked)} solved paths: {valid} valid; "
              f"failures: { {k: v for k, v in why.items() if k} or 'none' }")
        for line in details[:5]:
            print("  " + line)

    out_path = args.out or str(problem.REPO / "ik_benchmark" / "results" / f"constrained_{args.problems}_crab_gpu.npz")
    Path(out_path).parent.mkdir(exist_ok=True)
    np.savez(out_path, solved=ok, status=status, iters=stat[:, 1], lat_ms=lat, batch_ms=batch_ms,
             path_len=np.where(ok, plen, 0))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
