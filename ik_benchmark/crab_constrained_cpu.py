"""CRAB's CPU constrained RRT-Connect (src/cpu/crrtc.hpp) on McVAMP's Panda line / plane problem sets (crab env). Run:
    python ik_benchmark/crab_constrained_cpu.py --problems plane [--limit N] [--threads 8]
    python ik_benchmark/crab_constrained_cpu.py --problems line --edge-mode 1 --smooth --freeze --closer   # ablations

Problem files: McVAMP's experiments/assets/problems/benchmark/{plane,line}_problems.json, copied to
ik_benchmark/targets/constrained/ (or --file). Each problem has its own cuboids, a task-space region and a start and
goal ON the region. As in McVAMP's benchmark the orientation bounds are opened to +-1000 unless --keep-orientation.
In its files `eef_transforms` is the region's reference frame (wTr) and `eef_transforms_ref_frame_w_world` the offset
frame in the end-effector frame (rTe), the order McVAMP's constructor receives them in (eef_to_offset first).
Reference (McVAMP, one core, from prrax's RESULTS.md): plane 1000/1000 solved, median 0.50 ms; line 758/766, 0.15 ms.

Settings of McVAMP's benchmark are the defaults: range 1, no dynamic domain, 15 projection iterations, perturbation 0.1,
all waypoints inserted. Nothing about the region is a planner constant: bounds and the per-row tolerance (--tol)
are written into each problem's Tsr. Paths are re-checked in float64: endpoints, joint limits, every node on the region
(within 2 x --tol), collision-free along linear interpolation between nodes at --res rad.
Saves results/constrained_<problems>_crab_cpu.npz.
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

MAXP = 512


def urdf_limits(joints, robot_key):
    """The URDF's own joint limits (problem.limits() also caps revolute joints at +-pi, which is wrong for Panda:
    joint 6 reaches 3.75 and McVAMP's problems use it)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(problem.common.ROBOTS[robot_key].urdf_path).getroot()
    lim = {j.get("name"): j.find("limit") for j in root.findall("joint")}
    return (np.array([float(lim[n].get("lower")) for n in joints]), np.array([float(lim[n].get("upper")) for n in joints]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", choices=("plane", "line"), default="plane")
    ap.add_argument("--file", default=None)
    ap.add_argument("--keep-orientation", action="store_true")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=200000)
    ap.add_argument("--max-samples", type=int, default=500000, help="nodes per tree")
    ap.add_argument("--range", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    g = ap.add_argument_group("region")
    g.add_argument("--tol", type=float, default=1e-3, help="per-row slack at which a row counts as satisfied")
    g = ap.add_argument_group("projection")
    g.add_argument("--proj-iters", type=int, default=15)
    g.add_argument("--lambda", dest="lam", type=float, default=1e-2, help="LM damping")
    g.add_argument("--rate", type=float, default=1.0, help="step scale")
    g.add_argument("--stall2", type=float, default=1e-12)
    g.add_argument("--no-hold", action="store_true", help="drop rows already inside their bounds from the Jacobian")
    g = ap.add_argument_group("local planner")
    g.add_argument("--lanes", type=int, default=8)
    g.add_argument("--perturb", type=float, default=0.1)
    g.add_argument("--slack", type=float, default=2.0)
    g.add_argument("--reached2", type=float, default=1e-2)
    g.add_argument("--endpoint-only", action="store_true", help="add only the endpoint of each extension to the tree")
    g = ap.add_argument_group("cpRRTC ablations (all off by default)")
    g.add_argument("--edge-mode", type=int, choices=(0, 1), default=0, help="0 McVAMP lanes + gap sweep, 1 dense chain")
    g.add_argument("--smooth", action="store_true", help="edge-mode 1: pull adjacent waypoints to the chord spacing")
    g.add_argument("--freeze", action="store_true", help="edge-mode 1: stop updating the satisfied prefix")
    g.add_argument("--closer", action="store_true", help="connect: a hop must get closer to its target")
    g.add_argument("--closer2", type=float, default=1.0 / 1024.0, help="--closer: squared progress a hop must make")
    g.add_argument("--smooth-rate", type=float, default=0.99)
    g.add_argument("--spacing", type=float, default=1.0 / 32.0, help="edge-mode 1: waypoint spacing (rad)")
    ap.add_argument("--profile", action="store_true",
                    help="print cycles per call of tsr_err / tsr_eval / tsr_step / fkcc (rdtsc; slows the run a little)")
    ap.add_argument("--validate", type=int, default=50)
    ap.add_argument("--res", type=float, default=0.05, help="validation resolution (rad)")
    ap.add_argument("--endpoint-tol", type=float, default=0.5,
                    help="validation: sanity bound (rad, joint space) on how far a path end may be from the file's start "
                         "/ goal; the planner projects them onto the region first, as McVAMP's benchmark does, and the "
                         "real check is in region-error space")
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
    from crab_codegen.robot_header import build_cpu_robot_header
    from crab_io.urdf import load_urdf, urdf_to_robot
    from crab_kinematics.constraints import tsr_error_and_jacobian
    from crusty_jit import build_simple_jit_function
    from cuda_host import compile_cpp
    from src.rrtc import build_cpu_crrtc_source

    c = problem.CHAINS["mbm"]
    lo, hi = urdf_limits(c.joints, c.robot)
    dof = len(lo)
    checker = problem.SelfCollision(problem.ChainFk("mbm"))
    robot = urdf_to_robot(load_urdf(str(problem.common.ROBOTS[c.robot].urdf_path)), [c.tools[0]]).finalize()
    q_index = [robot.joints[nm].id for nm in c.joints]
    header = build_cpu_robot_header(robot, q_index, lo, hi, checker.pair_i.tolist(), checker.pair_j.tolist())
    eigen = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")
    exe = compile_cpp(build_cpu_crrtc_source(header, build_tsr_header(robot, gpu=False)), f"crab_crrtc_cpu_panda{'_prof' if args.profile else ''}",
                      problem.REPO / "ik_benchmark" / "build",
                      flags=(f"-I{eigen}", *(("-DSHRIMP_PROF",) if args.profile else ())))

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
                        args.smooth_rate, args.spacing, args.proj_iters, args.lam, args.rate, args.stall2,
                        0 if args.no_hold else 1], np.float32)
    print(f"crab cpu constrained rrtc, panda {args.problems}: {n} problems on {args.threads} thread(s), "
          f"max {args.max_iters} iterations, range {args.range}, edge-mode {args.edge_mode}"
          f"{' smooth' if args.smooth else ''}{' freeze' if args.freeze else ''}{' closer' if args.closer else ''}")
    with tempfile.TemporaryDirectory() as d:
        f = {k: Path(d) / k for k in ("cp", "starts", "goals", "env", "env_n", "tsr", "path", "len", "stat", "ms")}
        for k, v, dt in (("cp", cparams, np.float32), ("starts", starts, np.float32), ("goals", goals, np.float32),
                         ("env", env, np.float32), ("env_n", env_n, np.int32), ("tsr", tsr, np.float32)):
            np.ascontiguousarray(v, dt).tofile(f[k])
        out = subprocess.check_output(
            [str(exe), str(n), str(args.threads), str(args.max_iters), str(args.max_samples), str(args.range),
             str(args.seed), str(MS), str(MB), str(MC), *(str(f[k]) for k in
             ("cp", "starts", "goals", "env", "env_n", "tsr", "path", "len", "stat", "ms"))], text=True)
        wall_ms = float(out)
        paths = np.fromfile(f["path"], np.float32).reshape(n, MAXP, dof)
        plen = np.fromfile(f["len"], np.int32)
        stat = np.fromfile(f["stat"], np.int32).reshape(n, 5)
        ms = np.fromfile(f["ms"], np.float32).astype(np.float64)

    status = stat[:, 0]
    ok = (status == 1) & (plen > 0)

    # float64 validation: endpoints, limits, every node on the region, collision between nodes
    kern = build_simple_jit_function(robot, tsr_error_and_jacobian, name="crrtc_val_tsr",
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
        # The planner projects the file's start / goal onto the region (as McVAMP's benchmark does), so a path end may
        # differ from the file's: by at most the end's own violation, in region-error space (joint space: sanity bound).
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

    print(f"solved {ok.sum()}/{n} ({ok.mean() * 100:.1f}%), wall {wall_ms:.1f} ms")
    print(f"unsolved: {int((status == 0).sum())} out of iterations, {int((status == -1).sum())} tree full, "
          f"{int((status == -2).sum())} start/goal not on the region, {int(((status == 1) & (plen <= 0)).sum())} path too long")
    if ok.any():
        s = ms[ok]
        print(f"planning ms per solved problem (one core): median {np.median(s):.2f}  mean {s.mean():.2f}  "
              f"p95 {np.percentile(s, 95):.2f}  max {s.max():.2f}")
        print(f"per solved problem (median): {np.median(stat[ok, 1]):.0f} iterations, {np.median(stat[ok, 2]):.0f} config "
              f"checks, {np.median(stat[ok, 3]):.0f} full + {np.median(stat[ok, 4]):.0f} error-only task-space "
              f"evaluations, path {np.median(plen[ok]):.0f} nodes")
    if len(checked):
        print(f"float64 validation of {len(checked)} solved paths: {valid} valid; "
              f"failures: { {k: v for k, v in why.items() if k} or 'none' }")
        for line in details[:5]:
            print("  " + line)

    out_path = args.out or str(problem.REPO / "ik_benchmark" / "results" / f"constrained_{args.problems}_crab_cpu.npz")
    Path(out_path).parent.mkdir(exist_ok=True)
    np.savez(out_path, solved=ok, status=status, iters=stat[:, 1], plan_ms=ms, path_len=np.where(ok, plen, 0),
             checks=stat[:, 2], evals=stat[:, 3], err_evals=stat[:, 4], wall_ms=wall_ms, threads=args.threads)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
