"""CRAB's own IK (crab env, GPU). Run: python ik_benchmark/crab_ik.py [--method fused|torch]
    [--chain arm|full|bimanual|mbm|mbm_baxter] [--seeds 8,32,128] [--max-iter 60] [--warm] [--set coll]

--method fused (default): crab_codegen/ik_kernel.py, ONE CUDA kernel, a whole LM per thread, calling
CRAB's SymForce-generated pose + Jacobian as device functions (nvcc + Eigen, float32). Timed like
crocket_ik.py (CUDA events over back-to-back launches), so comparable to its rows. The rest of this
docstring describes --method torch:

Batched Levenberg-Marquardt in torch around CRAB's Caspar CUDA kernels. Forward kinematics and the analytic geometric Jacobian are CRAB's symbolic expressions compiled by
Caspar (one CUDA thread per configuration, float32). The LM loop itself is NOT fused into a kernel:
every iteration is a few Caspar launches plus torch ops (residual, batched solve, accept/reject), over
all still-unconverged (target, seed) pairs at once. Same algorithm as crocket_ik.py's lmmulti: weights
50/10, pose residual [p - p*, log(R R*^T)], damped least squares with lam /3 on accept and *4 on reject,
joint-limit clamp, stop at 1e-4 m / 1e-3 rad. Collision-blind. Same seeds as crocket_ik.py (rng 1234).

Timing differs from crocket_ik.py and is NOT directly comparable:
  seconds                 wall-clock of one target's S seeds (host sync before and after), first
                          --latency-targets targets; includes Python and launch overhead
  batch_seconds_per_problem  all targets x seeds in one batched solve, wall-clock / targets
Results go to ik_benchmark/results/<chain>_crab_caspar_lm_<S>[_warm][_<set>].npz for report.py.
"""

import argparse
import os
import time

os.environ["CRAB_PRECISION"] = "float32"       # before crab_model is imported anywhere

import numpy as np  # noqa: E402
import torch  # noqa: E402

import problem  # noqa: E402

POS_W, ORI_W = 50.0, 10.0
EPS_POS, EPS_ORI = 1e-4, 1e-3
LAM0, LAM_MIN, LAM_MAX = 5e-3, 1e-7, 1e7


class CrabFk:
    """Poses and stacked Jacobian of the chain's tools on the GPU (CRAB codegen via Caspar)."""

    def __init__(self, chain: str):
        import crab_model  # noqa: F401
        from crab_io.urdf import load_urdf, urdf_to_robot
        from crab_kinematics.derived import q_to_ee_jacobian_analytic, q_to_ee_pose
        from crusty_cuda import build_cuda_function

        c = problem.CHAINS[chain]
        urdf = load_urdf(str(problem.common.ROBOTS[c.robot].urdf_path))
        robots = [urdf_to_robot(urdf, [tool]).finalize() for tool in c.tools]    # one per tool
        self.nq = robots[0].nq
        self.q_index = torch.tensor([robots[0].joints[n].id for n in c.joints], device="cuda")
        self.pose = [build_cuda_function(rb, q_to_ee_pose, name=f"ikpose_{c.robot}_{i}")
                     for i, rb in enumerate(robots)]
        self.jac = [build_cuda_function(rb, q_to_ee_jacobian_analytic, name=f"ikjac_{c.robot}_{i}")
                    for i, rb in enumerate(robots)]

    def full(self, q: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(q), self.nq, device=q.device, dtype=q.dtype)
        out[:, self.q_index] = q
        return out

    def poses(self, q: torch.Tensor) -> torch.Tensor:
        """(B, tools, 7) as [qx qy qz qw x y z] (SymForce Pose3 storage)."""
        full = self.full(q)
        return torch.stack([k(full) for k in self.pose], dim=1)

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """(B, 6 * tools, dof): per tool rows (v; w) world-aligned, columns in the chain's joint order."""
        full = self.full(q)
        return torch.cat([k(full).reshape(-1, 6, self.nq)[:, :, self.q_index] for k in self.jac], dim=1)


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def residual(poses: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, 6 * tools): per tool [p - p*, log(R R*^T)]. poses [qx qy qz qw x y z]; target [x y z qw qx qy qz]."""
    pos = poses[..., 4:7] - target[..., 0:3]
    conj = target[..., [4, 5, 6, 3]] * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=target.device)
    e = qmul(poses[..., 0:4], conj)
    e = torch.where(e[..., 3:4] < 0, -e, e)
    v, w = e[..., :3], e[..., 3:4]
    s = v.norm(dim=-1, keepdim=True)
    scale = torch.where(s > 1e-7, 2.0 * torch.atan2(s, w) / s.clamp_min(1e-7), 2.0 / w.clamp_min(1e-7))
    return torch.cat([pos, v * scale], dim=-1).reshape(len(poses), -1)


def converged(r: torch.Tensor) -> torch.Tensor:
    r = r.reshape(len(r), -1, 6)
    ok = ((r[..., :3] ** 2).sum(-1) < EPS_POS ** 2) & ((r[..., 3:] ** 2).sum(-1) < EPS_ORI ** 2)
    return ok.all(dim=1)


@torch.no_grad()
def solve(fk: CrabFk, q0: torch.Tensor, target: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
          max_iter: int) -> torch.Tensor:
    """LM on every row of q0 (B, dof) against its target (B, tools, 7); returns the final q (B, dof)."""
    q = q0.clone()
    B, dof = q.shape
    tools = target.shape[1]
    w2 = torch.tensor([POS_W] * 3 + [ORI_W] * 3, device=q.device).repeat(tools) ** 2     # (M,)
    eye = torch.eye(dof, device=q.device)
    lam = torch.full((B,), LAM0, device=q.device)
    r = residual(fk.poses(q), target)
    cost = 0.5 * (w2 * r * r).sum(1)
    for _ in range(max_iter):
        idx = (~converged(r)).nonzero().squeeze(1)       # only the unconverged keep iterating
        if len(idx) == 0:
            break
        qa, ra, la, ca = q[idx], r[idx], lam[idx], cost[idx]
        J = fk.jacobian(qa)                                          # (n, M, dof)
        JtW = J.transpose(1, 2) * w2                                 # (n, dof, M)
        A = JtW @ J + la[:, None, None] * eye
        dq = torch.linalg.solve(A, JtW @ ra[..., None])[..., 0]
        trial = torch.minimum(torch.maximum(qa - dq, lo), hi)
        rt = residual(fk.poses(trial), target[idx])
        ct = 0.5 * (w2 * rt * rt).sum(1)
        ok = torch.isfinite(ct) & (ct < ca)
        q[idx] = torch.where(ok[:, None], trial, qa)
        r[idx] = torch.where(ok[:, None], rt, ra)
        cost[idx] = torch.where(ok, ct, ca)
        lam[idx] = torch.where(ok, (la / 3).clamp_min(LAM_MIN), (la * 4).clamp_max(LAM_MAX))
    return q


def timed(fn) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_fused(args, t) -> None:
    """The fused path: crab_codegen.ik_kernel's single CUDA kernel (CRAB's SymForce pose + Jacobian as
    device functions, whole LM per thread), host program and timing as crocket_ik.py's (CUDA events)."""
    import crab_model  # noqa: F401  (float32 epsilon via CRAB_PRECISION)
    from crab_codegen.ik_kernel import build_ik_source
    from crab_io.urdf import load_urdf, urdf_to_robot
    from cuda_host import MAIN, compile_cu, run_exe

    c = problem.CHAINS[args.chain]
    urdf = load_urdf(str(problem.common.ROBOTS[c.robot].urdf_path))
    robots = [urdf_to_robot(urdf, [tool]).finalize() for tool in c.tools]
    q_index = [robots[0].joints[n].id for n in c.joints]
    lo, hi = t["lo"], t["hi"]
    tg = t["target"]                                              # (P, tools, 7) as [x y z qw qx qy qz]
    P, dof = len(tg), len(lo)
    target = np.concatenate([tg[..., 3:], tg[..., :3]], axis=-1).reshape(P, -1)    # per tool [qw qx qy qz x y z]

    eigen = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")
    src = build_ik_source(robots, q_index, lo, hi, args.max_iter) + MAIN
    exe = compile_cu(src, f"crab_fused_{args.chain}_{args.max_iter}", problem.REPO / "ik_benchmark" / "build",
                     flags=("--expt-relaxed-constexpr", f"-I{eigen}"))

    out_dir = problem.REPO / "ik_benchmark" / "results"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(1234)
    for S in [int(s) for s in args.seeds.split(",")]:
        seeds = rng.uniform(lo, hi, (P, S, dof))
        if args.warm:
            seeds[:, 0] = t["start"]
        q_all, latency_ms, batch_ms = run_exe(exe, "fk" if args.bench == "fk" else "thread", seeds, target)
        if args.bench == "fk":                                    # microbenchmark: print only, nothing saved
            print(f"FK-only x{args.max_iter} (crab fused, S={S}): latency median {np.median(latency_ms):.3f} "
                  f"ms/target, batch {batch_ms / P * 1e3:.2f} us/target")
            continue
        name = f"crab_fused_lm_{S}{'_warm' if args.warm else ''}"
        fname = f"{args.chain}_{name}{'_' + args.tset if args.tset else ''}"
        np.savez(out_dir / f"{fname}.npz", q_all=q_all, seconds=latency_ms / 1e3,
                 batch_seconds_per_problem=batch_ms / 1e3 / P, chain=args.chain, tset=args.tset, name=name)
        print(f"{name}: latency median {np.median(latency_ms):.3f} ms/target, "
              f"batch {batch_ms / P * 1e3:.2f} us/target")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fused", "torch"), default="fused",
                        help="fused: one CUDA kernel, LM per thread; torch: Caspar kernels driven from torch")
    parser.add_argument("--chain", choices=sorted(problem.CHAINS), default="arm")
    parser.add_argument("--bench", choices=("", "fk"), default="",
                        help="fk (fused only): time max-iter x (pose + Jacobian) per seed, no solver; not saved")
    parser.add_argument("--seeds", default="8,32,128")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--set", default="", choices=("", "coll"), dest="tset")
    parser.add_argument("--warm", action="store_true", help="mbm: seed 0 of each target = the start config")
    parser.add_argument("--latency-targets", type=int, default=30)
    args = parser.parse_args()

    t = np.load(problem.target_path(args.chain, args.tset))
    if args.method == "fused":
        run_fused(args, t)
        return
    lo, hi = t["lo"], t["hi"]
    P, dof = len(t["target"]), len(lo)
    dev = "cuda"
    fk = CrabFk(args.chain)
    target = torch.tensor(t["target"], dtype=torch.float32, device=dev)          # (P, tools, 7)
    lo_t, hi_t = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (lo, hi))

    out_dir = problem.REPO / "ik_benchmark" / "results"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(1234)
    for S in [int(s) for s in args.seeds.split(",")]:
        seeds = rng.uniform(lo, hi, (P, S, dof))
        if args.warm:
            seeds[:, 0] = t["start"]
        seeds_t = torch.tensor(seeds, dtype=torch.float32, device=dev).reshape(P * S, dof)
        tgt = target.repeat_interleave(S, dim=0)                                 # (P * S, tools, 7)
        run = lambda a, b: solve(fk, seeds_t[a:b], tgt[a:b], lo_t, hi_t, args.max_iter)   # noqa: E731

        run(0, min(P, 4) * S)                                                    # warm-up: launches, caches
        result = {}
        batch = timed(lambda: result.update(q=run(0, P * S)))
        n_lat = min(P, args.latency_targets)
        seconds = np.array([timed(lambda p=p: run(p * S, (p + 1) * S)) for p in range(n_lat)])
        q_all = result["q"].cpu().numpy().astype(np.float64).reshape(P, S, dof)

        name = f"crab_caspar_lm_{S}{'_warm' if args.warm else ''}"
        fname = f"{args.chain}_{name}{'_' + args.tset if args.tset else ''}"
        np.savez(out_dir / f"{fname}.npz", q_all=q_all, seconds=seconds, batch_seconds_per_problem=batch / P,
                 chain=args.chain, tset=args.tset, name=name)
        print(f"{name}: latency median {np.median(seconds) * 1e3:.2f} ms/target (wall-clock, incl. Python), "
              f"batch {batch / P * 1e6:.1f} us/target")


if __name__ == "__main__":
    main()
