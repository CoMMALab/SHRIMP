"""CRAB's CPU RRT-Connect (shrimp/cpu/rrtc.hpp, scalar) on the MBM problems (crab env). Run:
    python ik_benchmark/crab_rrtc_cpu.py --chain mbm_baxter --all [--threads 8] [--limit N]
    python ik_benchmark/crab_rrtc_cpu.py --chain mbm_baxter --from ik_benchmark/results/mbm_baxter_crab_rrtc.npz

--all         plan every problem (CPU-only reference)
--from FILE   plan only the problems FILE left unsolved (solved == False or path_len == 0): the fallback set
Same problems and queries as crab_rrtc.py (start -> q_star, the problem's obstacles). `plan ms` is one core's time
per problem; --threads spreads problems over threads (wall time reported separately). Saves
results/<chain>_crab_cpu[_fallback].npz in the PATHS_NPZ.md layout over ALL problems (unplanned rows: solved
False, path_len 0) plus `planned`, `plan_ms` (NaN where unplanned), `iters`, `wall_ms`, `threads`.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

os.environ["CRAB_PRECISION"] = "float32"       # before crab_model is imported anywhere

import numpy as np  # noqa: E402

import problem  # noqa: E402
from crab_rrtc import build, valid_path  # noqa: E402

MAXP = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", choices=("mbm", "mbm_baxter"), default="mbm_baxter")
    ap.add_argument("--from", dest="src", default=None, help="results npz: plan only its unsolved problems")
    ap.add_argument("--all", action="store_true", help="plan every problem")
    ap.add_argument("--limit", type=int, default=0, help="only the first N of the selected problems")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=100000)
    ap.add_argument("--max-samples", type=int, default=100000, help="nodes per tree")
    ap.add_argument("--range", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--interleave", action="store_true",
                    help="test each link right after its FK step, root -> tip (default: full FK first, then "
                         "tip -> root with the per-link bounding-sphere broad phase)")
    ap.add_argument("--profile", action="store_true",
                    help="print cycles per config check split into FK / obstacles / self pairs "
                         "(not with --interleave)")
    ap.add_argument("--epsilon", type=float, default=0.0, help="NN approximation: within (1+eps) of the nearest")
    ap.add_argument("--weights", default="-", help="per-joint NN metric weights 'w0,w1,...' (chain order); - = ones")
    ap.add_argument("--validate", type=int, default=50, help="solved paths re-checked in float64")
    ap.add_argument("--res", type=float, default=0.05, help="validation resolution (rad)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if bool(args.src) == args.all:
        ap.error("give exactly one of --from FILE or --all")

    from cuda_host import compile_cpp

    t = np.load(problem.target_path(args.chain))
    P, lo, hi = len(t["q_star"]), t["lo"], t["hi"]
    dof = len(lo)
    checker = problem.SelfCollision(problem.ChainFk(args.chain))
    if args.src:
        r = np.load(args.src)
        todo = np.flatnonzero(~(np.asarray(r["solved"], bool) & (np.asarray(r["path_len"]) > 0)))
    else:
        todo = np.arange(P)
    if args.limit:
        todo = todo[:args.limit]
    n = len(todo)
    if n == 0:
        print("nothing to plan")
        return

    eigen = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")
    il = args.interleave
    if args.profile and il:
        ap.error("--profile does not work with --interleave")
    exe = compile_cpp(build(args.chain, lo, hi, checker, cpu=True, interleave=il),
                      f"crab_rrtc_cpu_{args.chain}{'_il' if il else ''}{'_prof' if args.profile else ''}",
                      problem.REPO / "ik_benchmark" / "build",
                      flags=(f"-I{eigen}", *(("-DSHRIMP_PROF",) if args.profile else ())))

    starts, goals = t["start"][todo], t["q_star"][todo]
    MS, MB, MC = t["sph"].shape[1], t["box"].shape[1], t["cap"].shape[1]
    env = np.concatenate([t["sph"][todo].reshape(n, -1), t["box"][todo].reshape(n, -1),
                          t["cap"][todo].reshape(n, -1)], 1)
    env_n = np.stack([t["n_sph"][todo], t["n_box"][todo], t["n_cap"][todo]], 1)
    if args.weights != "-" and len(args.weights.split(",")) != dof:
        ap.error(f"--weights needs {dof} values")
    print(f"crab cpu rrtc {args.chain}: planning {n} of {P} problems on {args.threads} thread(s), "
          f"max {args.max_iters} iterations, range {args.range}")
    with tempfile.TemporaryDirectory() as d:
        f = {k: Path(d) / k for k in ("starts", "goals", "env", "env_n", "path", "len", "stat", "ms")}
        np.ascontiguousarray(starts, np.float32).tofile(f["starts"])
        np.ascontiguousarray(goals, np.float32).tofile(f["goals"])
        np.ascontiguousarray(env, np.float32).tofile(f["env"])
        np.ascontiguousarray(env_n, np.int32).tofile(f["env_n"])
        out = subprocess.check_output(
            [str(exe), str(n), str(args.threads), str(args.max_iters), str(args.max_samples), str(args.range),
             str(args.seed), str(args.epsilon), args.weights, str(MS), str(MB), str(MC), *(str(f[k]) for k in
             ("starts", "goals", "env", "env_n", "path", "len", "stat", "ms"))], text=True)
        wall_ms = float(out)
        path = np.fromfile(f["path"], np.float32).reshape(n, MAXP, dof)
        plen = np.fromfile(f["len"], np.int32)
        stat = np.fromfile(f["stat"], np.int32).reshape(n, 3)
        ms = np.fromfile(f["ms"], np.float32).astype(np.float64)

    ok = (stat[:, 0] == 1) & (plen > 0)
    solved = np.zeros(P, bool)
    solved[todo] = ok
    path_len = np.zeros(P, np.int32)
    path_len[todo] = np.where(ok, plen, 0)
    paths = np.zeros((P, MAXP, dof), np.float32)
    paths[todo] = path
    paths[~solved] = 0
    plan_ms = np.full(P, np.nan)
    plan_ms[todo] = ms
    iters = np.zeros(P, np.int32)
    iters[todo] = stat[:, 1]
    planned = np.zeros(P, bool)
    planned[todo] = True
    cost = np.full(P, np.nan)
    for i in np.flatnonzero(solved):
        cost[i] = np.linalg.norm(np.diff(paths[i, :path_len[i]].astype(np.float64), axis=0), axis=1).sum()

    valid, why = 0, {}
    checked = np.flatnonzero(solved)[:args.validate]
    for i in checked:
        res = valid_path(paths[i, :path_len[i]].astype(np.float64), t["start"][i], t["q_star"][i], lo, hi, checker,
                         problem.obstacles_of(t, i), args.res)
        valid += res == ""
        why[res] = why.get(res, 0) + 1

    cat, names = t["category"], [str(x) for x in t["category_names"]]
    print(f"solved {ok.sum()}/{n} ({ok.mean() * 100:.1f}%), wall {wall_ms:.1f} ms")
    print(f"planning ms per problem (one core): median {np.median(ms):.2f}  mean {ms.mean():.2f}  "
          f"p95 {np.percentile(ms, 95):.2f}  max {ms.max():.2f}  sum {ms.sum() / 1e3:.2f} s")
    fk = stat[:, 2].astype(np.float64)
    print(f"per problem (median): {np.median(stat[:, 1]):.0f} iterations, {np.median(fk):.0f} config checks "
          f"({np.median(fk / np.maximum(stat[:, 1], 1)):.1f}/iteration); "
          f"overall {ms.sum() * 1e6 / max(fk.sum(), 1):.0f} ns per config check (all planning time included)")
    print(f"{'family':<34}{'planned':>8}{'solved':>8}{'med ms':>9}{'max ms':>9}")
    for ci, name in enumerate(names):
        m = planned & (cat == ci)
        if m.any():
            print(f"{name:<34}{m.sum():>8}{solved[m].sum():>8}{np.median(plan_ms[m]):>9.2f}{plan_ms[m].max():>9.2f}")
    if (~ok).any():
        full = int(((stat[:, 0] != 1) & (stat[:, 1] < args.max_iters)).sum())
        print(f"unsolved {int((~ok).sum())}: {int((~ok).sum()) - full} hit --max-iters, {full} filled a tree")
    if len(checked):
        print(f"float64 validation of {len(checked)} solved paths: {valid} valid; "
              f"failures: { {k: v for k, v in why.items() if k} or 'none' }")

    out_path = args.out or str(problem.REPO / "ik_benchmark" / "results" /
                               f"{args.chain}_crab_cpu{'_fallback' if args.src else ''}.npz")
    Path(out_path).parent.mkdir(exist_ok=True)
    np.savez(out_path, solved=solved, path_len=path_len, paths=paths, cost=cost, category=cat,
             category_names=np.array(names), planned=planned, plan_ms=plan_ms, iters=iters,
             wall_ms=wall_ms, threads=args.threads)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
