"""CPU arm of the GPU->CPU fallback experiment: VAMP's RRT-Connect on the MBM problems (a vamp build with the
robot). Run in the VAMP env:
    python ik_benchmark/plan_vamp.py --robot baxter --from ik_benchmark/results/mbm_baxter_crab_rrtc.npz [--workers 8]
    python ik_benchmark/plan_vamp.py --robot baxter --all [--limit 200]

--from FILE   plan only the problems FILE left unsolved (solved == False or path_len == 0): the fallback set
--all         plan every problem (CPU-only reference)
Same problems and queries as crab_rrtc.py / solve_prrax.py: rows of ik_benchmark/targets/<robot>_<chain>.npz,
start -> goals[0] only, the problem's obstacles; fresh search (no tree handoff); no simplification; Halton
sampler reset per problem (--skip shifts it). `plan_ms` is VAMP's own planning time (nanoseconds), one core per
problem; --workers spreads problems over processes (VAMP is single-threaded), wall time reported separately.

Saves ik_benchmark/results/<chain>_vamp[_fallback].npz in the PATHS_NPZ.md layout over ALL P problems (unplanned
rows: solved False, path_len 0), plus `planned` (P,) bool, `plan_ms` (P,) NaN where unplanned, `iters` (P,), so
validate_in_vamp.py and the other tools read it unchanged.
"""

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAXP = 512
CHAIN = {"panda": "mbm", "baxter": "mbm_baxter"}
_W = {}                                           # per-worker state


def _init(robot, targets, kw, skip):
    import vamp

    t = np.load(f"{targets}/{robot}_{CHAIN[robot]}.npz")
    with open(f"{targets}/{robot}_problems.pkl", "rb") as f:
        data = pickle.load(f)["problems"]
    names = [str(n) for n in t["category_names"]]
    mod, planner, settings, _ = vamp.configure_robot_and_planner_with_kwargs(robot, "rrtc", **kw)
    _W.update(vamp=vamp, t=t, by={n: {p.get("index", j): p for j, p in enumerate(data[n])} for n in names},
              names=names, planner=planner, settings=settings, sampler=getattr(mod, "halton")(), skip=skip)


def _plan(i):
    vamp, t, w = _W["vamp"], _W["t"], _W
    prob = w["by"][w["names"][int(t["category"][i])]][int(t["problem_index"][i])]
    env = vamp.problem_dict_to_vamp(prob)
    w["sampler"].reset()
    w["sampler"].skip(w["skip"])
    res = w["planner"](prob["start"], [prob["goals"][0]], env, w["settings"], w["sampler"])
    solved = bool(res.path)
    path = np.asarray(res.path.numpy(), np.float64) if solved else None
    return i, solved, res.nanoseconds / 1e6, int(res.iterations), path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=sorted(CHAIN), required=True)
    ap.add_argument("--targets", default=str(HERE / "targets"))
    ap.add_argument("--from", dest="src", default=None, help="results npz: plan only its unsolved problems")
    ap.add_argument("--all", action="store_true", help="plan every problem")
    ap.add_argument("--limit", type=int, default=0, help="only the first N of the selected problems")
    ap.add_argument("--workers", type=int, default=1, help="processes (one core each)")
    ap.add_argument("--max-iterations", type=int, default=100000, help="VAMP samples per problem")
    ap.add_argument("--range", type=float, default=None, help="RRT range (default: VAMP's per-robot value)")
    ap.add_argument("--skip", type=int, default=0, help="Halton samples to skip (changes the seed)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if bool(args.src) == args.all:
        ap.error("give exactly one of --from FILE or --all")

    chain = CHAIN[args.robot]
    t = np.load(f"{args.targets}/{args.robot}_{chain}.npz")
    P, dof = len(t["category"]), len(t["lo"])
    if args.src:
        r = np.load(args.src)
        todo = np.flatnonzero(~(np.asarray(r["solved"], bool) & (np.asarray(r["path_len"]) > 0)))
    else:
        todo = np.arange(P)
    if args.limit:
        todo = todo[:args.limit]
    kw = {"max_iterations": args.max_iterations, "max_samples": args.max_iterations}
    if args.range is not None:
        kw["range"] = args.range
    print(f"vamp rrtc {args.robot}: planning {len(todo)} of {P} problems on {args.workers} worker(s), "
          f"max {args.max_iterations} iterations")

    solved = np.zeros(P, bool)
    plen = np.zeros(P, np.int32)
    paths = np.zeros((P, MAXP, dof), np.float32)
    plan_ms = np.full(P, np.nan)
    iters = np.zeros(P, np.int32)
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.robot, args.targets, kw, args.skip)) as ex:
        for i, ok, ms, it, path in ex.map(_plan, [int(i) for i in todo], chunksize=4):
            solved[i], plan_ms[i], iters[i] = ok, ms, it
            if ok:
                n = min(len(path), MAXP)
                paths[i, :n], plen[i] = path[:n], n
    wall = time.perf_counter() - wall0

    planned = np.zeros(P, bool)
    planned[todo] = True
    ms = plan_ms[todo]
    names, cat = [str(n) for n in t["category_names"]], t["category"]
    print(f"solved {solved[todo].sum()}/{len(todo)} ({solved[todo].mean() * 100:.1f}%), wall {wall:.2f} s")
    print(f"planning ms per problem (VAMP, one core): median {np.median(ms):.2f}  mean {ms.mean():.2f}  "
          f"p95 {np.percentile(ms, 95):.2f}  max {ms.max():.2f}  sum {ms.sum() / 1e3:.2f} s")
    print(f"{'family':<34}{'planned':>8}{'solved':>8}{'med ms':>9}{'max ms':>9}")
    for ci, name in enumerate(names):
        m = planned & (cat == ci)
        if m.any():
            print(f"{name:<34}{m.sum():>8}{solved[m].sum():>8}{np.median(plan_ms[m]):>9.2f}{plan_ms[m].max():>9.2f}")

    out = args.out or str(HERE / "results" / f"{chain}_vamp{'_fallback' if args.src else ''}.npz")
    Path(out).parent.mkdir(exist_ok=True)
    cost = np.full(P, np.nan)
    for i in np.flatnonzero(solved):
        cost[i] = np.linalg.norm(np.diff(paths[i, :plen[i]].astype(np.float64), axis=0), axis=1).sum()
    np.savez(out, solved=solved, path_len=plen, paths=paths, cost=cost, category=cat,
             category_names=np.array(names), planned=planned, plan_ms=plan_ms, iters=iters,
             wall_s=wall, workers=args.workers)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
