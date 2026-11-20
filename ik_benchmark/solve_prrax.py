"""prrax (external/prrax, JAX + GRiM) on the SAME MBM problems as crab_rrtc.py. Run in the prrax env:
    python ik_benchmark/solve_prrax.py [--chain mbm|mbm_baxter] [--limit N] [--resources DIR]

Reads ik_benchmark/targets/<robot>_<chain>.npz (written by generate_mbm.py) to pick the problems (family +
problem_index), then takes start / goals[0] / obstacles from ik_benchmark/targets/<robot>_problems.pkl, so the
problem set, start and single goal equal crab_rrtc.py's. Defaults are prrax's own benchmark's
(evaluate_mbm_batched.py): 64 new samples/iter, granularity 16, balance mode 2, dynamic domain, step size
0.5 (Panda) / 0.25 (Baxter), max_nodes 1e6, 5000 iterations, batches of 25 problems.

Reports like crab_rrtc.py: solved per family, alone latency (wall ms, second call so the per-shape JIT is
warm; and the kernel ms prrax reports), batched wall ms per problem, path cost. Saves
ik_benchmark/results/<chain>_prrax.npz including the paths, which are NOT re-checked here (prrax's own
verdict); `box`-family cylinders follow pyroffi's own obstacle conversion, not generate_mbm.py's boxes.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyroffi as pk
import yourdfpy
from pyroffi.collision._obstacles import create_collision_environment
from pyroffi.collision._robot_collision import RobotCollisionSpherized

from prrax.jax import prrtc as PRRTC
from prrax.jax import utils as U

REPO = Path(__file__).resolve().parents[1]
ROBOT = {"mbm": "panda", "mbm_baxter": "baxter"}
STEP = {"panda": 0.5, "baxter": 0.25}
MAXP = 512
VAMP_JOINTS = {"baxter": [f"{side}_{j}" for side in ("left", "right") for j in ("s0", "s1", "e0", "e1", "w0", "w1", "w2")]}


def signature(p: dict) -> str:
    return json.dumps({k: p.get(k, []) for k in ("sphere", "cylinder", "box")}, sort_keys=True,
                      separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", choices=sorted(ROBOT), default="mbm")
    ap.add_argument("--resources", default=str(REPO / "external" / "pyroffi" / "resources"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=25, help="problems per batched solve")
    ap.add_argument("--lat-n", type=int, default=30, help="problems timed alone")
    ap.add_argument("--max-iterations", type=int, default=5000)
    ap.add_argument("--num-new-samples", type=int, default=64)
    ap.add_argument("--granularity", type=int, default=16)
    ap.add_argument("--max-nodes", type=int, default=1_000_000)
    ap.add_argument("--balance-mode", type=int, default=2)
    ap.add_argument("--tree-ratio", type=float, default=1.0)
    ap.add_argument("--no-dynamic-domain", action="store_true")
    ap.add_argument("--collision-checker", default="binary",
                    choices=["binary", "binary_coarse", "binary_coarse_coop"],
                    help="binary_coarse = bounding-sphere pre-pass before the fine spheres")
    args = ap.parse_args()
    robot = ROBOT[args.chain]

    t = np.load(REPO / "ik_benchmark" / "targets" / f"{robot}_{args.chain}.npz")
    with open(REPO / "ik_benchmark" / "targets" / f"{robot}_problems.pkl", "rb") as f:
        data = pickle.load(f)["problems"]
    names = [str(n) for n in t["category_names"]]
    by = {n: {p.get("index", j): p for j, p in enumerate(data[n])} for n in names}
    P = len(t["category"]) if not args.limit else min(args.limit, len(t["category"]))
    probs = [by[names[int(c)]][int(i)] for c, i in zip(t["category"][:P], t["problem_index"][:P])]
    starts = np.asarray([p["start"] for p in probs], np.float32)
    goals = np.asarray([p["goals"][0] for p in probs], np.float32)[:, None, :]       # goals[0] only

    res_dir = Path(args.resources) / robot
    urdf = yourdfpy.URDF.load(str(res_dir / f"{robot}_spherized.urdf"))
    model = pk.Robot.from_urdf(urdf)
    coll = RobotCollisionSpherized.from_urdf(urdf, srdf_path=str(res_dir / f"{robot}.srdf"))
    lo = jnp.asarray(model.joints.lower_limits, jnp.float32)
    hi = jnp.asarray(model.joints.upper_limits, jnp.float32)

    # VAMP's problems.pkl joint order vs pyroffi's (URDF file order: Baxter's right arm comes first there).
    # Plan in the model's order, then return paths in VAMP's order.
    model_names = list(model.joints.actuated_names)
    vamp_names = VAMP_JOINTS.get(robot)
    if vamp_names and set(vamp_names) == set(model_names):
        to_model = [vamp_names.index(n) for n in model_names]       # model joint m <- VAMP column to_model[m]
        to_vamp = [model_names.index(n) for n in vamp_names]        # VAMP joint v <- model column to_vamp[v]
    else:
        to_model = to_vamp = list(range(len(model_names)))
    print(f"joint order: pyroffi {model_names[:2]}.. / VAMP {(vamp_names or model_names)[:2]}.. "
          f"-> {'permuted' if to_model != sorted(to_model) else 'identical'}")
    starts, goals = starts[:, to_model], goals[:, :, to_model]
    cache = {}
    ctxs = []
    for p in probs:
        s = signature(p)
        if s not in cache:
            cache[s] = U.build_prrtc_collision_context(model, coll, create_collision_environment(p))
        ctxs.append(cache[s])

    kw = dict(max_iterations=args.max_iterations, step_size=STEP[robot], num_new_samples=args.num_new_samples,
              granularity=args.granularity, max_nodes=args.max_nodes, balance_mode=args.balance_mode,
              tree_ratio=args.tree_ratio, collision_checker=args.collision_checker, dynamic_domain=not args.no_dynamic_domain, min_vals=lo, max_vals=hi)

    results, batch_ms = [None] * P, 0.0
    for c0 in range(0, P, args.chunk):
        sl = slice(c0, min(c0 + args.chunk, P))
        run = lambda: PRRTC.prrtc_plan_batch(jnp.asarray(starts[sl]), jnp.asarray(goals[sl]),   # noqa: E731
                                             collision_context=ctxs[sl], **kw)
        if c0 == 0:
            U.synchronize_prrtc_results(run())                                   # warm-up (JIT, build)
        t0 = time.perf_counter()
        out = run()
        U.synchronize_prrtc_results(out)
        batch_ms += (time.perf_counter() - t0) * 1e3
        results[sl] = out

    lat_idx = np.arange(min(P, args.lat_n)) * P // min(P, args.lat_n)
    lat, kern = [], []
    for i in lat_idx:
        one = lambda: PRRTC.prrtc_plan(jnp.asarray(starts[i]), jnp.asarray(goals[i]),        # noqa: E731
                                       collision_context=ctxs[i], **kw)
        U.synchronize_prrtc_result(one())                                        # per-shape JIT warm
        r, ms = U.timed_prrtc_solve(one)
        lat.append(ms)
        kern.append(float(r.kernel_time_ms) if r.kernel_time_ms is not None else np.nan)
    lat, kern = np.array(lat), np.array(kern)

    solved = np.array([r.solved for r in results])
    cost = np.array([float(r.cost) if r.solved else np.nan for r in results])
    dof = starts.shape[1]
    paths = np.zeros((P, MAXP, dof), np.float32)
    plen = np.zeros(P, np.int32)
    for i, r in enumerate(results):
        if r.solved and r.path is not None:
            n = min(len(r.path), MAXP)
            paths[i, :n] = np.asarray(r.path)[:n][:, to_vamp]           # back to VAMP's joint order
            plen[i] = n

    cat = t["category"][:P]
    print(f"prrax {args.chain}: {P} problems, {dof} DOF, batches of {args.chunk}, batch {batch_ms:.1f} ms "
          f"({batch_ms / P * 1e3:.1f} us/problem), alone median {np.median(lat):.2f} ms wall / "
          f"{np.nanmedian(kern):.2f} ms kernel (first {len(lat)})")
    print(f"{'family':<34}{'n':>5}{'solved':>8}{'lat ms':>9}{'kern ms':>9}{'cost med':>10}")
    for ci, name in enumerate(names):
        m = cat == ci
        if not m.any():
            continue
        in_fam = m[lat_idx]
        print(f"{name:<34}{m.sum():>5}{solved[m].mean() * 100:>7.1f}%"
              f"{(np.median(lat[in_fam]) if in_fam.any() else float('nan')):>9.2f}"
              f"{(np.nanmedian(kern[in_fam]) if in_fam.any() and not np.isnan(kern[in_fam]).all() else float('nan')):>9.2f}"
              f"{(np.nanmedian(cost[m]) if solved[m].any() else float('nan')):>10.2f}")
    print(f"{'all':<34}{P:>5}{solved.mean() * 100:>7.1f}%")

    out_dir = REPO / "ik_benchmark" / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f"{args.chain}_prrax.npz", solved=solved, cost=cost, lat_ms=lat, kernel_ms=kern,
             batch_ms=batch_ms, category=cat, category_names=np.array(names), paths=paths, path_len=plen)


if __name__ == "__main__":
    main()
