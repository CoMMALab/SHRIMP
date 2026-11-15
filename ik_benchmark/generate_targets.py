"""Generate hard IK targets for RBY1, each with a known-reachable ground-truth configuration.
Run in the crab env: python ik_benchmark/generate_targets.py [--chain arm|full|bimanual] [--n 200]
                                                             [--set coll]

Every target is FK(q_star) of a q_star we construct (for every tool of the chain), so it is
reachable by construction (q_star is NOT the only solution; a solver succeeds with any in-limit q
reaching all the poses). With `--set coll`, q_star is also self-collision-free (spheres; see
problem.SelfCollision), so a target is reachable without colliding; random 23-DOF configurations
often collide, and those are rejected. Categories:
  free            q_star uniform in the limits (control group)
  at_limit        a few random joints pinned exactly at a limit, the rest uniform
  near_limit      same, pinned 2% of the range inside the limit
  singular        lowest 2% sigma_min of the stacked Jacobian (of a 20000 draw pool)
  limit_singular  limit-pinned q with the lowest 5% sigma_min of a 20000 draw pool
(how many joints get pinned is per chain; bimanual pins more since it has 23 joints)
"""

import argparse

import numpy as np

import problem

SEED = 0
POOL = 20000


def pin(rng, n, lo, hi, k_choices, inset):
    dof = len(lo)
    q = rng.uniform(lo, hi, (n, dof))
    pinned = np.zeros(n, dtype=int)
    span = hi - lo
    for i in range(n):
        k = int(rng.choice(k_choices))
        idx = rng.choice(dof, size=k, replace=False)
        upper = rng.integers(0, 2, size=k).astype(bool)
        q[i, idx] = np.where(upper, hi[idx] - inset * span[idx], lo[idx] + inset * span[idx])
        pinned[i] = k
    return q, pinned


def sigma_min(fk, q: np.ndarray) -> np.ndarray:
    J = np.stack([fk.jacobian(row) for row in q])  # (n, 6 * tools, dof)
    return np.linalg.svd(J, compute_uv=False)[:, -1]


def free_mask(checker, q: np.ndarray) -> np.ndarray:
    if checker is None:
        return np.ones(len(q), dtype=bool)
    return np.array([not checker.collides(row) for row in q])


def take(checker, draw, n):
    """n valid draws (collision-free if a checker is given); also returns the acceptance rate."""
    qs, ks, kept, tried = [], [], 0, 0
    for _ in range(500):
        q, k = draw(max(4 * n, 256))
        keep = free_mask(checker, q)
        qs.append(q[keep])
        ks.append(k[keep])
        kept += int(keep.sum())
        tried += len(q)
        if kept >= n:
            return np.concatenate(qs)[:n], np.concatenate(ks)[:n], kept / tried
    raise RuntimeError(f"only {kept}/{tried} draws were collision-free; not enough for {n} targets")


def lowest_sigma(fk, checker, q_pool, k_pool, n):
    keep = free_mask(checker, q_pool)
    q_pool, k_pool = q_pool[keep], k_pool[keep]
    if len(q_pool) < n:
        raise RuntimeError(f"pool has only {len(q_pool)} valid draws for {n} targets")
    order = np.argsort(sigma_min(fk, q_pool))[:n]
    return q_pool[order], k_pool[order], keep.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", choices=sorted(problem.CHAINS), default="arm")
    parser.add_argument("--n", type=int, default=200, help="targets per category")
    parser.add_argument("--set", default="", choices=("", "coll"), dest="tset",
                        help="coll: ground-truth configurations are self-collision-free")
    args = parser.parse_args()
    chain = problem.CHAINS[args.chain]

    fk = problem.ChainFk(args.chain)
    checker = problem.SelfCollision(fk) if args.tset == "coll" else None
    measure = problem.SelfCollision(fk) if checker is None else checker  # clearance for the report
    lo, hi = problem.limits(fk.joints)
    rng = np.random.default_rng(SEED)
    n = args.n
    zeros = lambda m: np.zeros(m, dtype=int)  # noqa: E731

    q_free, k_free, a_free = take(checker, lambda m: (rng.uniform(lo, hi, (m, fk.dof)), zeros(m)), n)
    q_lim, k_lim, a_lim = take(checker, lambda m: pin(rng, m, lo, hi, chain.pin_counts, 0.0), n)
    q_near, k_near, a_near = take(checker, lambda m: pin(rng, m, lo, hi, chain.pin_counts, 0.02), n)
    pool = rng.uniform(lo, hi, (POOL, fk.dof))
    q_sing, k_sing, a_sing = lowest_sigma(fk, checker, pool, zeros(POOL), n)
    pool_pin, k_pool = pin(rng, POOL, lo, hi, chain.pin_counts_hard, 0.0)
    q_ls, k_ls, a_ls = lowest_sigma(fk, checker, pool_pin, k_pool, n)

    groups = {
        "free": (q_free, k_free, a_free),
        "at_limit": (q_lim, k_lim, a_lim),
        "near_limit": (q_near, k_near, a_near),
        "singular": (q_sing, k_sing, a_sing),
        "limit_singular": (q_ls, k_ls, a_ls),
    }
    names = list(groups)
    q_star = np.concatenate([groups[g][0] for g in names])
    n_pinned = np.concatenate([groups[g][1] for g in names])
    category = np.concatenate([np.full(n, i) for i in range(len(names))])
    target = np.stack([fk.pose(q) for q in q_star])     # (N, tools, 7)
    sig = sigma_min(fk, q_star)
    clearance = np.array([measure.clearance(q) for q in q_star])

    path = problem.target_path(args.chain, args.tset)
    path.parent.mkdir(exist_ok=True)
    np.savez(path, q_star=q_star, target=target, category=category, category_names=np.array(names),
             sigma_min=sig, n_pinned=n_pinned, lo=lo, hi=hi, joints=np.array(fk.joints),
             tools=np.array(fk.tools), chain=args.chain, tset=args.tset, clearance=clearance,
             pos_tol=problem.POS_TOL, rot_tol=problem.ROT_TOL)

    # Sanity row for report.py: handing back q_star itself must score 100% (checks the scoring).
    results = problem.REPO / "ik_benchmark" / "results"
    results.mkdir(exist_ok=True)
    suffix = f"_{args.tset}" if args.tset else ""
    np.savez(results / f"oracle_{args.chain}{suffix}.npz", q=q_star, seconds=np.zeros(len(q_star)),
             chain=args.chain, tset=args.tset, name="oracle (q_star)")

    print(f"wrote {path}: {len(q_star)} targets, {fk.dof} DOF, tools {list(fk.tools)}, "
          f"{measure.n_spheres} spheres, {measure.n_pairs} checked pairs")
    print(f"  {'':<15} {'sigma_min med':>14} {'clearance med':>14}  {'min':>8}  {'draws collision-free':>21}")
    for i, g in enumerate(names):
        s, c = sig[category == i], clearance[category == i]
        print(f"  {g:<15} {np.median(s):>14.4f} {np.median(c):>14.4f}  {c.min():>8.4f}  {groups[g][2]:>20.1%}")


if __name__ == "__main__":
    main()
