"""Turn VAMP's MBM problems (Panda or Baxter) into an IK benchmark: the goal configuration's end-effector
pose (panda_grasptarget / right_gripper) is the target, the problem's obstacles are the world. Run in
the crab env, with VAMP's problems.pkl copied to ik_benchmark/targets/<robot>_problems.pkl:
    python ik_benchmark/generate_mbm.py [--robot panda|baxter] [--pkl PATH]
(chains: mbm = Panda 7 DOF; mbm_baxter = all 14 Baxter joints, the left arm free)

Per valid problem: q_goal = goals[0] (the first goal VAMP stores), target = FK(q_goal) with CRAB's
float64 FK, so the pose is reachable AND reachable collision-free (checked here against self-collision
and the obstacles; problems whose goal fails this checker are dropped and counted). A solution is any
in-limit q reaching the pose without colliding (report.py judges; solvers don't see obstacles yet).
Obstacles follow vamp.problem_dict_to_vamp: spheres; boxes (R = Rz(phi) Ry(theta) Rx(rho) from
orientation_euler_xyz); cylinders are CAPSULES (segment p1-p2 swept by the radius; VAMP's "cylinder" is this capsule), except in the `box`
family where they become boxes with half extents [r, r, length / 2]. Categories = MBM families.
`start` (the problem's start configuration) is saved too, for warm-start solves (--warm).
"""

import argparse
import pickle

import numpy as np

import problem

CHAIN = {"panda": "mbm", "baxter": "mbm_baxter"}   # --robot -> problem.CHAINS key


def rot_xyz(rho: float, theta: float, phi: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = np.cos(rho), np.sin(rho), np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def obstacles(p: dict, family: str) -> tuple:
    sph = [[*map(float, s["position"]), float(s["radius"])] for s in p["sphere"]]
    box, cap = [], []
    for b in p["box"]:
        r = rot_xyz(*map(float, b["orientation_euler_xyz"]))
        box.append([*map(float, b["position"]), *r.ravel(), *map(float, b["half_extents"])])
    for c in p["cylinder"]:
        pos, r = np.array(c["position"], dtype=float), rot_xyz(*map(float, c["orientation_euler_xyz"]))
        rad, half = float(c["radius"]), float(c["length"]) / 2
        if family == "box":
            box.append([*pos, *r.ravel(), rad, rad, half])
        else:
            axis = r @ np.array([0.0, 0.0, half])
            cap.append([*(pos + axis), *(pos - axis), rad])
    return sph, box, cap


def pad(rows: list, width: int, m: int) -> np.ndarray:
    out = np.zeros((len(rows), m, width))
    for i, r in enumerate(rows):
        if len(r):
            out[i, :len(r)] = np.array(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=sorted(CHAIN), default="panda")
    parser.add_argument("--pkl", default=None, help="default: ik_benchmark/targets/<robot>_problems.pkl")
    args = parser.parse_args()
    chain = CHAIN[args.robot]

    pkl = args.pkl or str(problem.REPO / "ik_benchmark" / "targets" / f"{args.robot}_problems.pkl")
    with open(pkl, "rb") as f:
        data = pickle.load(f)["problems"]
    fk = problem.ChainFk(chain)
    checker = problem.SelfCollision(fk)
    lo, hi = problem.limits(fk.joints, fk.robot)

    names = list(data)
    keep = {k: [] for k in ("q", "start", "cat", "sph", "box", "cap", "clr", "idx")}
    print(f"{'family':<34}{'valid':>6}{'kept':>6}{'out of limits':>15}{'self collides':>15}{'env collides':>15}")
    for ci, family in enumerate(names):
        valid = [p for p in data[family] if p["valid"]]
        bad_lim = bad_col = bad_self = 0
        worst = 0.0                                    # largest joint-limit violation (rad)
        for p in valid:
            q = np.asarray(p["goals"][0], dtype=float)
            viol = float(np.max(np.maximum(lo - q, q - hi)))
            if viol > 1e-4:
                bad_lim += 1
                worst = max(worst, viol)
                continue
            sph, box, cap = obstacles(p, family)
            obs = {"sph": np.array(sph).reshape(-1, 4), "box": np.array(box).reshape(-1, 15),
                   "cap": np.array(cap).reshape(-1, 7)}
            clr = checker.clearance(q, obs)
            if clr < 0:
                if bad_self + bad_col == 0:            # why the first dropped goal of the family collides
                    print(f"  [{family}] first dropped goal: {checker.diagnose(q, obs)}")
                if checker.clearance(q) < 0:
                    bad_self += 1                      # the robot collides with itself
                else:
                    bad_col += 1                       # ... with the obstacles
                continue
            for k, v in (("q", q), ("start", np.asarray(p["start"], dtype=float)), ("cat", ci), ("clr", clr),
                         ("idx", int(p.get("index", len(keep["idx"]))))):
                keep[k].append(v)
            keep["sph"].append(sph)
            keep["box"].append(box)
            keep["cap"].append(cap)
        print(f"{family:<34}{len(valid):>6}{len(valid) - bad_lim - bad_col - bad_self:>6}{bad_lim:>15}"
              f"{bad_self:>15}{bad_col:>15}   worst limit violation {worst:.4f} rad")

    if not keep["q"]:
        raise SystemExit("every goal was dropped: see the table above (limits / self / env columns)")
    q_star = np.array(keep["q"])
    target = np.stack([fk.pose(q) for q in q_star])               # (N, 1, 7)
    sig = np.linalg.svd(np.stack([fk.jacobian(q) for q in q_star]), compute_uv=False)[:, -1]
    category = np.array(keep["cat"])
    counts = {k: np.array([len(r) for r in keep[k]]) for k in ("sph", "box", "cap")}

    path = problem.target_path(chain)
    path.parent.mkdir(exist_ok=True)
    np.savez(path, q_star=q_star, start=np.array(keep["start"]), target=target, category=category,
             category_names=np.array(names), sigma_min=sig, lo=lo, hi=hi, joints=np.array(fk.joints),
             tools=np.array(fk.tools), chain=chain, tset="", clearance=np.array(keep["clr"]),
             problem_index=np.array(keep["idx"]), pos_tol=problem.POS_TOL, rot_tol=problem.ROT_TOL,
             sph=pad(keep["sph"], 4, max(1, counts["sph"].max())), n_sph=counts["sph"],
             box=pad(keep["box"], 15, max(1, counts["box"].max())), n_box=counts["box"],
             cap=pad(keep["cap"], 7, max(1, counts["cap"].max())), n_cap=counts["cap"])

    results = problem.REPO / "ik_benchmark" / "results"
    results.mkdir(exist_ok=True)
    np.savez(results / f"oracle_{chain}.npz", q=q_star, seconds=np.zeros(len(q_star)), chain=chain, tset="",
             name="oracle (q_goal)")

    print(f"\nwrote {path}: {len(q_star)} problems, {fk.dof} DOF, tool {list(fk.tools)}, "
          f"{checker.n_spheres} spheres / {checker.n_pairs} self pairs")
    print(f"  {'':<34}{'sigma_min med':>14}{'clearance med':>15}{'min':>9}{'obstacles med':>15}")
    for i, family in enumerate(names):
        m = category == i
        if not m.any():
            print(f"  {family:<34}  (no problems kept)")
            continue
        n_obs = counts["sph"][m] + counts["box"][m] + counts["cap"][m]
        print(f"  {family:<34}{np.median(sig[m]):>14.4f}{np.median(np.array(keep['clr'])[m]):>15.4f}"
              f"{np.min(np.array(keep['clr'])[m]):>9.4f}{np.median(n_obs):>15.0f}")


if __name__ == "__main__":
    main()
