"""RBY1 hard-IK problem definitions + ground-truth FK (CRAB's float64 CPU kernels; crab env).

Each problem: reach a pose for every tool in `tools` with the chain's joints; every other joint
stays at 0.
  arm       left_arm_0..6 (7 DOF, one redundant dof), tool: left. Genuinely hard at limits: the
            self-motion manifold is 1-D and often short inside the limits.
  full      planar base (x, y, rz) + torso (6) + left arm (7) = 16 DOF, tool: left.
  bimanual  planar base + torso + both arms = 23 DOF, tools: left AND right at once (12 task dims).
            The torso and base are shared, so the two arms' targets constrain each other.
Joint limits are the URDF's, except the planar base's +-1000 m, clamped to +-1 m (see `limits`).
A solution is ANY q within the limits whose poses are all within tolerance; it needn't be q_star.
"""

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]            # SHRIMP root (shrimp/, crab/, ik_benchmark/, external/)
sys.path.insert(0, str(REPO))                          # shrimp
sys.path.insert(0, str(REPO / "crab"))                 # crab_codegen, crab_kinematics, ...
sys.path.insert(0, str(REPO / "crab" / "baseline_comparisons"))

import common  # noqa: E402  (baseline_comparisons/common.py: RobotSpec for rby1)

POS_TOL = 1e-3   # m
ROT_TOL = 1e-2   # rad

LEFT, RIGHT = "left_arm/tool", "right_arm/tool"
LEFT_ARM = tuple(f"left_arm/left_arm_{i}" for i in range(7))
RIGHT_ARM = tuple(f"right_arm/right_arm_{i}" for i in range(7))
BASE_TORSO = (("base/base_x", "base/base_y", "base/base_rz")
              + tuple(f"torso/torso_{i}" for i in range(6)))


@dataclass(frozen=True)
class Chain:
    joints: tuple
    tools: tuple
    pin_counts: tuple      # how many joints `at_limit` targets pin
    pin_counts_hard: tuple  # ... and for `limit_singular`
    robot: str = "rby1"    # key into common.ROBOTS


PANDA = tuple(f"panda_joint{i}" for i in range(1, 8))

CHAINS = {
    # VAMP's MBM problems (generate_mbm.py): Panda, goal pose of panda_grasptarget, obstacles in the file
    "mbm": Chain(PANDA, ("panda_grasptarget",), (), (), robot="panda"),
    # same for Baxter: all 14 joints, pose of right_gripper (VAMP's end effector); the left arm is free
    "mbm_baxter": Chain(tuple(common.ROBOTS["baxter"].joints), ("right_gripper",), (), (), robot="baxter"),
    "arm": Chain(LEFT_ARM, (LEFT,), (1, 2, 3), (2, 3)),
    "full": Chain(BASE_TORSO + LEFT_ARM, (LEFT,), (1, 2, 3), (2, 3)),
    "bimanual": Chain(BASE_TORSO + LEFT_ARM + RIGHT_ARM, (LEFT, RIGHT), (3, 5, 8), (5, 8)),
}


def limits(joints, robot: str = "rby1") -> tuple:
    """(lo, hi) arrays. Prismatic joints (the planar base) clamp to +-1 m, revolute to +-pi when
    the URDF gives none; otherwise the URDF's own limits."""
    root = ET.parse(common.ROBOTS[robot].urdf_path).getroot()
    info = {j.get("name"): (j.get("type"), j.find("limit")) for j in root.findall("joint")}
    lo, hi = [], []
    for name in joints:
        jtype, lim = info[name]
        cap = 1.0 if jtype == "prismatic" else math.pi
        lo.append(max(float(lim.get("lower")), -cap) if lim is not None and lim.get("lower") else -cap)
        hi.append(min(float(lim.get("upper")), cap) if lim is not None and lim.get("upper") else cap)
    return np.array(lo), np.array(hi)


def rot_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Rotation angle between wxyz quaternion rows (float32-safe chord formula)."""
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    d = np.minimum(np.linalg.norm(q1 - q2, axis=-1), np.linalg.norm(q1 + q2, axis=-1))
    return 4.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))


class ChainFk:
    """Float64 poses and stacked Jacobian of the chain's tools as functions of its joints."""

    def __init__(self, chain: str):
        import crab_model  # noqa: F401 -- must precede any symforce.symbolic import
        from crab_io.urdf import load_urdf, urdf_to_robot
        from crab_kinematics.derived import q_to_ee_pose, q_to_ee_jacobian_analytic
        from crusty_jit import build_simple_jit_function

        self.robot = CHAINS[chain].robot
        spec = common.ROBOTS[self.robot]
        self.joints, self.tools = CHAINS[chain].joints, CHAINS[chain].tools
        self.dof = len(self.joints)
        urdf = load_urdf(str(spec.urdf_path))
        robots = [urdf_to_robot(urdf, [tool]).finalize() for tool in self.tools]  # one per tool
        self.nq = robots[0].nq
        self.q_index = [robots[0].joints[n].id for n in self.joints]
        self._pose = [build_simple_jit_function(rb, q_to_ee_pose, name=f"ik_ee_pose_{self.robot}_{i}")
                      for i, rb in enumerate(robots)]
        self._jac = [build_simple_jit_function(rb, q_to_ee_jacobian_analytic, name=f"ik_ee_jac6_{self.robot}_{i}")
                     for i, rb in enumerate(robots)]

    def _full(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros(self.nq)
        out[self.q_index] = q
        return out

    def pose(self, q: np.ndarray) -> np.ndarray:
        """(n_tools, 7): [x, y, z, qw, qx, qy, qz] per tool, qw >= 0."""
        full, rows = self._full(q), []
        for kernel in self._pose:
            o = np.asarray(kernel(full))  # SymForce storage [qx, qy, qz, qw, x, y, z]
            quat = np.array([o[3], o[0], o[1], o[2]])
            rows.append(np.concatenate([o[4:7], quat if quat[0] >= 0 else -quat]))
        return np.stack(rows)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """(6 * n_tools, dof): per tool rows (v; w) world-aligned, columns in self.joints order."""
        full = self._full(q)
        blocks = [np.asarray(k(full)).reshape(6, self.nq)[:, self.q_index] for k in self._jac]
        return np.vstack(blocks)


def target_path(chain: str, tset: str = "") -> Path:
    """Targets file for a chain; tset "coll" = built from collision-free configurations."""
    return REPO / "ik_benchmark" / "targets" / f"{CHAINS[chain].robot}_{chain}{'_' + tset if tset else ''}.npz"


def obstacles_of(t, i: int):
    """Problem i's world obstacles from a targets file, or None when the file has none (MBM files do):
    {"sph": (k, 4) [x y z r], "box": (k, 15) [center, R row-major, half extents],
     "cap": (k, 7) [p1, p2, r]} (capsule = segment p1-p2 swept by r; VAMP's "cylinder" is this capsule)."""
    if "n_sph" not in t.files:
        return None
    return {"sph": t["sph"][i, :t["n_sph"][i]], "box": t["box"][i, :t["n_box"][i]],
            "cap": t["cap"][i, :t["n_cap"][i]]}


def world_clearance(centers: np.ndarray, radii: np.ndarray, obs) -> float:
    """Smallest gap between the robot spheres and the obstacles (< 0 = penetrating; for a sphere whose
    centre is inside a box the gap is -radius, not the exact penetration)."""
    gaps = [np.inf]
    if len(obs["sph"]):
        d = np.linalg.norm(centers[:, None] - obs["sph"][None, :, :3], axis=2)
        gaps.append(float(np.min(d - radii[:, None] - obs["sph"][None, :, 3])))
    for b in obs["box"]:
        local = (centers - b[:3]) @ b[3:12].reshape(3, 3)         # R^T (c - p), row-vector form
        out = np.linalg.norm(np.maximum(np.abs(local) - b[12:15], 0.0), axis=1)
        gaps.append(float(np.min(out - radii)))
    for c in obs["cap"]:
        seg = c[3:6] - c[:3]
        s = np.clip((centers - c[:3]) @ seg / (seg @ seg), 0.0, 1.0)
        d = np.linalg.norm(centers - (c[:3] + s[:, None] * seg), axis=1)
        gaps.append(float(np.min(d - radii - c[6])))
    return min(gaps)


def srdf_disabled(spec) -> set:
    """Link pairs (frozensets) the SRDF disables for collision checking (stdlib only)."""
    srdf = sorted(spec.urdf_path.parent.glob("*.srdf"))[0]   # rby1.srdf / panda.srdf
    return {frozenset((e.get("link1"), e.get("link2")))
            for e in ET.parse(srdf).getroot().findall("disable_collisions")}


def rigid_key(spec) -> dict:
    """link -> frozenset of the movable joints between it and the root (stdlib only). Two links with
    equal sets never move relative to each other, so their sphere gaps are constant: such pairs can't
    be fixed by any q, and (e.g. Baxter's gripper fingers) may overlap in the model itself."""
    root = ET.parse(spec.urdf_path).getroot()
    up = {j.find("child").get("link"): (j.get("name"), j.get("type"), j.find("parent").get("link"))
          for j in root.findall("joint")}
    out = {}
    for el in root.findall("link"):
        movable, cur = set(), el.get("name")
        while cur in up:
            name, kind, cur = up[cur]
            if kind != "fixed":
                movable.add(name)
        out[el.get("name")] = frozenset(movable)
    return out


def urdf_spheres(spec) -> list:
    """(link, (x, y, z), radius) of every collision sphere on a non-root link, in URDF order (stdlib
    only, so the crocket env can use it). Root-link spheres are skipped, as CRAB's FK kernel does."""
    root = ET.parse(spec.urdf_path).getroot()
    children = {j.find("child").get("link") for j in root.findall("joint")}
    out = []
    for link in root.findall("link"):
        if link.get("name") not in children:
            continue
        for col in link.findall("collision"):
            sph = col.find("geometry/sphere")
            if sph is None:
                continue
            org = col.find("origin")
            xyz = tuple(float(v) for v in org.get("xyz", "0 0 0").split()) if org is not None else (0.0,) * 3
            out.append((link.get("name"), xyz, float(sph.get("radius"))))
    return out


class SelfCollision:
    """Float64 collision check with the URDF's collision SPHERES (its boxes are not checked yet):
    self-collision, plus the world obstacles of an MBM problem when given (`obs`, see obstacles_of).

    Self pairs: spheres of two different links, unless the SRDF lists the link pair as disabled
    (adjacent/welded links). Collision = a pair closer than the sum of radii + margin; the
    clearance is the smallest such gap (negative when colliding). World sphere centres come from
    CRAB's FK kernel, so this shares the FK used everywhere else here."""

    def __init__(self, fk: ChainFk, margin: float = 0.0):
        from crab_codegen.sphere_order import robot_sphere_list
        from crab_io.urdf import load_urdf, urdf_to_robot
        from crab_kinematics.derived import forward_kinematics_spheres
        from crusty_jit import build_simple_jit_function

        spec = common.ROBOTS[fk.robot]
        self.fk, self.margin = fk, margin
        robot = urdf_to_robot(load_urdf(str(spec.urdf_path)), [fk.tools[0]]).finalize()
        spheres = robot_sphere_list(robot)                     # same order as the FK kernel's output
        self._spheres_fk = build_simple_jit_function(robot, forward_kinematics_spheres,
                                                     name=f"ik_spheres_{fk.robot}")
        self.n_spheres = len(spheres)
        self.links = [s["link_name"] for s in spheres]
        by_link = {}
        for i, s in enumerate(spheres):
            by_link.setdefault(s["link_name"], []).append(i)

        disabled = srdf_disabled(spec)
        key = rigid_key(spec)
        links = sorted(by_link)
        pairs = [(i, j) for a in range(len(links)) for b in range(a + 1, len(links))
                 if frozenset((links[a], links[b])) not in disabled and key[links[a]] != key[links[b]]
                 for i in by_link[links[a]] for j in by_link[links[b]]]
        self.pair_i = np.array([p[0] for p in pairs])
        self.pair_j = np.array([p[1] for p in pairs])
        self.n_pairs = len(pairs)

    def clearance(self, q: np.ndarray, obs=None) -> float:
        """Smallest (distance - radius_i - radius_j) over the self pairs (and robot sphere vs obstacle
        gaps when `obs` is given); < margin = colliding."""
        out = np.asarray(self._spheres_fk(self.fk._full(q))).reshape(-1, 4)
        d = np.linalg.norm(out[self.pair_i, :3] - out[self.pair_j, :3], axis=1)
        gap = float(np.min(d - out[self.pair_i, 3] - out[self.pair_j, 3]))
        if obs is not None:
            gap = min(gap, world_clearance(out[:, :3], out[:, 3], obs))
        return gap

    def diagnose(self, q: np.ndarray, obs=None) -> str:
        """Why q collides: the worst self pair (gap, the two links) and the worst obstacle gap."""
        out = np.asarray(self._spheres_fk(self.fk._full(q))).reshape(-1, 4)
        d = np.linalg.norm(out[self.pair_i, :3] - out[self.pair_j, :3], axis=1)
        gap = d - out[self.pair_i, 3] - out[self.pair_j, 3]
        k = int(np.argmin(gap))
        text = (f"self gap {gap[k]:.4f} between {self.links[self.pair_i[k]]} and {self.links[self.pair_j[k]]}"
                f" ({int((gap < 0).sum())} pairs penetrating)")
        if obs is not None:
            text += f"; world gap {world_clearance(out[:, :3], out[:, 3], obs):.4f}"
        return text

    def collides(self, q: np.ndarray, obs=None) -> bool:
        return self.clearance(q, obs) < self.margin
