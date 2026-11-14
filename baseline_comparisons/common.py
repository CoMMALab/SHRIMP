"""Shared helpers for the FK comparison scripts (stdlib + numpy only; runs in any env).

Every script times FK of one link of one robot and writes results/<robot>/<library>.json;
compare.py reads them. All libraries load the *same* URDF file (robots/<robot>/) and use the
same joint configs, mapped by joint name (each library orders joints differently).
"""

import argparse
import json
import math
import pathlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"

BATCH_SIZES = [1, 16, 256, 4096, 65536, 262144, 1048576]
CPU_MAX_BATCH = 4096  # slow CPU baselines stop here
N_ACCURACY = 64


@dataclass(frozen=True)
class RobotSpec:
    name: str
    urdf: str            # relative to REPO
    base_link: str
    link: str            # link whose pose FK is timed/compared
    joints: tuple        # sampled joints, in this order (all other actuated joints stay at 0)
    extra_frames: tuple = ()  # more frames to request from libraries that prune joints not
                              # on the path to a requested frame (cuRobo), so all `joints` survive

    @property
    def urdf_path(self) -> pathlib.Path:
        return REPO / self.urdf

    @property
    def dof(self) -> int:
        return len(self.joints)

    def limits(self) -> tuple:
        """(lo, hi) arrays for self.joints from the URDF. Missing/huge limits are clamped
        (revolute/continuous to +-pi, prismatic to +-1 m, e.g. the planar base)."""
        root = ET.parse(self.urdf_path).getroot()
        info = {}
        for j in root.findall("joint"):
            lim = j.find("limit")
            lo = float(lim.get("lower")) if lim is not None and lim.get("lower") else None
            hi = float(lim.get("upper")) if lim is not None and lim.get("upper") else None
            info[j.get("name")] = (j.get("type"), lo, hi)
        lo_out, hi_out = [], []
        for name in self.joints:
            jtype, lo, hi = info[name]
            cap = 1.0 if jtype == "prismatic" else math.pi
            lo = -cap if lo is None else max(lo, -cap)
            hi = cap if hi is None else min(hi, cap)
            lo_out.append(lo)
            hi_out.append(hi)
        return np.array(lo_out), np.array(hi_out)


ROBOTS = {
    "panda": RobotSpec(
        name="panda",
        urdf="robots/panda/panda_spherized.urdf",
        base_link="panda_link0",
        link="panda_hand",
        joints=tuple(f"panda_joint{i}" for i in range(1, 8)),
    ),
    "rby1": RobotSpec(
        name="rby1",
        urdf="robots/rby1/rby1.urdf",
        base_link="world",
        link="left_arm/tool",
        extra_frames=("right_arm/tool",),
        joints=(("base/base_x", "base/base_y", "base/base_rz")
                + tuple(f"torso/torso_{i}" for i in range(6))
                + tuple(f"left_arm/left_arm_{i}" for i in range(7))
                + tuple(f"right_arm/right_arm_{i}" for i in range(7))),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=sorted(ROBOTS), default="panda")
    parser.add_argument("--task", choices=("fk", "jac"), default="fk")  # compare.py only
    args = parser.parse_args()
    args.spec = ROBOTS[args.robot]
    return args


def parse_robot() -> RobotSpec:
    return parse_args().spec


def jac_to_spec_order(jac: np.ndarray, column_of: list) -> np.ndarray:
    """jac: (N, 3, n_cols) -> (N, 3 * dof) row-major with columns in spec.joints order;
    column_of[k] is the library's column for spec.joints[k]."""
    return jac[:, :, column_of].reshape(jac.shape[0], -1)


def sample_configs(spec: RobotSpec, n: int, seed: int) -> np.ndarray:
    lo, hi = spec.limits()
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, spec.dof))


def accuracy_configs(spec: RobotSpec) -> np.ndarray:
    return sample_configs(spec, N_ACCURACY, seed=123)


def embed(spec: RobotSpec, q: np.ndarray, library_joint_names, n_total: int = None) -> np.ndarray:
    """Place sampled columns (spec.joints order) into a library's own joint order; any
    joint of the library not in spec.joints (e.g. gripper fingers) stays 0."""
    names = list(library_joint_names)
    full = np.zeros((q.shape[0], n_total or len(names)), dtype=np.float64)
    full[:, [names.index(n) for n in spec.joints]] = q
    return full


def to_pose_row(position, quat_wxyz) -> list:
    """[x, y, z, qw, qx, qy, qz] with the quaternion sign canonicalized so qw >= 0."""
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat[0] < 0:
        quat = -quat
    return [float(v) for v in (*np.asarray(position, dtype=np.float64), *quat)]


def median_us(fn, n_calls: int, n_warmup: int = 20) -> float:
    """Median wall-clock microseconds per call. fn must block until its result is ready."""
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_calls):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)) * 1e6


def amortized_us(fn, sync, n_inner: int = 50, n_runs: int = 9, n_warmup: int = 3) -> float:
    """Per-call microseconds with launch/sync overhead amortized: n_inner back-to-back
    asynchronous calls, one sync, divided by n_inner; median of n_runs. This is crocket's own
    protocol (benchmarks/README.md), used for every GPU library so they're comparable.
    fn() launches work and returns something; sync(last_result) blocks until it is done."""
    for _ in range(n_warmup):
        sync(fn())
    samples = []
    for _ in range(n_runs):
        start = time.perf_counter()
        last = None
        for _ in range(n_inner):
            last = fn()
        sync(last)
        samples.append((time.perf_counter() - start) / n_inner)
    return float(np.median(samples)) * 1e6


def save(spec: RobotSpec, library: str, device: str, precision: str, latency_us: float,
         per_config_us_by_batch: dict, accuracy_poses: list, note: str = "",
         build_seconds: float = None, task: str = "fk") -> None:
    """task "jac": accuracy_poses holds (N, 3*dof) row-major position Jacobians in spec.joints
    order, saved as jac_<library>.json. latency_us: synchronized wall-clock of one batch-1 call (None if not measured).
    per_config_us_by_batch: amortized (see amortized_us) time per configuration."""
    out_dir = RESULTS / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "robot": spec.name,
        "dof": spec.dof,
        "library": library,
        "device": device,
        "precision": precision,
        "link": spec.link,
        "latency_us_batch1": latency_us,
        "per_config_us_by_batch": {str(k): v for k, v in per_config_us_by_batch.items()},
        "accuracy_poses": accuracy_poses,
        "build_seconds": build_seconds,
        "note": note,
        "task": task,
    }
    path = out_dir / (f"{library}.json" if task == "fk" else f"{task}_{library}.json")
    path.write_text(json.dumps(record, indent=1))
    lat = f"{latency_us:.2f} us" if latency_us is not None else "n/a"
    print(f"[{spec.name}/{library}] batch-1 wall-clock {lat}; wrote {path}")
